from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls
from PIL import Image


class _Tensor:
    def __init__(self, array: np.ndarray, **options: Any) -> None:
        self.array, self.options = array, options

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array

    @classmethod
    def from_numpy(cls, array, copy=True, **options):
        return cls(array.copy() if copy else array, **options)


class _TextEncoder:
    """Stands in for pyneat.Model and its runner: returns the next of the given embeddings for each prompt it runs,
    and keeps the MPKs it loaded, the tensors it ran and whether it was closed."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings, self.mpks, self.inputs, self.closed = embeddings, [], [], False

    def __call__(self, mpk: str, options: object) -> _TextEncoder:
        self.mpks.append(mpk)
        return self

    def build(self, inputs: list[_Tensor]) -> _TextEncoder:
        return self

    def run(self, inputs: list[_Tensor], timeout_ms: int) -> list[_Tensor]:
        self.inputs.append(inputs[0])
        return [_Tensor(self.embeddings[len(self.inputs) - 1][None])]

    def close(self) -> None:
        self.closed = True


def _text_encoder(script: ModuleType, monkeypatch: pytest.MonkeyPatch, embeddings: np.ndarray) -> _TextEncoder:
    encoder = _TextEncoder(embeddings)
    pyneat = MagicMock()
    pyneat.Model = encoder
    pyneat.Tensor.from_numpy = lambda array, copy=True, **options: _Tensor(array.copy(), **options)
    monkeypatch.setattr(script, "pyneat", pyneat)
    return encoder


def test_mpk_arguments_name_exactly_both_encoders(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    mpks = script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz", "text_encoder=/data/text_mpk.tar.gz"])
    assert mpks == {"image_encoder": Path("/data/image_mpk.tar.gz"), "text_encoder": Path("/data/text_mpk.tar.gz")}
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and text_encoder=<path>"):
        script.parse_mpks(["model=/data/siglip2_mpk.tar.gz"])


def test_class_scores_compare_the_image_embedding_against_every_class(load_script):
    script = load_script("accuracy")
    rng = np.random.default_rng(0)
    image = rng.normal(size=script.EMBED_DIM).astype(np.float32)
    classes = script.normalize(rng.normal(size=(1000, script.EMBED_DIM)).astype(np.float32))
    scores = script.class_scores([_Tensor(image[None] * 3)], classes)
    assert scores.shape == (1000,)
    np.testing.assert_allclose(scores, classes @ (image / np.linalg.norm(image)), rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="one image embedding"):
        script.class_scores([_Tensor(image), _Tensor(np.zeros(1000))], classes)
    with pytest.raises(ValueError, match="expected an image embedding"):
        script.class_scores([_Tensor(np.zeros(1000))], classes)


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_normalization_rejects_invalid_embeddings(values, load_script):
    with pytest.raises(ValueError, match="invalid embedding"):
        load_script("accuracy").normalize(np.array(values, np.float32))


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_the_text_check_holds_each_prompts_tokens_rows_and_embedding(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    check = script.load_text_check()
    count = len(check["prompts"])
    assert check["token_ids"].shape == (count, 64)
    assert check["rows"].shape == (len(check["row_ids"]), script.EMBED_DIM)
    assert check["text_embeds"].shape == (count, script.EMBED_DIM)
    assert set(check["token_ids"].ravel().tolist()) == set(check["row_ids"].tolist()), "every token ID has its row"
    assert np.isfinite(check["rows"]).all() and np.isfinite(check["text_embeds"]).all()
    for ids in check["token_ids"]:
        # The prompt's tokens, then <eos> (1), then <pad> (0) up to 64 tokens: SigLIP 2's tokenizer adds no <bos>, and
        # a prompt's own tokens are never <pad>, <eos>, <bos> or <unk> (0 to 3).
        end = int(np.argmax(ids == 1))
        assert end > 0 and (ids[:end] > 3).all() and ids[end] == 1 and (ids[end + 1 :] == 0).all()
    # A changed file is refused until the script names its SHA-256.
    content = bytearray(script.TEXT_CHECK.read_bytes())
    content[len(content) // 2] ^= 1
    (tmp_path / script.TEXT_CHECK.name).write_bytes(content)
    with pytest.raises(ValueError, match="TEXT_CHECK_SHA256"):
        script.load_text_check(tmp_path / script.TEXT_CHECK.name)


def test_the_text_check_passes_the_onnxs_embeddings_and_fails_others(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    check = script.load_text_check()
    expected = check["text_embeds"]
    # A longer embedding in the same direction passes. Each prompt reaches the encoder as its token rows, in a tensor
    # declared HWC as Neat sizes the input, and the check closes the runner.
    encoder = _text_encoder(script, monkeypatch, expected * 3)
    assert script.check_text_encoder(Path("text_mpk.tar.gz")) == pytest.approx([1.0] * len(expected), abs=1e-5)
    assert encoder.mpks == ["text_mpk.tar.gz"]
    assert encoder.closed, "the check closes the text encoder's runner"
    rows = dict(zip(check["row_ids"].tolist(), check["rows"]))
    for given, ids in zip(encoder.inputs, check["token_ids"], strict=True):
        assert given.array.shape == (1, 64, script.EMBED_DIM)
        assert given.options["layout"] == script.pyneat.TensorLayout.HWC
        np.testing.assert_array_equal(given.array[0], [rows[token] for token in ids.tolist()])
    # One prompt embedded as another prompt, or as NaN, fails.
    for stray in (expected[1], np.nan):
        embeddings = expected.copy()
        embeddings[0] = stray
        _text_encoder(script, monkeypatch, embeddings)
        with pytest.raises(ValueError, match="strays from the FP32 ONNX"):
            script.check_text_encoder(Path("text_mpk.tar.gz"))


def test_a_stray_text_encoder_stops_the_stage_before_any_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    encoder = _text_encoder(script, monkeypatch, -script.load_text_check()["text_embeds"])
    arguments = [
        "--mpk",
        "image_encoder=image.tar.gz",
        "--mpk",
        "text_encoder=text.tar.gz",
        "--dataset",
        f"imagenet-2012-val={tmp_path}",
        "--resources",
        str(tmp_path / "package"),
        "--output",
        str(tmp_path / "report"),
    ]
    # The dataset folder is empty, so reading it would fail otherwise: the check ran first.
    with pytest.raises(ValueError, match="strays from the FP32 ONNX"):
        script.main(arguments)
    assert encoder.mpks == ["text.tar.gz"]
    assert not (tmp_path / "report").exists()


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_class_vectors_are_generated_with_the_compiled_encoder_and_normalized_per_prompt(monkeypatch, load_script):
    from types import SimpleNamespace

    script = load_script("accuracy")
    monkeypatch.setattr(script, "EMBED_DIM", 3)
    monkeypatch.setattr(script, "TEXT_LENGTH", 4)
    monkeypatch.setattr(script, "TEMPLATES", ("photo {}", "drawing {}"))
    table = np.zeros((100, 3), np.float32)
    table[:4, 0] = np.arange(4)
    features = np.array([[4, 0, 0], [0, 1, 0], [-3, 0, 0], [0, 0, 9]], np.float32)
    given = []

    class Tokenizer:
        def encode_batch(self, texts):
            given.extend(texts)
            offset = 0 if texts[0].endswith("cat") else 2
            return [SimpleNamespace(ids=[offset + i, 99, 99, 99]) for i in range(2)]

    class Runner:
        closed = False

        def run(self, inputs, timeout_ms):
            (tensor,) = inputs
            assert tensor.options["layout"] == script.pyneat.TensorLayout.HWC
            feature = features[int(tensor.array[0, 0, 0])]
            return [_Tensor(feature[None])]

        def close(self):
            self.closed = True

    runner = Runner()
    monkeypatch.setattr(script.pyneat, "Tensor", _Tensor)
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: SimpleNamespace(build=lambda inputs: runner))
    monkeypatch.setattr(script, "tensor_options", lambda: None)
    actual = script.build_class_embeddings(Path("text.tar.gz"), Tokenizer(), table, ("cat", "dog"))
    np.testing.assert_allclose(actual, np.array([[1, 1, 0], [-1, 0, 1]]) / np.sqrt(2), atol=1e-6)
    assert given == ["photo cat", "drawing cat", "photo dog", "drawing dog"]
    assert runner.closed


def test_accuracy_uses_all_imagenet_classes(load_script):
    script = load_script("accuracy")
    assert len(script.CLASS_NAMES) == 1000
    assert script.CLASS_NAMES[0] == "tench" and script.CLASS_NAMES[-1] == "toilet paper"


def test_accuracy_uses_big_visions_81_canonicalized_imagenet_templates(load_script):
    script = load_script("accuracy")
    assert len(script.TEMPLATES) == 81
    texts = script.class_prompts("American robin, Turdus migratorius")
    assert "a photo of a american robin" in texts
    assert texts[-1] == "american robin"
    assert not any("." in text or "," in text for text in texts)


def test_accuracy_preprocessing_uses_the_big_vision_resize_and_center_crop(load_script):
    accuracy = load_script("accuracy")
    # At 256x256 the resize is identity. Cropping removes the border, rather than squeezing it into the image.
    pixels = np.zeros((256, 256, 3), np.uint8)
    pixels[16:240, 16:240] = (128, 192, 255)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    result = accuracy.preprocess_image(stream.getvalue())
    assert result.shape == (224, 224, 3) and result.dtype == np.float32
    expected = np.broadcast_to((np.array([128, 192, 255], np.float32) / 255 - 0.5) / 0.5, result.shape)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_accuracy_resize_does_not_antialias_before_sampling(load_script):
    accuracy = load_script("accuracy")
    # At 768 -> 256, half-pixel bilinear centers land exactly on source pixels 1, 4, 7, ... .
    # An antialiasing filter would average these with their neighbors.
    pixels = np.zeros((768, 768, 3), np.uint8)
    pixels[1::3, 1::3] = 255
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    np.testing.assert_array_equal(accuracy.preprocess_image(stream.getvalue()), np.ones((224, 224, 3)))


def test_text_inputs_use_packaged_weights_without_a_checkpoint(tmp_path, monkeypatch, load_script):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    script = load_script("accuracy")
    monkeypatch.setattr(script, "EMBED_DIM", 2)
    directory = tmp_path / "text_encoder"
    directory.mkdir()
    (directory / "tokenizer.json").write_text("{}")
    vocab = 256000
    weights = np.arange(vocab * 2, dtype=np.float32).reshape(vocab, 2)
    np.save(directory / "token_embeddings.npy", weights, allow_pickle=False)
    tokenizer = MagicMock()

    def from_file(path):
        assert Path(path).read_text() == "{}"
        return tokenizer

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=from_file)))
    for name in ("torch", "safetensors", "huggingface_hub"):
        monkeypatch.setitem(sys.modules, name, None)
    actual_tokenizer, table = script.load_text_inputs(tmp_path)
    assert actual_tokenizer is tokenizer
    assert isinstance(table, np.memmap)
    np.testing.assert_array_equal(table, weights)
    tokenizer.enable_truncation.assert_called_once_with(max_length=script.TEXT_LENGTH)
    assert tokenizer.enable_padding.call_args.kwargs["length"] == script.TEXT_LENGTH
    # A mismatched table must fail before an input can reach the compiled encoder.
    np.save(directory / "token_embeddings.npy", weights[:2], allow_pickle=False)
    with pytest.raises(ValueError, match="unexpected token embedding table"):
        script.load_text_inputs(tmp_path)
    np.save(directory / "token_embeddings.npy", np.zeros(weights.shape, np.float16), allow_pickle=False)
    with pytest.raises(ValueError, match="unexpected token embedding table"):
        script.load_text_inputs(tmp_path)
