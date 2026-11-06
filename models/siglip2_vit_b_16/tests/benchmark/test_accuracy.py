from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class _Tensor:
    def __init__(self, array: np.ndarray, **options: Any) -> None:
        self.array, self.options = array, options

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array


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


def test_class_logits_are_told_apart_from_the_embedding_by_size(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    logits = np.arange(1000, dtype=np.float32)

    for outputs in ([_Tensor(np.zeros((1, 768), np.float32)), _Tensor(logits[None])],
                    [_Tensor(logits[None]), _Tensor(np.zeros((1, 768), np.float32))]):
        np.testing.assert_array_equal(script.class_logits(outputs), logits)
    with pytest.raises(ValueError, match="output sizes"):
        script.class_logits([_Tensor(logits[None])])


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
    arguments = ["--mpk", "image_encoder=image.tar.gz", "--mpk", "text_encoder=text.tar.gz",
                 "--dataset", f"imagenet-2012-val={tmp_path}", "--output", str(tmp_path / "report")]
    # The dataset folder is empty, so reading it would fail otherwise: the check ran first.
    with pytest.raises(ValueError, match="strays from the FP32 ONNX"):
        script.main(arguments)
    assert encoder.mpks == ["text.tar.gz"]
    assert not (tmp_path / "report").exists()


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
