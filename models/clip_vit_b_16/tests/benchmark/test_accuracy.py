from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class _Tensor:
    def __init__(self, array: np.ndarray, **options: Any) -> None:
        self.array, self.options = array, options

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array

    @classmethod
    def from_numpy(cls, array: np.ndarray, copy: bool = False, **options: Any) -> _Tensor:
        return cls(array.copy() if copy else array, **options)


class _TextEncoder:
    """A stand-in for the text encoder's runner, which records its inputs."""

    def __init__(self, encode: Callable[[np.ndarray], np.ndarray]) -> None:
        self.encode, self.inputs, self.closed = encode, [], False

    def build(self, tensors: list[_Tensor]) -> _TextEncoder:
        return self

    def run(self, tensors: list[_Tensor], timeout_ms: int) -> list[_Tensor]:
        (tensor,) = tensors
        self.inputs.append(tensor)
        return [_Tensor(self.encode(tensor.array))]

    def close(self) -> None:
        self.closed = True


def _causal_sum(embeds: np.ndarray) -> np.ndarray:
    """A causal encoder in miniature: each output row sums the token embeddings up to it."""
    return np.cumsum(embeds, axis=1)


def _probe(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Two prompts padded with the end-of-text token, the rows their IDs use, and _causal_sum's end-of-text rows."""
    token_ids = np.full((2, 77), 49407, np.int32)
    token_ids[0, :5] = [49406, 320, 1125, 269, 49407]
    token_ids[1, :9] = [49406, 320, 1125, 539, 320, 1149, 634, 269, 49407]
    row_ids = np.unique(token_ids)
    rows = rng.normal(size=(len(row_ids), 512)).astype(np.float32)
    table = dict(zip(row_ids.tolist(), rows))
    embeds = [np.stack([table[token] for token in ids.tolist()]) for ids in token_ids]
    text_embeds = np.stack([_causal_sum(prompt[None])[0, np.argmax(ids)] for prompt, ids in zip(embeds, token_ids)])
    prompts = np.asarray(["a photo.", "a photo of a tench."])
    return {"prompts": prompts, "token_ids": token_ids, "row_ids": row_ids, "rows": rows, "text_embeds": text_embeds}


def test_mpk_arguments_name_exactly_both_encoders(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    mpks = script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz", "text_encoder=/data/text_mpk.tar.gz"])
    assert mpks == {"image_encoder": Path("/data/image_mpk.tar.gz"), "text_encoder": Path("/data/text_mpk.tar.gz")}
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and text_encoder=<path>"):
        script.parse_mpks(["model=/data/clip_mpk.tar.gz"])


def test_class_logits_are_told_apart_from_the_embedding_by_size(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    logits = np.arange(1000, dtype=np.float32)

    for outputs in (
        [_Tensor(np.zeros((1, 512), np.float32)), _Tensor(logits[None])],
        [_Tensor(logits[None]), _Tensor(np.zeros((1, 512), np.float32))],
    ):
        np.testing.assert_array_equal(script.class_logits(outputs), logits)
    with pytest.raises(ValueError, match="output sizes"):
        script.class_logits([_Tensor(logits[None])])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_the_text_probe_holds_padded_prompts_and_only_the_rows_they_use(
    load_script: Callable[[str], ModuleType],
) -> None:
    script = load_script("accuracy")
    probe = script.load_text_probe()

    token_ids, row_ids = probe["token_ids"], probe["row_ids"]
    assert token_ids.shape == (len(probe["prompts"]), 77) == (8, 77)
    ends = token_ids.argmax(axis=1)
    for ids, end in zip(token_ids, ends):
        # The start token, the prompt, then the end-of-text token up to 77 tokens.
        assert ids[0] == 49406 and (ids[1:end] < 49406).all() and (ids[end:] == 49407).all()
    np.testing.assert_array_equal(row_ids, np.unique(token_ids))
    assert probe["rows"].shape == (len(row_ids), 512) and probe["text_embeds"].shape == (8, 512)
    assert np.isfinite(probe["rows"]).all() and np.isfinite(probe["text_embeds"]).all()
    assert script.token_embeddings(probe, 0).shape == script.TEXT_SHAPE
    with pytest.raises(ValueError, match="does not match TEXT_PROBE_SHA256"):
        script.load_text_probe(Path(__file__))


def test_the_text_encoder_check_runs_each_prompts_token_rows(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    probe = _probe(np.random.default_rng(0))
    encoder = _TextEncoder(_causal_sum)
    monkeypatch.setattr(script.pyneat, "Model", lambda mpk, options: encoder)
    monkeypatch.setattr(script.pyneat, "Tensor", _Tensor)

    script.check_text_encoder(Path("text_encoder.tar.gz"), probe)

    assert len(encoder.inputs) == 2 and encoder.closed
    table = dict(zip(probe["row_ids"].tolist(), probe["rows"]))
    for ids, tensor in zip(probe["token_ids"], encoder.inputs):
        # Each token's row, looked up on the CPU, in a tensor declared HWC as Neat sizes the input.
        np.testing.assert_array_equal(tensor.array, np.stack([table[token] for token in ids.tolist()])[None])
        assert tensor.options["layout"] == script.pyneat.TensorLayout.HWC


def test_the_text_encoder_check_stops_on_an_end_of_text_row_far_from_fp32(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    probe = _probe(np.random.default_rng(0))
    monkeypatch.setattr(script.pyneat, "Tensor", _Tensor)

    monkeypatch.setattr(script.pyneat, "Model", lambda mpk, options: _TextEncoder(_causal_sum))
    probe["text_embeds"][1] *= -1
    with pytest.raises(ValueError, match=r"FP32: 'a photo of a tench.' \(-1.0000\)$"):
        script.check_text_encoder(Path("text_encoder.tar.gz"), probe)
    # A non-finite row fails the check too.
    monkeypatch.setattr(script.pyneat, "Model", lambda mpk, options: _TextEncoder(lambda embeds: embeds * np.nan))
    with pytest.raises(ValueError, match=r"'a photo.' \(nan\), 'a photo of a tench.' \(nan\)"):
        script.check_text_encoder(Path("text_encoder.tar.gz"), probe)


def test_a_stray_text_encoder_stops_the_stage_before_any_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    script = load_script("accuracy")
    loaded = []

    def model(mpk: str, options: object) -> _TextEncoder:
        # Every MPK runs as _causal_sum, whose end-of-text rows are far from the stored FP32 rows.
        loaded.append(mpk)
        return _TextEncoder(_causal_sum)

    monkeypatch.setattr(script.pyneat, "Model", model)
    monkeypatch.setattr(script.pyneat, "Tensor", _Tensor)
    # The labels name one image that does not exist, so reading it would fail: the check must stop the stage first.
    labels = {
        "categories": [{"id": index} for index in range(1000)],
        "images": [{"id": 1, "file_name": "ILSVRC2012_val_00000001.JPEG"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
    }
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "labels_val2012.json").write_text(json.dumps(labels), encoding="utf-8")
    arguments = [
        "--mpk",
        "image_encoder=image.tar.gz",
        "--mpk",
        "text_encoder=text.tar.gz",
        "--dataset",
        f"imagenet-2012-val={tmp_path}",
        "--output",
        str(tmp_path / "report"),
    ]

    with pytest.raises(ValueError, match="text encoder end-of-text rows below cosine 0.99 with FP32"):
        script.main(arguments)
    assert loaded == ["text.tar.gz"]
    assert not (tmp_path / "report").exists()


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
