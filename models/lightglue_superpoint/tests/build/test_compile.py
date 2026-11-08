from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# Ground truth for a pairs line: two identity intrinsics and an identity pose.
IDENTITY = " ".join(["1 0 0 0 1 0 0 0 1"] * 2 + ["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"])


def _write_dataset(root: Path, split: str, counts: dict[str, int]) -> Path:
    """A yfcc100m-keypoints variant: each image's keypoints, scores and descriptors, and pairs of consecutive images."""
    names = sorted(counts)
    for index, name in enumerate(names):
        directory = root / split / Path(name).with_suffix("")
        directory.mkdir(parents=True)
        keypoints = np.full((counts[name], 2), [320.0 + index, 240.0], np.float32)
        np.save(directory / "keypoints.npy", keypoints)
        np.save(directory / "scores.npy", np.full(counts[name], 0.5, np.float32))
        np.save(directory / "descriptors.npy", np.full((counts[name], 256), 1 / 16, np.float32))
    (root / "annotations").mkdir()
    rows = [f"{name} 1280 960 0 {counts[name]}" for name in names]
    (root / "annotations" / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    pairs = [f"{first} {second} 0 0 {IDENTITY}" for first, second in zip(names, names[1:])]
    (root / "annotations" / f"pairs_{split}.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    return root


def test_inputs_hold_each_images_512_real_keypoints(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    counts = {"a/0.jpg": 512, "a/1.jpg": 512, "a/2.jpg": 511}
    root = _write_dataset(tmp_path / "yfcc100m-keypoints-cal", "cal", counts)

    pair, short_pair = calibration.read_pairs(root)
    inputs = calibration.pair_inputs(root, pair)

    assert {name: value.shape for name, value in inputs.items()} == calibration.INPUT_SHAPES
    # The first image's keypoints sit at the 640x480 input's centre, which LightGlue normalizes to the origin, where
    # every phase is 0: cosines 1 and sines 0. The second image's sit one pixel to the right.
    centre = np.concatenate([np.ones(64), np.zeros(64)])
    np.testing.assert_allclose(inputs["encoding0"][0], [centre] * 512, atol=1e-7)
    np.testing.assert_allclose(inputs["encoding1"][0], [calibration.encoding(np.asarray([1 / 320, 0]))] * 512)
    np.testing.assert_allclose(inputs["descriptors1"][0], 1 / 16)
    # The matcher has no padding mask: an image with fewer keypoints is refused, and its pairs are left out.
    with pytest.raises(ValueError, match="511 keypoints"):
        calibration.pair_inputs(root, short_pair)
    assert [(item.name0, item.name1) for item in calibration.usable_pairs(root)] == [("a/0.jpg", "a/1.jpg")]


def test_the_encoding_is_lightglues_rotary_encoding_of_the_keypoint(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    keypoint = np.asarray([0.25, -0.5])

    encoded = calibration.encoding(keypoint)

    phases = calibration.KEYPOINT_PROJECTION.astype(np.float64) @ keypoint
    assert encoded.shape == (calibration.ENCODING_DIM,) == (128,)
    # Each of the 32 phases appears twice in a row, as repeat_interleave(2) gives the 64-wide attention heads.
    np.testing.assert_allclose(encoded[:64:2], np.cos(phases), atol=1e-6)
    np.testing.assert_allclose(encoded[1:64:2], np.cos(phases), atol=1e-6)
    np.testing.assert_allclose(encoded[64::2], np.sin(phases), atol=1e-6)
    np.testing.assert_allclose(encoded[65::2], np.sin(phases), atol=1e-6)


def test_matches_need_a_mutual_best_pair_above_the_threshold(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    assignment = np.full((512, 512), 1e-4, np.float32)
    assignment[0, 3], assignment[1, 2] = 0.8, 0.09  # a match, and a mutual best pair below the threshold
    assignment[2, 400], assignment[5, 400] = 0.95, 0.5  # keypoint 400's best partner is 2, not 5

    matched = calibration.matches(assignment.reshape(1, 1, 512, 512))

    assert matched.tolist() == [[0, 3], [2, 400]]


def test_pairs_and_images_read_every_ground_truth_value(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    root = _write_dataset(tmp_path / "set", "val", {"a/0.jpg": 512, "a/1.jpg": 512})
    pairs_file = root / "annotations" / "pairs_val.txt"
    line = pairs_file.read_text(encoding="utf-8").strip()

    (pair,) = calibration.read_pairs(root)
    assert (pair.name0, pair.name1) == ("a/0.jpg", "a/1.jpg")
    np.testing.assert_array_equal(pair.pose, np.eye(4))
    assert calibration.read_images(root)["a/0.jpg"] == calibration.Image(1280, 960, 0, 512)
    pairs_file.write_text(line + " 7", encoding="utf-8")
    with pytest.raises(ValueError, match="39 fields"):
        calibration.read_pairs(root)


def test_calibration_reads_the_archive_and_fidelity_follows_it_in_one_seeded_order(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    counts = {f"s/{index}.jpg": 512 for index in range(12)}
    root = _write_dataset(tmp_path / "yfcc100m-keypoints-cal", "cal", counts)
    archive = tmp_path / "cal.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname=root.name)

    samples = calibration.calibration_inputs(archive, 4)
    later = calibration.fidelity_inputs(archive, 42, 4, 3)

    assert {name: value.shape[0] for name, value in samples.items()} == dict.fromkeys(calibration.INPUT_SHAPES, 4)
    order = calibration.seeded_pairs(root, 42)
    expected = [calibration.pair_inputs(root, pair)["encoding0"] for pair in order[4:7]]
    np.testing.assert_array_equal([inputs["encoding0"] for inputs in later], expected)
    with pytest.raises(ValueError, match="needs 12 pairs"):
        calibration.fidelity_inputs(archive, 42, 4, 8)


def test_only_the_int8_build_calibrates_before_the_fidelity_pairs(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe._metadata()
    selection = metadata["numerical_fidelity"]["selection"]

    for profile in metadata["build_profiles"]:
        parameters = recipe._parameters(metadata, profile["id"])
        (binding,) = profile["components"]
        assert parameters["precision"] == profile["id"]
        if recipe._calibrates(parameters["precision"]):
            assert binding["calibration"] == "yfcc100m-keypoints-cal"
            assert parameters["calibration_samples"] <= selection["offset"]
        else:
            assert binding.get("calibration") is None
            assert "calibration_samples" not in parameters
    assert metadata["numerical_fidelity"]["outputs"] == [recipe.OUTPUT_NAME]
    assert set(recipe.MATCH_OVERLAP_FLOORS) == {profile["id"] for profile in metadata["build_profiles"]}
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._parameters(metadata, "w32a32")


def test_bf16_skips_calibration_and_int8_calibrates_with_histogram_mse(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    # Each AFE constructor returns its name and arguments, so the test reads the recipe without the Model Compiler.
    for name in ("QuantizationParams", "SkipCalibration", "HistogramMSEMethod", "bfloat16_scheme",
                 "quantization_scheme"):
        monkeypatch.setattr(recipe, name, lambda *args, _name=name, **kwargs: (_name, args, kwargs))
    metadata = recipe._metadata()

    (_, _, int8), (_, _, bf16) = (recipe._quantization_config(recipe._parameters(metadata, profile))
                                  for profile in ("w8a8", "w16a16"))

    assert int8 == {
        "calibration_method": ("HistogramMSEMethod", (2048,), {}),
        "activation_quantization_scheme": ("quantization_scheme", (), {"asymmetric": True, "per_channel": False,
                                                                       "bits": 8}),
        "weight_quantization_scheme": ("quantization_scheme", (), {"asymmetric": False, "per_channel": True,
                                                                   "bits": 8}),
    }
    assert bf16 == {
        "calibration_method": ("SkipCalibration", (), {}),
        "activation_quantization_scheme": ("bfloat16_scheme", (), {}),
        "weight_quantization_scheme": ("bfloat16_scheme", (), {}),
    }


def test_a_build_needs_its_matches_to_overlap_the_onnxs(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    expected = np.full((512, 512), 1e-4, np.float32)
    actual = expected.copy()
    for row in range(4):  # FP32 matches keypoints 0-3; the quantized graph keeps 0-2 and finds 5
        expected[row, row + 10] = 0.9
    for row in (0, 1, 2, 5):
        actual[row, row + 10] = 0.8

    overlap = recipe._OutputError("matches")
    overlap.add(recipe.match_matrix(expected), recipe.match_matrix(actual))

    # 3 shared matches of 4 and 4: 3 / sqrt(4 * 4).
    assert overlap.summary()["cosine_similarity"] == pytest.approx(0.75)
    recipe.require_match_overlap("w8a8", 0.75)
    with pytest.raises(RuntimeError, match="not above the 0.9 floor"):
        recipe.require_match_overlap("w16a16", 0.75)


def test_every_input_fills_whole_mla_words(load_script: Callable[[str], ModuleType]) -> None:
    """The MLA reads each input in 16-byte words, and Neat refuses an input that leaves a word partly filled."""
    calibration = load_script("calibration")

    for name, shape in calibration.INPUT_SHAPES.items():
        assert int(np.prod(shape)) % 16 == 0, name


def _mpk(tmp_path: Path, manifest: dict, members: tuple[str, ...] = ("model_stage1_mla.elf",)) -> Path:
    archive = tmp_path / "model_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content in [("model_mpk.json", json.dumps(manifest).encode()), *((name, b"") for name in members)]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    return archive


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_inputs(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    inputs = [{"name": name, "size": int(np.prod(shape)) * 4} for name, shape in recipe.INPUT_SHAPES.items()]
    manifest = {
        "model_sdk_version": recipe.COMPILER_VERSION,
        "input_nodes": inputs,
        "plugins": [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}],
    }

    assert recipe.audit_mpk(_mpk(tmp_path, manifest))["mla_plugins"] == 1
    with pytest.raises(RuntimeError, match="host=1"):
        recipe.audit_mpk(_mpk(tmp_path, {**manifest, "plugins": [*manifest["plugins"], {"processor": "A65"}]}))
    with pytest.raises(RuntimeError, match="input interface mismatch"):
        recipe.audit_mpk(_mpk(tmp_path, {**manifest, "input_nodes": inputs[:2]}))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--build-profile", "w8a8", "--compile-invocation", "w8a8"], "w8a8 calibrates; pass --dataset"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_incomplete_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_build_holds_the_matcher_and_follows_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    (item,) = outputs["outputs"]
    recipe.audit_mpk(compiled / item["mpk_path"])
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    metrics = {metric["name"]: metric for metric in fidelity["metrics"]}
    assert list(metrics) == ["assignment"]
    # compile.py stops a build whose matches overlap the ONNX's no more than MATCH_OVERLAP_FLOORS before the MLA
    # compile, so a build here kept enough of them; the accuracy stage checks its pose AUC against tests.yaml.
    assert all(np.isfinite(value) for name, value in metrics["assignment"].items() if name != "name"), metrics
    assert build_profile in recipe.MATCH_OVERLAP_FLOORS
