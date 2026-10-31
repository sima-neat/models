from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from model_registry.model import load_model as load_metadata
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "raft_stereo_realtime"


@pytest.fixture
def scene_flow(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("scene_flow")


def _png(value: int, size: tuple[int, int] = (960, 540)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _pfm(disparity: np.ndarray) -> bytes:
    """A little-endian one-channel PFM, which stores its rows from the bottom."""
    height, width = disparity.shape
    return f"Pf\n{width} {height}\n-1.0\n".encode() + np.flipud(disparity).astype("<f4").tobytes()


def _pair(index: int) -> tuple[str, str, str]:
    return (
        f"Monkaa/frames_finalpass/scene/left/{index:04d}.png",
        f"Monkaa/frames_finalpass/scene/right/{index:04d}.png",
        f"Monkaa/disparity/scene/left/{index:04d}.pfm",
    )


def _cal_archive(path: Path, pairs: int) -> Path:
    """The scene-flow cal layout: scene_flow/cal.txt and uniform 960x540 views, grey level = pair (+100 if right),
    each with a disparity map of the pair's index in every pixel."""
    lines, files = [], {}
    for index in range(pairs):
        left, right, disparity = _pair(index)
        lines.append(f"{left} {right} {disparity}\n")
        files.update({left: _png(index), right: _png(index + 100)})
        files[disparity] = _pfm(np.full((540, 960), index, np.float32))
    files["cal.txt"] = "".join(lines).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"scene_flow/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def test_preprocessing_takes_a_native_view_and_normalizes_it(scene_flow: ModuleType) -> None:
    result = scene_flow.preprocess_image(_png(255))

    assert result.shape == (540, 960, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], (1 - scene_flow.MEAN) / scene_flow.STD, rtol=1e-6)
    # The graph pads the rows itself, so a view of any other size is refused rather than resized.
    with pytest.raises(ValueError, match="expected a 960x540 view"):
        scene_flow.preprocess_image(_png(0, (1242, 375)))


def test_pfm_rows_are_read_from_the_top(scene_flow: ModuleType) -> None:
    disparity = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_array_equal(scene_flow.read_pfm(_pfm(disparity)), disparity)
    with pytest.raises(ValueError, match="one-channel"):
        scene_flow.read_pfm(b"PF\n4 3\n-1.0\n" + bytes(144))


def test_calibration_and_fidelity_take_spans_of_one_seeded_order(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 12)

    order = scene_flow.selected_pairs(dataset, seed=42, offset=0, samples=12)

    # Every listed pair once, the same order for the same seed, and later spans of that order for fidelity.
    assert sorted(order) == [_pair(i) for i in range(12)]
    assert scene_flow.selected_pairs(dataset, seed=42, offset=4, samples=6) == order[4:10]
    with pytest.raises(ValueError, match="needs 13 pairs"):
        scene_flow.selected_pairs(dataset, seed=42, offset=1, samples=12)


def test_read_pairs_returns_side_by_side_frames_and_true_disparities(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 4)

    frames, disparities = scene_flow.read_pairs(dataset, [_pair(3), _pair(1)])

    assert frames.shape == (2, 540, 1920, 3) and disparities.shape == (2, 540, 960)
    # The fixture's grey level is the pair index for the left view and 100 more for the right one; the left
    # view fills the first 960 columns of each frame and the right view the rest.
    grey = (np.array([3, 1, 103, 101], np.float32) / 255 - scene_flow.MEAN[0]) / scene_flow.STD[0]
    np.testing.assert_allclose(frames[:, 20, 959, 0], grey[:2], rtol=1e-6)
    np.testing.assert_allclose(frames[:, 20, 960, 0], grey[2:], rtol=1e-6)
    np.testing.assert_array_equal(disparities[:, 7, 7], [3, 1])
    with pytest.raises(ValueError, match="lacks 3 listed files"):
        scene_flow.read_pairs(dataset, [_pair(7)])


def test_end_point_error_scores_pixels_under_192(scene_flow: ModuleType) -> None:
    truth = np.array([[10.0, 20.0], [191.0, 250.0]], np.float32)
    disparity = np.array([[[[11.0, 18.0], [191.0, 0.0]]]], np.float32)

    # |11 - 10|, |18 - 20| and 0 over the three pixels under 192; the 250 px pixel is not scored.
    assert scene_flow.end_point_error(disparity, truth) == pytest.approx(1.0)


def _mpk_manifest(compile_module: ModuleType) -> dict:
    """A plugin chain of the compiled MPK: a cast of the frame, the MLA, then a slice and a cast back."""
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "stereo", "size": 1 * 3 * 540 * 1920 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "MLA", "config_params": {}},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "slice_transform"}},
            {
                "sequence": 4,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 1 * 1 * 540 * 960 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_inputs", "output", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_stereo_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "two_inputs":
        manifest["input_nodes"].append({"name": "right", "size": 1 * 3 * 540 * 960 * 4})
    elif case == "output":
        plugins[-1]["output_nodes"][0]["size"] = 1 * 1 * 544 * 960 * 4
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65", "config_params": {}})
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    assert compile_module._parameters("w16a16") == {
        "precision": "w16a16",
        "calibration_bins": 2048,
        "calibration_samples": 4,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w32a32")
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_module._quantization({"precision": "w4a16", "calibration_bins": 2048})


def test_model_calibrates_and_compares_disparity_on_scene_flow_cal() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16", "w8a8"]
    largest = max(profile["components"][0]["parameters"]["calibration_samples"] for profile in model["build_profiles"])
    fidelity = model["numerical_fidelity"]
    # Calibration takes the first pairs of the seeded order, so the fidelity pairs must start after them.
    assert fidelity["selection"]["offset"] >= largest
    assert fidelity["dataset"]["variant"] == "cal" and fidelity["outputs"] == ["disparity"]
    assert model["license"]["redistribution"] == "allowed"


def test_compiled_build_holds_the_audited_mpk_and_a_disparity_fidelity_report(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    (metric,) = report["metrics"]
    assert metric["name"] == "disparity"
    assert np.isfinite([metric["mean_absolute_error"], metric["cosine_similarity"]]).all()
