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
MODEL_DIR = ROOT / "models" / "lightstereo_s"


@pytest.fixture
def scene_flow(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("scene_flow")


def _png(value: int, size: tuple[int, int] = (960, 540)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _pair(index: int) -> tuple[str, str]:
    return tuple(f"Monkaa/frames_finalpass/scene/{view}/{index:04d}.png" for view in ("left", "right"))


def _cal_archive(path: Path, pairs: int) -> Path:
    """The scene-flow cal layout: scene_flow/cal.txt and uniform 960x540 views, grey level = pair (+100 if right)."""
    lines, files = [], {}
    for index in range(pairs):
        names = _pair(index)
        lines.append(f"{names[0]} {names[1]} Monkaa/disparity/scene/left/{index:04d}.pfm\n")
        files.update({names[0]: _png(index), names[1]: _png(index + 100)})
    files["cal.txt"] = "".join(lines).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"scene_flow/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def test_pad_geometry_adds_four_rows_above_a_960x540_view(scene_flow: ModuleType) -> None:
    # Scene Flow views are 960x540; OpenStereo's RightTopPad brings them to 544x960.
    assert scene_flow.pad_geometry(540, 960) == (4, 0)
    assert scene_flow.pad_geometry(530, 950) == (14, 10)
    with pytest.raises(ValueError, match="exceeds"):
        scene_flow.pad_geometry(540, 1024)


def test_preprocessing_edge_pads_top_and_right_then_normalizes(scene_flow: ModuleType) -> None:
    image = np.zeros((540, 956, 3), np.uint8)
    image[0] = 255  # the top row, which edge padding repeats above the image
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")

    result = scene_flow.preprocess_image(stream.getvalue())

    assert result.shape == (544, 960, 3) and result.dtype == np.float32
    white, black = (1 - scene_flow.MEAN) / scene_flow.STD, -scene_flow.MEAN / scene_flow.STD
    # Rows 0-3 are padding copied from the white top row; row 4 is that row; column 959 repeats column 955.
    np.testing.assert_allclose(result[:5, 0], np.tile(white, (5, 1)), rtol=1e-6)
    np.testing.assert_allclose(result[5, 959], black, rtol=1e-6)
    np.testing.assert_allclose(white, [2.2489, 2.4286, 2.64], atol=1e-4)


def test_fidelity_pairs_follow_the_declared_seeded_selection(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 12)

    compare = scene_flow.fidelity_pairs(dataset, seed=42, offset=0, samples=6)

    # Six distinct listed pairs, the same ones for the same seed, and offset skips into the same shuffle.
    assert len(set(compare)) == 6 and set(compare) <= {_pair(i) for i in range(12)}
    assert scene_flow.fidelity_pairs(dataset, seed=42, offset=0, samples=6) == compare
    assert scene_flow.fidelity_pairs(dataset, seed=42, offset=2, samples=4) == compare[2:]
    with pytest.raises(ValueError, match="needs 13 pairs"):
        scene_flow.fidelity_pairs(dataset, seed=42, offset=1, samples=12)


def test_read_pairs_returns_left_and_right_batches_in_the_given_order(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 4)

    pairs = scene_flow.read_pairs(dataset, [_pair(3), _pair(1)])

    assert list(pairs) == ["left", "right"]
    assert pairs["left"].shape == pairs["right"].shape == (2, 544, 960, 3)
    # The fixture's grey level is the pair index for the left view and 100 more for the right one.
    grey = (np.array([3, 1, 103, 101], np.float32) / 255 - scene_flow.MEAN[0]) / scene_flow.STD[0]
    np.testing.assert_allclose(pairs["left"][:, 20, 0, 0], grey[:2], rtol=1e-6)
    np.testing.assert_allclose(pairs["right"][:, 20, 0, 0], grey[2:], rtol=1e-6)
    with pytest.raises(ValueError, match="lacks 2 listed images"):
        scene_flow.read_pairs(dataset, [_pair(7)])


def _mpk_manifest(compile_module: ModuleType) -> dict:
    """The plugin chain of the compiled w16a16 MPK: a BF16 cast per view, the MLA, then a slice and a cast back."""
    view = 1 * 3 * 544 * 960 * 4
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "left", "size": view}, {"name": "right", "size": view}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 3, "processor": "MLA", "config_params": {}},
            {"sequence": 4, "processor": "EV74", "config_params": {"kernel": "slice_transform"}},
            {
                "sequence": 5,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 1 * 1 * 544 * 960 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "one_input", "output", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_stereo_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "one_input":
        manifest["input_nodes"].pop()
    elif case == "output":
        plugins[-1]["output_nodes"][0]["size"] //= 4
    elif case == "two_mla":
        plugins.insert(3, {"sequence": 3, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(3, {"sequence": 3, "processor": "A65", "config_params": {}})
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

    assert compile_module._parameters("w16a16") == {"precision": "w16a16"}
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w32a32")


def test_model_is_internal_only_and_compares_disparity_on_scene_flow() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    # W8A8 measured 1.555 px Scene Flow EPE and W8A16 fails the compiler's check, so only BF16 ships.
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16"]
    assert not model.get("calibrations")
    # Upstream declares no license; the issue's decision is internal evaluation, never publication.
    assert model["license"]["redistribution"] == "prohibited"
    assert model["numerical_fidelity"]["dataset"]["variant"] == "cal"
    assert model["numerical_fidelity"]["outputs"] == ["disparity"]


def test_compiled_build_holds_the_audited_mpk_and_a_disparity_fidelity_report(
    compiled: Path, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    (metric,) = report["metrics"]
    assert metric["name"] == "disparity"
    assert np.isfinite([metric["mean_absolute_error"], metric["cosine_similarity"]]).all()
