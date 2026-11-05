from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


def _png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="PNG")
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_resizes_to_640x480_and_applies_imagenet_normalization(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # A 320x320 image, red above blue, stretches to 640x480 without cropping: rows 0-239 red, 240-479 blue.
    image = Image.new("RGB", (320, 320), (255, 0, 0))
    image.paste((0, 0, 255), (0, 160, 320, 320))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert result.shape == (480, 640, 3)
    assert result.dtype == np.float32
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[0, 0], red, atol=1e-5)
    np.testing.assert_allclose(result[230, 639], red, atol=1e-5)
    np.testing.assert_allclose(result[250, 0], blue, atol=1e-5)
    np.testing.assert_allclose(result[479, 639], blue, atol=1e-5)


def test_selection_is_a_seeded_shuffle_of_the_pittsburgh_images_only(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # pitts30k/cal keeps its images under pitts30k/images/<tar>/, beside the split, the list and the README.
    images = {
        f"pitts30k/images/00{index % 2}/{index:06d}_pitch1_yaw1.jpg": _png(8, 8, (index, 0, 0)) for index in range(6)
    }
    archive = _write_archive(
        tmp_path / "cal.tgz",
        {
            **images,
            "pitts30k/pitts30k_train.mat": b"spec",
            "pitts30k/cal.txt": b"list",
            "pitts30k/README.txt": b"readme",
        },
    )

    shuffled = calibration.select_images(archive, 42, 0, 6)

    assert sorted(shuffled) == sorted(images)
    assert calibration.select_images(archive, 42, 2, 3) == shuffled[2:5]
    assert calibration.load_images(archive, shuffled[:2]).shape == (2, 480, 640, 3)
    with pytest.raises(ValueError, match="has 6 Pittsburgh images, but 7 are required"):
        calibration.select_images(archive, 42, 4, 3)


def test_calibration_images_precede_the_fidelity_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    root = tmp_path / "cal"
    for index in range(80):
        (root / "pitts30k" / "images" / "000").mkdir(parents=True, exist_ok=True)
        (root / "pitts30k" / "images" / "000" / f"{index:06d}_pitch1_yaw1.jpg").write_bytes(b"")
    selection = recipe._metadata()["numerical_fidelity"]["selection"]
    fidelity = recipe.select_images(root, selection["seed"], selection["offset"], selection["samples"])

    calibration = recipe.calibration_names(root, recipe._parameters("w8a16")["calibration_samples"])

    # model.yaml calibrates on 50 images and compares 10 more, starting at offset 50.
    assert (len(calibration), len(fidelity)) == (50, 10)
    assert not set(calibration) & set(fidelity)
    metadata = recipe._metadata()
    metadata["numerical_fidelity"]["selection"]["offset"] = 49
    monkeypatch.setattr(recipe, "_metadata", lambda: metadata)
    with pytest.raises(ValueError, match="must begin after the calibration images"):
        recipe.calibration_names(root, 50)


@pytest.mark.parametrize(("precision", "per_token"), [("w8a16", True), ("w16a16", False)])
def test_only_w8a16_quantizes_activations_per_token(
    precision: str, per_token: bool, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    builder = MagicMock()
    monkeypatch.setattr(compile_module, "default_quantization", builder)

    compile_module._quantization(precision, 2048)

    # The recipe ends every configuration with with_per_token_quant, which the YOLO26 W8A16 recipes also set.
    (last,) = [item for item in builder.mock_calls if str(item).endswith(f".with_per_token_quant({per_token})")]
    assert last.args == (per_token,)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe._parameters("w16a16") == {"precision": "w16a16"}
    assert recipe._parameters("w8a16") == {"precision": "w8a16", "calibration_bins": 2048, "calibration_samples": 50}
    assert recipe._metadata()["numerical_fidelity"]["outputs"] == [recipe.OUTPUT]
    with pytest.raises(ValueError, match="no build profile 'w32a32'"):
        recipe._parameters("w32a32")


def _mpk_manifest(recipe: ModuleType) -> dict[str, Any]:
    return {
        "model_sdk_version": recipe.COMPILER_VERSION,
        # One float32 480x640 RGB image in, one float32 4096-value descriptor out.
        "input_nodes": [{"name": "image", "size": 3 * 480 * 640 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 4096 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict[str, Any]) -> None:
    _write_archive(path, {"model.elf": b"elf", "netvlad_mpk.json": json.dumps(manifest).encode()})


@pytest.mark.parametrize("case", ["valid", "two_mla", "host", "compiler", "input", "output", "mla_output"])
def test_mpk_audit_requires_one_mla_segment_and_the_descriptor_interface(
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    manifest = _mpk_manifest(recipe)
    plugins = manifest["plugins"]
    if case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    elif case == "input":
        manifest["input_nodes"][0]["size"] = 3 * 224 * 224 * 4
    elif case == "output":
        plugins[-1]["output_nodes"] = [{"size": 512 * 4}]
    elif case == "mla_output":
        # The descriptor must come back through the float32 conversion, not straight from the MLA.
        plugins[1]["output_nodes"] = plugins.pop()["output_nodes"]
    archive = tmp_path / "netvlad_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        recipe.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(archive)


def test_compile_reports_the_compilers_silent_exit_as_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    loaded = MagicMock()
    # AFE raised a bare SystemExit when it could not segment NetVLAD's 32768-wide L2 normalization at INT8.
    loaded.quantize.return_value.compile.side_effect = SystemExit()
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: loaded)

    with pytest.raises(RuntimeError, match="failed to compile"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", "w16a16", None)


def test_the_benchmarks_read_the_directory_the_runners_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The runners look through the archive's one top-level directory, so its files sit directly under --dataset."""
    (tmp_path / "pitts30k_test.tsv").write_text(
        "database\t000/d.jpg\t1\t2\nquery\tqueries_real/q.jpg\t3\t4\n", encoding="utf-8"
    )

    database, _, queries, _ = load_script("accuracy").read_listing(tmp_path)

    assert list(database) == ["000/d.jpg"] and list(queries) == ["queries_real/q.jpg"]
    # performance.py parses its arguments and runs as it loads.
    arguments = ["--mpk", "model=m.tar.gz", "--dataset", f"pitts30k-val={tmp_path}", "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["performance.py", *arguments])
    with pytest.raises(SystemExit, match=f"{tmp_path} has no images under images/"):
        load_script("performance")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
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


def test_compiled_mpk_has_one_mla_segment_and_the_descriptor_interface(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    recipe.audit_mpk(compiled / "model" / f"netvlad_{build_profile}_mpk.tar.gz")


def test_compiled_build_reports_descriptor_fidelity(compiled: Path, build_profile: str) -> None:
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))

    assert (report["build_profile_id"], report["samples"]) == (build_profile, 10)
    (metric,) = report["metrics"]
    assert metric["name"] == "descriptor"
    assert np.isfinite([metric["cosine_similarity"], metric["normalized_rmse"]]).all()
