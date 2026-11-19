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
from PIL import Image


def test_preprocessing_stretches_and_normalizes(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    image = np.full((300, 400, 3), 128, np.uint8)

    result = pipeline.preprocess(image)

    assert (result.shape, result.dtype) == ((384, 384, 3), np.float32)
    np.testing.assert_allclose(
        result, np.broadcast_to((128 - pipeline.PIXEL_MEAN) / pipeline.PIXEL_STD, result.shape), atol=1e-5
    )


def test_decode_picks_the_top_query_class_pairs_and_scores_masks(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    classes = np.full((pipeline.CLASSES + 1, pipeline.QUERIES), -10.0, np.float32)
    classes[pipeline.CLASSES] = 10.0  # every query "no object" ...
    classes[:, 3] = -10.0
    classes[5, 3] = 10.0  # ... except query 3, class 5
    masks = np.full((pipeline.QUERIES, pipeline.MASK_SIZE, pipeline.MASK_SIZE), -10.0, np.float32)
    # Query 3's mask covers the top-left quarter of the 384x384 input, stretched over a 300x400 image.
    masks[3, :48, :48] = 10.0

    label, score, binary = pipeline.decode(classes[None, :, None, :], masks[None], (300, 400))[0]

    # The mask's border pixels sit near logit 0 after the bilinear resizes, which lowers the mean probability a little.
    assert label == 5 and score == pytest.approx(1.0, abs=1e-2)
    assert binary.shape == (300, 400)
    # The top-left quarter of the input is the top-left 150x200 of the image.
    assert binary[:145, :195].all() and not binary[155:].any() and not binary[:, 205:].any()


def test_every_component_input_comes_from_an_earlier_output(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    shapes = {"image": pipeline.INPUT_SHAPES["backbone_0"]["image"]}

    for component in pipeline.COMPONENTS:
        assert pipeline.route(component, shapes) == list(pipeline.INPUT_SHAPES[component].values())
        shapes.update(pipeline.OUTPUT_SHAPES[component])


def test_performance_script_routes_like_the_pipeline(load_script: Callable[[str], ModuleType]) -> None:
    pipeline, performance = load_script("pipeline"), load_script("performance")

    assert performance.COMPONENTS == pipeline.COMPONENTS
    assert performance.INPUTS == {c: tuple(pipeline.INPUT_SHAPES[c]) for c in pipeline.COMPONENTS}
    assert performance.OUTPUTS == {c: tuple(pipeline.OUTPUT_SHAPES[c]) for c in pipeline.COMPONENTS}


def test_compile_binds_every_component(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters("w16a16")

    assert list(parameters) == [
        "backbone_0",
        "backbone_1",
        *(f"pixel_decoder_{index}" for index in range(13)),
        "transformer_decoder",
    ]
    assert {item["precision"] for item in parameters.values()} == {"w16a16"}
    with pytest.raises(ValueError, match="no build profile 'w8a8'"):
        recipe._component_parameters("w8a8")


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in [*(f"cal2017/{index:02d}.jpg" for index in range(6)), "val2017/x.jpg"]:
            content = io.BytesIO()
            Image.new("RGB", (64, 48), (200, 100, 50)).save(content, format="JPEG")
            member = tarfile.TarInfo(f"./{name}")
            member.size = content.tell()
            content.seek(0)
            stream.addfile(member, content)

    images = recipe.calibration_images(archive, 4)

    assert [image.shape for image in images] == [(48, 64, 3)] * 4
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        recipe.calibration_images(archive, 7)


def _mpk(path: Path, processors: list[str]) -> Path:
    manifest = {"plugins": [{"processor": p} for p in processors]}
    content = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("model_mpk.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return path


def test_audit_requires_one_mla_segment_and_no_host_code(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74"]), "backbone_0")
    for name, processors in [("two", ["MLA", "MLA"]), ("host", ["MLA", "A65"])]:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors), "backbone_0")


def test_compile_main_rejects_an_unknown_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err
