from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from PIL import Image

# The least cosine with reference.onnx a build keeps on the 50 fidelity images, per output. Model Compiler
# 3.0.0.dev0+master.402 measured the least output on 16 of them at w16a16 0.999665; CI compiles with Model Compiler
# 3.0.0.2. A floor allows twice the measured loss, 1 - cosine, when measured on all 50 images and three times when
# measured on fewer, rounded down to three decimals.
FIDELITY_FLOORS = {"w16a16": 0.998}


def _png(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
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


def _mpk(path: Path, plugins: list[dict], inputs: list[dict]) -> Path:
    manifest = {"plugins": plugins, "input_nodes": inputs}
    return _write_archive(path, {"model_mpk.json": json.dumps(manifest).encode(), "model.elf": b"elf"})


def test_preprocessing_is_imagenets_mean_and_std(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.SIZE == 224
    assert calibration.RESCALE == pytest.approx(1 / 255)
    np.testing.assert_allclose(calibration.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(calibration.STD, [0.229, 0.224, 0.225])


def test_preprocessing_resizes_straight_to_the_model_input(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, squeezes to 224x224 without a crop: 112 rows of each.
    image = Image.new("RGB", (128, 256), (255, 0, 0))
    image.paste((0, 0, 255), (0, 128, 128, 256))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert (result.shape, result.dtype) == ((224, 224, 3), np.float32)
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[100, 112], red, atol=1e-5)
    np.testing.assert_allclose(result[124, 112], blue, atol=1e-5)


def test_preprocessing_follows_metas_antialiased_bilinear_resize(load_script: Callable[[str], ModuleType]) -> None:
    """Meta's make_transform, with torchvision, on a fine pattern that an unfiltered resize would alias."""
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision.transforms.v2")
    calibration = load_script("calibration")
    pixels = np.random.default_rng(0).integers(0, 256, (333, 500, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    meta = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    expected = meta(Image.fromarray(pixels)).numpy().transpose(1, 2, 0)

    # PIL rounds the resized image to 8 bits, one level being 1 / 255 / std.
    np.testing.assert_allclose(calibration.preprocess_image(stream.getvalue()), expected, rtol=0, atol=0.02)


def test_selection_is_seeded_and_reads_an_archive(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    names = [f"imagenet/cal2012/n01440764_{index:02d}.JPEG" for index in range(6)]
    # An image outside cal2012 is never selected.
    files = {name: _png(300, 260) for name in [*names, "imagenet/val2012/n01440764_00.JPEG"]}
    archive = _write_archive(tmp_path / "cal.tar.gz", files)
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.select_names(archive, 42, 2, 3) == expected[2:5]
    assert calibration.load_images(archive, expected[:2]).shape == (2, 224, 224, 3)
    with pytest.raises(ValueError, match="needs 7 images"):
        calibration.select_names(archive, 42, 1, 6)


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(tmp_path / "cal.tar.gz", {"imagenet/cal2012/a.JPEG": b"a"})

    assert calibration.read_images(archive, ["imagenet/cal2012/a.JPEG"]) == {"imagenet/cal2012/a.JPEG": b"a"}
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, ["imagenet/cal2012/a.JPEG", "imagenet/cal2012/b.JPEG"])


def test_every_profile_names_its_precision_and_does_not_calibrate(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(recipe.MODEL_DIR)

    for profile in metadata["build_profiles"]:
        parameters = recipe._parameters(metadata, profile["id"])
        (binding,) = profile["components"]
        assert profile["id"] == parameters["precision"]
        assert binding.get("calibration") is None
    assert {profile["id"] for profile in metadata["build_profiles"]} == set(FIDELITY_FLOORS)
    assert metadata["numerical_fidelity"]["outputs"] == list(recipe.OUTPUTS)
    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._parameters(metadata, "w32a32")


def test_bf16_activations_skip_calibration(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    # Each AFE constructor returns its name and arguments, so the test reads the recipe without the Model Compiler.
    for name in ("QuantizationParams", "SkipCalibration", "bfloat16_scheme", "quantization_scheme"):
        monkeypatch.setattr(recipe, name, lambda *args, _name=name, **kwargs: (_name, args, kwargs))

    (_, _, int8_weights), (_, _, bf16) = (recipe._quantization(precision) for precision in ("w8a16", "w16a16"))

    bf16_scheme = ("bfloat16_scheme", (), {})
    skip = ("SkipCalibration", (), {})
    assert int8_weights == {
        "calibration_method": skip,
        "activation_quantization_scheme": bf16_scheme,
        "weight_quantization_scheme": (
            "quantization_scheme",
            (),
            {"asymmetric": False, "per_channel": True, "bits": 8},
        ),
        "per_token_quant": True,
    }
    assert bf16 == {
        "calibration_method": skip,
        "activation_quantization_scheme": bf16_scheme,
        "weight_quantization_scheme": bf16_scheme,
    }
    with pytest.raises(ValueError, match="only the declared Modalix BF16-activation builds"):
        recipe._quantization("w8a8")


def test_a_build_quantizes_without_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    loaded = MagicMock(name="loaded")
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: loaded)

    recipe.quantize(tmp_path / "model.onnx", "w16a16", "w16a16")

    assert loaded.quantize.call_args.kwargs["calibration_data"] is None


def test_a_compiler_failure_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    # AFE ends a failed quantization or compile with a bare SystemExit, whose exit status would be 0.
    recipe = load_script("compile")
    loaded = MagicMock(name="loaded")
    loaded.quantize.side_effect = SystemExit()
    monkeypatch.setattr(recipe, "load_model", lambda *args, **kwargs: loaded)
    quantized = MagicMock(name="quantized")
    quantized.compile.side_effect = SystemExit()

    with pytest.raises(RuntimeError, match="failed to quantize"):
        recipe.quantize(tmp_path / "model.onnx", "w16a16", "w16a16")
    with pytest.raises(RuntimeError, match="failed to compile"):
        recipe.compile_model(quantized, tmp_path / "model")


def test_fidelity_error_sums_match_the_metrics_over_all_samples(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    error = recipe._OutputError("class_embedding")
    error.add(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]]))
    error.add(np.asarray([[0.0, 2.0]]), np.asarray([[0.0, 1.0]]))
    summary = error.summary()

    assert summary["name"] == "class_embedding"
    assert summary["cosine_similarity"] == pytest.approx(3 / np.sqrt(5 * 2))
    assert summary["max_absolute_error"] == pytest.approx(1.0)
    assert summary["mean_absolute_error"] == pytest.approx(0.25)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        error.add(np.zeros((1, 2)), np.zeros((1, 3)))


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = [{"processor": "EV74"}, {"processor": "MLA"}, {"processor": "EV74"}]
    inputs = [{"name": "pixel_values", "size": 3 * 224 * 224 * 4}]
    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", plugins, inputs))
    for name, arguments, message in (
        ("cpu", ([*plugins, {"processor": "A65"}], inputs), "host=1"),
        ("input", (plugins, [{"name": "pixel_values", "size": 4}]), "input interface"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--compile-invocation", "w8a16", "--fidelity-dataset", "cal.tgz"],
            "a build profile compiles in one invocation named after it",
        ),
        (["--compile-invocation", "w16a16"], "the following arguments are required: --fidelity-dataset"),
        (
            ["--compile-invocation", "w16a16", "--fidelity-dataset", "cal.tgz", "--model", "model.onnx"],
            "--model must be the converted weights.pth",
        ),
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
    source = tmp_path / "weights.pth"
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


def test_the_rebuilt_network_equals_timms_convnext_with_tanh_gelus(load_script: Callable[[str], ModuleType]) -> None:
    """On timm's convnext_base with seeded random weights, as the checkpoint's weights are not in the repository, and
    timm's GELUs in their tanh approximation, the form surgery.py writes as x * sigmoid(x * (x * x * b + a))."""
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    surgery = load_script("surgery")
    torch.manual_seed(0)
    model = timm.create_model("convnext_base", pretrained=False, num_classes=0).eval()
    for block in (block for stage in model.stages for block in stage.blocks):
        block.mlp.act = torch.nn.GELU(approximate="tanh")
    # timm initializes LayerScale to 1e-6, which would hide each block; random values exercise it.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("gamma", "norm.weight", "norm.bias")):
                value.uniform_(0.5, 1.5)
    rebuilt = surgery.build_network()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    pixels = torch.from_numpy(surgery.synthetic(123))

    with torch.inference_mode():
        class_embedding, patch_embeddings = rebuilt.eval()(pixels)
        features = model.forward_features(pixels)
        expected_patches = model.head.norm(features).flatten(2).transpose(1, 2)
        expected_class = model(pixels)

    np.testing.assert_allclose(class_embedding.numpy(), expected_class.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(patch_embeddings.numpy(), expected_patches.numpy(), rtol=1e-4, atol=1e-4)


def test_output_comparison_rejects_shape_nan_and_a_low_cosine(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    expected = np.random.default_rng(0).normal(size=surgery.INTERFACE["class_embedding"])

    assert surgery.compare_outputs("class_embedding", expected, 2 * expected) == pytest.approx(1.0)
    for actual, message in (
        (expected[:, :1], "Unexpected shape"),
        (np.full_like(expected, np.nan), "Non-finite"),
        (expected + np.random.default_rng(1).normal(size=expected.shape), "cosine similarity"),
    ):
        with pytest.raises(ValueError, match=message):
            surgery.compare_outputs("class_embedding", expected, actual)


def test_export_must_keep_the_embedding_interface(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in surgery.INTERFACE.items()
    ]
    nodes = [
        helper.make_node(
            "Constant", [], [value.name], value=helper.make_tensor(value.name, 1, shape, np.zeros(shape).ravel())
        )
        for value, shape in zip(values[1:], list(surgery.INTERFACE.values())[1:])
    ]
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            helper.make_graph(nodes, "fixture", values[:1], values[1:]),
            opset_imports=[helper.make_opsetid("", surgery.OPSET)],
        ),
        path,
    )
    surgery.validate_export(path)
    values[2].type.tensor_type.shape.dim[1].dim_value = 50
    onnx.save(
        helper.make_model(
            helper.make_graph(nodes, "fixture", values[:1], values[1:]),
            opset_imports=[helper.make_opsetid("", surgery.OPSET)],
        ),
        path,
    )
    with pytest.raises(ValueError, match="patch_embeddings"):
        surgery.validate_export(path)


def test_build_holds_one_audited_mpk_that_follows_the_reference(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    (archive,) = sorted((compiled / "model").glob("*_mpk.tar.gz"))
    recipe.audit_mpk(archive)
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in fidelity["metrics"]] == list(recipe.OUTPUTS)
    floor = FIDELITY_FLOORS[build_profile]
    assert all(metric["cosine_similarity"] >= floor for metric in fidelity["metrics"]), fidelity["metrics"]
