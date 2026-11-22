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
from model_registry.errors import ValidationError
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[2]
# The least cosine with reference.onnx a build keeps on the 5 fidelity images, per output. Model Compiler
# 3.0.0.dev0+develop.2291 measured the least output on the 50 cal images at positions 51 to 100 of the seed-42 shuffled
# order, whose first 5 are the fidelity images: w16a16 0.99934, w8a16 0.99554; CI compiles with Model Compiler 3.0.0.2.
# A floor allows twice the measured loss, 1 - cosine, when measured on all 50 images and three times when measured on
# fewer, rounded down to three decimals.
FIDELITY_FLOORS = {"w16a16": 0.998, "w8a16": 0.991}


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
    meta = transforms.Compose([
        transforms.ToImage(), transforms.Resize((224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

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


def test_the_rebuilt_encoder_equals_timms_dinov3(load_script: Callable[[str], ModuleType]) -> None:
    """On timm's own DINOv3 with random weights, as the checkpoint's weights are not in the repository; every rewrite
    must be exact.

    The initial LayerScale of 1e-5 would leave each block's output almost its input, so LayerScale and the norms'
    affines are drawn at random as well.
    """
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    surgery = load_script("surgery")
    torch.manual_seed(0)
    model = timm.create_model("vit_huge_plus_patch16_dinov3", pretrained=False, num_classes=0, img_size=224,
                              dynamic_img_size=False).eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("gamma_1", "gamma_2", "norm1.weight", "norm2.weight", "norm.weight")):
                value.uniform_(0.5, 1.5)
            elif name.endswith(("norm1.bias", "norm2.bias", "norm.bias")):
                value.normal_(0, 0.1)
    weights = {**model.state_dict(), surgery.ROTARY_EMBEDDING: model.rope.get_embed()}
    pixels = torch.from_numpy(surgery.synthetic_pixels(0))

    with torch.inference_mode():
        tokens = model.forward_features(pixels)
        class_embedding, patch_embeddings = surgery.build_encoder(
            {name: value.to(torch.float64) for name, value in weights.items()})(pixels)

    torch.testing.assert_close(class_embedding, tokens[:, 0], rtol=0, atol=1e-4)
    torch.testing.assert_close(patch_embeddings, tokens[:, surgery.PREFIX:], rtol=0, atol=1e-4)


def test_rotate_half_is_timms_on_each_heads_channels(load_script: Callable[[str], ModuleType]) -> None:
    torch = pytest.importorskip("torch")
    pos_embed_sincos = pytest.importorskip("timm.layers.pos_embed_sincos")
    surgery = load_script("surgery")
    # Unique values show any slip in the pairing or the signs; timm rotates the last axis, surgery.py axis 1.
    heads = torch.arange(surgery.HEAD_SIZE * surgery.HEADS * 3, dtype=torch.float64).reshape(
        1, surgery.HEAD_SIZE, surgery.HEADS, 3)

    expected = pos_embed_sincos.rope_rotate_half(heads.movedim(1, -1)).movedim(-1, 1)
    torch.testing.assert_close(surgery.rotate_half(heads), expected)


def test_weights_are_read_from_the_indexed_reference_initializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    torch = pytest.importorskip("torch")
    surgery = load_script("surgery")
    # Two small weights stand in for the encoder's, so the test needs no 3.4 GB reference.onnx.
    monkeypatch.setattr(surgery, "weight_shapes", lambda: {
        "norm.weight": [3], "blocks.0.attn.proj.weight": [2, 3], surgery.ROTARY_EMBEDDING: [2, 4]})
    norm, projection = np.arange(3, dtype=np.float32), np.arange(6, dtype=np.float32).reshape(2, 3)
    initializers = [onnx.numpy_helper.from_array(norm, "norm"), onnx.numpy_helper.from_array(projection.T, "matmul")]
    graph = onnx.helper.make_graph([], "reference", [], [], initializer=initializers)
    onnx.save(onnx.helper.make_model(graph), tmp_path / "reference.onnx")
    rotary = torch.ones(2, 4)
    index = {"norm.weight": ("norm", False), "blocks.0.attn.proj.weight": ("matmul", True)}
    torch.save({surgery.ROTARY_EMBEDDING: rotary, surgery.INITIALIZERS: index}, tmp_path / "weight-index.pth")

    weights = surgery.load_weights(tmp_path / "weight-index.pth", tmp_path / "reference.onnx")
    assert weights["norm.weight"].tolist() == norm.tolist()
    assert weights["blocks.0.attn.proj.weight"].tolist() == projection.tolist()
    assert weights[surgery.ROTARY_EMBEDDING].tolist() == rotary.tolist()

    # Read as is, the transposed projection has the wrong shape.
    index["blocks.0.attn.proj.weight"] = ("matmul", False)
    torch.save({surgery.ROTARY_EMBEDDING: rotary, surgery.INITIALIZERS: index}, tmp_path / "untransposed.pth")
    with pytest.raises(ValueError, match="are not DINOv3"):
        surgery.load_weights(tmp_path / "untransposed.pth", tmp_path / "reference.onnx")


def _export_like_model(surgery: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
              for name, shape in surgery.INTERFACE.items()]
    if case == "order":
        values[1], values[2] = values[2], values[1]
    elif case == "shape":
        values[2].type.tensor_type.shape.dim[1].dim_value = surgery.PREFIX + surgery.PATCHES
    nodes = []
    for value in values[1:]:
        zeros = np.zeros([d.dim_value for d in value.type.tensor_type.shape.dim], np.float32)
        nodes.append(helper.make_node("Constant", [], [value.name], value=onnx.numpy_helper.from_array(zeros)))
    if case == "dynamic":
        nodes.append(helper.make_node("Shape", ["pixel_values"], ["shape"]))
    graph = helper.make_graph(nodes, "export", values[:1], values[1:])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else surgery.OPSET)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "shape", "dynamic"])
def test_the_export_interface(tmp_path: Path, load_script: Callable[[str], ModuleType], case: str) -> None:
    surgery = load_script("surgery")
    path = tmp_path / "model.onnx"
    onnx.save(_export_like_model(surgery, case), path)
    if case == "valid":
        surgery.validate_export(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_export(path)


def test_the_export_must_keep_the_references_embeddings(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    rng = np.random.default_rng(0)
    expected = [rng.normal(size=shape).astype(np.float32) for shape in surgery.OUTPUTS.values()]

    surgery.compare_embeddings(expected, [value * 1.01 + 1e-4 for value in expected])
    for broken in ([expected[0], -expected[1]], [expected[0], expected[1][:, 1:]],
                   [np.full_like(expected[0], np.nan), expected[1]]):
        with pytest.raises(ValueError):
            surgery.compare_embeddings(expected, broken)


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(recipe.MODEL_DIR)

    for profile in metadata["build_profiles"]:
        parameters = recipe.component_parameters(recipe.MODEL_DIR, profile["id"], recipe.COMPONENT_ID)
        assert parameters == {"precision": profile["id"]}
        # BF16 activations take no calibration images.
        assert "calibration" not in profile["components"][0]
    assert metadata["numerical_fidelity"]["outputs"] == list(recipe.OUTPUTS)
    assert {profile["id"] for profile in metadata["build_profiles"]} == set(FIDELITY_FLOORS)
    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(recipe.MODEL_DIR, "w32a32", recipe.COMPONENT_ID)


def test_bf16_activations_skip_calibration(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    # Each AFE constructor returns its name and arguments, so the test reads the recipe without the Model Compiler.
    for name in ("QuantizationParams", "SkipCalibration", "bfloat16_scheme", "quantization_scheme"):
        monkeypatch.setattr(recipe, name, lambda *args, _name=name, **kwargs: (_name, args, kwargs))

    (_, _, bf16), (_, _, int8_weights) = (recipe._quantization_config(precision) for precision in ("w16a16", "w8a16"))

    assert bf16 == {
        "calibration_method": ("SkipCalibration", (), {}),
        "activation_quantization_scheme": ("bfloat16_scheme", (), {}),
        "weight_quantization_scheme": ("bfloat16_scheme", (), {}),
    }
    assert int8_weights == {
        "calibration_method": ("SkipCalibration", (), {}),
        "activation_quantization_scheme": ("bfloat16_scheme", (), {}),
        "weight_quantization_scheme": ("quantization_scheme", (), {"asymmetric": False, "per_channel": True,
                                                                   "bits": 8}),
        "per_token_quant": True,
    }
    with pytest.raises(ValueError, match="only the declared Modalix BF16-activation builds"):
        recipe._quantization_config("w8a8")


@pytest.mark.parametrize("profile", ["w16a16", "w8a16"])
def test_bf16_builds_quantize_without_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], profile: str
) -> None:
    recipe = load_script("compile")
    calls = {}

    class Loaded:
        def quantize(self, **arguments):
            calls.update(arguments)

    monkeypatch.setattr(recipe, "onnx_source", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(recipe, "load_model", lambda importer, target: Loaded())
    monkeypatch.setattr(recipe, "_quantization_config", lambda precision: f"{precision} config")

    recipe.quantize(tmp_path / "model.onnx", profile, profile)

    assert (calls["calibration_data"], calls["quantization_config"]) == (None, f"{profile} config")


def test_a_quantization_failure_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    # AFE ends a failed quantization with a bare SystemExit, whose exit status would be 0.
    recipe = load_script("compile")

    class Loaded:
        def quantize(self, **arguments):
            raise SystemExit()

    monkeypatch.setattr(recipe, "onnx_source", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(recipe, "load_model", lambda importer, target: Loaded())
    monkeypatch.setattr(recipe, "_quantization_config", lambda precision: f"{precision} config")

    with pytest.raises(RuntimeError, match="failed to quantize"):
        recipe.quantize(tmp_path / "model.onnx", "w16a16", "w16a16")


def test_a_compiler_failure_fails_the_build(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    # AFE ends a failed compile with a bare SystemExit, whose exit status would be 0.
    recipe = load_script("compile")

    class Quantized:
        def compile(self, **arguments):
            raise SystemExit()

    with pytest.raises(RuntimeError, match="failed to compile"):
        recipe.compile_model(Quantized(), tmp_path / "model")


def test_fidelity_compares_the_quantized_embeddings_with_reference_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    fidelity = recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]
    image_values = (1.0, 2.0)
    images = np.stack([np.full((224, 224, 3), value, np.float32) for value in image_values])
    reference_inputs = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            reference_inputs.append(feeds["pixel_values"].shape)
            return [np.full(recipe.OUTPUTS[name], feeds["pixel_values"][0, 0, 0, 0], np.float32) for name in outputs]

    quantized = MagicMock()
    # The outputs come back out of order, and measure_fidelity matches them by shape. Each image's patch embeddings
    # come back 1 too high.
    quantized.execute.side_effect = [
        [np.full(recipe.OUTPUTS["patch_embeddings"], value + 1), np.full(recipe.OUTPUTS["class_embedding"], value)]
        for value in image_values
    ]
    monkeypatch.setattr(recipe, "select_names", lambda dataset, seed, offset, samples: ["a.JPEG", "b.JPEG"])
    monkeypatch.setattr(recipe, "load_images", lambda dataset, names: images)
    monkeypatch.setattr(recipe, "ort", MagicMock(InferenceSession=Session))

    report = recipe.measure_fidelity(tmp_path / "reference.onnx", quantized, tmp_path / "cal", fidelity, "w8a16")

    assert reference_inputs == [(1, 3, 224, 224)] * 2
    assert report["model_id"] == "dinov3_vit_h_plus_16"
    assert (report["build_profile_id"], report["samples"]) == ("w8a16", 2)
    class_embedding, patch_embeddings = report["metrics"]
    assert (class_embedding["name"], class_embedding["max_absolute_error"]) == ("class_embedding", 0.0)
    assert (patch_embeddings["name"], patch_embeddings["mean_absolute_error"]) == ("patch_embeddings", 1.0)


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
        (["--compile-invocation", "w16a16"], "the following arguments are required: --fidelity-dataset"),
        (["--compile-invocation", "model", "--fidelity-dataset", "cal.tar.gz"],
         "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16", "--fidelity-dataset", "cal.tar.gz", "--model", "model.onnx"],
         "--model must be the converted weight-index.pth"),
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
    source = tmp_path / "weight-index.pth"
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
