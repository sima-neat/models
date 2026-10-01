from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "ssd_mobilenet_v3"


def load_module(name: str) -> ModuleType:
    path = MODEL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ssd_mobilenet_v3_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    local_dependencies = {
        "compile": ("calibration", "surgery", "contract", "source_bundle", "onnx_audit"),
        "surgery": ("contract", "source_bundle", "onnx_audit"),
    }.get(name, ())
    previous_dependencies = {dependency: sys.modules.pop(dependency, None) for dependency in local_dependencies}
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
        for dependency, previous in previous_dependencies.items():
            sys.modules.pop(dependency, None)
            if previous is not None:
                sys.modules[dependency] = previous
    return module


def load_qat_module(name: str) -> ModuleType:
    path = MODEL_DIR / "qat" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ssd_mobilenet_v3_qat_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    previous_contract = sys.modules.pop("contract", None)
    sys.path.insert(0, str(MODEL_DIR / "qat"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR / "qat"))
        sys.modules.pop("contract", None)
        if previous_contract is not None:
            sys.modules["contract"] = previous_contract
    return module


def test_validation_restart_preserves_component(tmp_path: Path, monkeypatch) -> None:
    recipe = load_module("compile")
    source = tmp_path / "source.onnx"
    source.touch()
    profile = "modalix_int8"
    monkeypatch.setattr(sys, "argv", [
        "compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"),
        "--build-profile", profile, "--component", "model",
    ])
    monkeypatch.setattr(recipe, "compile_model", lambda *args: None)
    captured = []

    def capture_restart(executable, argv):
        captured.extend(argv)
        raise SystemExit(0)

    monkeypatch.setattr(recipe.os, "execv", capture_restart)
    with pytest.raises(SystemExit) as stopped:
        recipe.main()
    assert stopped.value.code == 0
    assert captured[captured.index("--component") + 1] == "model"
    assert captured[captured.index("--build-profile") + 1] == profile
    assert "--validation-only" in captured


def synthetic_source(path: Path, contract: ModuleType, *, qat: bool) -> Path:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    previous = contract.INPUT_NAME
    if qat:
        initializers.extend(
            [
                numpy_helper.from_array(np.asarray(0.01, dtype=np.float32), "activation_scale"),
                numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "activation_zero"),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [previous, "activation_scale", "activation_zero"],
                    ["input_q"],
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["input_q", "activation_scale", "activation_zero"],
                    ["input_dq"],
                ),
            ]
        )
        previous = "input_dq"
    input_channels = 3
    for index in range(contract.CONVOLUTION_COUNT):
        weight = f"weight_{index}"
        convolution_input = weight
        if qat:
            scale = f"weight_scale_{index}"
            zero = f"weight_zero_{index}"
            convolution_input = f"weight_dq_{index}"
            initializers.extend(
                [
                    numpy_helper.from_array(np.ones((2, input_channels, 1, 1), dtype=np.int8), weight),
                    numpy_helper.from_array(np.full((2,), 0.01, dtype=np.float32), scale),
                    numpy_helper.from_array(np.zeros((2,), dtype=np.int8), zero),
                ]
            )
            nodes.append(helper.make_node("DequantizeLinear", [weight, scale, zero], [convolution_input], axis=0))
        else:
            initializers.append(numpy_helper.from_array(np.ones((2, input_channels, 1, 1), dtype=np.float32), weight))
        output = f"conv_{index}"
        nodes.append(helper.make_node("Conv", [previous, convolution_input], [output], name=output))
        previous = output
        input_channels = 2
    for name, shape in contract.OUTPUTS:
        shape_name = f"{name}_shape"
        initializers.append(numpy_helper.from_array(np.asarray(shape, dtype=np.int64), shape_name))
        nodes.append(helper.make_node("ConstantOfShape", [shape_name], [name]))
    graph = helper.make_graph(
        nodes,
        "ssd_source_fixture",
        [helper.make_tensor_value_info(contract.INPUT_NAME, TensorProto.FLOAT, contract.INPUT_SHAPE)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in contract.OUTPUTS],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def synthetic_bundle(path: Path, tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    contract = load_qat_module("contract")
    audit = load_qat_module("onnx_audit")
    fp32 = synthetic_source(tmp_path / "fp32.onnx", contract, qat=False)
    int8 = synthetic_source(tmp_path / "int8.onnx", contract, qat=True)
    members = {"fp32.onnx": fp32.read_bytes(), "int8.onnx": int8.read_bytes()}
    manifest = {
        "schema_version": 1,
        "architecture": "torchvision.ssdlite320_mobilenet_v3_large",
        "accuracy": {"fp32_ap": 0.213, "qat_ap": 0.211, "ap_drop_points": 0.2},
        "sources": {
            "bf16": {"file": "fp32.onnx", "audit": audit.audit_onnx(fp32, expect_qat=False)},
            "int8": {"file": "int8.onnx", "audit": audit.audit_onnx(int8, expect_qat=True)},
        },
    }
    members["manifest.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return path, members


def test_surgery_selects_immutable_precision_sources(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source, members = synthetic_bundle(tmp_path / "source.tar.gz", tmp_path)
    bf16 = surgery.prepare_model(source, tmp_path / "prepared-bf16.onnx")
    int8_first = surgery.prepare_int8_model(source, tmp_path / "prepared-int8-first.onnx")
    int8_second = surgery.prepare_int8_model(source, tmp_path / "prepared-int8-second.onnx")
    assert bf16.read_bytes() == members["fp32.onnx"]
    assert int8_first.read_bytes() == members["int8.onnx"]
    assert int8_first.read_bytes() == int8_second.read_bytes()


def test_surgery_rejects_unqualified_source_bundle(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source, members = synthetic_bundle(tmp_path / "source.tar.gz", tmp_path)
    manifest = json.loads(members["manifest.json"])
    manifest["accuracy"]["ap_drop_points"] = 1.50001
    manifest["accuracy"]["qat_ap"] = manifest["accuracy"]["fp32_ap"] - 0.0150001
    members["manifest.json"] = json.dumps(manifest).encode()
    with tarfile.open(source, "w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="exceeds the 1.5 AP-point"):
        surgery.prepare_int8_model(source, tmp_path / "rejected.onnx")


def test_int8_reference_evaluator_adapts_opset17_qdq_dispatch(tmp_path: Path) -> None:
    compile_module = load_module("compile")
    contract = load_qat_module("contract")
    source = synthetic_source(tmp_path / "int8.onnx", contract, qat=True)
    evaluator = compile_module.build_reference_evaluator(source, "int8")
    outputs = evaluator.run(
        None,
        {contract.INPUT_NAME: np.zeros(contract.INPUT_SHAPE, dtype=np.float32)},
    )
    assert [output.shape for output in outputs] == [shape for _name, shape in contract.OUTPUTS]
    assert all(np.isfinite(output).all() for output in outputs)


def test_model_sources_match_pinned_v3_assets() -> None:
    calibration = load_module("calibration")
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    component = metadata["components"][0]
    assert component["source"] == {
        "type": "file",
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/502158604",
        "sha256": "5655500a6f5046b60c3e90b24d18d9c67977a732abeeccdbacadf9dfbce3ac5a",
        "size_bytes": 15784877,
        "media_type": "application/x-gtar",
    }
    assert metadata["calibrations"][0]["source"] == {
        "type": "file",
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/500803386",
        "sha256": calibration.ARCHIVE_SHA256,
        "size_bytes": 55112720,
        "media_type": "application/x-npz",
    }
    assert component["surgery"]["parameters"]["bf16_source"] == "ssd_mobilenet_v3_torchvision_fp32.onnx"
    assert component["surgery"]["parameters"]["int8_source"] == "ssd_mobilenet_v3_torchvision_qat_int8.onnx"


def test_calibration_split_has_128_representative_and_32_disjoint_samples() -> None:
    calibration = load_module("calibration")
    assert len(calibration.CALIBRATION_IMAGE_IDS) == 128
    assert len(calibration.VALIDATION_IMAGE_IDS) == 32
    assert set(calibration.CALIBRATION_IMAGE_IDS).isdisjoint(calibration.VALIDATION_IMAGE_IDS)
    assert len(set(calibration.CALIBRATION_IMAGE_IDS)) == 128
    assert len(set(calibration.VALIDATION_IMAGE_IDS)) == 32
    assert calibration.IMAGE_SIZE == 320
    assert calibration.IMAGE_DATASET_REVISION == "9e50abcdc1361852f34841af4939cbcd2d37c92f"


def test_compile_recipe_uses_verified_calibration_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_module("compile")
    verified = tmp_path / "calibration.npz"
    verified.write_bytes(b"verified")
    monkeypatch.setenv("CALIBRATION_PATH", str(verified))

    class UnexpectedFetcher:
        def fetch(self, _source: Any) -> None:
            raise AssertionError("verified calibration must not be downloaded again")

    monkeypatch.setattr(recipe, "Fetcher", UnexpectedFetcher)
    assert recipe.resolve_calibration_path() == verified.resolve()


def test_calibration_rejects_wrong_sample_shape() -> None:
    calibration = load_module("calibration")
    with pytest.raises(ValueError, match="calibration must be"):
        calibration.validate_arrays(
            np.zeros((1, 320, 320, 3), dtype=np.float32),
            np.zeros((32, 320, 320, 3), dtype=np.float32),
            np.asarray(calibration.CALIBRATION_IMAGE_IDS, dtype=np.int64),
            np.asarray(calibration.VALIDATION_IMAGE_IDS, dtype=np.int64),
            check_hashes=False,
        )


def test_compiler_converts_pinned_calibration_to_imagenet_normalization() -> None:
    compile_module = load_module("compile")
    samples = np.asarray([[[[-1.0, 0.0, 1.0]]]], dtype=np.float32)
    normalized = compile_module.imagenet_normalize(samples)
    expected = (np.asarray([0.0, 0.5, 1.0], dtype=np.float32) - compile_module.IMAGENET_MEAN) / (
        compile_module.IMAGENET_STD
    )
    assert normalized.dtype == np.float32
    assert normalized.shape == samples.shape
    assert np.allclose(normalized[0, 0, 0], expected)


def test_torchvision_qat_contract_uses_six_anchors_on_every_level() -> None:
    contract = load_qat_module("contract")
    assert contract.ANCHORS_PER_LOCATION == (6, 6, 6, 6, 6, 6)
    assert contract.TOTAL_ANCHORS == 3234
    assert contract.CONVOLUTION_COUNT == 98
    assert [name for name, _shape in contract.OUTPUTS] == [
        *(f"ssd_loc_{level}" for level in range(6)),
        *(f"ssd_confidence_{level}" for level in range(6)),
    ]
    assert contract.OUTPUTS[0][1] == (1, 24, 20, 20)
    assert contract.OUTPUTS[6][1] == (1, 546, 20, 20)


def test_qat_environment_is_python310_and_fully_pinned() -> None:
    qat_dir = MODEL_DIR / "qat"
    direct = {
        line.strip().lower()
        for line in (qat_dir / "requirements.in").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    locked = {
        line.strip().lower()
        for line in (qat_dir / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert direct <= locked
    assert "torch==2.3.1" in locked
    assert "torchvision==0.18.1" in locked
    assert "onnx==1.17.0" in locked
    bootstrap = (qat_dir / "bootstrap.sh").read_text(encoding="utf-8")
    assert "python install 3.10.18" in bootstrap
    assert 'qat_version == "1.0.0.dev0+develop.20"' in bootstrap
    assert "SIMA_QAT_WHEEL" in bootstrap
    assert "SIMA_QAT_SITE_PACKAGES" in bootstrap


def test_qat_onnx_audit_requires_per_channel_weights(tmp_path: Path) -> None:
    contract = load_qat_module("contract")
    audit = load_qat_module("onnx_audit")
    nodes = []
    initializers = []
    input_name = contract.INPUT_NAME
    activation_scale = numpy_helper.from_array(np.asarray(0.01, dtype=np.float32), "activation_scale")
    activation_zero = numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "activation_zero")
    initializers.extend([activation_scale, activation_zero])
    nodes.extend(
        [
            helper.make_node("QuantizeLinear", [input_name, "activation_scale", "activation_zero"], ["input_q"]),
            helper.make_node("DequantizeLinear", ["input_q", "activation_scale", "activation_zero"], ["input_dq"]),
        ]
    )
    previous = "input_dq"
    input_channels = 3
    for index in range(contract.CONVOLUTION_COUNT):
        weight = f"weight_{index}"
        scale = f"weight_scale_{index}"
        zero = f"weight_zero_{index}"
        weight_dq = f"weight_dq_{index}"
        output = f"conv_{index}"
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((2, input_channels, 1, 1), dtype=np.int8), weight),
                numpy_helper.from_array(np.full((2,), 0.01, dtype=np.float32), scale),
                numpy_helper.from_array(np.zeros((2,), dtype=np.int8), zero),
            ]
        )
        nodes.append(helper.make_node("DequantizeLinear", [weight, scale, zero], [weight_dq], axis=0))
        nodes.append(helper.make_node("Conv", [previous, weight_dq], [output]))
        previous = output
        input_channels = 2
    for name, shape in contract.OUTPUTS:
        shape_name = f"{name}_shape"
        initializers.append(numpy_helper.from_array(np.asarray(shape, dtype=np.int64), shape_name))
        nodes.append(helper.make_node("ConstantOfShape", [shape_name], [name]))
    graph = helper.make_graph(
        nodes,
        "qat_audit_fixture",
        [helper.make_tensor_value_info(contract.INPUT_NAME, TensorProto.FLOAT, contract.INPUT_SHAPE)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in contract.OUTPUTS],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "qat.onnx"
    onnx.save(model, path)
    report = audit.audit_onnx(path, expect_qat=True)
    assert report["per_channel_weight_count"] == contract.CONVOLUTION_COUNT
    assert report["activation_qdq_count"] == 1


def plugin(
    processor: str,
    sequence: int,
    *,
    kernel: str = "",
    params: dict[str, Any] | None = None,
    output_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "processor": processor,
        "config_params": {"kernel": kernel, "params": params or {}},
        "input_nodes": [],
        "output_nodes": output_nodes or [],
    }


def write_mpk(path: Path, compile_module: ModuleType, precision: str, tessellation_mla: bool) -> Path:
    sequence = 1
    plugins = []
    if precision == "int8":
        plugins.append(plugin("EV74", sequence, kernel="quantization_transform", params={"output_data_type": "int8"}))
        sequence += 1
    plugins.append(plugin("MLA", sequence, output_nodes=[{"name": f"MLA_{i}"} for i in range(12)]))
    sequence += 1
    for _ in range(12):
        kernel = "dequantization_transform" if precision == "int8" else "cast_transform"
        params = {"input_data_type": "int8"} if precision == "int8" else {
            "in_dtype": "bfloat16", "out_dtype": "float32"
        }
        plugins.append(plugin("EV74", sequence, kernel=kernel, params=params))
        sequence += 1
    plugins.append(
        plugin(
            "EV74",
            sequence,
            kernel="pass_through",
            output_nodes=[{"size": size} for size in compile_module.PUBLIC_OUTPUT_SIZES],
        )
    )
    manifest = json.dumps(
        {
            "model_sdk_version": compile_module.COMPILER_VERSION,
            "input_nodes": [{"name": compile_module.INPUT_NAME, "size": compile_module.PUBLIC_INPUT_SIZE}],
            "plugins": plugins,
        }
    ).encode()
    with tarfile.open(path, "w:gz") as stream:
        members = [("ssd_mpk.json", manifest), ("ssd.elf", b"ELF")]
        if not tessellation_mla:
            stage = "0_quanttess.json" if precision == "int8" else "0_tessellate.json"
            members.append((stage, b"{}"))
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    return path


@pytest.mark.parametrize("precision", ["int8", "bf16"])
def test_mpk_audit_proves_default_variants(tmp_path: Path, precision: str) -> None:
    compile_module = load_module("compile")
    tessellation_mla = True
    variant = f"modalix_{precision}"
    archive = write_mpk(tmp_path / f"{variant}_mpk.tar.gz", compile_module, precision, tessellation_mla)
    report = compile_module.audit_mpk(archive, variant)
    assert report["output_transport_tensor_count"] == 12
    assert report["tessellation"] == ("mla" if tessellation_mla else "ev74")
    expected_stages = []
    if not tessellation_mla:
        expected_stages = ["0_quanttess.json" if precision == "int8" else "0_tessellate.json"]
    assert report["tessellation_stages"] == expected_stages
    assert report["public_output_dtype"] == "float32"


@pytest.mark.parametrize(("precision", "scalar_type"), [("int8", 6), ("bf16", 13)])
def test_quantized_topology_audit_proves_compute_and_weight_precision(
    tmp_path: Path, precision: str, scalar_type: int
) -> None:
    compile_module = load_module("compile")
    layers = []
    for index in range(98):
        config: dict[str, Any] = {
            "_status": "SIMA_QUANTIZED",
            "conv_attrs.input_type": {"value": scalar_type},
        }
        if precision == "int8":
            config.update({"per_channel": True, "weight_quant_data": f"weights_{index}"})
        else:
            config["weights_attrs.data"] = f"weights_{index}"
        layers.append({"class_name": f"MLA_0/ConvAddActivation_{index}", "config": config})
    path = tmp_path / f"{precision}.sima.json"
    path.write_text(
        json.dumps({"modelTopology": {"model_config": {"config": {"layers": layers}}}}),
        encoding="utf-8",
    )
    report = compile_module.audit_quantized_model(path, precision)
    assert report["internal_precision"] == precision
    assert report["compute_convolution_count"] == 98


def test_quantization_quality_report_measures_every_raw_head() -> None:
    compile_module = load_module("compile")
    reference = [
        np.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
        for _, shape in compile_module.OUTPUTS
    ]
    candidate = [value * np.float32(0.99) + np.float32(0.001) for value in reference]
    report = compile_module.quantization_quality_report([reference, reference], [candidate, candidate])
    assert report["sample_count"] == 2
    assert len(report["outputs"]) == 12
    assert report["minimum_cosine_similarity"] > 0.999
    assert report["maximum_normalized_root_mean_square_error"] < 0.02


def test_quantization_quality_report_rejects_non_finite_output() -> None:
    compile_module = load_module("compile")
    reference = [np.ones(shape, dtype=np.float32) for _, shape in compile_module.OUTPUTS]
    candidate = [value.copy() for value in reference]
    candidate[0].flat[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        compile_module.quantization_quality_report([reference], [candidate])


def test_validation_output_normalization_handles_compiler_layouts() -> None:
    compile_module = load_module("compile")
    expected_shape = (1, 2, 3, 4)
    nchw = np.arange(np.prod(expected_shape), dtype=np.float32).reshape(expected_shape)
    nhwc = nchw.transpose(0, 2, 3, 1)
    nwch = nchw.transpose(0, 3, 1, 2)
    assert np.array_equal(compile_module._normalize_validation_output(nhwc, "head", expected_shape), nchw)
    assert np.array_equal(compile_module._normalize_validation_output(nwch, "head", expected_shape), nchw)


@pytest.mark.parametrize(
    ("precision", "passing", "failing"),
    [
        (
            "int8",
            {"minimum_cosine_similarity": 0.985, "maximum_normalized_root_mean_square_error": 0.18},
            {"minimum_cosine_similarity": 0.94209, "maximum_normalized_root_mean_square_error": 0.33537},
        ),
        (
            "bf16",
            {"minimum_cosine_similarity": 0.9995, "maximum_normalized_root_mean_square_error": 0.04},
            {"minimum_cosine_similarity": 0.998, "maximum_normalized_root_mean_square_error": 0.04},
        ),
    ],
)
def test_quantization_quality_gate_passes_and_rejects_regression(
    precision: str,
    passing: dict[str, float],
    failing: dict[str, float],
) -> None:
    compile_module = load_module("compile")
    assert compile_module.enforce_quantization_quality(passing, precision)["gate"]["passed"] is True
    with pytest.raises(RuntimeError, match=rf"{precision.upper()} quantization quality failed"):
        compile_module.enforce_quantization_quality(failing, precision)


def test_quantization_validation_selects_executor_explicitly() -> None:
    compile_module = load_module("compile")

    class Reference:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert outputs is None
            self.shapes.append(inputs[compile_module.INPUT_NAME].shape)
            return [np.ones(shape, dtype=np.float32) for _, shape in compile_module.OUTPUTS]

    class Executor:
        def __init__(self) -> None:
            self.use_jax: list[bool | None] = []

        def execute(self, *, inputs: dict[Any, np.ndarray], use_jax: bool | None = None) -> list[np.ndarray]:
            del inputs
            self.use_jax.append(use_jax)
            return [np.ones(shape, dtype=np.float32) for _, shape in compile_module.OUTPUTS]

    floating = Reference()
    quantized = Executor()
    validation = np.zeros((2, 320, 320, 3), dtype=np.float32)
    report = compile_module.validate_quantization(
        floating,
        quantized,
        validation,
        lambda name: name,
    )
    assert report["executor"] == "onnx_reference_and_jax_quantized"
    assert floating.shapes == [(1, 3, 320, 320), (1, 3, 320, 320)]
    assert quantized.use_jax == [True, True]


def test_recipe_preserves_quantization_and_public_io_policy() -> None:
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    assert "with_activation_quantization" in source
    assert "with_weight_quantization" in source
    assert "with_smooth_quant" not in source
    assert "with_bias_correction" not in source
    assert "BiasCorrectionType" not in source
    assert "quantization_mla" not in source
    assert "public_input_types" not in source
    assert "public_output_types" not in source
    assert "use_jax=True" in source
    assert "os.execv" in source
    assert "Model.load" in source
    assert "--validation-only" in source


def test_model_declares_default_precision_profiles() -> None:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    bindings = [profile["components"][0] for profile in metadata["build_profiles"]]
    matrix = {
        binding["parameters"]["precision"]
        for binding in bindings
    }
    assert matrix == {"int8", "bf16"}
    assert all(binding["parameters"]["calibration_samples"] == 128 for binding in bindings)
    assert {
        (binding["parameters"]["precision"], binding["parameters"]["calibration_method"])
        for binding in bindings
    } == {("int8", "min_max"), ("bf16", "mse")}
    int8_parameters = [
        binding["parameters"]
        for binding in bindings
        if binding["parameters"]["precision"] == "int8"
    ]
    assert all(parameters["quantization_method"] == "qat" for parameters in int8_parameters)
    assert all(parameters.get("smooth_quant", False) is False for parameters in int8_parameters)
    assert all(parameters.get("bias_correction", "none") == "none" for parameters in int8_parameters)
    bf16_parameters = [
        binding["parameters"]
        for binding in bindings
        if binding["parameters"]["precision"] == "bf16"
    ]
    assert all(parameters["quantization_method"] == "ptq" for parameters in bf16_parameters)
    status = {
        binding["parameters"]["precision"]: binding.get(
            "status", "supported"
        )
        for binding in bindings
    }
    assert status == {
        "int8": "supported",
        "bf16": "supported",
    }
