from __future__ import annotations

import importlib.util
import io
import json
import logging
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
    local_dependencies = ("calibration", "surgery") if name == "compile" else ()
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


def synthetic_source(path: Path, surgery: ModuleType) -> Path:
    nodes: list[onnx.NodeProto] = []
    initializers = [
        numpy_helper.from_array(np.zeros((1, 3, 1, 1), dtype=np.float32), "weights_0"),
        numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "bias_0"),
    ]
    nodes.append(helper.make_node("Conv", [surgery.INPUT_NAME, "weights_0", "bias_0"], ["conv_0"]))
    for index in range(1, surgery.CONVOLUTION_COUNT):
        weights = f"weights_{index}"
        bias = f"bias_{index}"
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), weights),
                numpy_helper.from_array(np.zeros((1,), dtype=np.float32), bias),
            ]
        )
        nodes.append(helper.make_node("Conv", [f"conv_{index - 1}", weights, bias], [f"conv_{index}"]))
    for name, shape in surgery.OUTPUTS:
        shape_name = f"{name}_shape"
        initializers.append(numpy_helper.from_array(np.asarray(shape, dtype=np.int64), shape_name))
        nodes.append(helper.make_node("ConstantOfShape", [shape_name], [name]))
    graph = helper.make_graph(
        nodes,
        "ssd_source_fixture",
        [helper.make_tensor_value_info(surgery.INPUT_NAME, TensorProto.FLOAT, surgery.INPUT_SHAPE)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in surgery.OUTPUTS],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    helper.set_model_props(
        model,
        {
            "sima.output_order": "loc[0..5],confidence[0..5]",
            "source.archive.sha256": surgery.UPSTREAM_ARCHIVE_SHA256,
        },
    )
    onnx.save(model, path)
    return path


def test_surgery_validates_static_heads_and_copies_source_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_module("surgery")
    source = synthetic_source(tmp_path / "source.onnx", surgery)
    monkeypatch.setattr(surgery, "_sha256", lambda path: surgery.SOURCE_SHA256)
    destination = surgery.prepare_model(source, tmp_path / "prepared.onnx")
    prepared = onnx.load(destination, load_external_data=False)
    surgery.validate_prepared_model(prepared)
    assert source.read_bytes() == destination.read_bytes()
    assert [value.name for value in prepared.graph.output] == [name for name, _ in surgery.OUTPUTS]
    assert not any(node.op_type == "Identity" for node in prepared.graph.node)


def test_int8_surgery_stabilizes_only_negligible_nonzero_weight_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_module("surgery")
    source = synthetic_source(tmp_path / "source.onnx", surgery)
    model = onnx.load(source)
    weights = next(value for value in model.graph.initializer if value.name == "weights_1")
    weights.CopyFrom(numpy_helper.from_array(np.full((1, 1, 1, 1), 1e-8, dtype=np.float32), weights.name))
    onnx.save(model, source)
    monkeypatch.setattr(surgery, "_sha256", lambda path: surgery.SOURCE_SHA256)
    monkeypatch.setattr(surgery, "EXPECTED_STABILIZED_CHANNELS", 1)

    first = surgery.prepare_int8_model(source, tmp_path / "prepared-first.onnx")
    second = surgery.prepare_int8_model(source, tmp_path / "prepared-second.onnx")
    prepared = onnx.load(first, load_external_data=False)
    surgery.validate_int8_prepared_model(prepared)

    stabilized = numpy_helper.to_array(next(value for value in prepared.graph.initializer if value.name == "weights_1"))
    untouched_zero = numpy_helper.to_array(
        next(value for value in prepared.graph.initializer if value.name == "weights_0")
    )
    assert np.array_equal(stabilized, np.full((1, 1, 1, 1), surgery.NEGLIGIBLE_WEIGHT_FLOOR))
    assert np.count_nonzero(untouched_zero) == 0
    assert first.read_bytes() == second.read_bytes()
    assert sum(node.op_type == "Conv" for node in prepared.graph.node) == surgery.CONVOLUTION_COUNT


def test_model_sources_match_pinned_v3_assets() -> None:
    surgery = load_module("surgery")
    calibration = load_module("calibration")
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    assert metadata["source"] == {
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/500803387",
        "sha256": surgery.SOURCE_SHA256,
        "size_bytes": 13075527,
        "media_type": "application/onnx",
    }
    assert metadata["calibration"]["source"] == {
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/500803386",
        "sha256": calibration.ARCHIVE_SHA256,
        "size_bytes": 55112720,
        "media_type": "application/x-npz",
    }
    assert metadata["surgery"]["parameters"]["source_archive_sha256"] == surgery.UPSTREAM_ARCHIVE_SHA256
    assert metadata["surgery"]["parameters"]["int8_negligible_weight_floor"] == pytest.approx(
        float(surgery.NEGLIGIBLE_WEIGHT_FLOOR)
    )
    assert metadata["surgery"]["parameters"]["int8_stabilized_weight_channels"] == (
        surgery.EXPECTED_STABILIZED_CHANNELS
    )


def test_calibration_split_has_128_representative_and_32_disjoint_samples() -> None:
    calibration = load_module("calibration")
    assert len(calibration.CALIBRATION_IMAGE_IDS) == 128
    assert len(calibration.VALIDATION_IMAGE_IDS) == 32
    assert set(calibration.CALIBRATION_IMAGE_IDS).isdisjoint(calibration.VALIDATION_IMAGE_IDS)
    assert len(set(calibration.CALIBRATION_IMAGE_IDS)) == 128
    assert len(set(calibration.VALIDATION_IMAGE_IDS)) == 32
    assert calibration.IMAGE_SIZE == 320
    assert calibration.IMAGE_DATASET_REVISION == "9e50abcdc1361852f34841af4939cbcd2d37c92f"


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
    plugins.append(plugin("MLA", sequence))
    sequence += 1
    internal_type = "int8" if precision == "int8" or not tessellation_mla else "bfloat16"
    plugins.append(
        plugin(
            "EV74",
            sequence,
            kernel="unpack_transform",
            params={"tensor_types": [internal_type] * 12},
        )
    )
    sequence += 1
    if precision == "int8":
        for _ in range(12):
            plugins.append(
                plugin("EV74", sequence, kernel="dequantization_transform", params={"input_data_type": "int8"})
            )
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
@pytest.mark.parametrize("tessellation_mla", [False, True])
def test_mpk_audit_proves_all_four_variants(tmp_path: Path, precision: str, tessellation_mla: bool) -> None:
    compile_module = load_module("compile")
    suffix = "tess_mla" if tessellation_mla else "tess_off_mla"
    variant = f"modalix_{precision}_{suffix}"
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


def test_quantization_warning_audit_rejects_zeroed_weights() -> None:
    compile_module = load_module("compile")
    audit = compile_module.QuantizationWarningAudit()
    audit.emit(
        logging.LogRecord(
            "afe.ir.defines",
            logging.WARNING,
            __file__,
            1,
            "Weights were set to zero to avoid numeric saturation",
            (),
            None,
        )
    )
    with pytest.raises(RuntimeError, match="zero-weight quantization warnings"):
        audit.enforce()


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


def test_recipe_uses_supported_compiler_2_1_3_api_only() -> None:
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    assert "with_activation_quantization" in source
    assert "with_weight_quantization" in source
    assert "with_smooth_quant(True)" in source
    assert "with_bias_correction" in source
    assert "BiasCorrectionType.ITERATIVE" in source
    assert "quantization_mla" not in source
    assert "public_input_types" not in source
    assert "public_output_types" not in source
    assert "use_jax=True" in source
    assert "os.execv" in source
    assert "Model.load" in source
    assert "--validation-only" in source


def test_model_declares_exact_four_variant_cross_product() -> None:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matrix = {
        (variant["parameters"]["precision"], variant["parameters"]["tessellation_mla"])
        for variant in metadata["compile_variants"]
    }
    assert matrix == {("int8", False), ("int8", True), ("bf16", False), ("bf16", True)}
    assert all(variant["parameters"]["calibration_samples"] == 128 for variant in metadata["compile_variants"])
    assert {
        (variant["parameters"]["precision"], variant["parameters"]["calibration_method"])
        for variant in metadata["compile_variants"]
    } == {("int8", "min_max"), ("bf16", "mse")}
    int8_parameters = [
        variant["parameters"]
        for variant in metadata["compile_variants"]
        if variant["parameters"]["precision"] == "int8"
    ]
    assert all(parameters["smooth_quant"] is True for parameters in int8_parameters)
    assert all(parameters["bias_correction"] == "iterative" for parameters in int8_parameters)
    status = {
        (variant["parameters"]["precision"], variant["parameters"]["tessellation_mla"]): variant.get(
            "status", "supported"
        )
        for variant in metadata["compile_variants"]
    }
    assert status == {
        ("int8", False): "unsupported",
        ("int8", True): "unsupported",
        ("bf16", False): "supported",
        ("bf16", True): "supported",
    }
