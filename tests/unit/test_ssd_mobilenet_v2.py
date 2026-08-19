from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "ssd_mobilenet_v2"


def load_module(name: str) -> ModuleType:
    path = MODEL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ssd_mobilenet_v2_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    shadowed = {dependency: sys.modules.pop(dependency, None) for dependency in ("calibration", "surgery")}
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
        for dependency, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(dependency, None)
            else:
                sys.modules[dependency] = previous
    return module


def synthetic_source(path: Path, surgery: ModuleType) -> Path:
    nodes = []
    initializers = []
    strides = (16, 32, 64, 128, 256, 300)
    for level, spec in enumerate(surgery.LEVELS):
        for role, channels in (("BoxEncodingPredictor", spec[2]), ("ClassPredictor", spec[3])):
            prefix = f"BoxPredictor_{level}/{role}/BiasAdd"
            weights = f"{prefix}/weights"
            bias = f"{prefix}/bias"
            initializers.extend(
                [
                    numpy_helper.from_array(np.zeros((channels, 3, 1, 1), dtype=np.float32), weights),
                    numpy_helper.from_array(np.zeros((channels,), dtype=np.float32), bias),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Conv",
                    [surgery.SOURCE_INPUT, weights, bias],
                    [f"{prefix}:0"],
                    name=prefix,
                    strides=[strides[level], strides[level]],
                )
            )
    graph = helper.make_graph(
        nodes,
        "ssd_source_fixture",
        [helper.make_tensor_value_info(surgery.SOURCE_INPUT, TensorProto.FLOAT, surgery.INPUT_SHAPE)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in surgery.SOURCE_OUTPUTS],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


def test_surgery_renames_and_reorders_static_heads_without_compute_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_module("surgery")
    source = synthetic_source(tmp_path / "source.onnx", surgery)
    monkeypatch.setattr(surgery, "_sha256", lambda path: surgery.SOURCE_SHA256)
    monkeypatch.setattr(surgery, "EXPECTED_CONVOLUTION_COUNT", 12)
    monkeypatch.setattr(surgery, "EXPECTED_OPERATOR_COUNTS", Counter({"Conv": 12}))
    destination = surgery.prepare_model(source, tmp_path / "prepared.onnx")
    prepared = onnx.load(destination, load_external_data=False)
    surgery.validate_prepared_model(prepared)
    assert [value.name for value in prepared.graph.output] == [name for name, _ in surgery.OUTPUTS]
    assert len(prepared.graph.node) == 12
    assert all(node.op_type == "Conv" for node in prepared.graph.node)
    assert not any(node.op_type == "Identity" for node in prepared.graph.node)


def test_model_pins_exact_immutable_v2_source_contract() -> None:
    surgery = load_module("surgery")
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    assert metadata["source"] == {
        "uri": ("hf://models/Kalray/ssd-mobilenet-v2@25a67532e30418eb68d1f49888374c16e8abfa73/ssd-mobilenet-v2.onnx"),
        "sha256": surgery.SOURCE_SHA256,
        "size_bytes": 67387440,
        "media_type": "application/onnx",
    }
    assert metadata["interface"]["opset"] == 13
    assert surgery.EXPECTED_CONVOLUTION_COUNT == 76


def test_calibration_split_has_128_representative_and_32_disjoint_samples() -> None:
    calibration = load_module("calibration")
    assert len(calibration.CALIBRATION_IMAGE_IDS) == 128
    assert len(calibration.VALIDATION_IMAGE_IDS) == 32
    assert set(calibration.CALIBRATION_IMAGE_IDS).isdisjoint(calibration.VALIDATION_IMAGE_IDS)
    assert len(set(calibration.CALIBRATION_IMAGE_IDS)) == 128
    assert len(set(calibration.VALIDATION_IMAGE_IDS)) == 32


def test_calibration_rejects_wrong_sample_shape() -> None:
    calibration = load_module("calibration")
    with pytest.raises(ValueError, match="calibration must be"):
        calibration.validate_arrays(
            np.zeros((1, 300, 300, 3), dtype=np.float32),
            np.zeros((32, 300, 300, 3), dtype=np.float32),
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
    for index in range(compile_module.EXPECTED_CONVOLUTION_COUNT):
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
    assert report["compute_convolution_count"] == compile_module.EXPECTED_CONVOLUTION_COUNT


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


@pytest.mark.parametrize(
    ("precision", "cosine", "nrmse"),
    [("int8", 0.9848, 0.1742), ("bf16", 0.9994, 0.0346)],
)
def test_quantization_quality_gate_accepts_qualified_measurements(
    precision: str, cosine: float, nrmse: float
) -> None:
    compile_module = load_module("compile")
    report = {
        "minimum_cosine_similarity": cosine,
        "maximum_normalized_root_mean_square_error": nrmse,
    }
    assert compile_module.enforce_quantization_quality(report, precision)["gate"]["passed"] is True


def test_quantization_quality_gate_rejects_regression() -> None:
    compile_module = load_module("compile")
    report = {
        "minimum_cosine_similarity": 0.97,
        "maximum_normalized_root_mean_square_error": 0.21,
    }
    with pytest.raises(RuntimeError, match="INT8 quantization quality failed"):
        compile_module.enforce_quantization_quality(report, "int8")


def test_quantization_quality_report_normalizes_afe_output_layouts() -> None:
    compile_module = load_module("compile")
    reference = [np.ones(shape, dtype=np.float32) for _, shape in compile_module.OUTPUTS]
    candidate = [value.transpose(0, 2, 3, 1) for value in reference]
    report = compile_module.quantization_quality_report([reference], [candidate])
    assert report["minimum_cosine_similarity"] == pytest.approx(1.0)
    assert report["maximum_normalized_root_mean_square_error"] == pytest.approx(0.0)


def test_quantization_quality_report_rejects_malformed_output_shape() -> None:
    compile_module = load_module("compile")
    reference = [np.ones(shape, dtype=np.float32) for _, shape in compile_module.OUTPUTS]
    candidate = [value.copy() for value in reference]
    candidate[0] = candidate[0].reshape(1, -1)
    with pytest.raises(RuntimeError, match="unsupported shape"):
        compile_module.quantization_quality_report([reference], [candidate])


def test_quantization_validation_executes_all_32_held_out_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_module = load_module("compile")
    monkeypatch.setattr(compile_module, "OUTPUTS", (("head", (1, 1, 1, 1)),))

    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, **_: Any) -> list[np.ndarray]:
            self.calls += 1
            return [np.ones((1, 1, 1, 1), dtype=np.float32)]

    floating = Executor()
    quantized = Executor()
    validation = np.zeros((32, 1, 1, 3), dtype=np.float32)
    report = compile_module.validate_quantization(floating, quantized, validation, lambda name: name)
    assert report["sample_count"] == 32
    assert floating.calls == 32
    assert quantized.calls == 32


def test_first_qualification_thresholds_are_committed_from_vulcan_evidence() -> None:
    specification = yaml.safe_load((MODEL_DIR / "tests" / "correctness.yaml").read_text(encoding="utf-8"))
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    assert specification["compiler_raw_head_thresholds"] == {
        "int8": {
            "minimum_cosine_similarity": 0.98,
            "maximum_normalized_root_mean_square_error": 0.20,
        },
        "bf16": {
            "minimum_cosine_similarity": 0.999,
            "maximum_normalized_root_mean_square_error": 0.05,
        },
    }
    assert "enforce_quantization_quality" in source


def test_recipe_uses_supported_compiler_2_1_3_api_only() -> None:
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    assert "with_activation_quantization" in source
    assert "with_weight_quantization" in source
    assert "quantization_mla" not in source
    assert "public_input_types" not in source
    assert "public_output_types" not in source


def test_model_declares_exact_four_variant_cross_product() -> None:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matrix = {
        (variant["parameters"]["precision"], variant["parameters"]["tessellation_mla"])
        for variant in metadata["compile_variants"]
    }
    assert matrix == {("int8", False), ("int8", True), ("bf16", False), ("bf16", True)}
    assert all(variant["parameters"]["calibration_samples"] == 128 for variant in metadata["compile_variants"])
