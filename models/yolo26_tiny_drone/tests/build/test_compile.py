from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from model_registry.artifact import validate_compile_outputs
from model_registry.fingerprints import COMPILE_TOOLING, compile_fingerprint
from model_registry.model import load_model as load_metadata
from PIL import Image


def test_contract_is_the_four_level_1600_one_class_export(load_script):
    contract = load_script("contract").DEFAULT_CONTRACT
    assert contract.input_shape == (1, 3, 1600, 1600)
    assert contract.outputs == tuple(
        (f"{role}_{i}", (1, channels, size, size))
        for role, channels in (("bbox", 4), ("class_logit", 1))
        for i, size in enumerate((400, 200, 100, 50))
    )


@pytest.mark.parametrize("size", [0, 16, 1599, 1601])
def test_contract_rejects_invalid_size(load_script, size):
    with pytest.raises(ValueError, match="divisible"):
        load_script("contract").DeploymentContract(size)


def test_only_the_qat_build_and_registered_calibration_are_declared(model_dir, load_script):
    metadata = load_metadata(model_dir)
    assert [profile["id"] for profile in metadata["build_profiles"]] == ["w8a8"]
    assert metadata["calibrations"][0]["dataset"]["id"] == "anti-uav"
    assert "source" not in metadata["calibrations"][0]
    assert "qualification_evidence" not in metadata
    parameters = load_script("compile")._parameters("w8a8")
    assert parameters["model_size"] == 1600
    assert parameters["calibration_samples"] == 128
    assert "quantized_source" not in parameters
    assert not any((model_dir / "scripts" / f"{name}.py").exists() for name in ("quantized_snapshot", "output_labels"))
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w8a16")


def _source_graph(contract):
    """Small constants with the pinned source's operator/precision contract; not numerical qualification."""
    helper, tensors = onnx.helper, onnx.numpy_helper
    initializers = [
        tensors.from_array(np.array(0.125, np.float32), "scale"),
        tensors.from_array(np.array(-7, np.int8), "zero"),
        tensors.from_array(np.ones((1, 3, 1, 1), np.int8), "weights"),
        tensors.from_array(np.array([0.25], np.float32), "weight_scale"),
        tensors.from_array(np.array([0], np.int8), "weight_zero"),
        tensors.from_array(np.zeros((1, 1, 2, 2), np.float32), "attention"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["input_q"]),
        helper.make_node("DequantizeLinear", ["input_q", "scale", "zero"], ["input_dq"]),
        helper.make_node("DequantizeLinear", ["weights", "weight_scale", "weight_zero"], ["weights_dq"], axis=0),
        helper.make_node("QuantizeLinear", ["attention", "scale", "zero"], ["attention_q"]),
        helper.make_node("DequantizeLinear", ["attention_q", "scale", "zero"], ["attention_dq"]),
    ]
    for i in range(129):
        nodes.append(helper.make_node("Conv", ["input_dq", "weights_dq"], [f"conv_{i}"], name=f"conv_{i}"))
    for op, count in (("Einsum", 8), ("Softmax", 4)):
        for i in range(count):
            name = f"{op}_{i}"
            inputs = ["attention_dq"] * (2 if op == "Einsum" else 1)
            attrs = {"equation": "abij,abjk->abik"} if op == "Einsum" else {"axis": -1}
            nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
            nodes.append(helper.make_node("QuantizeLinear", [name, "scale", "zero"], [f"{name}_q"]))
    for name, shape in contract.outputs:
        # Head values are independent constants: this fixture qualifies audit
        # decisions only, not the detector's numerical or connected topology.
        initializers.append(tensors.from_array(np.zeros(shape, np.float32), f"{name}_value"))
        nodes.append(helper.make_node("Identity", [f"{name}_value"], [name]))
    graph = helper.make_graph(
        nodes,
        "source-audit-fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, contract.input_shape)],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in contract.outputs],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("mutation", [None, "head_order", "dtype", "weight_scale", "weight_zero", "attention", "conv"])
def test_read_only_qdq_audit_detects_contract_and_precision_damage(load_script, tmp_path, mutation):
    audit = load_script("onnx_audit")
    contract = audit.DeploymentContract(32)
    graph = _source_graph(contract)
    if mutation == "head_order":
        first, other = graph.graph.output[1], graph.graph.output[4]
        a, b = onnx.ValueInfoProto(), onnx.ValueInfoProto()
        a.CopyFrom(first)
        b.CopyFrom(other)
        first.CopyFrom(b)
        other.CopyFrom(a)
    elif mutation == "dtype":
        graph.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif mutation in {"weight_scale", "weight_zero"}:
        tensor = next(item for item in graph.graph.initializer if item.name == mutation)
        value = np.array([np.nan], np.float32) if mutation == "weight_scale" else np.array([1], np.int8)
        tensor.CopyFrom(onnx.numpy_helper.from_array(value, mutation))
    elif mutation == "attention":
        node = next(item for item in graph.graph.node if item.op_type == "Einsum")
        node.attribute[0].s = b"abij,abjk->abki"
    elif mutation == "conv":
        node = next(item for item in graph.graph.node if item.op_type == "Conv")
        node.input[0] = "images"
    path = tmp_path / "source.onnx"
    onnx.save(graph, path)
    before = path.read_bytes()
    if mutation:
        with pytest.raises(RuntimeError):
            audit.audit_onnx(path, contract)
    else:
        result = audit.audit_onnx(path, contract)
        assert result["convolution_count"] == 129
        assert result["attention"]["matmul_count"] == 8
    assert path.read_bytes() == before


def _calibration_archive(path, calibration, contract, mutation=None):
    entries, images, digest = [], [], hashlib.sha256()
    for i in range(128):
        image = Image.new("RGB", (20, 10), (i, 20, 30))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        payload = encoded.getvalue()
        name = f"{calibration.ARCHIVE_ROOT}/images/sequence/{i:03}.png"
        images.append({"file": name, "sequence": "sequence", "sha256": hashlib.sha256(payload).hexdigest()})
        entries.append((name, payload))
        sample = calibration.preprocess_rgb_image(image, contract)
        digest.update(memoryview(sample).cast("B"))
    tensor_digest = digest.hexdigest()
    manifest = {
        "schema_version": 1,
        "count": 128,
        "tensor_sha256": tensor_digest,
        "preprocessing": f"rgb_letterbox_{contract.model_size}_fill_114_divide_255_nhwc_float32",
        "images": images,
    }
    if mutation == "last_image":
        entries[-1] = (entries[-1][0], b"corrupt")
    elif mutation == "order":
        images.reverse()
    elif mutation == "duplicate":
        images[1] = images[0]
    elif mutation == "extra":
        entries.append(("extra.txt", b"extra"))
    elif mutation == "traversal":
        entries.append(("../escape", b"bad"))
    entries.insert(0, (calibration.MANIFEST_NAME, json.dumps(manifest).encode()))
    with tarfile.open(path, "w:gz") as stream:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    return tensor_digest


def test_calibration_is_repeatable_batch_one_nhwc_without_stacking(load_script, tmp_path, monkeypatch):
    calibration = load_script("calibration")
    contract = calibration.DeploymentContract(32)
    path = tmp_path / "calibration.tar.gz"
    digest = _calibration_archive(path, calibration, contract)
    monkeypatch.setattr(np, "stack", lambda *_a, **_k: pytest.fail("calibration must not stack all frames"))
    samples = calibration.CalibrationSamples(path, contract, digest)
    assert len(samples) == 128
    for _ in range(2):
        seen = []
        for values in samples:
            assert set(values) == {"images"}
            sample = values["images"]
            assert sample.shape == (1, 32, 32, 3) and sample.dtype == np.float32
            seen.append(round(float(sample[0, 16, 16, 0]) * 255))
            np.testing.assert_allclose(sample[0, 0, 0], np.array([114, 114, 114]) / 255, rtol=1e-6)
        assert seen == list(range(128))
    path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed after validation"):
        next(iter(samples))


@pytest.mark.parametrize("mutation", ["last_image", "order", "duplicate", "extra", "traversal"])
def test_all_calibration_bytes_are_checked_before_compiler_admission(load_script, tmp_path, mutation):
    calibration = load_script("calibration")
    contract = calibration.DeploymentContract(32)
    path = tmp_path / "calibration.tar.gz"
    digest = _calibration_archive(path, calibration, contract, mutation)
    with pytest.raises(ValueError):
        calibration.CalibrationSamples(path, contract, digest)


def test_quantization_uses_public_qdq_path_without_ptq_weight_changes(load_script, monkeypatch):
    compile_module = load_script("compile")
    config = MagicMock()
    for method in (
        "with_activation_quantization",
        "with_weight_quantization",
        "with_requantization_mode",
        "with_calibration",
        "with_bias_correction",
        "with_channel_equalization",
        "with_smooth_quant",
    ):
        getattr(config, method).return_value = config
    monkeypatch.setattr(compile_module, "default_quantization", config)
    monkeypatch.setattr(compile_module, "quantization_scheme", lambda **options: options)
    assert compile_module._quantization() is config
    config.with_activation_quantization.assert_called_once_with({"asymmetric": True, "per_channel": False, "bits": 8})
    config.with_weight_quantization.assert_called_once_with({"asymmetric": False, "per_channel": True, "bits": 8})
    config.with_bias_correction.assert_called_once_with(compile_module.BiasCorrectionType.NONE)
    config.with_channel_equalization.assert_called_once_with(False)
    config.with_smooth_quant.assert_called_once_with(False)


def test_compile_checks_all_inputs_before_sdk_admission(load_script, model_dir, tmp_path, monkeypatch):
    compile_module = load_script("compile")
    source_hash = load_metadata(model_dir)["components"][0]["source"]["sha256"]
    monkeypatch.setattr(compile_module, "audit_onnx", lambda *_: {"sha256": source_hash})
    imported = MagicMock()
    monkeypatch.setattr(compile_module, "onnx_source", imported)

    def reject_calibration(*_args):
        raise ValueError("last image is corrupt")

    monkeypatch.setattr(compile_module, "CalibrationSamples", reject_calibration)
    with pytest.raises(ValueError, match="last image"):
        compile_module.compile_model(tmp_path / "source.onnx", tmp_path / "out", "w8a8", tmp_path / "cal.tgz")
    imported.assert_not_called()
    monkeypatch.setattr(compile_module, "audit_onnx", lambda *_: {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="pinned QAT source"):
        compile_module.compile_model(tmp_path / "source.onnx", tmp_path / "out", "w8a8", tmp_path / "cal.tgz")
    imported.assert_not_called()


@pytest.mark.parametrize("reject_precision", [False, True])
def test_compile_forwards_original_qdq_and_reiterable_samples(
    load_script, model_dir, tmp_path, monkeypatch, reject_precision
):
    """Orchestration test only; the compiled-stage test checks the real SDK artifact."""
    compile_module = load_script("compile")
    source = tmp_path / "exact-qat.onnx"
    source_hash = load_metadata(model_dir)["components"][0]["source"]["sha256"]
    monkeypatch.setattr(compile_module, "audit_onnx", lambda *_: {"sha256": source_hash})
    samples = [{"images": np.zeros((1, 32, 32, 3), np.float32)}]
    monkeypatch.setattr(compile_module, "CalibrationSamples", lambda *_: samples)
    importer, imported = MagicMock(), MagicMock()
    importer.return_value = imported
    monkeypatch.setattr(compile_module, "onnx_source", importer)
    loaded, quantized = MagicMock(), MagicMock()
    loaded.quantize.return_value = quantized
    monkeypatch.setattr(compile_module, "load_model", MagicMock(return_value=loaded))
    layout = {"actual_input": "HWC", "actual_output": "HWC16"}
    monkeypatch.setattr(compile_module, "_mla_tessellate_params", lambda _: layout)

    def write_output(**kwargs):
        output = Path(kwargs["output_path"])
        output.mkdir()
        _write_mpk(output / "yolo26_tiny_drone_w8a8_mpk.tar.gz", _mpk_manifest(compile_module))

    quantized.compile.side_effect = write_output
    precision_audit = MagicMock(return_value={"mla_nodes_with_int8_ports": 1})
    monkeypatch.setattr(compile_module, "audit_quantized_model", precision_audit)
    if reject_precision:
        precision_audit.side_effect = RuntimeError("promoted compute port")
        with pytest.raises(RuntimeError, match="promoted compute port"):
            compile_module.compile_model(source, tmp_path / "out", "w8a8", tmp_path / "cal.tgz")
        quantized.compile.assert_not_called()
        assert not (tmp_path / "out").exists()
        return
    result = compile_module.compile_model(source, tmp_path / "out", "w8a8", tmp_path / "cal.tgz")
    assert precision_audit.call_args.args[:2] == (quantized, source)
    assert result.is_file()
    importer.assert_called_once_with(str(source), shape_dict={"images": (1, 3, 1600, 1600)}, layout="NCHW")
    assert loaded.quantize.call_args.kwargs["calibration_data"] is samples
    assert "is_quantized" not in loaded.quantize.call_args.kwargs
    assert quantized.compile.call_args.kwargs["tessellate_parameters"] == layout
    assert quantized.compile.call_args.kwargs["batch_size"] == 1


def test_all_compile_helpers_but_not_tests_enter_the_existing_fingerprint(model_dir, tmp_path):
    repository = model_dir.parents[1]
    for directory in ("schemas", "taxonomies", "datasets", "deps"):
        shutil.copytree(repository / directory, tmp_path / directory)
    for relative in COMPILE_TOOLING:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    copied = tmp_path / "models" / model_dir.name
    shutil.copytree(model_dir, copied)

    def fingerprint():
        return compile_fingerprint(tmp_path, model_dir.name, "w8a8", "w8a8", "pinned-test-compiler")

    original = fingerprint()
    (copied / "tests" / "build" / "test_compile.py").write_text("# changed contract tests\n")
    assert fingerprint() == original
    for helper in ("contract.py", "onnx_audit.py", "precision_audit.py", "calibration.py", "compile.py"):
        path = copied / "scripts" / helper
        content = path.read_text()
        path.write_text(content + "\n# changed compile input\n")
        assert fingerprint() != original, helper
        path.write_text(content)


@pytest.mark.parametrize("tuple_output", [True, False])
def test_layout_mapping_uses_actual_mla_names_and_all_outputs(load_script, monkeypatch, tuple_output):
    compile_module = load_script("compile")
    monkeypatch.setitem(sys.modules, "afe.ir.node", SimpleNamespace(node_is_tuple=lambda _: tuple_output))
    monkeypatch.setattr(compile_module, "TensorTessellateParameters", lambda **kwargs: kwargs)
    output = SimpleNamespace(name="single", input_node_names=["raw_0", "raw_1"])
    mla = SimpleNamespace(
        input_names=["actual_input"], ir=SimpleNamespace(nodes={"output": output}, output_node_name="output")
    )
    model = SimpleNamespace(_net=SimpleNamespace(nodes={"MLA_0": mla}))
    params = compile_module._mla_tessellate_params(model)
    assert set(params) == (
        {"actual_input", "raw_0_output", "raw_1_output"} if tuple_output else {"actual_input", "single_output"}
    )
    assert params["actual_input"]["enable_mla"] is True
    assert params["actual_input"]["dram_layout"] == compile_module.TensorDRAMLayout.HWC
    assert all(
        value["dram_layout"] == compile_module.TensorDRAMLayout.HWC16
        for name, value in params.items()
        if name != "actual_input"
    )


def _mpk_manifest(compile_module):
    contract = compile_module.DEFAULT_CONTRACT
    outputs = [{"name": f"MLA_0/{name}", "size": math.prod(shape) * 4} for name, shape in contract.outputs]
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "images", "size": math.prod(contract.input_shape) * 4}],
        "plugins": [
            {
                "sequence": 0,
                "processor": "EV74",
                "config_params": {"kernel": "quantization_transform", "params": {"output_data_type": "int8"}},
                "output_nodes": [{"name": "quantized_images"}],
            },
            {"sequence": 1, "processor": "MLA", "config_params": {}, "input_nodes": [{"name": "quantized_images"}]},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "dequantization_transform", "params": {"input_data_type": "int8"}},
                "output_nodes": outputs,
            },
            {
                "sequence": 3,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "input_nodes": outputs,
                "output_nodes": [{"name": f"output_{i}", "size": output["size"]} for i, output in enumerate(outputs)],
            },
        ],
    }


def _write_mpk(path, manifest):
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"fixture"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize(
    "mutation",
    [None, "equal_size_roles", "host", "two_mla", "compiler", "layout", "bf16", "size", "missing_dequantizer", "batch"],
)
def test_compiled_semantics_reject_wrong_roles_and_fallback(load_script, tmp_path, mutation):
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if mutation == "equal_size_roles":
        outputs = plugins[-1]["input_nodes"]
        assert outputs[1]["size"] == outputs[4]["size"]
        outputs[1], outputs[4] = outputs[4], outputs[1]
    elif mutation == "host":
        plugins[2]["processor"] = "APU"
    elif mutation == "two_mla":
        plugins[2]["processor"] = "MLA"
    elif mutation == "compiler":
        manifest["model_sdk_version"] = "2.1.3"
    elif mutation == "layout":
        plugins[0]["config_params"]["kernel"] = "tessellation_transform"
    elif mutation == "bf16":
        plugins[2]["config_params"]["params"]["input_data_type"] = "bf16"
    elif mutation == "size":
        plugins[-1]["output_nodes"][0]["size"] -= 1
    elif mutation == "missing_dequantizer":
        plugins[2]["output_nodes"] = plugins[2]["output_nodes"][:-1]
    elif mutation == "batch":
        plugins[1]["config_params"]["actual_batch_size"] = 2
    path = tmp_path / "model_mpk.tar.gz"
    _write_mpk(path, manifest)
    if mutation:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(path)
    else:
        compile_module.audit_mpk(path)


@pytest.mark.parametrize("canonical_labels", [False, True])
def test_public_aliases_and_multiple_packed_groups_keep_typed_provenance(load_script, tmp_path, canonical_labels):
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    if canonical_labels:
        for node, name in zip(manifest["plugins"][-1]["output_nodes"], compile_module.DEFAULT_CONTRACT.output_names()):
            node["name"] = name
    # Layout may pack several groups rather than one eight-head carrier. Each
    # unpack declares types for its own physical outputs, not the model total.
    for group in range(2):
        manifest["plugins"].insert(
            2,
            {
                "sequence": 1.5,
                "processor": "EV74",
                "config_params": {"kernel": "unpack_transform", "params": {"tensor_types": ["int8"] * 4}},
                "output_nodes": [{"name": f"group_{group}_{i}"} for i in range(4)],
            },
        )
    path = tmp_path / "model_mpk.tar.gz"
    _write_mpk(path, manifest)
    compile_module.audit_mpk(path)
    if canonical_labels:
        nodes = manifest["plugins"][-1]["output_nodes"]
        nodes[1]["name"], nodes[4]["name"] = nodes[4]["name"], nodes[1]["name"]
        _write_mpk(path, manifest)
        with pytest.raises(RuntimeError, match="public output labels/order"):
            compile_module.audit_mpk(path)


def test_actual_compiled_build_preserves_public_contract(compiled: Path, build_profile: str, load_script):
    """Runs on real stage output, not a mock, before the registry can publish it."""
    assert build_profile == "w8a8"
    validate_compile_outputs(
        compiled / "compile-outputs.json",
        compiled,
        model_id="yolo26_tiny_drone",
        build_profile_id=build_profile,
        compile_invocation_id=build_profile,
        component_ids=["model"],
    )
    archives = list((compiled / "model").glob("*_mpk.tar.gz"))
    assert len(archives) == 1
    module = load_script("compile")
    module.audit_mpk(archives[0])
    receipt = json.loads((compiled / "quantization-audit.json").read_text())
    assert receipt["source_sha256"] == load_metadata(module.MODEL_DIR)["components"][0]["source"]["sha256"]
    assert receipt["compiler"] == module.resolve_model_compiler(module.MODEL_DIR.parents[1])
    assert receipt["mpk_sha256"] == hashlib.sha256(archives[0].read_bytes()).hexdigest()
    assert receipt["source_operations"] == {"Conv": 129, "Einsum": 8, "Softmax": 4}
    assert receipt["direct_source_coverage"] == {"Conv": 129, "Softmax": 4}
    assert receipt["attention_coverage_authority"] == "paired_normalized_fp32_graph"
    assert receipt["normalized_attention_operations"] > 0
    for key in ("normalized_model_sha256", "normalized_typed_graph_sha256"):
        assert len(receipt[key]) == 64 and int(receipt[key], 16) >= 0
    assert receipt["mla_nodes_with_int8_ports"] > 0 and receipt["per_channel_int8_weights"] > 0


def _small_qdq_source(path: Path, input_scale: float, output_scale: float) -> None:
    """A 1x1 identity-weight Conv with explicit, nontrivial input/output QDQ."""
    helper, tensors = onnx.helper, onnx.numpy_helper
    constants = [
        ("input_scale", np.array(input_scale, np.float32)),
        ("input_zero", np.array(-7, np.int8)),
        ("weight", np.array([[[[2]]]], np.int8)),
        ("weight_scale", np.array([0.5], np.float32)),
        ("weight_zero", np.array([0], np.int8)),
        ("output_scale", np.array(output_scale, np.float32)),
        ("output_zero", np.array(3, np.int8)),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["images", "input_scale", "input_zero"], ["input_q"]),
            helper.make_node("DequantizeLinear", ["input_q", "input_scale", "input_zero"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight", "weight_scale", "weight_zero"], ["weight_dq"], axis=0),
            helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"], name="Conv_0", kernel_shape=[1, 1]),
            helper.make_node("QuantizeLinear", ["conv", "output_scale", "output_zero"], ["output_q"]),
            helper.make_node("DequantizeLinear", ["output_q", "output_scale", "output_zero"], ["output"]),
        ],
        "qat-hint-retention",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
        [tensors.from_array(value, name) for name, value in constants],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_sdk_probe_has_observable_input_and_output_qdq(tmp_path):
    from onnx.reference import ReferenceEvaluator

    path = tmp_path / "probe.onnx"
    _small_qdq_source(path, 0.25, 0.375)
    # Recent ONNX reference interpreters implement Q/DQ starting at opset19.
    # Convert only the reference copy through ONNX's supported version adapter;
    # the source file and real SDK probe keep the detector's opset17.
    model = onnx.version_converter.convert_version(onnx.load(path), 19)
    values = np.array([0.1875, 0.4375, 1.0, 40.0], np.float32).reshape(1, 1, 2, 2)
    result = ReferenceEvaluator(model).run(None, {"images": values})[0]
    np.testing.assert_array_equal(result.reshape(-1), [0.375, 0.375, 1.125, 33.375])
    # Prove that the test's output boundary is not redundant with the input
    # quantization grid, rather than merely expecting the unchanged input.
    del model.graph.node[-2:]
    model.graph.output[0].name = "conv"
    no_output_qdq = ReferenceEvaluator(model).run(None, {"images": values})[0]
    assert not np.array_equal(result, no_output_qdq)


def test_actual_sdk_retains_qdq_hints_across_calibration_ranges(compiled: Path, tmp_path: Path, load_script):
    """Real SDK behavior, not final-ELF precision qualification or a mocked import.

    Selected only in the compile stage, whose interpreter is the pinned Model
    Compiler. Missing SDK dependencies fail this test; they must not skip it.
    """
    from afe.apis.defines import InputName, gen2_target
    from afe.apis.loaded_net import load_model
    from afe.load.importers.general_importer import onnx_source

    assert compiled.is_dir()
    compile_module = load_script("compile")
    # Binary fractions avoid rounding ties and make the independent affine
    # quantizer oracle exact. Weight2 * scale0.5 is the identity operation.
    values = np.array([0.1875, 0.4375, 1.0, 40.0], np.float32).reshape(1, 2, 2, 1)

    def reference(scale, output_scale):
        codes = np.clip(np.rint(values / scale) - 7, -128, 127)
        activation = (codes + 7) * scale
        output_codes = np.clip(np.rint(activation / output_scale) + 3, -128, 127)
        return (output_codes - 3) * output_scale

    def execute(scale, output_scale, calibration_range):
        source = tmp_path / f"qdq-{scale}-{output_scale}-{calibration_range}.onnx"
        _small_qdq_source(source, scale, output_scale)
        loaded = load_model(
            onnx_source(str(source), shape_dict={"images": (1, 1, 2, 2)}, layout="NCHW"), target=gen2_target
        )
        calibration = [
            {InputName("images"): np.full_like(values, magnitude)}
            for magnitude in (-calibration_range, calibration_range)
        ]
        model = loaded.quantize(
            calibration_data=calibration, quantization_config=compile_module._quantization(), model_name="qat_hint_test"
        )
        if (scale, output_scale, calibration_range) == (0.25, 0.375, 0.03125):
            # Public serialization preserves the actual pinned compiler's
            # quantized representation for precision inspection, not reload.
            saved = tmp_path / "saved-qdq"
            saved.mkdir()
            model.save(model_name="qat_hint_test", output_directory=str(saved))
            precision = load_script("precision_audit").audit_saved_quantization(saved / "qat_hint_test.sima", source)
            assert precision["source_operations"] == {"Conv": 1}
            assert precision["per_channel_int8_weights"] == 1
        outputs = model.execute(inputs={InputName("images"): values})
        assert len(outputs) == 1
        result = np.asarray(outputs[0]).reshape(values.shape)
        np.testing.assert_allclose(result, reference(scale, output_scale), rtol=0, atol=1e-6)
        return result

    narrow = execute(0.25, 0.375, 0.03125)
    wide = execute(0.25, 0.375, 4.0)
    np.testing.assert_array_equal(narrow, wide)
    changed_input = execute(0.75, 0.375, 0.03125)
    changed_output = execute(0.25, 0.25, 0.03125)
    assert not np.array_equal(narrow, changed_input), "changing exported input QDQ was ignored"
    assert not np.array_equal(narrow, changed_output), "changing exported output QDQ was ignored"
    # The large sample saturates the input in the first model and the output
    # in the last model. These values make the nonzero zero points observable,
    # rather than merely canceling them algebraically for in-range samples.
    assert narrow.reshape(-1)[-1] == 33.375
    assert changed_output.reshape(-1)[-1] == (127 - 3) * 0.25


def _precision_fixture(path, source, mutation):
    """Compiler 3.0 save-format contract; not a compiler or numerical substitute."""
    import yaml

    _small_qdq_source(source, 0.25, 0.375)
    graph = onnx.load(source)
    graph.graph.node.extend(
        [
            onnx.helper.make_node("Einsum", ["input_dq", "input_dq"], ["attention"], name="attention"),
            onnx.helper.make_node("Softmax", ["attention"], ["probabilities"], name="softmax"),
        ]
    )
    onnx.save(graph, source)
    tensor = {
        "__decode_key__": "TensorType",
        "scalar": {"__decode_key__": "ScalarType", "value": 1},
        "shape": {"__decode_key__": "tuple", "vals": [1, 1, 2, 2]},
    }
    config = {
        "_status": "SIMA_QUANTIZED",
        "Source names": ["Conv_0"],
        "_input_types": [[copy.deepcopy(tensor)]],
        "_output_types": [copy.deepcopy(tensor)],
        # This reference deliberately differs from the archive's namespace.
        "weight_quant_data": "json_weight",
        "per_channel": True,
        "conv_attrs.input_type": {"__decode_key__": "ScalarType", "value": 6},
    }
    conv = {"class_name": "MLA_0/ConvAddActivation_0", "name": "MLA_0/conv", "config": config}
    fused = {
        "class_name": "MLA_0/BatchMatmul_1",
        "name": "MLA_0/normalized_attention",
        "inbound_nodes": [[["MLA_0/conv", 0, 0, {}], ["MLA_0/conv", 0, 0, {}]]],
        "config": {
            "_status": "SIMA_QUANTIZED",
            "Source names": ["softmax"],
            "_input_types": [[copy.deepcopy(tensor)]],
            "_output_types": [copy.deepcopy(tensor)],
        },
    }
    structural = {
        "class_name": "MLA_0/Tuple_None",
        "name": "MLA_0/tuple_out",
        "config": {
            "_status": "RELAY",
            "_input_types": [[copy.deepcopy(tensor)]],
            "_output_types": [copy.deepcopy(tensor)],
        },
    }
    layers = [conv, fused, structural]
    archived_structural = {"name": "MLA_0/tuple_out", "status": "relay", "ir": {"op": "tuple"}}
    archived = {
        "name": "MLA_0/conv",
        "status": "sima_quantized",
        "source_node_names": ["Conv_0"],
        "ir": {"quant": {"weight_quant_data": "archive_weight", "per_channel": True}},
    }
    # The full compiler drops original Einsum labels but preserves paired
    # normalized identities. Names are deliberately unrelated to the ONNX name.
    attrs = {"op": "batch_matmul", "input_shapes": [[1, 1, 2, 2], [1, 1, 2, 2]], "transpose_b": False}
    attention = {
        "name": fused["name"],
        "status": "sima_quantized",
        "inputs": ["MLA_0/conv"] * 2,
        "ir": {"op": "batch_matmul", "quant": {"op": "batch_matmul", "attrs": copy.deepcopy(attrs)}},
    }
    normalized_attention = copy.deepcopy(attention)
    normalized_attention["status"] = "relay"
    normalized_attention["ir"] = {"op": "batch_matmul", "attrs": attrs, "quant": None}
    normalized_layers = copy.deepcopy(layers)
    arrays = {"archive_weight": np.ones((1, 1, 1, 1, 1), np.int8), "json_weight": np.ones(1, np.float32)}
    if mutation == "missing_source":
        fused["config"]["Source names"] = ["attention"]
    elif mutation == "ambiguous_source":
        duplicate = copy.deepcopy(fused)
        duplicate["name"] = "MLA_0/other"
        layers.append(duplicate)
    elif mutation in {"input_float", "output_float", "other_float"}:
        target = fused if mutation == "other_float" else conv
        field = "_input_types" if mutation == "input_float" else "_output_types"
        value = target["config"][field][0]
        (value[0] if isinstance(value, list) else value)["scalar"]["value"] = 6
    elif mutation == "missing_type":
        del config["_output_types"]
    elif mutation == "status":
        config["_status"] = "RELAY"
    elif mutation == "weight_dtype":
        arrays["archive_weight"] = arrays["archive_weight"].astype(np.float32)
    elif mutation == "missing_weight":
        del arrays["archive_weight"]
    elif mutation == "empty_weight":
        arrays["archive_weight"] = np.empty(0, np.int8)
    elif mutation == "per_channel":
        archived["ir"]["quant"]["per_channel"] = False
    elif mutation == "archive_source":
        archived["source_node_names"] = ["other"]
    elif mutation == "structural_float":
        structural["config"]["_output_types"][0]["scalar"]["value"] = 6
    elif mutation == "structural_source":
        structural["config"]["Source names"] = ["attention"]
    elif mutation == "named_tuple_compute":
        archived_structural["ir"]["op"] = "add_activation"
    elif mutation == "attention_missing":
        layers.remove(fused)
    elif mutation == "attention_class":
        fused["class_name"] = "CPU/BatchMatmul_1"
    elif mutation == "attention_status":
        attention["status"] = "relay"
    elif mutation == "attention_op":
        attention["ir"]["op"] = "add_activation"
    elif mutation == "attention_quant_op":
        attention["ir"]["quant"]["op"] = "add_activation"
    elif mutation == "attention_shape":
        fused["config"]["_output_types"][0]["shape"]["vals"][-1] = 3
    elif mutation == "attention_attribute_shape":
        attention["ir"]["quant"]["attrs"]["input_shapes"][0][-1] = 3
    elif mutation == "attention_transpose":
        attention["ir"]["quant"]["attrs"]["transpose_b"] = True
    elif mutation == "attention_edge":
        fused["inbound_nodes"][0][0][0] = "other"
    elif mutation == "attention_archive_edge":
        attention["inputs"][0] = "other"
    elif mutation == "attention_absent_normalized":
        normalized_attention["ir"]["op"] = "add_activation"
    document = {"modelTopology": {"model_config": {"config": {"layers": layers}}}}
    path.with_suffix(".sima.json").write_text(json.dumps(document))
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    network = [
        "This YAML document contains a serialized Network",
        "Format version 1",
        {"commit_id": None},
        {"nodes": [archived, attention, archived_structural]},
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.sima.yaml", yaml.safe_dump(network))
        archive.writestr("model.sima.npz", payload.getvalue())
    normalized = path.with_name(path.stem + ".fp32" + path.suffix)
    normalized.with_suffix(".sima.json").write_text(
        json.dumps({"modelTopology": {"model_config": {"config": {"layers": normalized_layers}}}})
    )
    network[3]["nodes"] = [normalized_attention]
    with zipfile.ZipFile(normalized, "w") as archive:
        archive.writestr("model.sima.yaml", yaml.safe_dump(network))


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_source",
        "ambiguous_source",
        "input_float",
        "output_float",
        "other_float",
        "missing_type",
        "status",
        "weight_dtype",
        "missing_weight",
        "empty_weight",
        "per_channel",
        "archive_source",
        "structural_float",
        "structural_source",
        "named_tuple_compute",
        "attention_missing",
        "attention_op",
        "attention_class",
        "attention_status",
        "attention_quant_op",
        "attention_shape",
        "attention_attribute_shape",
        "attention_transpose",
        "attention_edge",
        "attention_archive_edge",
        "attention_absent_normalized",
    ],
)
def test_saved_precision_checks_typed_ports_provenance_and_bound_weight_bytes(
    load_script, tmp_path, monkeypatch, mutation
):
    audit = load_script("precision_audit")
    scalar = MagicMock(return_value=SimpleNamespace(value=1))
    monkeypatch.setattr(audit, "scalar_type_from_dtype", scalar)
    saved, source = tmp_path / "model.sima", tmp_path / "source.onnx"
    _precision_fixture(saved, source, mutation)
    if mutation:
        with pytest.raises(RuntimeError):
            audit.audit_saved_quantization(saved, source)
    else:
        result = audit.audit_saved_quantization(saved, source)
        assert result["source_operations"] == {"Conv": 1, "Einsum": 1, "Softmax": 1}
        assert result["mla_nodes_with_int8_ports"] == 3  # Fused compute plus typed structural packing.
        assert result["per_channel_int8_weights"] == 1
        assert result["direct_source_coverage"] == {"Conv": 1, "Softmax": 1}
        assert result["normalized_attention_operations"] == 1
        assert result["attention_coverage_authority"] == "paired_normalized_fp32_graph"
        scalar.assert_called_once_with("int8")
        assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result["quantized_model_sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()


def test_precision_gate_uses_public_save_without_reloading(load_script, tmp_path, monkeypatch):
    audit = load_script("precision_audit")
    model = MagicMock()
    source, directory = tmp_path / "source.onnx", tmp_path / "saved"
    inspected = MagicMock(return_value={"mla_nodes_with_int8_ports": 1})
    monkeypatch.setattr(audit, "audit_saved_quantization", inspected)
    assert audit.audit_quantized_model(model, source, directory) == {"mla_nodes_with_int8_ports": 1}
    model.save.assert_called_once_with(
        model_name="quantization", output_directory=str(directory), include_unquantized_net=True
    )
    model.load.assert_not_called()
    inspected.assert_called_once_with(directory / "quantization.sima", source)


def test_actual_sdk_admits_int8_multi_output_structure(compiled: Path, tmp_path: Path, load_script):
    """Tuple packaging is RELAY in Compiler 3.0; that is not floating compute."""
    from afe.apis.defines import gen2_target
    from afe.apis.loaded_net import load_model
    from afe.load.importers.general_importer import onnx_source

    recipe = load_script("compile")
    source = tmp_path / "multi.onnx"
    _small_qdq_source(source, 0.25, 0.375)
    graph = onnx.load(source)
    graph.graph.node.extend(
        [
            onnx.helper.make_node("Add", ["conv", "conv"], ["twice"], name="add_probe"),
            onnx.helper.make_node("QuantizeLinear", ["twice", "output_scale", "output_zero"], ["second_q"]),
            onnx.helper.make_node("DequantizeLinear", ["second_q", "output_scale", "output_zero"], ["second_output"]),
        ]
    )
    graph.graph.output.append(onnx.helper.make_tensor_value_info("second_output", onnx.TensorProto.FLOAT, [1, 1, 2, 2]))
    onnx.checker.check_model(graph)
    onnx.save(graph, source)
    loaded = load_model(
        onnx_source(str(source), shape_dict={"images": (1, 1, 2, 2)}, layout="NCHW"), target=gen2_target
    )
    model = loaded.quantize(
        calibration_data=[{"images": np.full((1, 2, 2, 1), value, np.float32)} for value in (-0.03125, 0.03125)],
        quantization_config=recipe._quantization(),
        model_name="tuple_probe",
    )
    result = recipe.audit_quantized_model(model, source, tmp_path / "saved")
    assert result["source_operations"] == {"Conv": 1}
    assert result["per_channel_int8_weights"] == 1
    assert result["mla_nodes_with_int8_ports"] > result["per_channel_int8_weights"]


def test_actual_sdk_binds_normalized_attention_to_quantized_ports(compiled: Path, tmp_path: Path, load_script):
    """Public paired saves retain normalized attention even without source labels."""
    from afe.apis.defines import gen2_target
    from afe.apis.loaded_net import load_model
    from afe.load.importers.general_importer import onnx_source

    assert compiled.is_dir()
    recipe = load_script("compile")
    source = tmp_path / "attention.onnx"
    _small_qdq_source(source, 0.25, 0.375)
    graph = onnx.load(source)
    conv = next(node for node in graph.graph.node if node.op_type == "Conv")
    conv.CopyFrom(
        onnx.helper.make_node(
            "Einsum", ["input_dq", "input_dq"], ["conv"], name="source_attention", equation="abij,abjk->abik"
        )
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, source)
    loaded = load_model(
        onnx_source(str(source), shape_dict={"images": (1, 1, 2, 2)}, layout="NCHW"), target=gen2_target
    )
    model = loaded.quantize(
        calibration_data=[{"images": np.full((1, 2, 2, 1), value, np.float32)} for value in (-1.0, 1.0)],
        quantization_config=recipe._quantization(),
        model_name="attention_probe",
    )
    result = recipe.audit_quantized_model(model, source, tmp_path / "saved")
    assert result["source_operations"] == {"Einsum": 1}
    assert result["direct_source_coverage"] == {}
    assert result["normalized_attention_operations"] == 1
    assert result["mla_nodes_with_int8_ports"] >= 1
