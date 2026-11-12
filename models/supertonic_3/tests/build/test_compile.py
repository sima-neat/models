from __future__ import annotations

import builtins
import hashlib
import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from model_registry.model import load_model, mpk_filename

PROFILE_ID = "w8a16_w16a16"
COMPONENT_IDS = ("vector_estimator", "vocoder")
RUNTIME_DATA = "supertonic_runtime_data.npz"


def _fixture_mpk(directory: Path, name: str, manifest: dict[str, Any]) -> Path:
    archive = directory / name
    with tarfile.open(archive, "w:gz") as stream:
        for member_name, data in (("model.elf", b"elf"), ("fixture_mpk.json", json.dumps(manifest).encode())):
            member = tarfile.TarInfo(member_name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    return archive


def test_vector_surgery_import_does_not_require_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    original_import = builtins.__import__

    def without_runtime(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_runtime)
    assert callable(load_script("vector_transforms").optimize)


def test_vector_estimator_runs_both_guidance_branches_in_one_call(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters(load_model(recipe.MODEL_DIR), PROFILE_ID)
    assert list(parameters) == list(COMPONENT_IDS)
    assert recipe._batch_size(parameters["vector_estimator"]) == 2
    assert recipe._batch_size(parameters["vocoder"]) == 1
    with pytest.raises(ValueError, match="positive"):
        recipe._batch_size({"batch_size": 0})


def test_the_cpu_files_come_from_the_revision_the_graphs_compile_from(model_dir: Path) -> None:
    # The text encoder feeds the vector estimator, so both come from one Supertone revision.
    authored = yaml.safe_load((model_dir / "model.yaml").read_text(encoding="utf-8"))
    compiled_from = authored["compilation"]["source"]
    for resource in authored["resources"].values():
        pin = resource["source"]
        assert (pin["repository"], pin["revision"]) == (compiled_from["repository"], compiled_from["revision"])


@pytest.mark.parametrize("component_id", COMPONENT_IDS)
@pytest.mark.parametrize("wrong_output_size", [False, True])
def test_recipe_quantizes_with_the_declared_name_and_audits_the_mpk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    component_id: str,
    wrong_output_size: bool,
) -> None:
    recipe = load_script("compile")
    contract = recipe.COMPONENTS[component_id]
    graph = tmp_path / "model.onnx"
    inputs = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) for name, shape in contract["inputs"].items()
    }
    importer, target, quantization = object(), object(), object()
    model = load_model(recipe.MODEL_DIR)
    parameters = recipe._component_parameters(model, PROFILE_ID)[component_id]
    model_name = mpk_filename(model, PROFILE_ID, component_id).removesuffix("_mpk.tar.gz")
    batch_size = recipe._batch_size(parameters)
    output_sizes = [int(np.prod(shape)) * 4 * batch_size for shape in contract["outputs"].values()]

    def onnx_source(path: str, shape_dict: Any = None, *, layout: str | None = None) -> object:
        assert (path, shape_dict, layout) == (str(graph), contract["inputs"], "NCHW")
        return importer

    class Quantized:
        def compile(
            self, output_path: str, *, batch_size: int, compress: bool, preserve: bool, log_level: int,
            layer_norm_use_fp32_intermediates: bool,
        ) -> None:
            assert batch_size == expected_batch
            assert layer_norm_use_fp32_intermediates is True
            manifest = {
                "model_sdk_version": recipe.COMPILER_VERSION,
                "input_nodes": [
                    {"name": name, "size": int(np.prod(shape)) * 4 * batch_size}
                    for name, shape in contract["inputs"].items()
                ],
                "plugins": [{
                    "processor": "MLA",
                    "sequence": 0,
                    "output_nodes": [{"size": size + int(wrong_output_size)} for size in output_sizes],
                }],
            }
            _fixture_mpk(Path(output_path), f"{model_name}_mpk.tar.gz", manifest)

    expected_batch = batch_size
    quantized = Quantized()

    class Loaded:
        def quantize(self, *, calibration_data: Any, quantization_config: Any, model_name: str, log_level: int) -> Any:
            assert quantization_config is quantization
            assert model_name == mpk_filename(model, PROFILE_ID, component_id).removesuffix("_mpk.tar.gz")
            assert calibration_data is None
            return quantized

    loaded = Loaded()

    def load_model_stub(params: object, *, target: object) -> Loaded:
        assert params is importer
        return loaded

    modules = {
        "afe.apis.defines": {"InputName": str, "gen2_target": target},
        "afe.apis.loaded_net": {"load_model": load_model_stub},
        "afe.load.importers.general_importer": {"onnx_source": onnx_source},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(recipe, "_quantization", lambda _parameters: quantization)

    actual_loaded, actual_quantized, _ = recipe._load_and_quantize(graph, component_id, model_name, parameters, inputs)
    assert (actual_loaded, actual_quantized) == (loaded, quantized)
    if wrong_output_size:
        with pytest.raises(RuntimeError, match="public output contract mismatch"):
            recipe._compile(quantized, tmp_path, component_id, batch_size)
    else:
        archive = recipe._compile(quantized, tmp_path, component_id, batch_size)
        assert archive.name == mpk_filename(model, PROFILE_ID, component_id)
        assert recipe.audit_mpk(archive, component_id, batch_size)["output_sizes"] == output_sizes


def test_one_invocation_writes_both_mpks_and_compile_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    model = load_model(recipe.MODEL_DIR)
    phases: list[tuple[str, Path]] = []

    def run_phase(_name: str, _target: Any, *, source: Path, temporary_dir: Path, profile_id: str,
                  component_id: str, parameters: dict[str, Any]) -> None:
        phases.append((component_id, source))
        work_dir = temporary_dir / component_id
        (work_dir / "compiled").mkdir(parents=True)
        name = mpk_filename(model, profile_id, component_id)
        _fixture_mpk(work_dir / "compiled", name, {})
        report = {"mpk": name, "derived_graph_sha256": component_id * 2, "batch_size": recipe._batch_size(parameters)}
        (work_dir / recipe.COMPILE_REPORT).write_text(json.dumps(report), encoding="utf-8")
        if component_id == "vector_estimator":
            (work_dir / RUNTIME_DATA).write_bytes(b"tables")

    monkeypatch.setattr(recipe, "_run_phase", run_phase)
    onnx_dir = tmp_path / "source" / "onnx"
    onnx_dir.mkdir(parents=True)
    source = onnx_dir / "vector_estimator.onnx"
    source.write_bytes(b"vector")
    (onnx_dir / "vocoder.onnx").write_bytes(b"vocoder")
    output = tmp_path / "out"

    archives = recipe.compile_model(source, PROFILE_ID, output)

    assert phases == [("vector_estimator", source), ("vocoder", onnx_dir / "vocoder.onnx")]
    for component_id in COMPONENT_IDS:
        assert archives[component_id] == output / component_id / mpk_filename(model, PROFILE_ID, component_id)
        assert archives[component_id].is_file()
    outputs = json.loads((output / "compile-outputs.json").read_text(encoding="utf-8"))
    assert (outputs["model_id"], outputs["build_profile_id"], outputs["compile_invocation_id"]) == (
        "supertonic_3", PROFILE_ID, PROFILE_ID
    )
    assert [item["component_id"] for item in outputs["outputs"]] == list(COMPONENT_IDS)
    assert [item["mpk_path"] for item in outputs["outputs"]] == [
        f"{component_id}/{mpk_filename(model, PROFILE_ID, component_id)}" for component_id in COMPONENT_IDS
    ]
    assert outputs["derived_graphs"] == [
        {"component_id": component_id, "sha256": component_id * 2} for component_id in COMPONENT_IDS
    ]
    assert (output / RUNTIME_DATA).read_bytes() == b"tables"
    # The package carries the runtime tables, so the compile declares them with their checksum.
    assert outputs["runtime_files"] == [{"path": RUNTIME_DATA, "sha256": hashlib.sha256(b"tables").hexdigest()}]
    report = json.loads((output / "compile-report.json").read_text(encoding="utf-8"))
    assert report["components"]["vector_estimator"]["batch_size"] == 2
    assert set(report["source_sha256"]) == {"vector_estimator.onnx", "vocoder.onnx"}
    with pytest.raises(FileExistsError):
        recipe.compile_model(source, PROFILE_ID, output)


def test_runtime_inputs_are_deterministic_and_length_selected(load_script: Callable[[str], ModuleType]) -> None:
    runtime = load_script("runtime_inputs")

    first = runtime.build_time_table(8)
    np.testing.assert_array_equal(first, runtime.build_time_table(8))
    assert first.shape == (8, 1, 64, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        runtime.build_time_table(0)

    rope_bank = np.zeros(runtime.ROPE_BANK_SHAPE, dtype=np.float32)
    for index in range(runtime.MAX_SEQUENCE_LENGTH):
        rope_bank[index].fill(index + 1)
    latent_mask = np.zeros((1, 1, 1, runtime.MAX_SEQUENCE_LENGTH), dtype=np.float32)
    text_mask = np.zeros((1, runtime.MAX_SEQUENCE_LENGTH, 1, 1), dtype=np.float32)
    latent_mask[..., :3] = 1
    text_mask[:, :5, ...] = 1

    packed = runtime.pack_rope_input(rope_bank, latent_mask, text_mask)
    assert packed.shape == runtime.ROPE_INPUT_SHAPE
    np.testing.assert_array_equal(packed[:, :64], rope_bank[2])
    np.testing.assert_array_equal(packed[:, 64:], rope_bank[4])


def test_performance_takes_both_components_in_run_order(load_script: Callable[[str], ModuleType]) -> None:
    performance = load_script("performance")
    mpks = performance.parse_mpks(["vector_estimator=/a.tar.gz", "vocoder=/b.tar.gz"])
    assert mpks == {"vector_estimator": Path("/a.tar.gz"), "vocoder": Path("/b.tar.gz")}
    for wrong in (["vocoder=/b", "vector_estimator=/a"], ["vector_estimator=/a"], ["vector_estimator", "vocoder=/b"]):
        with pytest.raises(SystemExit):
            performance.parse_mpks(wrong)


def test_build_holds_both_mpks_at_their_batch_sizes_and_the_runtime_data(
    compiled: Path, build_profile: str, model_dir: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    runtime = load_script("runtime_inputs")
    model = load_model(model_dir)
    parameters = recipe._component_parameters(model, build_profile)
    for component_id in COMPONENT_IDS:
        archive = compiled / component_id / mpk_filename(model, build_profile, component_id)
        recipe.audit_mpk(archive, component_id, recipe._batch_size(parameters[component_id]))
    with np.load(compiled / RUNTIME_DATA) as archive:
        rope_bank = runtime.validate_rope_bank(archive[runtime.ROPE_TABLE_KEY])
        assert rope_bank.shape == runtime.ROPE_BANK_SHAPE
        for name in ("conditional_style_key", "unconditional_text", "unconditional_style_key",
                     "unconditional_style_value"):
            assert np.isfinite(archive[name]).all(), name
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((compiled / RUNTIME_DATA).read_bytes()).hexdigest()
    assert outputs["runtime_files"] == [{"path": RUNTIME_DATA, "sha256": digest}]


def test_quantized_components_agree_with_the_float_graphs(compiled: Path) -> None:
    report = json.loads((compiled / "compile-report.json").read_text(encoding="utf-8"))
    for component_id in COMPONENT_IDS:
        for output in report["components"][component_id]["quantized_verification"]:
            assert output["cosine_similarity"] > 0.9, (component_id, output)


@pytest.mark.parametrize("precision", ["w8a16", "w16a16"])
def test_neither_precision_calibrates(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    defines = ModuleType("afe.apis.defines")
    for name in ("QuantizationParams", "SkipCalibration", "bfloat16_scheme", "quantization_scheme"):
        setattr(defines, name, MagicMock(name=name))
    defines.RequantizationMode = SimpleNamespace(sima="sima")
    monkeypatch.setitem(sys.modules, "afe.apis.defines", defines)
    recipe._quantization({"precision": precision})
    options = defines.QuantizationParams.call_args.kwargs
    # BF16 activations for both; INT8 per-channel weights only for w8a16.
    assert options["activation_quantization_scheme"] is defines.bfloat16_scheme.return_value
    if precision == "w8a16":
        defines.quantization_scheme.assert_called_once_with(asymmetric=False, per_channel=True, bits=8)
        assert options["weight_quantization_scheme"] is defines.quantization_scheme.return_value
    else:
        defines.quantization_scheme.assert_not_called()
        assert options["weight_quantization_scheme"] is defines.bfloat16_scheme.return_value
    defines.SkipCalibration.assert_called_once_with()
    assert options["calibration_method"] is defines.SkipCalibration.return_value
    assert (options["requantization_mode"], options["per_token_quant"]) == ("sima", False)
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe._quantization({"precision": "w4a8"})
