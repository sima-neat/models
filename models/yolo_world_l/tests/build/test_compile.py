from pathlib import Path

import numpy as np
import onnx
import pytest


def test_finite_clip_preserves_ordinary_normalization(load_script):
    surgery = load_script("surgery")
    h = onnx.helper
    graph = h.make_model(h.make_graph([
        h.make_node("Clip", ["norm", "eps", ""], ["bounded"], name="norm_clip")
    ], "normalization", [h.make_tensor_value_info("norm", onnx.TensorProto.FLOAT, [1])],
       [h.make_tensor_value_info("bounded", onnx.TensorProto.FLOAT, [1])],
       [onnx.numpy_helper.from_array(np.array(1e-12, np.float32), "eps")]),
       opset_imports=[h.make_opsetid("", 13)], ir_version=10)
    surgery.finite_clip_bounds(graph)
    bound = onnx.numpy_helper.to_array(graph.graph.initializer[-1])
    assert np.isfinite(bound) and bound > 1e30
    assert graph.graph.node[0].input[2] == "norm_clip/finite_max"
    for value in (0., 1e-15, .9, 1., 1.01):
        assert np.clip(value, 1e-12, bound) == max(value, 1e-12)
    onnx.checker.check_model(graph)


def test_calibration_keeps_image_prompt_pairing_and_sparse_masks(load_script):
    samples = load_script("calibration_samples").SAMPLES
    assert len(samples) == 128
    assert len({row["file_name"] for row in samples}) == 128
    assert {row["active_count"] for row in samples} == {1, 2, 3, 8, 16, 32, 80}
    for row in samples:
        assert len(row["sha256"]) == 64
        assert len(row["prompts"]) == 80
        assert 1 <= row["active_count"] <= 80


def test_supplied_calibration_bytes_are_verified(tmp_path, load_script, monkeypatch):
    calibration = load_script("calibration")
    path = tmp_path / "000000000001.jpg"
    path.write_bytes(b"wrong-image")
    monkeypatch.setattr(calibration, "SAMPLES", [{"file_name": path.name, "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="checksum"):
        calibration.verified_images(tmp_path)


def test_bf16_text_configuration_skips_calibration(load_script, monkeypatch):
    compiler = load_script("compile")
    sentinel = object()
    monkeypatch.setattr(compiler, "QuantizationParams", lambda **values: values)
    monkeypatch.setattr(compiler, "SkipCalibration", lambda: sentinel)
    config = compiler.quantization("text_encoder", {"precision": "w16a16", "context_tokens": 77})
    assert config["calibration_method"] is sentinel
    assert config["per_token_quant"] is False


def test_prompt_rows_preserve_order_and_share_a_content_bound_cache(tmp_path, load_script, monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace

    calibration = load_script("calibration")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(default_matmul_precision=lambda _: contextlib.nullcontext()))
    monkeypatch.setattr(sys.modules["afe.apis.defines"], "InputName", str)
    prompts = [str(i) for i in range(65)]
    table = np.zeros((1, 512), np.float32)
    graph, tokenizer = tmp_path / "text.onnx", tmp_path / "tokenizer.json"
    graph.write_bytes(b"first-graph")
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(calibration, "tokenize", lambda *args: (
        np.arange(len(prompts), dtype=np.float32)[:, None, None, None], np.zeros((len(prompts), 1, 96, 1), np.float32)))
    calls = []

    class Encoder:
        def execute(self, *, inputs, use_jax):
            values = next(iter(inputs.values()))
            assert values.shape == (1, 32, 1, 1) and use_jax
            calls.append(values.copy())
            return [np.eye(512, dtype=np.float32)[values.reshape(-1).astype(int)][None, :, None, :]]

    arguments = (prompts, graph, tokenizer, table, Encoder)
    first = calibration.cached_prompt_vectors(*arguments, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(first, np.eye(512, dtype=np.float32)[:65])
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[-1].reshape(-1), np.full(32, 64))
    np.testing.assert_array_equal(calibration.cached_prompt_vectors(*arguments, cache_dir=tmp_path / "cache"), first)
    assert len(calls) == 3  # A second model never executes the shared encoder.
    graph.write_bytes(b"changed-graph")
    calibration.cached_prompt_vectors(*arguments, cache_dir=tmp_path / "cache")
    assert len(calls) == 6


def test_compiled_components_are_entirely_on_mla(compiled: Path, load_script):
    compiler = load_script("compile")
    manifests = {}
    for component in ("text_encoder", "detector"):
        (archive,) = (compiled / component).glob("*_mpk.tar.gz")
        manifests[component] = compiler.audit_mpk(archive)
    assert [row["name"] for row in manifests["text_encoder"]["input_nodes"]] == ["token_embeddings", "eos_selector"]
    assert [row["name"] for row in manifests["detector"]["input_nodes"]] == ["images", "text_embeddings", "prompt_mask"]


def test_compile_recipe_records_grouped_graphs_and_final_archives(tmp_path, load_script):
    import hashlib
    import io
    import tarfile
    from unittest.mock import patch

    from model_registry.artifact import validate_compile_outputs
    from model_registry.errors import ValidationError

    compiler = load_script("compile")
    prepared = tmp_path / "graphs"
    prepared.mkdir()
    for component in compiler.COMPONENTS:
        (prepared / f"{component}.onnx").write_bytes(f"{component} graph".encode())
    np.save(prepared / "token_embedding_table.npy", np.zeros((1, 512), np.float32))
    (prepared / "tokenizer.json").write_text("{}")

    compiler_archives = {}

    class Backend:
        def quantize(self, **kwargs):
            return self

        def compile(self, *, output_path, **kwargs):
            directory = Path(output_path)
            directory.mkdir(parents=True)
            with tarfile.open(directory / "fixture_mpk.tar.gz", "w:gz") as archive:
                for name, content in {
                    "model.elf": b"fixture",
                    "model_mpk.json": b'{"plugins":[{"processor":"MLA"}]}',
                }.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            compiler_archives[directory.name] = (directory / "fixture_mpk.tar.gz").read_bytes()

    output = tmp_path / "build"
    profile = "w8a8_mixed_w16a16"
    with patch.object(compiler, "load_model", return_value=Backend()), \
            patch.object(compiler, "onnx_source"), patch.object(compiler, "calibration_inputs"):
        compiler.compile_prepared(prepared, output, profile)
    identity = dict(model_id=compiler.MODEL_DIR.name, build_profile_id=profile,
                    compile_invocation_id=profile, component_ids=list(compiler.COMPONENTS))
    manifest = output / "compile-outputs.json"
    result = validate_compile_outputs(manifest, output, **identity)
    assert result["derived_graphs"] == [
        {"component_id": component, "sha256": hashlib.sha256(f"{component} graph".encode()).hexdigest()}
        for component in compiler.COMPONENTS
    ]
    # Preserve the compiler bytes; CPU files travel beside the archives in the outer package.
    for component, payload in compiler_archives.items():
        assert (output / component / "fixture_mpk.tar.gz").read_bytes() == payload
    assert result["runtime_files"] == [
        {"path": f"prompt_assets/{name}", "sha256": hashlib.sha256((prepared / name).read_bytes()).hexdigest()}
        for name in ("token_embedding_table.npy", "tokenizer.json")
    ]
    for item in result["runtime_files"]:
        assert (output / item["path"]).read_bytes() == (prepared / Path(item["path"]).name).read_bytes()
    table = output / "prompt_assets/token_embedding_table.npy"
    payload = table.read_bytes()
    table.write_bytes(b"changed runtime file")
    with pytest.raises(ValidationError, match="runtime file.*SHA-256"):
        validate_compile_outputs(manifest, output, **identity)
    table.write_bytes(payload)
    (output / "detector" / "fixture_mpk.tar.gz").write_bytes(b"changed after manifest")
    with pytest.raises(ValidationError, match="MPK SHA-256 does not match"):
        validate_compile_outputs(manifest, output, **identity)
