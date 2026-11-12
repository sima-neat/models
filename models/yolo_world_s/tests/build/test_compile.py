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


@pytest.mark.parametrize("archive_input", [False, True])
def test_calibration_selects_only_cal2017_and_preserves_order(tmp_path, load_script, monkeypatch, archive_input):
    import tarfile

    calibration = load_script("calibration")
    directory = tmp_path / "dataset"
    for split, count in (("cal2017", 68), ("val2017", 5)):
        (directory / split).mkdir(parents=True)
        for index in range(count):
            (directory / split / f"{index:012}.jpg").write_text(str(index))
    dataset = directory
    if archive_input:
        dataset = tmp_path / "dataset.tgz"
        with tarfile.open(dataset, "w:gz") as archive:
            for path in reversed(sorted(directory.rglob("*.jpg"))):
                archive.add(path, arcname=path.relative_to(directory).as_posix())
    selected = calibration.calibration_names(dataset, 64)
    assert selected == calibration.calibration_names(directory, 64)
    assert len(selected) == len(set(selected)) == 64
    assert all(Path(name).parent.name == "cal2017" for name in selected)
    monkeypatch.setattr(calibration, "preprocess_image", lambda content: np.array([int(content)], np.float32))
    images = calibration.calibration_images(dataset, 64)
    np.testing.assert_array_equal(images[:, 0], [int(Path(name).stem) for name in selected])
    with pytest.raises(ValueError, match="cannot select"):
        calibration.calibration_names(dataset, 69)


def test_calibration_pairs_every_image_with_all_coco_prompts(load_script, monkeypatch):
    import sys

    calibration = load_script("calibration")
    monkeypatch.setattr(sys.modules["afe.apis.defines"], "InputName", str)
    images = np.arange(36, dtype=np.float32).reshape(3, 2, 2, 3)
    vectors = np.eye(512, dtype=np.float32)[:80]
    monkeypatch.setattr(calibration, "calibration_images", lambda dataset, count: images)

    def cached(prompts, *args):
        assert len(prompts) == len(set(prompts)) == 80
        assert (prompts[0], prompts[-1]) == ("person", "toothbrush")
        return vectors

    monkeypatch.setattr(calibration, "cached_prompt_vectors", cached)
    samples = calibration.calibration_inputs(Path("dataset.tgz"), None, None, None, None, 3)
    for index, sample in enumerate(samples):
        np.testing.assert_array_equal(sample["images"], images[index:index + 1])
        np.testing.assert_array_equal(sample["text_embeddings"], vectors[None])
        np.testing.assert_array_equal(sample["prompt_mask"], np.ones((1, 80), np.float32))


def test_calibration_is_pinned_in_model_registry(model_dir):
    from model_registry.model import load_model

    model = load_model(model_dir)
    assert model["calibrations"] == [{"id": "coco-2017-cal", "dataset": {
        "id": "coco-2017", "variant": "cal", "sha256": "54c64d84122eba371ac014d7b43849a461551c2e6c5c430b5ead90ec1dec12b1",
        "size_bytes": 187510866,
    }}]
    (profile,) = model["build_profiles"]
    assert profile["components"][0]["calibration"] == "coco-2017-cal"
    detector = next(item for item in profile["components"] if item["id"] == "detector")
    assert detector["parameters"]["calibration_samples"] == 64


def test_bf16_text_configuration_skips_calibration(load_script, monkeypatch):
    compiler = load_script("compile")
    sentinel = object()
    monkeypatch.setattr(compiler, "QuantizationParams", lambda **values: values)
    monkeypatch.setattr(compiler, "SkipCalibration", lambda: sentinel)
    config = compiler.quantization("text_encoder", {"precision": "w16a16", "context_tokens": 77})
    assert config["calibration_method"] is sentinel
    assert config["per_token_quant"] is False


@pytest.mark.parametrize("profile,parameters", [
    ("w16a16_w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["unknown"]}),
    ("w16a16_w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["projection_attention"] * 2}),
    ("w16a16_w8a8_mixed", {"precision": "w16a16", "bf16_regions": ["projection_attention"]}),
    ("w16a16_w8a8_mixed", {"precision": "w8a8"}),
    ("w16a16_w8a8", {"precision": "w8a8", "bf16_regions": ["projection_attention"]}),
])
def test_bf16_regions_reject_incompatible_precision_or_profile(load_script, profile, parameters):
    compiler = load_script("compile")
    with pytest.raises(ValueError):
        compiler._bf16_regions(profile, {"text_encoder": {}, "detector": parameters})


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
    import shutil
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

    def prepare(source, directory, *, bf16_regions):
        assert source == tmp_path / "weights.pth"
        assert bf16_regions == ["projection_attention"]
        shutil.copytree(prepared, directory, dirs_exist_ok=True)

    output = tmp_path / "build"
    profile = "w16a16_w8a8_mixed"
    with patch.object(compiler, "load_model", return_value=Backend()), \
            patch.object(compiler, "onnx_source"), patch.object(compiler, "calibration_inputs"), \
            patch.object(compiler, "prepare_model", side_effect=prepare):
        compiler.compile_model(tmp_path / "weights.pth", output, profile, dataset=tmp_path / "calibration.tgz")
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
