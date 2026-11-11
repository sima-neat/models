from __future__ import annotations

import io
import json
import random
import tarfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def _synthetic_source(path: Path, surgery: ModuleType) -> Path:
    """Twelve strided Convs at Kalray's head names, box and class head per level, reading the source input."""
    nodes, initializers = [], []
    strides = (16, 32, 64, 128, 256, 300)
    for level, spec in enumerate(surgery.LEVELS):
        for role, channels in (("BoxEncodingPredictor", spec[2]), ("ClassPredictor", spec[3])):
            prefix = f"BoxPredictor_{level}/{role}/BiasAdd"
            initializers += [
                numpy_helper.from_array(np.zeros((channels, 3, 1, 1), np.float32), f"{prefix}/weights"),
                numpy_helper.from_array(np.zeros((channels,), np.float32), f"{prefix}/bias"),
            ]
            nodes.append(
                helper.make_node(
                    "Conv",
                    [surgery.SOURCE_INPUT, f"{prefix}/weights", f"{prefix}/bias"],
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
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    return path


def test_surgery_renames_and_orders_the_twelve_heads_without_new_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = _synthetic_source(tmp_path / "source.onnx", surgery)
    monkeypatch.setattr(surgery, "_sha256", lambda path: surgery.SOURCE_SHA256)
    monkeypatch.setattr(surgery, "EXPECTED_OPERATOR_COUNTS", Counter({"Conv": 12}))

    prepared = onnx.load(surgery.prepare_model(source, tmp_path / "prepared.onnx"))

    surgery.validate_prepared_model(prepared)
    assert [value.name for value in prepared.graph.output] == [name for name, _ in surgery.OUTPUTS]
    assert [node.op_type for node in prepared.graph.node] == ["Conv"] * 12


def test_surgery_rejects_another_source(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = _synthetic_source(tmp_path / "source.onnx", surgery)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        surgery.prepare_model(source, tmp_path / "prepared.onnx")


def test_stretch_samples_rows_and_columns_as_tensorflow_1_does(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A 600-pixel ramp halves to 300: TF1 samples source column 2x exactly, with no half-pixel offset.
    ramp = np.tile(np.arange(600, dtype=np.float32)[None, :, None], (450, 1, 3))

    stretched = calibration.stretch(ramp)

    assert stretched.shape == (300, 300, 3)
    np.testing.assert_allclose(stretched[0, :, 0], np.arange(0, 600, 2))
    # Rows sample 1.5 apart, so a vertical ramp lands on 1.5 * row.
    vertical = np.tile(np.arange(450, dtype=np.float32)[:, None, None], (1, 600, 3))
    np.testing.assert_allclose(calibration.stretch(vertical)[:, 0, 0], np.arange(300) * 1.5, atol=1e-4)


def test_preprocessing_returns_rgb_hwc_scaled_to_minus_one_to_one(load_script: Callable[[str], ModuleType]) -> None:
    import cv2

    calibration = load_script("calibration")
    image = np.zeros((40, 60, 3), np.uint8)
    image[..., 0] = 255  # red in RGB
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok

    result = calibration.preprocess_image(encoded.tobytes())

    assert result.shape == (300, 300, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result[150, 150], [1.0, -1.0, -1.0], atol=1e-6)


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"coco/{split}/{index:012d}.jpg")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _mpk_manifest(compile_module: ModuleType, precision: str) -> dict:
    scalar_bytes = 1 if precision == "w8a8" else 2
    mla = {"sequence": 1, "processor": "MLA", "config_params": {}, "output_nodes": []}
    plugins = [{"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}}, mla]
    terminal_inputs = []
    for index, (name, (batch, channels, height, width)) in enumerate(compile_module.OUTPUTS):
        shape = [batch, height, width, channels]
        padded_shape = [batch, height, width, channels + (8 if index % 2 == 0 else 0)]
        source = {"name": f"MLA_0_{index}", "size": int(np.prod(padded_shape)) * scalar_bytes}
        mla["output_nodes"].append(source)
        if index % 2 == 0:
            sliced = {"name": f"slice_{index}", "size": int(np.prod(shape)) * scalar_bytes}
            plugins.append(
                {
                    "sequence": len(plugins),
                    "processor": "EV74",
                    "input_nodes": [source.copy()],
                    "output_nodes": [sliced.copy()],
                    "config_params": {"kernel": "slice_transform", "params": {"output_shapes": [shape]}},
                }
            )
            source = sliced
        converted = {"name": f"convert_{index}/{name}", "size": int(np.prod(shape)) * 4}
        params = {"input_shapes": [shape], "output_shapes": [shape]}
        if precision == "w8a8":
            kernel = "dequantization_transform"
            params["input_data_type"] = "int8"
        else:
            kernel = "cast_transform"
            params.update(in_dtype="bfloat16", out_dtype="float32")
        plugins.append(
            {
                "sequence": len(plugins),
                "processor": "EV74",
                "input_nodes": [source.copy()],
                "output_nodes": [converted.copy()],
                "config_params": {"kernel": kernel, "params": params},
            }
        )
        terminal_inputs.append(converted)
    plugins.append(
        {
            "sequence": len(plugins),
            "processor": "EV74",
            "config_params": {"kernel": "pass_through"},
            "input_nodes": terminal_inputs,
            "output_nodes": [
                {"name": f"pass_through_out_{index}", "size": size}
                for index, size in enumerate(compile_module.OUTPUT_BYTES)
            ],
        }
    )
    return {
        "model_sdk_version": compile_module.MODEL_SDK_VERSION,
        "input_nodes": [{"name": "images", "size": 1 * 3 * 300 * 300 * 4}],
        "plugins": plugins,
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("precision", ["w8a8", "w16a16"])
@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, precision: str, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module, precision)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[6] = outputs[6], outputs[0]
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive, precision)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive, precision)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    expected = {"precision": profile_id}
    if profile_id == "w8a8":
        expected |= {"calibration_bins": 2048, "calibration_samples": 512}

    assert load_script("compile")._parameters(profile_id) == expected


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w32a32")


@pytest.mark.parametrize("precision", ["w8a8", "w16a16"])
@pytest.mark.parametrize("case", ["identity", "shape", "type", "lineage", "size", "duplicate", "slice"])
def test_mpk_audit_rejects_corrupted_head_contracts(
    tmp_path: Path, precision: str, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module, precision)
    terminal = manifest["plugins"][-1]
    converter = next(
        p for p in manifest["plugins"] if p.get("output_nodes", [{}])[0].get("name") == "convert_0/ssd_loc_0"
    )
    params = converter["config_params"]["params"]
    if case == "identity":
        # Rename the head while keeping every byte count unchanged: the old audit accepted this.
        converter["output_nodes"][0]["name"] = terminal["input_nodes"][0]["name"] = "convert_0/ssd_loc_1"
    elif case == "shape":
        # NCHW substituted for NHWC has exactly the same byte count.
        params["input_shapes"] = params["output_shapes"] = [[1, 12, 19, 19]]
    elif case == "type":
        params["input_data_type" if precision == "w8a8" else "out_dtype"] = "float16"
    elif case == "lineage":
        converter["input_nodes"][0]["name"] = "not_an_mla_output"
    elif case == "size":
        converter["output_nodes"][0]["size"] += 4
    elif case == "duplicate":
        terminal["output_nodes"][1]["name"] = terminal["output_nodes"][0]["name"]
    elif case == "slice":
        manifest["plugins"][2]["config_params"]["params"]["output_shapes"] = [[1, 12, 19, 19]]
    archive = tmp_path / "bad_mpk.tar.gz"
    _write_mpk(archive, manifest)

    with pytest.raises(RuntimeError):
        compile_module.audit_mpk(archive, precision)


def test_compiled_mpk_has_the_declared_head_contract(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    archives = list((compiled / "model").glob("*_mpk.tar.gz"))
    assert len(archives) == 1, "the compile output must contain exactly one model MPK"
    load_script("compile").audit_mpk(archives[0], build_profile)


def test_compiled_mpk_preserves_the_padded_mla_head_storage(
    mpk_json: Callable[[str], dict],
    build_profile: str,
) -> None:
    manifest = mpk_json("model")
    # The artifact audit above owns the SDK-version requirement; package identity is a separate workflow check.
    mla = [plugin for plugin in manifest["plugins"] if plugin.get("processor") == "MLA"]
    assert len(mla) == 1
    # Independent physical-byte expectations, not compile.py's logical FP32
    # interface. BF16 uses C8 padding; it is not twice the INT8 C16 storage.
    expected = {
        "w8a8": [5776, 3200, 800, 288, 128, 32, 103968, 56000, 14000, 5040, 2240, 560],
        "w16a16": [11552, 4800, 1200, 432, 192, 48, 202160, 110400, 27600, 9936, 4416, 1104],
    }
    assert [node["size"] for node in mla[0]["output_nodes"]] == expected[build_profile]


def _archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for info, content in entries:
            archive.addfile(info, io.BytesIO(content) if info.isfile() else None)


def _file(name, content=b"image bytes"):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    return info, content


def test_directory_and_archive_keep_the_same_canonical_selection_order(load_script, tmp_path):
    module = load_script("calibration")
    dataset = tmp_path / "dataset"
    names = [f"coco/cal2017/{i:012d}.jpg" for i in range(8)]
    entries = []
    for i, name in enumerate(reversed(names)):
        content = f"image{i}".encode()
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(_file(name, content))
    archive = tmp_path / "calibration.tgz"
    _archive(archive, entries)
    expected = random.Random(module.CALIBRATION_SEED).sample(names, 4)
    assert module.calibration_names(dataset, 4) == module.calibration_names(archive, 4) == expected
    assert list(module._image_bytes(dataset, expected)) == list(module._image_bytes(archive, expected))


@pytest.mark.parametrize("alias", ["coco/cal2017/a.jpg", "./coco/cal2017/a.jpg"])
def test_duplicate_archive_names_reject_before_selection(load_script, tmp_path, alias):
    module = load_script("calibration")
    archive = tmp_path / "duplicates.tgz"
    _archive(archive, [_file("coco/cal2017/a.jpg", b"first"), _file(alias, b"last")])
    with pytest.raises(ValueError, match="duplicate dataset member"):
        module.calibration_names(archive, 2)


@pytest.mark.parametrize("name", ["/cal2017/a.jpg", "../cal2017/a.jpg", "root/../cal2017/a.jpg"])
def test_unsafe_archive_names_reject(load_script, tmp_path, name):
    module = load_script("calibration")
    archive = tmp_path / "unsafe.tgz"
    _archive(archive, [_file(name)])
    with pytest.raises(ValueError, match="unsafe dataset member"):
        module.calibration_names(archive, 1)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_links_and_special_archive_members_reject(load_script, tmp_path, kind):
    module = load_script("calibration")
    info = tarfile.TarInfo("coco/cal2017/link.jpg")
    info.type = kind
    info.linkname = "../outside.jpg"
    archive = tmp_path / "link.tgz"
    _archive(archive, [(info, b"")])
    with pytest.raises(ValueError, match="unsafe dataset member"):
        module.calibration_names(archive, 1)


def test_directory_symlink_cannot_escape_the_dataset(load_script, tmp_path):
    module = load_script("calibration")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside bytes")
    dataset = tmp_path / "dataset"
    (dataset / "cal2017").mkdir(parents=True)
    (dataset / "cal2017/link.jpg").symlink_to(outside)
    with pytest.raises(ValueError, match="outside dataset"):
        module.calibration_names(dataset, 1)
    with pytest.raises(ValueError, match="outside dataset"):
        list(module._image_bytes(dataset, ["../outside.jpg"]))


def test_read_checks_directory_boundary_again_after_selection(load_script, tmp_path):
    module = load_script("calibration")
    dataset = tmp_path / "dataset"
    path = dataset / "cal2017/a.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"inside bytes")
    names = module.calibration_names(dataset, 1)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside bytes")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside dataset"):
        list(module._image_bytes(dataset, names))


@pytest.mark.parametrize("precision", ["w8a8", "w16a16"])
def test_quantization_uses_explicit_precision_specific_observers(load_script, monkeypatch, precision):
    module = load_script("compile")
    seen = {}

    class Params:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def with_requantization_mode(self, mode):
            seen["requantization_mode"] = mode
            return self

        def with_per_token_quant(self, enabled):
            seen["per_token_quant"] = enabled
            return self

    monkeypatch.setattr(module, "QuantizationParams", Params)
    monkeypatch.setattr(module, "HistogramMSEMethod", lambda bins: ("histogram_mse", bins))
    monkeypatch.setattr(module, "SkipCalibration", lambda: "skip_calibration")
    monkeypatch.setattr(module, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(module, "quantization_scheme", lambda **kwargs: kwargs)
    module._quantization(precision, 2048 if precision == "w8a8" else None)
    assert seen["per_token_quant"] is False
    assert seen["node_names"] == {""}
    if precision == "w8a8":
        assert seen["calibration_method"] == ("histogram_mse", 2048)
        assert seen["activation_quantization_scheme"] == {"asymmetric": True, "per_channel": False, "bits": 8}
        assert seen["weight_quantization_scheme"] == {"asymmetric": False, "per_channel": True, "bits": 8}
        assert seen["biascorr_type"] == module.BiasCorrectionType.REGULAR
    else:
        assert seen["calibration_method"] == "skip_calibration"
        assert seen["activation_quantization_scheme"] == seen["weight_quantization_scheme"] == "bf16"
        assert "biascorr_type" not in seen


def test_bf16_quantization_receives_no_dummy_calibration_data(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = load_script("compile")
    monkeypatch.setattr(module, "_parameters", lambda _: {"precision": "w16a16"})
    monkeypatch.setattr(module, "_quantization", lambda *args: "bf16_skip_calibration")
    monkeypatch.setattr(module, "prepare_model", lambda *args: tmp_path / "heads.onnx")
    monkeypatch.setattr(module, "onnx_source", lambda *args, **kwargs: "imported")

    def no_samples(*args, **kwargs):
        raise AssertionError("BF16 must not read or synthesize calibration images")

    class ReachedQuantize(Exception):
        pass

    def quantize(**kwargs):
        assert kwargs["calibration_data"] is None
        assert kwargs["quantization_config"] == "bf16_skip_calibration"
        raise ReachedQuantize

    monkeypatch.setattr(module, "calibration_images", no_samples)
    monkeypatch.setattr(module, "DataGenerator", no_samples)
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: SimpleNamespace(quantize=quantize))
    with pytest.raises(ReachedQuantize):
        module.compile_model(tmp_path / "source.onnx", tmp_path / "output", "w16a16", None)
