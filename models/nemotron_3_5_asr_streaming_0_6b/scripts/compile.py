"""Compile the full encoder and package separate dynamic INT8 CPU decoding models."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import os
import resource
import shutil
import site
import subprocess
import sysconfig
import tarfile
import tempfile
import venv
from importlib.metadata import version
from pathlib import Path

import model_registry
import numpy as np
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]
# Kept in this compile input so dependency changes invalidate compilation, not conversion.
EXPORT_REQUIREMENTS = (
    "torch==2.8.0",
    "transformers==5.16.1",
    "safetensors==0.8.0",
    "numpy==1.26.4",
    "onnx==1.17.0",
    "onnxsim==0.7.3",
    "onnxruntime==1.21.1",
    "librosa==0.11.0",
    "scipy==1.14.1",
)
RUNTIME_FILES = (
    "predictor_cpu_int8.onnx",
    "joiner_cpu_int8.onnx",
    "tokenizer.json",
    "processor_config.json",
    "runtime_config.json",
    "frontend.npz",
    "LICENSE.txt",
    "NOTICE.txt",
)


def runtime_contract(archive: Path, logical_outputs: list[str]) -> dict:
    """Read native tensor shapes without modifying the compiler-produced archive."""
    with tarfile.open(archive, "r|gz") as stream:
        manifests = [json.load(stream.extractfile(member)) for member in stream if member.name.endswith("_mpk.json")]
    if len(manifests) != 1:
        raise ValueError("Expected exactly one compiled manifest")
    plugins = manifests[0]["plugins"]
    accelerators = [plugin for plugin in plugins if plugin["processor"] == "MLA"]
    casts = [plugin for plugin in plugins if plugin["config_params"].get("kernel") == "cast_transform"]
    host = [plugin for plugin in plugins if plugin["processor"] != "MLA"]
    if len(accelerators) != 1 or any(
        plugin["config_params"].get("kernel") not in {"cast_transform", "pass_through"} for plugin in host
    ):
        raise ValueError(
            "Encoder must contain one accelerator region with only boundary conversions and identity output publication"
        )
    accelerator = accelerators[0]
    produced = {port["name"]: plugin for plugin in casts for port in plugin["output_nodes"]}
    consumed = {port["name"]: plugin for plugin in casts for port in plugin["input_nodes"]}
    inputs, outputs = [], []
    for port in accelerator["input_nodes"]:
        cast = produced[port["name"]]
        params = cast["config_params"]["params"]
        if params["out_dtype"] != "bfloat16":
            raise ValueError("Expected native BF16 encoder inputs")
        inputs.append(dict(name=cast["input_nodes"][0]["name"], shape=params["output_shapes"][0], bytes=port["size"]))
    for name, port in zip(logical_outputs, accelerator["output_nodes"], strict=True):
        params = consumed[port["name"]]["config_params"]["params"]
        if params["in_dtype"] != "bfloat16":
            raise ValueError("Expected native BF16 encoder outputs")
        outputs.append(dict(name=name, shape=params["input_shapes"][0], bytes=port["size"]))
    return dict(inputs=inputs, outputs=outputs)


def cpu_model(source: Path, destination: Path) -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        # Fold checkpoint Transpose/Reshape constants before dynamic quantization,
        # which requires constant MatMul weights to be initializers.
        optimized = Path(directory) / "optimized.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(optimized)
        session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        del session
        quantize_dynamic(
            optimized,
            destination,
            per_channel=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Gather", "LSTM", "MatMul"],
        )
    onnx.checker.check_model(str(destination))


def prepare_model(source: Path, temporary: Path, component: Path) -> None:
    """Release all export and CPU-quantization objects before the encoder compiler starts."""
    import onnx
    from surgery import export_model

    graph_path = export_model(source, temporary / "graphs")
    graph = onnx.load(graph_path, load_external_data=False)
    config = json.loads(next(item.value for item in graph.metadata_props if item.key == "sima.asr"))
    for filename, document in config.pop("assets").items():
        if filename not in {"tokenizer.json", "processor_config.json"}:
            raise ValueError(f"Unexpected runtime asset: {filename}")
        (component / filename).write_text(json.dumps(document) + "\n")
    for name in ("predictor", "joiner"):
        cpu_model(graph_path.parent / (name + ".onnx"), component / (name + "_cpu_int8.onnx"))
    (temporary / "config.json").write_text(json.dumps(config))


def prepare_in_environment(source: Path, temporary: Path, component: Path) -> None:
    """Install export dependencies in a disposable environment, keeping the compiler unchanged."""
    with tempfile.TemporaryDirectory(dir=temporary, prefix="export-env-") as directory:
        directory = Path(directory)
        venv.EnvBuilder(with_pip=True).create(directory)
        python = directory / "bin/python"
        packages = Path(sysconfig.get_path("purelib", vars={"base": str(directory), "platbase": str(directory)}))
        # Reuse compatible compiler packages as fallbacks. Pip installs missing or conflicting versions only
        # in the export environment, whose own packages precede these paths. Include CI's registry tooling.
        inherited = [
            *site.getsitepackages(),
            *os.environ.get("PYTHONPATH", "").split(os.pathsep),
            str(Path(model_registry.__file__).resolve().parent.parent),
        ]
        paths = list(dict.fromkeys(str(Path(path).resolve()) for path in inherited if path))
        (packages / "compiler-packages.pth").write_text("\n".join(paths) + "\n")
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                *EXPORT_REQUIREMENTS,
            ],
            check=True,
        )
        # Isolated mode prevents PYTHONPATH or user packages from overriding the installed export pins.
        bootstrap = (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from transformers import AutoModelForRNNT, AutoModelForTDT, AutoProcessor; "
            "from compile import prepare_model; prepare_model(*(Path(p) for p in sys.argv[2:]))"
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                bootstrap,
                str(MODEL_DIR / "scripts"),
                str(source),
                str(temporary),
                str(component),
            ],
            check=True,
        )


def compile_model(source: Path, profile: str, output: Path) -> Path:
    metadata = load_metadata(MODEL_DIR)
    selected = next(item for item in metadata["build_profiles"] if item["id"] == profile)
    if selected["components"][0]["parameters"]["precision"] != "w8a16":
        raise ValueError("Only INT8 weights with BF16 activations are supported")
    name = metadata["id"] + "_" + profile
    component = output / "model"
    component.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=output, prefix="encoder-") as temporary:
        temporary = Path(temporary)
        prepare_in_environment(source, temporary, component)
        shutil.copyfile(temporary / "graphs/source-validation.json", output / "source-validation.json")
        config = json.loads((temporary / "config.json").read_text())
        encoder_path = temporary / "graphs/model.onnx"
        parameters = selected["components"][0]["parameters"]
        keys = ("chunk_ms", "lookahead")
        expected = {"precision": "w8a16", **{key: config[key] for key in keys}}
        if parameters != expected:
            raise ValueError(f"Build parameters must match the pinned encoder: {expected}")
        from afe.apis.defines import (
            RequantizationMode,
            SkipCalibration,
            bfloat16_scheme,
            default_quantization,
            gen2_target,
            quantization_scheme,
        )
        from afe.apis.error_handling_variables import enable_verbose_error_messages
        from afe.apis.loaded_net import load_model
        from afe.load.importers.general_importer import onnx_source

        enable_verbose_error_messages()
        for package in ("sima-frontend", "sima-mlc"):
            print(f"Compiler package: {package}={version(package)}", flush=True)
        imported = load_model(
            onnx_source(str(encoder_path), layout="NCHW", flexible_batch_size=False), target=gen2_target
        )
        quantization = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(SkipCalibration())
        )
        print(f"BF16 activation per-token quantization: {quantization.per_token_quant}", flush=True)
        quantized = imported.quantize(
            calibration_data=None, quantization_config=quantization, model_name=name, log_level=logging.INFO
        )
        # Compilation does not reuse the imported FP32 model or the quantized graph.
        del imported
        gc.collect()

        def memory_stats():
            try:
                groups = Path("/proc/self/cgroup").read_text().splitlines()
                group = next(line[3:] for line in groups if line.startswith("0::"))
                directory = Path("/sys/fs/cgroup") / group.lstrip("/")
                events = (directory / "memory.events").read_text().strip().replace("\n", ", ")
                peak = (directory / "memory.peak").read_text().strip()
                return f"{events}; peak_bytes={peak}"
            except (OSError, StopIteration):
                return "unavailable"

        print(f"Compiler memory before: {memory_stats()}", flush=True)
        try:
            quantized.compile(
                output_path=str(temporary / "compiled"),
                batch_size=1,
                log_level=logging.INFO,
                preserve=False,
            )
        finally:
            print(f"Compiler memory after: {memory_stats()}", flush=True)
            print(
                f"Peak RSS KiB: process={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, "
                f"child={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}",
                flush=True,
            )
        archives = list((temporary / "compiled").rglob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValueError("Expected one full encoder archive")
        archive = component / (name + "_mpk.tar.gz")
        shutil.copyfile(archives[0], archive)
    config.update(runtime_contract(archive, config["encoder_outputs"]))
    config.pop("encoder_outputs")
    frontend = config.pop("frontend")
    arrays = {
        key: np.frombuffer(base64.b64decode(value["data"]), dtype="<f4").reshape(value["shape"])
        for key, value in frontend.items()
    }
    np.savez(component / "frontend.npz", **arrays)
    for source_name, target_name in (
        ("LICENSE.txt", "LICENSE.txt"),
        ("NOTICE.txt", "NOTICE.txt"),
    ):
        shutil.copyfile(MODEL_DIR / "scripts" / source_name, component / target_name)
    (component / "runtime_config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_compile_outputs(output, archive, metadata["id"], profile)
    return archive


def write_compile_outputs(output: Path, archive: Path, model_id: str, profile: str) -> None:
    """Record the encoder and CPU assets using the registry's supporting-file contract."""
    component = archive.parent
    manifest = {
        "model_id": model_id,
        "build_profile_id": profile,
        "compile_invocation_id": profile,
        "outputs": [
            {
                "component_id": "model",
                "mpk_path": f"model/{archive.name}",
                "mpk_sha256": sha256_file(archive)[0],
            }
        ],
        "runtime_files": [
            {"path": f"model/{filename}", "sha256": sha256_file(component / filename)[0]} for filename in RUNTIME_FILES
        ],
    }
    (output / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("Compile invocation must equal the build profile")
    logging.basicConfig(level=logging.INFO)
    compile_model(args.model.resolve(), args.build_profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
