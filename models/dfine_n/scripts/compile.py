#!/usr/bin/env python3
"""Compile one D-FINE Nano build profile with the registry's existing flow.

model.yaml owns each component's precision and calibration. Only W8A8 backbones
calibrate; BF16-activation components, including every decoder, skip calibration. The mixed profile freezes a compact
precision map onto the prepared backbone and checks the compiler applied it. Full-model
fidelity is diagnostic because quantization can reorder selected queries; COCO
evaluation is the task gate.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import decoder_inputs, load_images, select_images, to_afe_layout
from model_registry.compile_script import compiler_log_level, parse_arguments
from model_registry.fetch import sha256_file
from model_registry.fidelity import write_fidelity_report
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    BACKBONE_OUTPUTS,
    DECODER_INPUTS,
    INPUT_SHAPE,
    OUTPUTS,
    annotate_backbone,
    prepare_model,
    region_precision_map,
    validate_bf16_regions,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dfine_n"
COMPONENTS = ("backbone", "decoder")
PRECISIONS = ("w16a16", "w8a8")
SHAPES = {"backbone": {"input": INPUT_SHAPE}, "decoder": DECODER_INPUTS}
OUTPUT_SHAPES = {"backbone": tuple(BACKBONE_OUTPUTS.values()), "decoder": tuple(OUTPUTS.values())}


def calibrates(parameters):
    return "calibration_samples" in parameters


def _component_parameters(metadata, profile_id):
    """Each component's build parameters, as model.yaml declares them."""
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}
    if list(parameters) != list(COMPONENTS):
        raise ValueError("Expected ordered backbone and decoder components")
    for name, value in parameters.items():
        if value.get("precision") not in PRECISIONS:
            raise ValueError(f"Unsupported {name} precision: {value.get('precision')!r}")
        if ("calibration_bins" in value) != calibrates(value):
            raise ValueError(f"The {name} declares calibration_bins and calibration_samples together or not at all")
        if calibrates(value) and any(
            type(value[key]) is not int or value[key] < 1 for key in ("calibration_bins", "calibration_samples")
        ):
            raise ValueError(f"Invalid {name} calibration parameters for {profile_id}")
    decoder = parameters["decoder"]
    if decoder["precision"] != "w16a16" or "bf16_regions" in decoder or calibrates(decoder):
        raise ValueError("Every profile keeps an uncalibrated W16A16 decoder without BF16 regions")
    regions = parameters["backbone"].get("bf16_regions")
    if regions is not None:
        validate_bf16_regions(regions)
        if parameters["backbone"]["precision"] != "w8a8":
            raise ValueError("BF16 regions apply to a W8A8 backbone")
    if (regions is not None) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"Build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return parameters


def quantization_config(parameters) -> QuantizationParams:
    """The compiler's activation and weight formats for a component's precision."""
    precision = parameters["precision"]
    calibration = (
        HistogramMSEMethod(int(parameters["calibration_bins"])) if calibrates(parameters) else SkipCalibration()
    )
    int8_weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    if precision == "w8a8":
        if not calibrates(parameters):
            raise ValueError("a W8A8 component must declare calibration_bins and calibration_samples")
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=int8_weights,
            # The measured Nano INT8 policy.
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"Unsupported precision: {precision!r}")


def quantize_component(graph, name, parameters, arrays):
    """Quantize one component; arrays holds its calibration inputs, or None when it skips calibration."""
    loaded = load_model(onnx_source(str(graph), layout="NCHW"))
    data = None
    if arrays is not None:
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Expected finite float32 calibration tensors")
        data = convert_data_generator_to_iterable(
            DataGenerator({InputName(key): to_afe_layout(value) for key, value in arrays.items()})
        )
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=quantization_config(parameters),
        model_name=name,
        log_level=compiler_log_level(),
    )
    if "bf16_regions" in parameters:
        require_mixed_precision(quantized)
    return quantized


def require_mixed_precision(quantized):
    """Check applied arithmetic, not only annotations, as RF-DETR does.

    Ignore placeholders and adapters: INT8 ingress does not prove an INT8
    convolution survived AFE's precision propagation and fusion.
    """
    counts = Counter(
        node.ir.quant_config.quantization_precision.value
        for node in quantized._net.iter_nodes_recursive()
        if type(getattr(node.ir, "operation", None)).__name__ == "ConvAddActivationOp"
        and getattr(node.ir, "quant_config", None) is not None
    )
    logging.info("Mixed backbone convolution precisions: %s", dict(counts))
    if not counts[QuantizationPrecision.BFLOAT_16] or not counts[QuantizationPrecision.INT_8]:
        raise RuntimeError(f"Expected BF16 and INT8 convolutions in the mixed backbone, found {dict(counts)}")


def audit_mpk(archive, component):
    """Check hardware placement and the public ABI, not incidental adapter topology."""
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe MPK member: {member.name}")
        names = [m.name for m in members]
        manifests = [n for n in names if n.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise ValueError("Expected one MPK manifest")
        manifest = json.load(stream.extractfile(manifests[0]))
    plugins = sorted(manifest["plugins"], key=lambda n: n["sequence"])
    mla = [n for n in plugins if n.get("processor") == "MLA"]
    if (
        len(mla) != 1
        or len([n for n in names if n.endswith(".elf")]) != 1
        or any(n.endswith(".so") or ".so." in n for n in names)
        or any(n.get("processor") not in ("MLA", "EV74") for n in plugins)
    ):
        raise ValueError("Expected one MLA partition and EV74 adapters, without host model fallback")
    inputs = {n["name"]: n["size"] for n in manifest["input_nodes"]}
    expected = {name: math.prod(shape) * 4 for name, shape in SHAPES[component].items()}
    terminal = plugins[-1]
    if (
        inputs != expected
        or terminal.get("config_params", {}).get("kernel") != "pass_through"
        or [n["size"] for n in terminal["output_nodes"]] != [math.prod(shape) * 4 for shape in OUTPUT_SHAPES[component]]
    ):
        raise ValueError(f"Unexpected {component} public input/output contract")
    return manifest


def compile_component(model, root, component):
    output = root / component
    output.mkdir()
    model.compile(output_path=str(output), batch_size=1, log_level=compiler_log_level())
    archives = list(output.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise ValueError(f"Expected one {component} model package")
    audit_mpk(archives[0], component)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def execute_backbone(model, image):
    tensors = tuple(
        np.asarray(v) for v in model.execute(inputs={InputName("input"): to_afe_layout(image[None])}, use_jax=True)
    )
    if len(tensors) != 4 or any(
        v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all()
        for v, shape in zip(tensors, OUTPUT_SHAPES["backbone"], strict=True)
    ):
        raise ValueError("Unexpected quantized backbone outputs")
    return tensors


def decoder_feed(outputs):
    return {
        name: np.ascontiguousarray(value) for name, value in zip(DECODER_INPUTS, decoder_inputs(outputs), strict=True)
    }


def prepare_phase(source, dataset, evaluation_names, work, parameters):
    backbone, decoder = prepare_model(source, work / "graphs")
    reference, before, after = session(source), session(backbone), session(decoder)
    values = {name: [] for name in OUTPUTS}
    for image in load_images(dataset, evaluation_names):
        expected = reference.run(None, {"input": image[None]})
        heads = before.run(None, {"input": image[None]})
        actual = after.run(None, decoder_feed(heads))
        for name, original, modified in zip(OUTPUTS, expected, actual, strict=True):
            np.testing.assert_allclose(
                modified,
                original,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"{name}: floating split/reference-logit approximation",
            )
            values[name].append(original[0])
    np.savez(work / "reference.npz", **{k: np.stack(v) for k, v in values.items()})
    if "bf16_regions" in parameters:
        # ONNX Runtime validates the unannotated floating split before AFE-only annotations.
        precision_map = region_precision_map(backbone, parameters["bf16_regions"])
        (work / "precision-map.json").write_text(json.dumps(precision_map, indent=2) + "\n")
        annotate_backbone(backbone, precision_map)


def backbone_phase(profile, parameters, dataset, calibration_names, evaluation_names, work):
    """Quantize and compile the backbone, saving its outputs for the decoder phase's fidelity check."""
    arrays = {"input": load_images(dataset, calibration_names)} if calibrates(parameters) else None
    model = quantize_component(work / "graphs/backbone.onnx", f"{MODEL_ID}_{profile}_backbone", parameters, arrays)
    evaluation = {name: [] for name in BACKBONE_OUTPUTS}
    for image in load_images(dataset, evaluation_names):
        for key, value in zip(BACKBONE_OUTPUTS, execute_backbone(model, image), strict=True):
            # AFE may recycle output storage between executions.
            evaluation[key].append(value[0].copy())
    np.savez(work / "backbone-evaluation.npz", **{k: np.stack(v) for k, v in evaluation.items()})
    compile_component(model, work, "backbone")


def decoder_phase(profile, parameters, work):
    model = quantize_component(work / "graphs/decoder.onnx", f"{MODEL_ID}_{profile}_decoder", parameters, None)
    compile_component(model, work, "decoder")
    compared = []
    with (
        np.load(work / "backbone-evaluation.npz", allow_pickle=False) as upstream,
        np.load(work / "reference.npz", allow_pickle=False) as reference,
    ):
        count = len(upstream[next(iter(BACKBONE_OUTPUTS))])
        for i in range(count):
            outputs = [upstream[name][i][None] for name in BACKBONE_OUTPUTS]
            actual = model.execute(inputs={InputName(k): v for k, v in decoder_feed(outputs).items()}, use_jax=True)
            # Copied, because the compiler may reuse an output's storage on the next execute.
            compared.append(
                {name: (reference[name][i][None], np.array(value)) for name, value in zip(OUTPUTS, actual, strict=True)}
            )
    write_fidelity_report(
        MODEL_DIR,
        profile,
        compared,
        work,
        reference="Original full ONNX (ONNX Runtime)",
        candidate=f"AFE profile {profile}: backbone -> CPU Top300/Gather -> bounded-reference decoder (JAX execute)",
    )


def execute_phase(target, arguments):
    # AFE can use SystemExit(None) on failure; it must not produce a successful worker.
    try:
        target(**arguments)
    except BaseException as error:
        raise RuntimeError(f"{target.__name__} failed; see its compiler log") from error


def run_phase(target, **arguments):
    """Release compiler/JAX memory between phases, as the RF-DETR recipe does."""
    process = multiprocessing.get_context("spawn").Process(
        name=target.__name__, target=execute_phase, args=(target, arguments)
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{target.__name__} failed with exit code {process.exitcode}; see its compiler log")


def calibration_images(parameters, dataset, fidelity_dataset, selection):
    """The backbone's calibration images, or none when it skips calibration."""
    if not calibrates(parameters["backbone"]):
        return ()
    count = parameters["backbone"]["calibration_samples"]
    if dataset is None:
        raise ValueError("calibrated components require --dataset")
    if dataset != fidelity_dataset or selection["offset"] < count:
        raise ValueError("Use the same verified dataset with disjoint calibration/fidelity images")
    return select_images(dataset, selection["seed"], 0, count)


def compile_model(source, profile, dataset, fidelity_dataset, output_dir):
    metadata = load_metadata(MODEL_DIR)
    parameters = _component_parameters(metadata, profile)
    selection = metadata["numerical_fidelity"]["selection"]
    calibration_names = calibration_images(parameters, dataset, fidelity_dataset, selection)
    evaluation_names = select_images(fidelity_dataset, selection["seed"], selection["offset"], selection["samples"])
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        work = Path(temporary)
        run_phase(
            prepare_phase,
            source=source,
            dataset=fidelity_dataset,
            evaluation_names=evaluation_names,
            work=work,
            parameters=parameters["backbone"],
        )
        run_phase(
            backbone_phase,
            profile=profile,
            parameters=parameters["backbone"],
            dataset=fidelity_dataset,
            calibration_names=calibration_names,
            evaluation_names=evaluation_names,
            work=work,
        )
        run_phase(decoder_phase, profile=profile, parameters=parameters["decoder"], work=work)
        output_dir.mkdir(parents=True)
        outputs = []
        for name in COMPONENTS:
            (archive,) = (work / name).glob("*_mpk.tar.gz")
            destination = output_dir / name
            destination.mkdir()
            target = destination / archive.name
            shutil.copyfile(archive, target)
            outputs.append(
                {
                    "component_id": name,
                    "mpk_path": str(target.relative_to(output_dir)),
                    "mpk_sha256": sha256_file(target)[0],
                }
            )
        shutil.copyfile(work / "fidelity.json", output_dir / "fidelity.json")
        document = {
            "model_id": MODEL_ID,
            "build_profile_id": profile,
            "compile_invocation_id": profile,
            "outputs": outputs,
            "derived_graphs": [
                {"component_id": name, "sha256": sha256_file(work / "graphs" / f"{name}.onnx")[0]}
                for name in COMPONENTS
            ],
        }
        (output_dir / "compile-outputs.json").write_text(json.dumps(document, indent=2) + "\n")
    return {entry["component_id"]: output_dir / entry["mpk_path"] for entry in outputs}


def main():
    arguments = parse_arguments(__doc__, MODEL_DIR)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    compile_model(
        arguments.model, arguments.build_profile, arguments.dataset, arguments.fidelity_dataset, arguments.output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
