"""Convert MediaPipe's BlazePose GHUM Heavy TFLite to ONNX with tf2onnx and check ONNX against TFLite."""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

TFLITE = "pose_landmark_heavy.tflite"
OPSET = 16
INTERFACE = {
    "input_1": [1, 256, 256, 3],
    "Identity": [1, 195],
    "Identity_1": [1, 1],
    "Identity_2": [1, 256, 256, 1],
    "Identity_3": [1, 64, 64, 39],
    "Identity_4": [1, 117],
}
OUTPUTS = list(INTERFACE)[1:]
# Largest ONNX-against-TFLite difference per output, about five times what seeds 123-132 measured:
# 0.008 px, 6e-8, 0.014, 0.002 and 1.6e-5 m.
TOLERANCE = {"Identity": 0.04, "Identity_1": 1e-5, "Identity_2": 0.07, "Identity_3": 0.01, "Identity_4": 1e-4}
SEEDS = range(123, 133)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", TFLITE):
        raise ValueError(f"Expected the TFLite file {TFLITE}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"BlazePose export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected input input_1 and outputs {', '.join(OUTPUTS)}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def name_constants(path):
    """Name the initializers in order of first use and clear the doc strings, which hold the source's path, so a
    repeated conversion writes the same bytes; tf2onnx numbers its folded constants differently on each run."""
    model = onnx.load(path)
    model.doc_string = model.graph.doc_string = ""
    initializers = {value.name: value for value in model.graph.initializer}
    names = {}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in initializers:
                node.input[index] = names.setdefault(name, f"constant_{len(names)}")
    if len(names) != len(initializers):
        raise ValueError(f"{len(initializers) - len(names)} initializers are not node inputs")
    for name, value in initializers.items():
        value.name = names[name]
    model.graph.initializer.sort(key=lambda value: int(value.name.removeprefix("constant_")))
    onnx.save(model, path)


def compare_outputs(expected, actual):
    """Return each output's largest ONNX-against-TFLite difference, raising beyond its tolerance."""
    errors = {}
    for name in OUTPUTS:
        if name not in expected or name not in actual:
            raise ValueError(f"Missing output {name}")
        if actual[name].shape != expected[name].shape or not np.isfinite(actual[name]).all():
            raise ValueError(f"Unexpected or non-finite {name}: {actual[name].shape} vs {expected[name].shape}")
        errors[name] = float(np.max(np.abs(actual[name] - expected[name])))
        if errors[name] > TOLERANCE[name]:
            raise ValueError(f"ONNX output {name} differs from TFLite by {errors[name]}")
    return errors


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    converter = [sys.executable, "-m", "tf2onnx.convert", "--opset", str(OPSET)]
    subprocess.run([*converter, "--tflite", str(source), "--output", str(output)], check=True)
    name_constants(output)
    validate_export(output)

    import onnxruntime as ort
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(source))
    interpreter.allocate_tensors()
    (image,) = interpreter.get_input_details()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = dict.fromkeys(OUTPUTS, 0.0)
    for seed in SEEDS:
        sample = np.random.RandomState(seed).uniform(0, 1, INTERFACE["input_1"]).astype(np.float32)
        interpreter.set_tensor(image["index"], sample)
        interpreter.invoke()
        expected = {item["name"]: interpreter.get_tensor(item["index"]) for item in interpreter.get_output_details()}
        actual = dict(zip(OUTPUTS, session.run(OUTPUTS, {"input_1": sample}), strict=True))
        for name, error in compare_outputs(expected, actual).items():
            errors[name] = max(errors[name], error)
    print(f"TFLite/ONNX check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}, largest errors {errors}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
