"""Export the pinned ResNeXt-101 64x4d checkpoint (timm, torchvision's ImageNet weights) to a static ONNX graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURE = "resnext101_64x4d"
CLASS_COUNT = 1000
INTERFACE = {"pixel_values": [1, 3, 224, 224], "logits": [1, CLASS_COUNT]}
OUTPUTS = ("logits",)
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json"])
SEEDS = range(123, 127)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = (config.get("architecture"), config.get("num_classes"))
    if declared != (ARCHITECTURE, CLASS_COUNT):
        raise ValueError(f"config.json declares {declared}, expected {(ARCHITECTURE, CLASS_COUNT)}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    require_onnx_opset_version(model)
    inputs, outputs = [value.name for value in model.graph.input], [value.name for value in model.graph.output]
    if inputs != ["pixel_values"] or outputs != list(OUTPUTS):
        raise ValueError(f"Expected the input pixel_values and the output logits, found {inputs} and {outputs}")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    if expected.shape != actual.shape or list(expected.shape) != INTERFACE["logits"]:
        raise ValueError("Unexpected shape for logits")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite logits")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg="logits")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"pixel_values": INTERFACE["pixel_values"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded export")
    onnx.save(model, path)


def convert(source, output_dir):
    import onnxruntime as ort
    import timm
    import torch
    from safetensors.torch import load_file

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(source.parent)
    model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=CLASS_COUNT)
    # strict=True raises on any missing or unexpected weight.
    model.load_state_dict(load_file(source), strict=True)
    model = model.cpu().float().eval()

    def synthetic(seed):
        values = np.random.RandomState(seed).standard_normal(INTERFACE["pixel_values"]).astype(np.float32)
        return torch.from_numpy(values)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (synthetic(SEEDS[0]),),
            str(output),
            input_names=["pixel_values"],
            output_names=list(OUTPUTS),
            opset_version=onnx_opset_version(),
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = synthetic(seed)
        with torch.inference_mode():
            expected = model(sample).numpy()
        (actual,) = session.run(list(OUTPUTS), {"pixel_values": sample.numpy()})
        compare_outputs(expected, actual)
    print(f"PyTorch/ONNX check passed on {len(SEEDS)} synthetic inputs: {output}")
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
