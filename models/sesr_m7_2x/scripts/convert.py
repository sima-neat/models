"""Export AIMET's FP32 SESR checkpoint to ONNX at one 224x224 tile and check ONNX against PyTorch.

The export keeps the trained, uncollapsed network exactly as AIMET Model Zoo scores it: each collapsible linear block
is still a wide convolution followed by a 1x1 convolution. surgery.py collapses them for the compiler.

AIMET's checkpoint is a training snapshot: besides `state_dict` it pickles the Adam optimizer's state, so loading it
with weights_only=True needs torch.optim.Adam, collections.defaultdict and dict allowed. No other code runs.
"""

import argparse
import collections
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata
from sesr import INPUT, INPUT_SHAPE, NUM_BLOCKS, OUTPUT, OUTPUT_SHAPE, SCALE, build_network

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = f"sesr_m{NUM_BLOCKS}_{SCALE}x_checkpoint_float32.pth.tar"
INTERFACE = {INPUT: INPUT_SHAPE, OUTPUT: OUTPUT_SHAPE}
SEEDS = range(123, 128)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def load_state(path: Path) -> dict:
    import torch

    with torch.serialization.safe_globals([torch.optim.Adam, collections.defaultdict, dict]):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(snapshot, dict) or "state_dict" not in snapshot:
        raise ValueError("Expected an AIMET training snapshot with a state_dict")
    return snapshot["state_dict"]


def build_trained_network(path: Path):
    network = build_network().eval()
    network.load_state_dict(load_state(path), strict=True)
    return network


def sample_inputs() -> list[np.ndarray]:
    """Four uniform-noise tiles and one smooth RGB gradient, in [0, 1]."""
    samples = [np.random.RandomState(seed).uniform(0, 1, INPUT_SHAPE).astype(np.float32) for seed in SEEDS[:-1]]
    y, x = np.meshgrid(*(np.linspace(0, 1, INPUT_SHAPE[2], dtype=np.float32),) * 2, indexing="ij")
    samples.append(np.stack([x, y, (x + y) / 2])[None])
    return samples


def validate_export(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    if [version.version for version in model.opset_import if not version.domain] != [17]:
        raise ValueError("expected ONNX opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"expected {INPUT} and {OUTPUT}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"unexpected input/output: {value.name}")


def convert(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.set_num_threads(4)
    network = build_trained_network(source)
    samples = sample_inputs()
    with torch.inference_mode():
        references = [network(torch.from_numpy(sample)).numpy() for sample in samples]
        torch.onnx.export(
            network,
            (torch.from_numpy(samples[0]),),
            str(output),
            opset_version=17,
            input_names=[INPUT],
            output_names=[OUTPUT],
            dynamo=False,
        )
    model, valid = simplify(onnx.load(output), perform_optimization=False)
    if not valid:
        raise RuntimeError("ONNX constant folding failed")
    onnx.save(model, output)
    validate_export(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    for sample, reference in zip(samples, references, strict=True):
        (actual,) = session.run(None, {INPUT: sample})
        if not np.isfinite(actual).all():
            raise ValueError("non-finite ONNX output")
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
    print(f"Exported SESR-M{NUM_BLOCKS} {SCALE}x and matched PyTorch on {len(samples)} tiles: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
