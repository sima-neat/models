"""Export the V-JEPA 2.1 ViT-B video encoder for a 16-frame 256x256 clip and check it against upstream PyTorch.

The weights are the hub's vjepa2_1_vit_base_384 encoder (Meta's vjepa2_1_vitb_dist_vitG_384.pt, key ema_encoder),
as a safetensors file. upstream's model code is vendored unchanged in vjepa2/, from facebookresearch/vjepa2 at the
commit model.yaml names. Upstream does not export to ONNX, so the exported module is encoder_4d.Encoder: the same
encoder in the compiler's 4D layout, with the clip's RoPE angles as constants. Its three parts (pipeline.py) are
exported side by side in one ONNX, since the whole encoder needs more memory to compile than a CI runner has.
Chained, they are checked against upstream's VisionTransformer on seeded clips before export, and each part's ONNX
against PyTorch after.
"""

import argparse
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pipeline import COMPONENTS, FRAME_SIZE, FRAMES, INPUT_SHAPES, OUTPUT_SHAPES, TUBELET, stack_clip, tensor_name

MODEL_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = Path(__file__).resolve().parent / "vjepa2"

CHECKPOINT = "model.safetensors"
OPSET = 17
SEEDS = range(123, 126)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def build(weights):
    """The encoder as upstream's src/hub/backbones.py builds vjepa2_1_vit_base_384, with the weights loaded strictly."""
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM_DIR))
    from app.vjepa_2_1.models import vision_transformer

    encoder = vision_transformer.vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=TUBELET,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder.load_state_dict(load_file(weights), strict=True)
    return encoder.eval()


def seeded_clip(seed):
    """A normalized [1, 3, FRAMES, FRAME_SIZE, FRAME_SIZE] clip; normalized pixels lie in about [-2.1, 2.7]."""
    shape = (1, 3, FRAMES, FRAME_SIZE, FRAME_SIZE)
    return np.random.RandomState(seed).uniform(-2.1, 2.7, shape).astype(np.float32)


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    # Features reach about 30, so the tolerance is relative to their scale.
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-3, err_msg=name)


def parts(four):
    """The side-by-side export module: each part on its own input."""
    import torch

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = four

        def forward(self, *inputs):
            return tuple(self.encoder.part(index, x, len(COMPONENTS)) for index, x in enumerate(inputs))

    return SideBySide().eval()


def chain(four, stacked):
    """The parts run one after another, as an application runs the components."""
    x = stacked
    for index in range(len(COMPONENTS)):
        x = four.part(index, x, len(COMPONENTS))
    return x


def check_encoder(encoder, four):
    """The 4D encoder's parts, chained, give upstream's features on seeded clips, and pipeline.stack_clip lays a
    clip out as encoder_4d.stack_tubelets does."""
    import torch
    from encoder_4d import stack_tubelets

    for seed in SEEDS[:2]:
        clip = torch.from_numpy(seeded_clip(seed))
        stacked = stack_tubelets(clip, TUBELET)
        np.testing.assert_array_equal(stack_clip(clip[0].permute(1, 2, 3, 0).numpy()), stacked.numpy())
        with torch.inference_mode():
            expected = encoder(clip)[0].numpy()
            actual = chain(four, stacked)[0, :, 0, :].T.numpy()
        compare(expected, actual, f"4D encoder (seed {seed})")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"V-JEPA export requires opset {OPSET}")
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    actual = ({v.name: shape(v) for v in model.graph.input}, {v.name: shape(v) for v in model.graph.output})
    expected = (
        {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()},
        {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()},
    )
    if actual != expected:
        raise ValueError(f"Unexpected interface: {actual}")
    if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for v in [*model.graph.input, *model.graph.output]):
        raise ValueError("Every input and output must be float32")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    # Folding only: onnxsim's optimizer would fuse token Linears into 2D Gemms, whose bias the compiler mis-types.
    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded V-JEPA export")
    onnx.save(model, path)


def check_onnx(four, path):
    """Each part's ONNX gives the 4D encoder's outputs on seeded clips; a later part takes the PyTorch tokens of the
    same clip."""
    import onnxruntime as ort
    import torch
    from encoder_4d import stack_tubelets

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for seed in SEEDS:
        x = stack_tubelets(torch.from_numpy(seeded_clip(seed)), TUBELET)
        feeds, expected = {}, []
        with torch.inference_mode():
            for index, name in enumerate(inputs):
                feeds[name] = x.numpy()
                x = four.part(index, x, len(COMPONENTS))
                expected.append(x.numpy())
        for name, want, got in zip(outputs, expected, session.run(outputs, feeds)):
            compare(want, got, f"{name} (seed {seed})")


def convert(source, output_dir):
    import torch
    from encoder_4d import Encoder

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    encoder = build(source)
    four = Encoder(encoder, FRAMES, FRAME_SIZE, FRAME_SIZE).eval()
    check_encoder(encoder, four)
    with tempfile.TemporaryDirectory(prefix="vjepa-", dir=output_dir) as temporary:
        exported = Path(temporary) / "model.onnx"
        torch.onnx.export(
            parts(four),
            tuple(torch.zeros(INPUT_SHAPES[c][n]) for c in COMPONENTS for n in INPUT_SHAPES[c]),
            str(exported),
            input_names=[tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]],
            output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        exported.replace(output)
    check_onnx(four, output)
    print(f"4D encoder and PyTorch/ONNX checks passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
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
