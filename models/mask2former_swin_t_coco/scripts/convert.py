"""Export Mask2Former Swin-T (COCO instance) as its three components side by side in one ONNX, and check it against
transformers.

The checkpoint is facebook/mask2former-swin-tiny-coco-instance, built with transformers' Mask2Former. Each exported
component is mask2former_4d's rearrangement of transformers' modules into the layout the compiler maps to the MLA,
for a fixed 384x384 input (pipeline.py). Each component is checked against its transformers counterpart on seeded
images before export, then each component's ONNX against PyTorch.

The masked attention masks a key by the sign of a mask logit, so a logit within float rounding of 0 can be masked
in one implementation and not another (different operation order, thread counts or CPUs round differently), and
through self-attention move every query's logits. The transformer decoder's checks therefore take mask features
scaled by MASK_SCALE: every mask logit scales with them, which makes such a logit MASK_SCALE times rarer, while the
masked path is still compared.

Loading reports two expected differences from transformers' current Swin, both checked: the checkpoint lacks the
Swin classifier's final LayerNorm, which the backbone never uses (it normalizes each feature map with its own
hidden_states_norms), and holds relative_position_index buffers that transformers now rebuilds itself.
"""

import argparse
import functools
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pipeline import (
    BACKBONE_SHAPES,
    COMPONENTS,
    IMAGE_SIZE,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    SCALE_SIZES,
    route,
    tensor_name,
)

MODEL_DIR = Path(__file__).resolve().parents[1]

FILES = ("config.json", "model.safetensors")
EXPORTER = "5.16.1"
OPSET = 17
SEEDS = range(123, 126)
MASK_SCALE = 100.0
SWIN = "model.pixel_level_module.encoder.swin"
EXPECTED_MISSING = {f"{SWIN}.layernorm.weight", f"{SWIN}.layernorm.bias"}


def validate_source(source):
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "config.json" or files != set(FILES):
        raise ValueError(f"the source must be Mask2Former's {', '.join(FILES)}, opened at config.json")


def build(directory):
    """transformers' Mask2Former from the checkpoint, with only the expected loading differences."""
    import transformers
    from transformers import Mask2FormerForUniversalSegmentation

    if transformers.__version__ != EXPORTER:
        raise RuntimeError(f"Expected transformers {EXPORTER}, found {transformers.__version__}")
    model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        directory, local_files_only=True, output_loading_info=True
    )
    unexpected = {key for key in info["unexpected_keys"] if not key.endswith(".relative_position_index")}
    if set(info["missing_keys"]) != EXPECTED_MISSING or unexpected or info["mismatched_keys"]:
        raise ValueError(f"unexpected checkpoint differences: {info}")
    if model.config.num_labels != 80 or model.config.num_queries != 100:
        raise ValueError("expected the 80-class COCO checkpoint with 100 queries")
    return model.eval()


def modules(model):
    """The side-by-side export module."""
    import torch
    from mask2former_4d import PixelDecoder4D, Swin4D, TransformerDecoder4D

    pixel = model.model.pixel_level_module

    class SideBySide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Swin4D(pixel.encoder, IMAGE_SIZE)
            self.pixel_decoder = PixelDecoder4D(pixel.decoder, [(shape[1], shape[2]) for shape in BACKBONE_SHAPES])
            self.transformer_decoder = TransformerDecoder4D(
                model.model.transformer_module, model.class_predictor, list(SCALE_SIZES)
            )
            layers = len(self.pixel_decoder.layers)
            self.parts = {
                "backbone_0": self.backbone.start,
                "backbone_1": self.backbone.finish,
                "pixel_decoder_0": self.pixel_decoder.start,
                **{
                    f"pixel_decoder_{2 * layer}": functools.partial(self.pixel_decoder.first_half, layer)
                    for layer in range(1, layers)
                },
                **{
                    f"pixel_decoder_{2 * layer + 1}": functools.partial(self.pixel_decoder.second_half, layer)
                    for layer in range(layers)
                },
                f"pixel_decoder_{2 * layers}": self.pixel_decoder.finish,
                "transformer_decoder": self.transformer_decoder,
            }
            self.parts = {component: self.parts[component] for component in COMPONENTS if component in self.parts}
            if list(self.parts) != list(COMPONENTS):
                raise ValueError(f"the pixel decoder has {layers} deformable layers, pipeline.py expects others")

        def run(self, component, inputs):
            outputs = self.parts[component](*inputs)
            return outputs if isinstance(outputs, tuple) else (outputs,)

        def forward(self, *inputs):
            """Every component side by side, each on its own inputs in session_inputs() order."""
            outputs, start = [], 0
            for component in COMPONENTS:
                count = len(INPUT_SHAPES[component])
                outputs.extend(self.run(component, inputs[start : start + count]))
                start += count
            return tuple(outputs)

    return SideBySide().eval()


def seeded_image(seed):
    """A smooth seeded image in the normalized range."""
    import torch

    noise = torch.from_numpy(np.random.RandomState(seed).uniform(-2.1, 2.6, (1, 3, 24, 24)).astype(np.float32))
    return torch.nn.functional.interpolate(noise, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear").numpy()


def check_parts(model, side_by_side):
    """The components give their transformers counterparts' outputs on seeded images: the two backbone parts,
    chained, its feature maps; the pixel decoder parts, chained on transformers' feature maps, its mask and
    multi-scale features; and the transformer decoder, on transformers' pixel decoder outputs with mask features
    scaled by MASK_SCALE, its class and mask logits."""
    import torch

    pixel = model.model.pixel_level_module
    for seed in SEEDS:
        image = torch.from_numpy(seeded_image(seed))
        with torch.inference_mode():
            features = pixel.encoder(image).feature_maps
            *early, stage3_input = side_by_side.backbone.start(image)
            for index, (want, got) in enumerate(zip(features, (*early, *side_by_side.backbone.finish(stage3_input)))):
                compare(want.numpy(), got.numpy(), f"backbone features{index} (seed {seed})")
            decoded = pixel.decoder(list(features))
            expected = (decoded.mask_features, *decoded.multi_scale_features)
            tokens, partial = side_by_side.pixel_decoder.start(*features[1:])
            tokens = side_by_side.pixel_decoder.second_half(0, tokens, partial)
            for layer in range(1, len(side_by_side.pixel_decoder.layers)):
                partial = side_by_side.pixel_decoder.first_half(layer, tokens)
                tokens = side_by_side.pixel_decoder.second_half(layer, tokens, partial)
            got = (side_by_side.pixel_decoder.finish(tokens, features[0]), *side_by_side.pixel_decoder.scales(tokens))
            for name, want, value in zip(("mask_features", "scale0", "scale1", "scale2"), expected, got):
                compare(want.numpy(), value.numpy(), f"pixel decoder {name} (seed {seed})")
            mask_features = decoded.mask_features * MASK_SCALE
            reference = model.model.transformer_module(list(decoded.multi_scale_features), mask_features)
            classes = model.class_predictor(reference.intermediate_hidden_states[-1].transpose(0, 1))
            scales = torch.cat([scale.reshape(1, scale.shape[1], 1, -1) for scale in decoded.multi_scale_features], 3)
            got_classes, got_masks = side_by_side.transformer_decoder(mask_features, scales)
        compare_decoder(classes.numpy(), got_classes[0, :, 0].T[None].numpy(), f"class logits (seed {seed})")
        compare_masks(reference.masks_queries_logits[-1].numpy(), got_masks.numpy(), f"mask logits (seed {seed})")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"Mask2Former export requires opset {OPSET}")
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    expected = (
        {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()},
        {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()},
    )
    actual = ({v.name: shape(v) for v in model.graph.input}, {v.name: shape(v) for v in model.graph.output})
    if actual != expected:
        raise ValueError(f"Unexpected interface: {actual}")
    if any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        for value in [*model.graph.input, *model.graph.output]
    ):
        raise ValueError("Every input and output must be float32")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded Mask2Former export")
    onnx.save(model, path)


def compare(expected, actual, name):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg=name)


def compare_decoder(expected, actual, name, masks=False):
    """The transformer decoder's logits, which the masked attention makes discontinuous (see the module docstring):
    a flipped mask decision moves one query's logits, and self-attention spreads a smaller change to others, by
    more than an element-wise tolerance allows on some CPUs. So the check is on what decoding uses: the typical
    error must stay within 1e-3 relative, and mask logits must agree in sign at 99.99% of pixels, or class logits
    pick the same class for 98 of 100 queries. A real export error moves far more."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite {name}: {actual.shape} vs {expected.shape}")
    typical = float(np.median(np.abs(actual - expected) / (1 + np.abs(expected))))
    if masks:
        agreement = float(np.mean((expected > 0) == (actual > 0)))
        limit = 0.9999
    else:
        # Class logits as [1, queries, classes], the layout transformers returns.
        agreement = float(np.mean(expected.argmax(-1) == actual.argmax(-1)))
        limit = 0.98
    if agreement < limit or typical > 1e-3:
        raise ValueError(f"{name}: agreement {agreement:.4%}, typical relative difference {typical:.2e}")


def compare_masks(expected, actual, name):
    compare_decoder(expected, actual, name, masks=True)


def check_onnx(side_by_side, path):
    """Each component's ONNX outputs match PyTorch's, each component fed the previous one's PyTorch outputs of the
    same seeded image; the transformer decoder takes those mask features scaled by MASK_SCALE (see the module
    docstring)."""
    import onnxruntime as ort
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = [tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]]
    for seed in SEEDS[:2]:
        available = {"image": torch.from_numpy(seeded_image(seed))}
        inputs = []
        with torch.inference_mode():
            for component in COMPONENTS:
                feeds = route(component, available)
                if component == "transformer_decoder":
                    feeds[0] = feeds[0] * MASK_SCALE
                inputs.extend(feeds)
                available.update(zip(OUTPUT_SHAPES[component], side_by_side.run(component, feeds)))
            expected = side_by_side(*inputs)
        feeds = {name: value.numpy() for name, value in zip(session_inputs(), inputs)}
        for name, want, got in zip(outputs, expected, session.run(outputs, feeds)):
            want = want.numpy()
            if name == tensor_name("transformer_decoder", "class_logits"):
                compare_decoder(want[0, :, 0].T[None], got[0, :, 0].T[None], f"{name} (seed {seed})")
            elif name == tensor_name("transformer_decoder", "mask_logits"):
                compare_masks(want, got, f"{name} (seed {seed})")
            else:
                compare(want, got, f"{name} (seed {seed})")


def session_inputs():
    return [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]


def convert(source, output_dir):
    import torch

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    model = build(source.parent)
    side_by_side = modules(model)
    check_parts(model, side_by_side)
    with tempfile.TemporaryDirectory(prefix="mask2former-") as temporary:
        exported = Path(temporary) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                side_by_side,
                tuple(torch.zeros(INPUT_SHAPES[c][n]) for c in COMPONENTS for n in INPUT_SHAPES[c]),
                str(exported),
                input_names=session_inputs(),
                output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
        fold_constants(exported)
        validate_export(exported)
        shutil.move(exported, output)
    check_onnx(side_by_side, output)
    seeds = f"{SEEDS.start}-{SEEDS.stop - 1}"
    print(f"4D components and PyTorch/ONNX checks passed with seeds {seeds}: {output}")
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
