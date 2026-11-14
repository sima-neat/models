"""Checks of the conversion's source pin, its tensor renaming, and the image encoder and weights it publishes."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.conversion import conversion_outputs
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models/edgetam"
_, WEIGHTS = conversion_outputs(MODEL_DIR)


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType) -> None:
    converter.validate_source(load_metadata(MODEL_DIR)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="edgetam.pt"):
        converter.validate_source({"uri": "hf://models/facebook/sam2-hiera-tiny@main/sam2_hiera_tiny.pt"})


@pytest.mark.parametrize(
    ("original", "renamed"),
    [
        ("image_encoder.trunk.body.stem.conv1.c.weight", "vision_encoder.backbone.timm_model.stem.conv1.c.weight"),
        (
            "sam_mask_decoder.transformer.layers.0.mlp.layers.1.weight",
            "mask_decoder.transformer.layers.0.mlp.proj_out.weight",
        ),
        (
            "sam_mask_decoder.output_hypernetworks_mlps.3.layers.1.weight",
            "mask_decoder.output_hypernetworks_mlps.3.layers.0.weight",
        ),
        ("memory_attention.layers.0.linear1.weight", "memory_attention.layers.0.mlp.up_proj.weight"),
        ("memory_encoder.fuser.layers.0.dwconv.weight", "memory_encoder.memory_fuser.layers.0.depthwise_conv.weight"),
        ("memory_encoder.mask_downsampler.encoder.12.weight", "memory_encoder.mask_downsampler.final_conv.weight"),
        ("spatial_perceiver.layers.0.attn.to_q.weight", "spatial_perceiver.layers.0.cross_attention.q_proj.weight"),
        ("obj_ptr_proj.layers.2.bias", "object_pointer_proj.proj_out.bias"),
        ("maskmem_tpos_enc", "memory_temporal_positional_encoding"),
    ],
)
def test_checkpoint_tensors_take_transformers_names(converter: ModuleType, original: str, renamed: str) -> None:
    assert converter.renamed_key(original) == renamed


def _shapes(values) -> dict[str, list[int]]:
    return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}


def test_converted_image_encoder_has_the_declared_interface(converted: Path, converter: ModuleType) -> None:
    model = onnx.load(converted, load_external_data=False)
    assert _shapes(model.graph.input) == {"image": [1, 3, 1024, 1024]}
    assert _shapes(model.graph.output) == converter.OUTPUTS


def test_converted_weights_hold_the_renamed_float32_checkpoint(converted: Path) -> None:
    torch = pytest.importorskip("torch")
    weights = torch.load(converted.with_name(WEIGHTS), weights_only=True)
    assert {"no_memory_embedding", "no_object_pointer", "memory_temporal_positional_encoding"} <= set(weights)
    prefixes = {name.split(".")[0] for name in weights}
    assert {"vision_encoder", "memory_attention", "memory_encoder", "spatial_perceiver", "mask_decoder"} <= prefixes
    # BatchNorm's num_batches_tracked counters are int64; every weight is float32.
    assert all(
        value.dtype == torch.float32 for name, value in weights.items() if not name.endswith("num_batches_tracked")
    )


def synthetic_clip(calibration: ModuleType, frames: int) -> tuple[list[np.ndarray], tuple[float, float, float, float]]:
    """Normalized frames of an ellipse moving across a striped background, and the first frame's object box."""
    from PIL import Image, ImageDraw

    clip = []
    for frame in range(frames):
        image = Image.new("RGB", (640, 480), (90, 120, 160))
        draw = ImageDraw.Draw(image)
        for stripe in range(0, 640, 40):
            draw.rectangle((stripe, 0, stripe + 20, 480), fill=(110, 140, 120))
        left, top = 200 + 12 * frame, 150 + 6 * frame
        draw.ellipse((left, top, left + 180, top + 120), fill=(220, 60, 40))
        clip.append(calibration.preprocess(image))
    scale_x, scale_y = 1024 / 640, 1024 / 480
    return clip, (200 * scale_x, 150 * scale_y, 380 * scale_x, 270 * scale_y)


def test_converted_graphs_track_like_transformers_edgetam(
    converted: Path, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """surgery.py's four graphs, run by calibration.py's tracker, give transformers' EdgeTAM video model's masks on
    every frame of a click-prompted and a box-prompted clip.

    It takes the conversion's output, so it runs when the conversion reruns; an edit to surgery.py or calibration.py
    alone is checked against transformers by surgery.py's own export check during compilation.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    import onnxruntime as ort
    from transformers import EdgeTamVideoModel, EdgeTamVisionConfig, EdgeTamVisionModel
    from transformers.models.edgetam_video.modeling_edgetam_video import EdgeTamVideoInferenceSession

    converter, surgery, calibration = (load_script(name) for name in ("convert", "surgery", "calibration"))
    graphs = surgery.export_components(converted, converted.with_name(WEIGHTS), tmp_path / "graphs")
    sessions = {
        name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for name, path in graphs.items()
    }
    # surgery's configuration leaves the vision part at its default; the reference takes EdgeTAM's vision model.
    reference = EdgeTamVideoModel(surgery.video_config())
    reference.vision_encoder = EdgeTamVisionModel(EdgeTamVisionConfig(backbone_config=converter.BACKBONE))
    reference.load_state_dict(torch.load(converted.with_name(WEIGHTS), weights_only=True), strict=True)
    reference.eval()
    frames, (left, top, right, bottom) = synthetic_clip(calibration, 6)
    prompts = {
        "click": calibration.Prompt((((left + right) / 2, (top + bottom) / 2),), ("foreground",)),
        "box": calibration.Prompt(((left, top), (right, bottom)), ("box_top_left", "box_bottom_right")),
    }
    for kind, prompt in prompts.items():
        tracker = calibration.Tracker(sessions)
        ours = [tracker.step(frame, prompt if index == 0 else None) for index, frame in enumerate(frames)]
        # transformers' point labels count from background as 0, one below surgery's POINT_TYPES.
        labels = [surgery.POINT_TYPES.index(name) - 1 for name in prompt.types]
        session = EdgeTamVideoInferenceSession(dtype=torch.float32)
        session.add_point_inputs(
            session.obj_id_to_idx(1),
            0,
            {
                "point_coords": torch.tensor([[prompt.points]], dtype=torch.float32),
                "point_labels": torch.tensor([[labels]], dtype=torch.int32),
            },
        )
        session.obj_with_new_inputs = [1]
        with torch.no_grad():
            theirs = [
                reference(inference_session=session, frame=torch.from_numpy(frame[0])).pred_masks[0, 0].numpy()
                for frame in frames
            ]
        for index, (mask, expected) in enumerate(zip(ours, theirs)):
            assert (expected > 0).any(), f"{kind} frame {index}: transformers found no object"
            agreement = float(((mask > 0) == (expected > 0)).mean())
            assert agreement > 0.999, f"{kind} frame {index}: masks agree on {agreement:.4f} of pixels"
