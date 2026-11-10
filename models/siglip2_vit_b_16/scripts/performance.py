"""Benchmark SigLIP 2's independent image and text encoders with Neat's benchmark API.

The image benchmark includes Neat preprocessing of one dataset image in a 720p
RGB frame. The text benchmark starts from one prompt's prepared 64x768 token
embeddings. Tokenization, CPU lookup and cosine scoring are outside timing.
Each encoder reports its own latency, throughput, telemetry and MLA statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat

TEXT_CHECK = Path(__file__).with_name("text_encoder_check.npz")
TEXT_CHECK_SHA256 = "273bfe20216fb3c1ef681f453335b37b066f687744dec2893154384e3c1305e5"
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
NUM_SAMPLES = 1000
MIN_DURATION_MS = 60000
VALIDATION_SPLIT = "val2012"
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def image_options() -> pyneat.ModelOptions:
    """The image encoder's: Neat stretches, normalizes, quantizes and tessellates the frame inside the pipeline."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    # Size Neat's input buffers for the benchmark frame, not the 1920x1080 default.
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    # Without explicit statistics Neat only scales pixels to [0, 1]; the model expects SigLIP 2's mean and std.
    options.preprocess.normalize.mean = MEAN
    options.preprocess.normalize.stddev = STD
    options.preprocess.normalize.has_explicit_stats = True
    return options


def tensor_options() -> pyneat.ModelOptions:
    """The text encoder's: its input is the prompt's token embeddings."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def sample_image(dataset: Path) -> Path:
    """One val2012 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / VALIDATION_SPLIT).glob("*.JPEG"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {VALIDATION_SPLIT}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the image encoder reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


@dataclass(frozen=True)
class Prompt:
    """A tokenized prompt and the embedding rows its tokens use."""

    token_ids: list[int]
    rows: dict[int, np.ndarray]


def load_prompt(path: Path = TEXT_CHECK) -> Prompt:
    """The first prompt of the text probe, from a file with the checksum the script pins."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != TEXT_CHECK_SHA256:
        raise SystemExit(f"{path} does not match TEXT_CHECK_SHA256")
    with np.load(io.BytesIO(content), allow_pickle=False) as probe:
        token_ids = probe["token_ids"][0]
        rows = dict(zip(probe["row_ids"].tolist(), probe["rows"]))
    return Prompt(token_ids.tolist(), rows)


def token_embeddings(prompt: Prompt) -> np.ndarray:
    """The CPU step before the text encoder: each of the prompt's token IDs replaced by its embedding row."""
    return np.stack([prompt.rows[token] for token in prompt.token_ids])[None]


def look_up(prompt: Prompt) -> pyneat.Tensor:
    """Prepare the prompt's token embeddings before timing, in the memory the text encoder reads.

    Neat sizes the text input as height 1, width 64 and depth 768, so the tensor declares HWC axes; untagged, a 3-D
    tensor counts as CHW, and its width, 768, exceeds the input's 64.
    """
    return pyneat.Tensor.from_numpy(
        token_embeddings(prompt), copy=True, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC
    )


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != {"image_encoder", "text_encoder"}:
        raise SystemExit("--mpk must name exactly image_encoder=<path> and text_encoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def benchmark_encoders(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path) -> None:
    path = sample_image(dataset)
    image, prompt = frame_tensor(path), load_prompt()
    reports = {}
    for component in ("image_encoder", "text_encoder"):
        is_image = component == "image_encoder"
        model = pyneat.Model(str(mpks[component]), image_options() if is_image else tensor_options())
        options = pyneat.BenchmarkOptions()
        options.inputs = [image if is_image else look_up(prompt)]
        options.num_samples = NUM_SAMPLES
        options.min_duration_ms = MIN_DURATION_MS
        options.include_mla_layer_stats = True
        options.include_mla_stage_stats = True
        workload = (
            f"{dataset_id} {path.name} in a 1280x720 RGB frame; Neat resize and SigLIP 2 normalization, "
            "image encoder with fixed 1000-class ImageNet head and tensor conversions"
            if is_image
            else "One prompt padded to 64 tokens; prepared FP32 token embeddings, "
            "text encoder with a pooled 768-value output and tensor conversions; "
            "tokenization and CPU embedding lookup excluded"
        )
        benchmark = model.benchmark(options)
        reports[component] = model_report.from_performance(benchmark, {component: model}, workload=workload)
        print(f"{component}: {benchmark.fps:.1f} inferences/s, mean latency {benchmark.latency_ms:.2f} ms", flush=True)
        del model
    model_report.from_component_performance(reports).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    benchmark_encoders(parse_mpks(args.mpk), dataset_id, Path(dataset_dir), args.output)


if __name__ == "__main__":
    main()
