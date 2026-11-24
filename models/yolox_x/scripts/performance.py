"""Benchmark one compiled MPK with Neat's YOLOX box decoder on a COCO image and save its report."""

import argparse
import random
from pathlib import Path

import model_report
import pyneat
from model_inputs.frame import FILL, FRAME_HEIGHT, FRAME_WIDTH, letterbox_to_720p, read_rgb

IMAGES = "val2017"


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="compiled model archive")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    image = sample_image(Path(dataset_dir))

    options = pyneat.ModelOptions()
    # Neat letterboxes, normalizes, quantizes and tessellates the frame inside the measured pipeline.
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.resize.width = options.preprocess.resize.height = 640
    options.preprocess.resize.pad_value = FILL
    # GenericPreproc normalizes (pixel / 255 - mean) / std; preserve YOLOX's raw pixel domain.
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = [0.0, 0.0, 0.0]
    options.preprocess.normalize.stddev = [1.0 / 255] * 3
    options.preprocess.normalize.has_explicit_stats = True
    options.num_classes = 80
    options.decode_type_option = pyneat.BoxDecodeTypeOption.Split3Interleaved
    options.decode_type = pyneat.BoxDecodeType.YoloX
    # An application's decoder keeps a few confident detections, not every candidate.
    options.top_k = 10
    options.score_threshold = 0.25
    options.nms_iou_threshold = 0.65
    model = pyneat.Model(args.mpk, options)
    # Request 1000 samples and a 60 s minimum measurement duration.
    benchmark_options = pyneat.BenchmarkOptions()
    benchmark_options.num_samples = 1000
    benchmark_options.min_duration_ms = 60000
    benchmark_options.include_mla_layer_stats = True
    benchmark_options.include_mla_stage_stats = True
    frame = letterbox_to_720p(read_rgb(image))
    benchmark_options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    postprocess = model.info().selection.selected_post_kind
    report = model_report.from_performance(
        model.benchmark(benchmark_options),
        {"model": model},
        workload=(
            f"{dataset_id} {image.name} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing, "
            f"YoloX box decoder, top 10 above 0.25, NMS IoU 0.65, {postprocess} post-processing"
        ),
    )
    report.save(args.output)
    return 0


if __name__ == "__main__":
    main()
