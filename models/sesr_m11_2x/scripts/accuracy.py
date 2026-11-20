"""Upscale DIV2K's 100 validation images 2x on the DevKit and score them against the originals.

Each low-resolution image is reflected outward by MARGIN pixels and cut into overlapping 224x224 tiles. The model
upscales every tile, and only each tile's centre, MARGIN input pixels in from every side, goes into the stitched
image, so no tile edge reaches the result. The upscaled image is clipped to [0, 1] and rounded to 8 bits.

Scores follow the standard super-resolution protocol: the luma channel (ITU-R BT.601, 16-235), with SCALE pixels
shaved from every border. PSNR uses a 255 peak; SSIM uses an 11x11 Gaussian window (sigma 1.5), as Wang et al.
define it. Both are averaged over the images.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

DATASET = "div2k-mini"
LOW_RES = "DIV2K_valid_LR_bicubic/X2"
HIGH_RES = "DIV2K_valid_HR"
IMAGES = 100
TILE = 224
SCALE = 2
# Wider than the network's receptive field: 2 + blocks + 2 input pixels each side, 11 for M7 and 15 for M11.
MARGIN = 16


def image_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Every validation image as (low-resolution, original) paths, in DIV2K's order."""
    low = sorted((root / LOW_RES).glob("*x2.png"))
    pairs = [(path, root / HIGH_RES / path.name.replace("x2.png", ".png")) for path in low]
    if len(pairs) != IMAGES or not all(high.is_file() for _, high in pairs):
        raise ValueError(f"expected {IMAGES} DIV2K validation pairs under {root}")
    return pairs


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def tile_starts(length: int) -> list[int]:
    """Starts, in the reflected image, of the tiles that cover `length` original pixels with their centres."""
    step = TILE - 2 * MARGIN
    starts = list(range(0, max(length - step, 0) + 1, step))
    if starts[-1] + step < length:
        starts.append(length - step)
    return starts


def upscale(image: np.ndarray, infer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Upscale an RGB uint8 image tile by tile; returns the 8-bit RGB result, SCALE times larger."""
    height, width = image.shape[:2]
    if min(height, width) < TILE - 2 * MARGIN:
        raise ValueError(f"image smaller than one tile centre: {image.shape}")
    padded = np.pad(image.astype(np.float32) / 255, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode="reflect")
    result = np.zeros((height * SCALE, width * SCALE, 3), np.float32)
    keep = (TILE - 2 * MARGIN) * SCALE
    for top in tile_starts(height):
        for left in tile_starts(width):
            tile = np.ascontiguousarray(padded[top : top + TILE, left : left + TILE])
            output = tile_output(infer(tile))
            centre = output[MARGIN * SCALE : MARGIN * SCALE + keep, MARGIN * SCALE : MARGIN * SCALE + keep]
            result[top * SCALE : top * SCALE + keep, left * SCALE : left * SCALE + keep] = centre
    return np.round(np.clip(result, 0, 1) * 255).astype(np.uint8)


def tile_output(output: np.ndarray) -> np.ndarray:
    """The upscaled tile as HWC float32, from the model's output in either layout."""
    output = np.squeeze(np.asarray(output, np.float32))
    side = TILE * SCALE
    if output.shape == (3, side, side):
        output = output.transpose(1, 2, 0)
    if output.shape != (side, side, 3):
        raise ValueError(f"unexpected SESR output shape {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("the model returned non-finite values")
    return output


def luma(image: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma (16-235) of an 8-bit RGB image, as float64."""
    rgb = image.astype(np.float64) / 255
    return 16 + rgb @ np.array([65.481, 128.553, 24.966])


def psnr(predicted: np.ndarray, truth: np.ndarray) -> float:
    error = np.mean((predicted - truth) ** 2)
    return float("inf") if error == 0 else float(10 * np.log10(255**2 / error))


def ssim(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Mean SSIM over the window positions that fit inside the image (Wang et al. 2004)."""
    import cv2

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    def filtered(values: np.ndarray) -> np.ndarray:
        return cv2.filter2D(values, -1, window)[5:-5, 5:-5]

    mu_x, mu_y = filtered(predicted), filtered(truth)
    var_x = filtered(predicted**2) - mu_x**2
    var_y = filtered(truth**2) - mu_y**2
    covariance = filtered(predicted * truth) - mu_x * mu_y
    index = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(index.mean())


def score(upscaled: np.ndarray, original: np.ndarray) -> dict[str, float]:
    """PSNR and SSIM on luma, with SCALE pixels shaved from every border."""
    if upscaled.shape != original.shape:
        raise ValueError(f"upscaled {upscaled.shape} does not match the original {original.shape}")
    predicted = luma(upscaled)[SCALE:-SCALE, SCALE:-SCALE]
    truth = luma(original)[SCALE:-SCALE, SCALE:-SCALE]
    return {"psnr_y": psnr(predicted, truth), "ssim_y": ssim(predicted, truth)}


def evaluate(pairs: list[tuple[Path, Path]], infer: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    rows = []
    started = time.monotonic()
    for count, (low, high) in enumerate(pairs, start=1):
        rows.append(score(upscale(read_rgb(low), infer), read_rgb(high)))
        if count == 1 or count % 10 == 0 or count == len(pairs):
            mean = {name: np.mean([row[name] for row in rows]) for name in rows[0]}
            print(f"{count}/{len(pairs)} images, {(time.monotonic() - started) / count:.2f} s/image, "
                  f"PSNR-Y {mean['psnr_y']:.3f} dB, SSIM-Y {mean['ssim_y']:.4f}", file=sys.stderr, flush=True)
    return {name: float(np.mean([row[name] for row in rows])) for name in ("psnr_y", "ssim_y")}


def main() -> None:
    import model_report
    import pyneat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<archive>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first images only, for local runs")
    args = parser.parse_args()
    mpks = [value.split("=", 1) for value in args.mpk]
    if len(mpks) != 1 or len(mpks[0]) != 2 or mpks[0][0] != "model":
        parser.error("--mpk must name exactly model=<archive>")
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        parser.error(f"--dataset must name {DATASET}")
    available = image_pairs(Path(directory))
    pairs = available[: args.limit] if args.limit else available

    options = pyneat.ModelOptions()
    # Each tile is already the model's input: RGB in [0, 1]. Neat only quantizes and tessellates it.
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    model = pyneat.Model(mpks[0][1], options)
    runner = None

    def infer(tile: np.ndarray) -> np.ndarray:
        nonlocal runner
        tensor = pyneat.Tensor.from_numpy(tile, copy=True, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        return runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=True)

    try:
        metrics = evaluate(pairs, infer)
    finally:
        if runner is not None:
            runner.close()
    model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": metrics,
                "split": "valid",
                "evaluated_samples": len(pairs),
                "available_samples": len(available),
                "details": {
                    "tiling": f"{TILE}x{TILE} tiles overlapping by {2 * MARGIN} pixels, centres stitched",
                    "scoring": f"BT.601 luma, {SCALE}-pixel border shaved, 8-bit output",
                },
            }
        }
    ).save(args.output)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
