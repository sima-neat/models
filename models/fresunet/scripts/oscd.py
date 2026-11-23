"""FresUNet's OSCD input and change-map scoring, shared by compile.py, accuracy.py and performance.py.

This reproduces SiMa's earlier model zoo pipeline (simazoo 2.0.0's OSCD dataset and OSCDPerformanceAnalyzer), which
produced the model's published 54.26% FP32 dice:

1. Each date's 13 Sentinel-2 bands are read from imgs_1/ and imgs_2/ (not the _rect folders). The 20 m bands are
   upsampled x2 and the 60 m bands x6 with scipy.ndimage.zoom (cubic spline), then cropped or edge-padded to the 10 m
   grid of B04.
2. The bands are stacked in the order B04 B03 B02 B08 B05 B06 B07 B8A B11 B12 B01 B09 B10 and normalized with the
   image's own mean and standard deviation over every band and pixel.
3. Each image is resized to 544x544 as skimage.transform.resize does by default (Gaussian anti-aliasing when
   downsampling, then linear interpolation), and the input is date 1's 13 channels followed by date 2's.
4. The change map is resized to 544x544 with nearest-neighbour interpolation.
5. The prediction is the argmax of the model's two output channels. Dice is the change class's F1, with true and false
   positives and negatives summed over every pixel of every test city.

The DevKit's PyNeat environment has NumPy, SciPy and OpenCV but not scikit-image, so resize() reproduces skimage's
default resize with SciPy, and OpenCV reads the TIFF and PNG files.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

SIZE = 544
CHANNELS = 26
INPUT_NAME = "new_input"
# Band name and zoom factor to the 10 m grid, in the model's channel order.
BANDS = (
    ("B04", 1), ("B03", 1), ("B02", 1), ("B08", 1),
    ("B05", 2), ("B06", 2), ("B07", 2), ("B8A", 2), ("B11", 2), ("B12", 2),
    ("B01", 6), ("B09", 6), ("B10", 6),
)


def dataset_root(path: Path) -> Path:
    """The folder holding train.txt and test.txt: the extracted archive or its only subfolder."""
    if (path / "test.txt").is_file():
        return path
    roots = [child for child in sorted(path.iterdir()) if (child / "test.txt").is_file()]
    if len(roots) != 1:
        raise ValueError(f"{path} holds no single OSCD directory with test.txt")
    return roots[0]


def cities(root: Path, split: str) -> list[str]:
    """The cities of the train or test split, in the file's order: one comma-separated line."""
    return [name.strip() for name in (root / f"{split}.txt").read_text(encoding="utf-8").strip().split(",")]


def _fit(band: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    band = band[: shape[0], : shape[1]]
    return np.pad(band, ((0, max(0, shape[0] - band.shape[0])), (0, max(0, shape[1] - band.shape[1]))), "edge")


def _read(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def read_bands(folder: Path) -> np.ndarray:
    """One date's 13 bands on the 10 m grid, [height, width, 13], normalized over the whole image."""
    # Every file is <prefix><band>.tif; the prefix is the first file's name without "B01.tif".
    prefix = sorted(os.listdir(folder))[0][:-7]
    reference = _read(folder / f"{prefix}B04.tif")
    stack = []
    for band, factor in BANDS:
        image = _read(folder / f"{prefix}{band}.tif")
        stack.append(image if factor == 1 else _fit(ndimage.zoom(image, factor), reference.shape))
    image = np.stack(stack, axis=2).astype(np.float64)
    return (image - image.mean()) / image.std()


def resize(image: np.ndarray) -> np.ndarray:
    """skimage.transform.resize(image, (544, 544)) for a float image: anti-aliasing, linear, mirror edges, clip."""
    factors = np.array([image.shape[0] / SIZE, image.shape[1] / SIZE] + [1.0] * (image.ndim - 2))
    if (factors > 1).any():
        filtered = ndimage.gaussian_filter(image, np.maximum(0, (factors - 1) / 2), mode="mirror")
    else:
        filtered = image
    out = ndimage.zoom(filtered, 1 / factors, order=1, mode="mirror", grid_mode=True)
    return np.clip(out, image.min(), image.max())


def city_input(root: Path, city: str) -> np.ndarray:
    """The model input for one city, [544, 544, 26] float32, channels last as the tensor route takes it."""
    dates = [resize(read_bands(root / city / folder)).astype(np.float32) for folder in ("imgs_1", "imgs_2")]
    return np.ascontiguousarray(np.concatenate(dates, axis=2))


def change_map(root: Path, city: str) -> np.ndarray:
    """The city's ground truth at 544x544, True where the area changed."""
    label = _read(root / city / "cm" / "cm.png")
    if label.ndim != 2:
        raise ValueError(f"{city}: expected a single-channel change map, found {label.shape}")
    changed = (label != 0).astype(np.uint8)
    factors = (SIZE / changed.shape[0], SIZE / changed.shape[1])
    return ndimage.zoom(changed, factors, order=0, mode="mirror", grid_mode=True) > 0


def prediction(output: np.ndarray) -> np.ndarray:
    """True where the model predicts change, from its two-class output in either layout."""
    output = np.squeeze(np.asarray(output))
    if output.shape == (SIZE, SIZE, 2):
        output = output.transpose(2, 0, 1)
    if output.shape != (2, SIZE, SIZE):
        raise ValueError(f"unexpected FresUNet output shape {output.shape}")
    return output.argmax(0) > 0


class ChangeScore:
    """True and false positives and negatives of the change class, summed over every pixel scored."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        self.tp += int(np.count_nonzero(predicted & truth))
        self.fp += int(np.count_nonzero(predicted & ~truth))
        self.fn += int(np.count_nonzero(~predicted & truth))
        self.tn += int(np.count_nonzero(~predicted & ~truth))

    def dice(self) -> float:
        """2TP / (2TP + FP + FN), which equals 2PR / (P + R)."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    def metrics(self) -> dict[str, float]:
        """The change class's F1, precision and recall, as the model report names them."""
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {"change_f1": self.dice(), "change_precision": precision, "change_recall": recall}
