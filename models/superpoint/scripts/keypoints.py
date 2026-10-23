"""SuperPoint's image preprocessing and the CPU decode of its raw heads, as transformers' SuperPoint decodes them.

NumPy and OpenCV only, so the conversion check, the compile gate and the DevKit accuracy stage share one reference.
"""

from __future__ import annotations

import numpy as np

HEIGHT, WIDTH = 480, 640
CELL = 8
KEYPOINT_CHANNELS = CELL * CELL + 1
DESCRIPTOR_SIZE = 256
# transformers' SuperPointConfig defaults for magic-leap-community/superpoint.
KEYPOINT_THRESHOLD = 0.005
NMS_RADIUS = 4
BORDER = 4


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as COCO does."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Return the model input for an RGB image: 640x480 grayscale scaled to [0, 1], HW float32.

    Resizes first, then mixes RGB with the NTSC weights transformers' SuperPointImageProcessor uses.
    """
    import cv2

    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255)
    return resized @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def homography(seed: int, width: int, height: int) -> np.ndarray:
    """A seeded, drone-like viewpoint change about the image centre: rotation, scale, shift and a little perspective."""
    rng = np.random.RandomState(seed)
    angle = np.radians(rng.uniform(-15, 15))
    scale = rng.uniform(0.85, 1.1)
    centre = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
    rotate = np.array([[scale * np.cos(angle), -scale * np.sin(angle), 0],
                       [scale * np.sin(angle), scale * np.cos(angle), 0], [0, 0, 1]])
    perspective = np.array([[1, 0, 0], [0, 1, 0], [rng.uniform(-1e-4, 1e-4), rng.uniform(-1e-4, 1e-4), 1]])
    back = np.array([[1, 0, width / 2 + rng.uniform(-30, 30)], [0, 1, height / 2 + rng.uniform(-30, 30)], [0, 0, 1]])
    return back @ perspective @ rotate @ centre


def _max_pool(values: np.ndarray, radius: int) -> np.ndarray:
    """max_pool2d with a (2r+1) window, stride 1 and -inf padding, separably."""
    height, width = values.shape
    padded = np.pad(values, radius, constant_values=-np.inf)
    rows = padded[:height].copy()
    for offset in range(1, 2 * radius + 1):
        np.maximum(rows, padded[offset:offset + height], out=rows)
    pooled = rows[:, :width].copy()
    for offset in range(1, 2 * radius + 1):
        np.maximum(pooled, rows[:, offset:offset + width], out=pooled)
    return pooled


def simple_nms(scores: np.ndarray, radius: int = NMS_RADIUS) -> np.ndarray:
    """transformers' simple_nms on one [H, W] score map."""
    zeros = np.zeros_like(scores)
    max_mask = scores == _max_pool(scores, radius)
    for _ in range(2):
        suppressed = _max_pool(max_mask.astype(scores.dtype), radius) > 0
        suppressed_scores = np.where(suppressed, zeros, scores)
        max_mask |= (suppressed_scores == _max_pool(suppressed_scores, radius)) & ~suppressed
    return np.where(max_mask, scores, zeros)


def score_map(logits: np.ndarray) -> np.ndarray:
    """Turn [65, H/8, W/8] keypoint logits into the [H, W] map of keypoint scores after NMS."""
    logits = np.asarray(logits, dtype=np.float32)
    exp = np.exp(logits - logits.max(0, keepdims=True))
    probabilities = (exp / exp.sum(0, keepdims=True))[:-1]
    cells_y, cells_x = probabilities.shape[1:]
    scores = probabilities.transpose(1, 2, 0).reshape(cells_y, cells_x, CELL, CELL)
    return simple_nms(scores.transpose(0, 2, 1, 3).reshape(cells_y * CELL, cells_x * CELL))


def sample_descriptors(keypoints: np.ndarray, raw_descriptors: np.ndarray) -> np.ndarray:
    """L2-normalize [256, H/8, W/8] coarse descriptors, sample them bilinearly at the keypoints, normalize again."""
    dense = np.asarray(raw_descriptors, dtype=np.float32)
    dense = dense / np.maximum(np.linalg.norm(dense, axis=0, keepdims=True), 1e-12)
    channels, height, width = dense.shape
    grid = (keypoints - CELL / 2 + 0.5) / np.array([width * CELL - CELL / 2 - 0.5, height * CELL - CELL / 2 - 0.5])
    # grid_sample with align_corners=True maps 0..1 onto pixel centres 0..size-1.
    x, y = grid[:, 0] * (width - 1), grid[:, 1] * (height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    sampled = np.zeros((len(keypoints), channels), np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            weight = (1 - np.abs(x - xi)) * (1 - np.abs(y - yi))
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            sampled[valid] += weight[valid, None] * dense[:, yi[valid], xi[valid]].T
    return sampled / np.maximum(np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12)


def decode(logits: np.ndarray, raw_descriptors: np.ndarray, threshold: float = KEYPOINT_THRESHOLD,
           limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's channels-first heads: (x, y) pixel keypoints, scores and unit descriptors.

    Without a limit the keypoints keep transformers' row-major order; with one, the best `limit` come best first.
    """
    scores = score_map(logits)
    ys, xs = np.nonzero(scores > threshold)
    inside = (ys >= BORDER) & (ys < scores.shape[0] - BORDER) & (xs >= BORDER) & (xs < scores.shape[1] - BORDER)
    ys, xs = ys[inside], xs[inside]
    values = scores[ys, xs]
    if limit is not None:
        order = np.argsort(-values, kind="stable")[:limit]
        ys, xs, values = ys[order], xs[order], values[order]
    keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
    return keypoints, values.astype(np.float32), sample_descriptors(keypoints, raw_descriptors)


def synthetic_image(seed: int) -> np.ndarray:
    """A seeded [H, W] grayscale scene of overlapping shapes, rich in corners, for checks that need no dataset."""
    import cv2

    rng = np.random.RandomState(seed)
    image = np.full((HEIGHT, WIDTH), rng.uniform(0.2, 0.8), np.float32)
    for _ in range(40):
        x, y = rng.randint(0, WIDTH), rng.randint(0, HEIGHT)
        w, h = rng.randint(20, 160), rng.randint(20, 120)
        corners = cv2.boxPoints(((x, y), (w, h), rng.uniform(0, 90))).astype(np.int32)
        cv2.fillConvexPoly(image, corners, float(rng.uniform(0, 1)))
    return cv2.GaussianBlur(image, (3, 3), 0.8)
