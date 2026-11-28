"""PointNet++ SSG's CPU steps, component interface and S3DIS blocks, shared by every script.

The network runs as eight MPKs, the four set-abstraction (SA) levels and the four feature-propagation (FP)
levels. Everything between them depends on point coordinates, which the MLA cannot gather by: `plan()` computes
farthest point sampling, ball queries and 3-NN interpolation weights for a block once, `group()` builds each SA
input, and `interpolate()` each FP input. NumPy does the work; 3-NN uses OpenCV's vectorized nearest-neighbour
search, imported where it runs. With channels_last, `group()` and `interpolate()` build the [H, W, C] tensor an MPK
takes, without NCHW copies.
"""

from __future__ import annotations

import collections
import multiprocessing
from collections.abc import Callable, Iterable, Iterator

import numpy as np

POINTS = 4096
FEATURES = 9  # xyz relative to the block centre, rgb in [0, 1], xyz divided by the room's extent
CLASSES = 13
CLASS_NAMES = (
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "chair", "table", "bookcase", "sofa", "board", "clutter",
)
# The S3DIS blocks that accuracy.py scores and performance.py streams. They live here, a compile input, so that a
# change to them reaches both stages' fingerprints through the builds.
DATASET = "s3dis-val"
BLOCK_SIZE = 1.0
STRIDE = 0.5
PADDING = 0.001
# Sampling a column's points is random; a fixed seed makes every run measure the same blocks.
SEED = 0
NEIGHBORS = 32
SA_POINTS = (1024, 256, 64, 16)
SA_RADII = (0.1, 0.2, 0.4, 0.8)
SA_CHANNELS = ((FEATURES, (32, 32, 64)), (64, (64, 64, 128)), (128, (128, 128, 256)), (256, (256, 256, 512)))
FP_CHANNELS = ((768, (256, 256)), (384, (256, 256)), (320, (256, 128)), (128, (128, 128, 128)))
# Run order. fp4 upsamples the coarsest level; fp1 also runs the classifier head.
COMPONENTS = ("sa1", "sa2", "sa3", "sa4", "fp4", "fp3", "fp2", "fp1")
LEVEL_POINTS = (POINTS, *SA_POINTS)
INPUT_SHAPES = {
    **{f"sa{level + 1}": (1, SA_CHANNELS[level][0] + 3, SA_POINTS[level], NEIGHBORS) for level in range(4)},
    **{f"fp{4 - i}": (1, FP_CHANNELS[i][0], LEVEL_POINTS[3 - i], 1) for i in range(4)},
}
OUTPUT_SHAPES = {
    **{f"sa{level + 1}": (1, SA_CHANNELS[level][1][-1], SA_POINTS[level], 1) for level in range(4)},
    **{f"fp{4 - i}": (1, FP_CHANNELS[i][1][-1], LEVEL_POINTS[3 - i], 1) for i in range(3)},
    "fp1": (1, CLASSES, POINTS, 1),
}


def input_name(component: str) -> str:
    return f"{component}_input"


def output_name(component: str) -> str:
    return f"{component}_output"


def farthest_point_sample(pos: np.ndarray, count: int) -> np.ndarray:
    """Indices of `count` points, each the farthest from those already chosen, starting from point 0.

    The loop is sequential, so each step reuses preallocated per-axis buffers: x, y and z squared and summed in that
    order, as the [N, 3] form sums them.
    """
    x, y, z = (np.ascontiguousarray(pos[:, axis], np.float32) for axis in range(3))
    picked = np.empty(count, np.int64)
    distance = np.full(len(pos), np.inf, np.float32)
    step, term = np.empty_like(distance), np.empty_like(distance)
    current = 0
    for i in range(count):
        picked[i] = current
        np.subtract(x, x[current], out=step)
        np.multiply(step, step, out=step)
        for axis in (y, z):
            np.subtract(axis, axis[current], out=term)
            np.multiply(term, term, out=term)
            step += term
        np.minimum(distance, step, out=distance)
        current = int(distance.argmax())
    return picked


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[len(a), len(b)] squared distances, summed one axis at a time without an [a, b, 3] temporary."""
    total = np.subtract.outer(a[:, 0], b[:, 0])
    total *= total
    for axis in (1, 2):
        difference = np.subtract.outer(a[:, axis], b[:, axis])
        difference *= difference
        total += difference
    return total


def ball_query(pos: np.ndarray, centres: np.ndarray, radius: float, chunk: int = 64) -> np.ndarray:
    """[S, 32] indices of the lowest-indexed points within `radius` of each centre, padded with the first.

    PointNet++'s reference query keeps the lowest indices; padding with a repeat leaves the max pool unchanged.
    Centres are taken `chunk` at a time in x order, and only points within the chunk's x range plus the radius are
    measured; the candidates stay in index order, so the result is the full query's.
    """
    pos, centres = np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(centres, np.float32)
    index = np.empty((len(centres), NEIGHBORS), np.int64)
    order = np.argsort(centres[:, 0], kind="stable")
    for start in range(0, len(order), chunk):
        chosen = order[start:start + chunk]
        low, high = centres[chosen, 0].min() - radius, centres[chosen, 0].max() + radius
        candidates = np.nonzero((pos[:, 0] >= low) & (pos[:, 0] <= high))[0]
        rows, columns = np.nonzero(squared_distances(centres[chosen], pos[candidates]) <= radius * radius)
        columns = candidates[columns]
        counts = np.bincount(rows, minlength=len(chosen))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slots = np.arange(len(rows)) - starts[rows]
        kept = slots < NEIGHBORS
        found = np.repeat(columns[starts][:, None], NEIGHBORS, axis=1)
        found[rows[kept], slots[kept]] = columns[kept]
        index[chosen] = found
    return index


def three_nearest(coarse: np.ndarray, fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 3] nearest coarse points of each fine point and their weights 1 / (d^2 + 1e-8), normalized."""
    import cv2  # OpenCV's brute-force K-nearest search, imported here so pointcloud.py loads without it

    _, nearest = cv2.batchDistance(np.ascontiguousarray(fine, np.float32), np.ascontiguousarray(coarse, np.float32),
                                   cv2.CV_32F, normType=cv2.NORM_L2SQR, K=3)
    nearest = nearest.astype(np.int64)
    offset = fine[:, None, :] - coarse[nearest]
    distance = np.einsum("nkc,nkc->nk", offset, offset)
    order = np.argsort(distance, axis=1, kind="stable")
    nearest, distance = np.take_along_axis(nearest, order, 1), np.take_along_axis(distance, order, 1)
    weight = 1.0 / (distance + 1e-8)
    return nearest, (weight / weight.sum(1, keepdims=True)).astype(np.float32)


def plan(pos: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Every coordinate-only index of one block: the points of each level, ball queries and interpolation."""
    pos = np.ascontiguousarray(pos, np.float32)
    levels, centres, groups = [pos], [], []
    for count, radius in zip(SA_POINTS, SA_RADII):
        picked = farthest_point_sample(levels[-1], count)
        groups.append(ball_query(levels[-1], levels[-1][picked], radius))
        centres.append(picked)
        levels.append(levels[-1][picked])
    interpolation = [three_nearest(levels[level + 1], levels[level]) for level in range(4)]
    return {"levels": levels, "centres": centres, "groups": groups, "interpolation": interpolation}


def group(features: np.ndarray, level: int, indices: dict[str, list[np.ndarray]], channels_last: bool = False
          ) -> np.ndarray:
    """SA input [1, C + 3, S, 32], or [S, 32, C + 3]: each neighbour's features, then its xyz minus the centre's."""
    pos, neighbours = indices["levels"][level], indices["groups"][level]
    grouped = np.empty((*neighbours.shape, features.shape[1] + 3), np.float32)
    grouped[..., :-3] = features[neighbours]
    np.subtract(pos[neighbours], pos[indices["centres"][level]][:, None, :], out=grouped[..., -3:])
    return grouped if channels_last else np.ascontiguousarray(grouped.transpose(2, 0, 1)[None])


def interpolate(coarse: np.ndarray, skip: np.ndarray | None, level: int, indices: dict,
                channels_last: bool = False) -> np.ndarray:
    """FP input [1, C + C_skip, N, 1], or [N, 1, C + C_skip]: the inverse-distance mix of the 3 nearest coarse
    features, then the skip."""
    nearest, weight = indices["interpolation"][level]
    width = coarse.shape[1] + (0 if skip is None else skip.shape[1])
    stacked = np.empty((len(nearest), 1, width), np.float32)
    np.einsum("nkc,nk->nc", coarse[nearest], weight, out=stacked[:, 0, :coarse.shape[1]])
    if skip is not None:
        stacked[:, 0, coarse.shape[1]:] = skip
    return stacked if channels_last else np.ascontiguousarray(stacked.transpose(2, 0, 1)[None])


def channels_last(value: np.ndarray) -> np.ndarray:
    """A component input [1, C, H, W] as the [H, W, C] tensor its MPK takes."""
    return np.ascontiguousarray(value[0].transpose(1, 2, 0))


def points_of(output: np.ndarray, channels: int, count: int) -> np.ndarray:
    """A component output as [points, channels], from NCHW [1, C, N, 1] or a channels-last layout such as Neat's
    [N, 1, C]. Every component has more than one channel, so the last axis tells the layouts apart."""
    output = np.asarray(output, np.float32)
    if output.size != channels * count:
        raise ValueError(f"expected {channels * count} values, got shape {output.shape}")
    if output.shape[-1] == channels:
        return output.reshape(count, channels)
    return output.reshape(channels, count).T


def synthetic_block(seed: int = 0) -> np.ndarray:
    """[4096, 9] features of a seeded block: a floor, two walls and a box of clutter, 1 m wide and 3 m high."""
    rng = np.random.RandomState(seed)
    parts = [
        np.c_[rng.uniform(-0.5, 0.5, (1500, 2)), rng.normal(0.0, 0.005, 1500)],
        np.c_[np.full(1000, -0.5), rng.uniform(-0.5, 0.5, 1000), rng.uniform(0, 3, 1000)],
        np.c_[rng.uniform(-0.5, 0.5, 896), np.full(896, 0.5), rng.uniform(0, 3, 896)],
        rng.uniform((-0.2, -0.2, 0.0), (0.2, 0.2, 0.8), (700, 3)),
    ]
    xyz = np.concatenate(parts)[rng.permutation(POINTS)].astype(np.float32)
    rgb = rng.uniform(0, 1, (POINTS, 3)).astype(np.float32)
    normalized = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0))
    return np.concatenate([xyz, rgb, normalized], axis=1).astype(np.float32)


def segment(x: np.ndarray, run: Callable[[str, np.ndarray], np.ndarray], indices: dict | None = None,
            channels_last: bool = False) -> np.ndarray:
    """[4096, 13] logits of one block, x being its [4096, 9] features; run(component, input) runs one MPK, its input
    NCHW, or [H, W, C] with channels_last. `indices` is plan(x[:, :3]) when the caller computed it already, for
    example in another process."""
    if x.shape != (POINTS, FEATURES):
        raise ValueError(f"a block is [{POINTS}, {FEATURES}], got {x.shape}")
    if indices is None:
        indices = plan(x[:, :3])
    features = [np.asarray(x, np.float32)]
    for level in range(4):
        component = f"sa{level + 1}"
        output = run(component, group(features[-1], level, indices, channels_last))
        features.append(points_of(output, OUTPUT_SHAPES[component][1], SA_POINTS[level]))
    upsampled = features[4]
    for level in range(3, -1, -1):
        component = f"fp{level + 1}"
        output = run(component, interpolate(upsampled, features[level] if level else None, level, indices,
                                            channels_last))
        upsampled = points_of(output, OUTPUT_SHAPES[component][1], LEVEL_POINTS[level])
    return upsampled


def blocks(
    room: dict[str, np.ndarray], rng: np.random.RandomState, protocol: str = "tiles"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Each [4096, 9] block of a room and the room indices of its points, under the given protocol."""
    xyz = room["xyz"].astype(np.float64)
    rgb = room["rgb"].astype(np.float64) / 255.0
    low, high = xyz.min(0), xyz.max(0)
    stride = BLOCK_SIZE if protocol == "tiles" else STRIDE
    columns_x = int(np.ceil((high[0] - low[0] - BLOCK_SIZE) / stride) + 1)
    columns_y = int(np.ceil((high[1] - low[1] - BLOCK_SIZE) / stride) + 1)
    for row in range(columns_y):
        for column in range(columns_x):
            end_x = min(low[0] + column * stride + BLOCK_SIZE, high[0])
            end_y = min(low[1] + row * stride + BLOCK_SIZE, high[1])
            start_x, start_y = end_x - BLOCK_SIZE, end_y - BLOCK_SIZE
            inside = np.where(
                (xyz[:, 0] >= start_x - PADDING) & (xyz[:, 0] <= end_x + PADDING)
                & (xyz[:, 1] >= start_y - PADDING) & (xyz[:, 1] <= end_y + PADDING)
            )[0]
            if inside.size == 0:
                continue
            if protocol == "tiles":
                # One block: 4096 distinct points, or every point and random repeats when the column has fewer.
                if inside.size >= POINTS:
                    order = rng.choice(inside, POINTS, replace=False)
                else:
                    order = np.concatenate([inside, rng.choice(inside, POINTS - inside.size, replace=True)])
                    rng.shuffle(order)
            else:
                total = int(np.ceil(inside.size / POINTS)) * POINTS
                extra = rng.choice(inside, total - inside.size, replace=total - inside.size > inside.size)
                order = np.concatenate([inside, extra])
                rng.shuffle(order)
            centred = xyz[order] - [start_x + BLOCK_SIZE / 2, start_y + BLOCK_SIZE / 2, 0.0]
            features = np.concatenate([centred, rgb[order], xyz[order] / high], axis=1).astype(np.float32)
            for start in range(0, len(order), POINTS):
                yield features[start : start + POINTS], order[start : start + POINTS]


def planned(items: Iterable[tuple], workers: int) -> Iterator[tuple]:
    """Each (..., features, points) item with plan(features' xyz) appended, in order. With workers, up to twice as
    many plans are computed ahead in spawned processes, which import only NumPy and pointcloud."""
    if not workers:
        for item in items:
            yield (*item, plan(item[-2][:, :3]))
        return
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        ahead = collections.deque()
        for item in items:
            ahead.append((item, pool.apply_async(plan, (np.ascontiguousarray(item[-2][:, :3]),))))
            if len(ahead) > 2 * workers:
                item, result = ahead.popleft()
                yield (*item, result.get())
        while ahead:
            item, result = ahead.popleft()
            yield (*item, result.get())
