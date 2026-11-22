"""Measure EdgeTAM's DAVIS 2017 J&F with its four MPKs on a Modalix board.

Each val sequence is tracked as the DAVIS semi-supervised protocol asks: every object starts from its mask on the
first frame and is tracked through the later frames on its own, sharing one image encoder run per frame. The
objects' masks merge into one label map, the object with the highest logit winning an overlap, and each object is
scored on every frame but the first and the last: J is its region overlap, F its boundary F-measure, and J&F their
mean, averaged over the objects of all sequences as the official DAVIS evaluation does.

Frames are stretched to 1024x1024 with Pillow's bicubic resize and normalized with the ImageNet mean and standard
deviation, as EdgeTAM's video predictor loads them. The first frame's memory comes from the object's ground-truth
mask, as upstream's. Its object pointer comes from a mask decoder run prompted with the mask's box, because the
mask decoder MPK takes points and boxes but no mask, where upstream prompts it with the mask itself. That one pointer
of the 16 is the only difference from the published protocol.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from PIL import Image

COMPONENTS = ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"]
SPLIT = "val"
# surgery.py's interfaces, channels last as the MPKs take and return them.
IMAGE_SIZE = 1024
MASKS, MASK_SIZE = 4, 256
HIDDEN, MEMORY_DIM = 256, 64
MEMORY_TOKENS, MEMORY_SLOTS, OBJECT_POINTERS = 512, 7, 16
PROMPT_POINTS = 3
POINT_TYPES = ["padding", "background", "foreground", "box_top_left", "box_bottom_right"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# calibration.preprocess's per-channel arithmetic, tabulated for every uint8 value.
NORMALIZE = ((np.arange(256, dtype=np.float32)[:, None] / 255.0 - MEAN) / STD).reshape(1, 256, 3)
# The tracker's mask logits for a frame where the object score says the object is absent.
NO_OBJECT_LOGIT = -1024.0
# DAVIS counts a boundary pixel as matched within this fraction of the frame's diagonal.
BOUNDARY_TOLERANCE = 0.008
TIMEOUT_MS = 120000
# Frames decode on a worker thread while the MPKs run; Pillow releases the GIL while it decodes and resizes.
PREFETCH = 4


def preprocess(content: bytes) -> np.ndarray:
    """An encoded frame as the image encoder's normalized [1, 1024, 1024, 3] input, with calibration.preprocess's
    values."""
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return cv2.LUT(np.asarray(resized), NORMALIZE)[None]


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(1) as pool:
        pending = collections.deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def cpu_tensor(array: np.ndarray) -> pyneat.Tensor:
    """A float32 array the CPU made, in the memory a component's input stage reads. Its shape is already the
    MPK's, so it carries no layout: Neat would read a 3D array as channels first and reject it."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(array, np.float32),
        copy=True,
        memory=pyneat.TensorMemory.EV74,
        layout=pyneat.TensorLayout.Unknown,
    )


def tensor_options() -> pyneat.ModelOptions:
    """Every input is a tensor the script prepared, which Neat only converts."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


class Component:
    """One MPK, with a runner built from the first inputs it runs."""

    def __init__(self, mpk: Path) -> None:
        self.model = pyneat.Model(str(mpk), tensor_options())
        self.runner = None

    def __call__(self, *inputs: np.ndarray | pyneat.Tensor) -> list[pyneat.Tensor]:
        tensors = [item if isinstance(item, pyneat.Tensor) else cpu_tensor(item) for item in inputs]
        if self.runner is None:
            self.runner = self.model.build(tensors)
        return self.runner.run(tensors, timeout_ms=TIMEOUT_MS)


def prompt_arrays(points: list[tuple[float, float]], types: list[str]) -> list[np.ndarray]:
    """The mask decoder's point_coords and point_types for (x, y) in 1024x1024 model pixels; unused slots are
    padding. Coordinates are shifted to pixel centres and normalized to [-1, 1], as calibration.Prompt does."""
    coords = np.zeros((1, 1, PROMPT_POINTS, 2), np.float32)
    one_hot = np.zeros((1, 1, PROMPT_POINTS, len(POINT_TYPES)), np.float32)
    one_hot[..., POINT_TYPES.index("padding")] = 1
    for slot, (point, kind) in enumerate(zip(points, types, strict=True)):
        coords[0, 0, slot] = (np.asarray(point, np.float32) + 0.5) / IMAGE_SIZE * 2 - 1
        one_hot[0, 0, slot] = np.eye(len(POINT_TYPES), dtype=np.float32)[POINT_TYPES.index(kind)]
    return [coords, one_hot]


def box_prompt(mask: np.ndarray) -> list[np.ndarray]:
    """The mask decoder's prompt for the box around a [1024, 1024] object mask."""
    rows, columns = np.nonzero(mask)
    corners = [(columns.min(), rows.min()), (columns.max() + 1, rows.max() + 1)]
    return prompt_arrays(corners, ["box_top_left", "box_bottom_right"])


@dataclass
class MemoryBank:
    """The frames one object's tracker remembers: the first frame, then the latest frames, newest first."""

    first_memory: np.ndarray
    first_pointer: np.ndarray
    memories: collections.deque = field(default_factory=lambda: collections.deque(maxlen=MEMORY_SLOTS - 1))
    pointers: collections.deque = field(default_factory=lambda: collections.deque(maxlen=OBJECT_POINTERS - 1))

    def add(self, memory: np.ndarray, pointer: np.ndarray) -> None:
        self.memories.appendleft(memory)
        self.pointers.appendleft(pointer)

    def arrays(self) -> list[np.ndarray]:
        """The memory attention's memories, memory_mask, object_pointers and object_pointer_mask. An unused slot
        stays zero with a mask of 0."""
        slots = [self.first_memory, *self.memories]
        memories = np.zeros((1, MEMORY_SLOTS * MEMORY_TOKENS, MEMORY_DIM), np.float32)
        memories[0, : len(slots) * MEMORY_TOKENS] = np.concatenate(slots)
        used_memories = np.zeros((1, MEMORY_SLOTS), np.float32)
        used_memories[0, : len(slots)] = 1
        kept = [self.first_pointer, *self.pointers]
        pointers = np.zeros((1, OBJECT_POINTERS, HIDDEN), np.float32)
        pointers[0, : len(kept)] = np.stack(kept)
        used_pointers = np.zeros((1, OBJECT_POINTERS), np.float32)
        used_pointers[0, : len(kept)] = 1
        return [memories, used_memories, pointers, used_pointers]


def choose_mask(
    masks: np.ndarray, iou_scores: np.ndarray, object_score: float, pointers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A tracked frame's [256, 256] mask logits and object pointer from the decoder's four masks: the best of
    masks 1-3 by predicted IoU, as calibration.select_mask keeps. A frame whose object score is not positive has
    no object."""
    best = 1 + int(np.argmax(iou_scores[1:]))
    mask = masks[..., best]
    if object_score <= 0:
        mask = np.full_like(mask, NO_OBJECT_LOGIT)
    return mask.copy(), pointers[best].copy()


class Tracker:
    """The four MPKs tracking the objects of one sequence, one memory bank per object."""

    def __init__(self, components: dict[str, Component]) -> None:
        self.components = components
        self.banks: dict[int, MemoryBank] = {}

    def encode(self, frame: np.ndarray) -> list[pyneat.Tensor]:
        """The frame's image embeddings and high-resolution features, copied to the memory the other components'
        input stages read."""
        return [tensor.cvu() for tensor in self.components["image_encoder"](frame)]

    def decode(
        self, embeddings: pyneat.Tensor, high_res: list[pyneat.Tensor], prompt: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """The mask decoder's masks, IoU scores, object score and object pointers; views valid until it runs
        again."""
        decoded = self.components["mask_decoder"](embeddings, *high_res, *prompt)
        masks, iou_scores, object_score, pointers = (tensor.to_numpy(copy=False) for tensor in decoded)
        return (
            masks.reshape(MASK_SIZE, MASK_SIZE, MASKS),
            iou_scores.reshape(MASKS),
            float(object_score.reshape(())),
            pointers.reshape(MASKS, HIDDEN),
        )

    def remember(self, embeddings: pyneat.Tensor, mask: np.ndarray) -> np.ndarray:
        """The memory of a frame and one object's [1024, 1024] mask on it."""
        (memory,) = self.components["memory_encoder"](embeddings, mask.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
        return memory.to_numpy(copy=True).reshape(MEMORY_TOKENS, MEMORY_DIM)

    def start(self, encoded: list[pyneat.Tensor], object_masks: dict[int, np.ndarray]) -> None:
        """Start tracking each object from its [1024, 1024] ground-truth mask on the encoded first frame."""
        embeddings, *high_res = encoded
        for object_id, mask in object_masks.items():
            # A prompt of two points keeps the single mask's pointer, as calibration.select_mask does.
            pointer = self.decode(embeddings, high_res, box_prompt(mask))[3][0].copy()
            self.banks[object_id] = MemoryBank(self.remember(embeddings, mask.astype(np.float32)), pointer)

    def track(self, encoded: list[pyneat.Tensor]) -> dict[int, np.ndarray]:
        """Each object's [256, 256] mask logits on the encoded frame; the frame joins every object's bank."""
        embeddings, *high_res = encoded
        logits = {}
        for object_id, bank in self.banks.items():
            (conditioned,) = self.components["memory_attention"](embeddings, *bank.arrays())
            logits[object_id], pointer = choose_mask(*self.decode(conditioned.cvu(), high_res, prompt_arrays([], [])))
            # A tracked frame's memory takes the upsampled mask's probabilities. OpenCV's exp overflows to
            # infinity for the no-object logit, which gives the probability 0 it should.
            upsampled = cv2.resize(logits[object_id], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            bank.add(self.remember(embeddings, 1 / (1 + cv2.exp(-upsampled))), pointer)
        return logits


def object_masks(object_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Each annotated object's mask at 1024x1024, by its ID: resized bilinearly and thresholded at 0.5, as
    EdgeTAM's video predictor takes a mask prompt."""
    masks = {}
    for object_id in np.unique(object_ids[object_ids > 0]).tolist():
        mask = (object_ids == object_id).astype(np.float32)
        masks[object_id] = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) >= 0.5
    return masks


def label_map(logits: dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    """The frame's predicted object IDs at its own size, 0 for background: each object's mask logits resized
    bilinearly to the frame, and each pixel given to the object with the highest logit when that is positive, as
    EdgeTAM's VOS inference keeps masks from overlapping."""
    object_ids = np.asarray(sorted(logits))
    resized = np.stack([
        cv2.resize(logits[object_id], (width, height), interpolation=cv2.INTER_LINEAR) for object_id in object_ids
    ])  # fmt: skip
    return np.where(resized.max(0) > 0, object_ids[resized.argmax(0)], 0).astype(np.uint8)


def region_overlap(predicted: np.ndarray, truth: np.ndarray) -> float:
    """DAVIS J: the masks' intersection over union, 1 when both are empty."""
    union = (predicted | truth).sum()
    return float((predicted & truth).sum() / union) if union else 1.0


def boundary(mask: np.ndarray) -> np.ndarray:
    """The mask's boundary pixels, as the DAVIS evaluation's seg2bmap marks them: a pixel that differs from its
    right, lower or lower-right neighbour."""
    east, south, south_east = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    east[:, :-1], south[:-1, :], south_east[:-1, :-1] = mask[:, 1:], mask[1:, :], mask[1:, 1:]
    edges = (mask ^ east) | (mask ^ south) | (mask ^ south_east)
    edges[-1, :] = mask[-1, :] ^ east[-1, :]
    edges[:, -1] = mask[:, -1] ^ south[:, -1]
    edges[-1, -1] = False
    return edges


def boundary_f_measure(predicted: np.ndarray, truth: np.ndarray) -> float:
    """DAVIS F: the F-measure of the two masks' boundaries, a pixel matching within BOUNDARY_TOLERANCE of the
    diagonal."""
    radius = int(np.ceil(BOUNDARY_TOLERANCE * np.linalg.norm(predicted.shape)))
    offsets = np.arange(-radius, radius + 1)
    disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius**2).astype(np.uint8)
    predicted_edges, true_edges = boundary(predicted), boundary(truth)
    predicted_count, true_count = predicted_edges.sum(), true_edges.sum()
    if not predicted_count or not true_count:
        precision, recall = float(not predicted_count), float(not true_count)
    else:
        near_truth = cv2.dilate(true_edges.astype(np.uint8), disk).astype(bool)
        near_predicted = cv2.dilate(predicted_edges.astype(np.uint8), disk).astype(bool)
        precision = (predicted_edges & near_truth).sum() / predicted_count
        recall = (true_edges & near_predicted).sum() / true_count
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def read_object_ids(path: Path) -> np.ndarray:
    """A frame's palette annotation: 0 is background, value k is object k."""
    return np.asarray(Image.open(path))


def score_sequence(components: dict[str, Component], dataset: Path, sequence: str) -> dict[int, tuple[float, float]]:
    """Each object's mean J and mean F over the sequence's frames, the first and the last left out."""
    frames = sorted(path.stem for path in (dataset / SPLIT / sequence).glob("*.jpg"))
    images = prefetched(lambda frame: preprocess((dataset / SPLIT / sequence / f"{frame}.jpg").read_bytes()), frames)
    first = read_object_ids(dataset / "annotations" / sequence / f"{frames[0]}.png")
    tracker = Tracker(components)
    tracker.start(tracker.encode(next(images)), object_masks(first))
    scores = {object_id: [] for object_id in tracker.banks}
    for frame, image in zip(frames[1:], images):
        logits = tracker.track(tracker.encode(image))
        # The last frame is tracked like the others and left unscored.
        if frame == frames[-1]:
            continue
        truth = read_object_ids(dataset / "annotations" / sequence / f"{frame}.png")
        labels = label_map(logits, *truth.shape)
        for object_id, values in scores.items():
            predicted, expected = labels == object_id, truth == object_id
            values.append((region_overlap(predicted, expected), boundary_f_measure(predicted, expected)))
    return {object_id: tuple(np.mean(values, axis=0).tolist()) for object_id, values in scores.items()}


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != set(COMPONENTS):
        raise SystemExit(f"--mpk must name exactly <component>=<path> for {', '.join(COMPONENTS)}")
    return {name: Path(mpks[name]) for name in COMPONENTS}


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="davis-2017-val=<extracted davis directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--resources", type=Path, help="package resource directory (license only)")
    parser.add_argument("--limit", type=int, help="measure a seeded sample of sequences")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)
    available = (dataset / f"{SPLIT}.txt").read_text(encoding="utf-8").split()
    sequences = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    components = {name: Component(mpk) for name, mpk in parse_mpks(args.mpk).items()}
    object_scores: list[tuple[float, float]] = []
    started = time.monotonic()
    for count, sequence in enumerate(sequences, start=1):
        scores = score_sequence(components, dataset, sequence)
        object_scores += scores.values()
        overlap, f_measure = np.mean(list(scores.values()), axis=0)
        log(f"{count}/{len(sequences)} {sequence}: {len(scores)} objects, J {overlap:.3f}, F {f_measure:.3f}")
    j_mean, f_mean = np.mean(object_scores, axis=0).tolist()

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {"jf_mean": (j_mean + f_mean) / 2, "j_mean": j_mean, "f_mean": f_mean},
            "split": SPLIT, "evaluated_samples": len(sequences), "available_samples": len(available),
            "details": {"objects": len(object_scores), "seconds": round(time.monotonic() - started, 1)},
        },
    })  # fmt: skip
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
