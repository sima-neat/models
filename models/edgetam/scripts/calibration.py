"""Make EdgeTAM's calibration inputs by tracking DAVIS 2017 objects through cal clips with the FP32 graphs.

Each clip is one cal sequence. Its first frame is prompted on the frame's largest annotated object, with a click, a
click and a background click, or the object's box; every later frame is tracked. Each graph calibrates on the inputs
this FP32 tracking gives it, so the memory attention sees real memory banks, full after 17 frames, and the mask decoder
sees both prompted and tracked frames. The last clip only checks the quantized components, so each component's
CalibrationSet holds samples spread over the earlier clips and frames spread over the last clip.

Frames are stretched to 1024x1024 with Pillow's bicubic resize and normalized with the ImageNet mean and standard
deviation, as EdgeTAM's video predictor loads them. The tracker below follows transformers' EdgeTAM video model:
see surgery.py for each graph's inputs and outputs.
"""

from __future__ import annotations

import io
import pickle
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from surgery import (
    COMPONENTS,
    IMAGE_SIZE,
    INTERFACES,
    MEMORY_SLOTS,
    MEMORY_TOKENS,
    OBJECT_POINTERS,
    POINT_TYPES,
    PROMPT_POINTS,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
PROMPTS = ["click", "click_and_background", "box"]
# The tracker's mask logits for a frame where the object score says the object is absent.
NO_OBJECT_LOGIT = -1024.0
# transformers' dynamic multimask: the single mask stays when its stability at +-0.05 is at least 0.98.
STABILITY_DELTA = 0.05
STABILITY_THRESHOLD = 0.98
SEED = 0


class DavisCal:
    """The cal sequences' frames and object masks, from the Dataset Registry archive or its extracted directory.

    Files go by their path inside the archive's davis folder, such as cal/bear/00000.jpg. Use it as a context
    manager, which closes the archive.
    """

    def __init__(self, dataset: Path):
        if dataset.is_dir():
            self.archive = None
            self.files = {_davis_path(str(path)): path for path in dataset.rglob("*") if path.is_file()}
        elif tarfile.is_tarfile(dataset):
            self.archive = tarfile.open(dataset)
            self.files = {_davis_path(member.name): member for member in self.archive.getmembers() if member.isfile()}
        else:
            raise ValueError(f"unsupported calibration dataset: {dataset}")

    def __enter__(self) -> DavisCal:
        return self

    def __exit__(self, *exception) -> None:
        if self.archive is not None:
            self.archive.close()

    def read(self, name: str) -> bytes:
        if self.archive is None:
            return self.files[name].read_bytes()
        return self.archive.extractfile(self.files[name]).read()

    def sequences(self) -> list[str]:
        return self.read("cal.txt").decode().split()

    def frames(self, sequence: str) -> list[str]:
        """The sequence's frame names without their extension, in order."""
        prefix = f"cal/{sequence}/"
        return sorted(Path(name).stem for name in self.files if name.startswith(prefix))

    def image(self, sequence: str, frame: str) -> Image.Image:
        return Image.open(io.BytesIO(self.read(f"cal/{sequence}/{frame}.jpg"))).convert("RGB")

    def object_ids(self, sequence: str, frame: str) -> np.ndarray:
        """The frame's palette mask: 0 is background, value k is object k."""
        return np.asarray(Image.open(io.BytesIO(self.read(f"annotations/{sequence}/{frame}.png"))))


def _davis_path(name: str) -> str:
    """A file's path inside the davis folder."""
    parts = Path(name).parts
    return "/".join(parts[len(parts) - parts[::-1].index("davis") :]) if "davis" in parts else name


def largest_object(object_ids: np.ndarray) -> np.ndarray:
    """The mask of the object that covers the most pixels."""
    ids, areas = np.unique(object_ids[object_ids > 0], return_counts=True)
    if not len(ids):
        raise ValueError("the frame has no annotated object")
    return object_ids == ids[np.argmax(areas)]


def preprocess(image: Image.Image) -> np.ndarray:
    """A frame as the image encoder's normalized [1, 3, 1024, 1024] input."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, np.float32).transpose(2, 0, 1)[None] / 255.0
    return (pixels - MEAN) / STD


@dataclass(frozen=True)
class Prompt:
    """Prompt points as (x, y) in 1024x1024 frame pixels, each with its POINT_TYPES type."""

    points: tuple[tuple[float, float], ...] = ()
    types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # At least one slot stays padding, as the prompt encoder always appends a padding point.
        if len(self.points) != len(self.types) or len(self.points) >= PROMPT_POINTS:
            raise ValueError(f"expected up to {PROMPT_POINTS - 1} points with one type each")

    def inputs(self) -> dict[str, np.ndarray]:
        """The mask decoder's prompt inputs; unused slots are padding.

        Coordinates are shifted to pixel centres and normalized to [-1, 1], as the prompt encoder does.
        """
        types = [*self.types, *["padding"] * (PROMPT_POINTS - len(self.types))]
        coords = np.zeros((PROMPT_POINTS, 2), np.float32)
        if self.points:
            coords[: len(self.points)] = (np.asarray(self.points, np.float32) + 0.5) / IMAGE_SIZE * 2 - 1
        one_hot = np.zeros((PROMPT_POINTS, len(POINT_TYPES)), np.float32)
        one_hot[np.arange(PROMPT_POINTS), [POINT_TYPES.index(kind) for kind in types]] = 1
        return {"point_coords": coords.T[None, :, None, :].copy(), "point_types": one_hot.T[None, :, None, :].copy()}


# A tracked frame's prompt: every slot is padding.
NO_PROMPT = Prompt()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def upsample(mask_logits: np.ndarray) -> np.ndarray:
    """A [256, 256] mask's logits bilinearly resized to [1, 1, 1024, 1024], as the tracker does."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(mask_logits, np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return resized.numpy()


def select_mask(outputs: dict[str, np.ndarray], multimask: bool) -> tuple[np.ndarray, np.ndarray]:
    """The tracker's mask logits and object pointer from the decoder's four masks.

    With one or no prompt point, it keeps the best of masks 1-3 by predicted IoU, and that mask's pointer. Otherwise it
    keeps mask 0 unless its stability is below the threshold, then the best of masks 1-3, and always pointer 0.
    """
    masks, scores, pointers = outputs["mask_logits"][0], outputs["iou_scores"][0], outputs["object_pointers"][0]
    best = 1 + int(np.argmax(scores[1:]))
    if multimask:
        mask, pointer = masks[best], pointers[best]
    else:
        # transformers scores a mask that is empty at both thresholds as stable.
        union = (masks[0] > -STABILITY_DELTA).sum()
        stable = union == 0 or (masks[0] > STABILITY_DELTA).sum() / union >= STABILITY_THRESHOLD
        mask, pointer = (masks[0] if stable else masks[best]), pointers[0]
    if outputs["object_score_logits"][0, 0] <= 0:
        mask = np.full_like(mask, NO_OBJECT_LOGIT)
    return mask, pointer


@dataclass
class Tracker:
    """One object's FP32 tracking state, recording every graph's inputs."""

    sessions: dict
    memories: list = field(default_factory=list)
    pointers: list = field(default_factory=list)
    recorded: dict = field(default_factory=lambda: {component: [] for component in COMPONENTS})

    def run(self, component: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.recorded[component].append(inputs)
        names = list(INTERFACES[component]["outputs"])
        return dict(zip(names, self.sessions[component].run(names, inputs)))

    def bank(self) -> dict[str, np.ndarray]:
        """The memory attention's bank: slot 0 is the prompted frame, slot k the frame k back; then the pointers."""
        slots = [self.memories[0], *self.memories[1:][::-1][: MEMORY_SLOTS - 1]]
        memories = np.zeros((1, MEMORY_SLOTS * MEMORY_TOKENS, slots[0].shape[-1]), np.float32)
        memory_mask = np.zeros((1, MEMORY_SLOTS), np.float32)
        for slot, memory in enumerate(slots):
            memories[0, slot * MEMORY_TOKENS : (slot + 1) * MEMORY_TOKENS] = memory[0]
            memory_mask[0, slot] = 1
        pointers = [self.pointers[0], *self.pointers[1:][::-1][: OBJECT_POINTERS - 1]]
        object_pointers = np.zeros((1, OBJECT_POINTERS, pointers[0].shape[-1]), np.float32)
        object_pointer_mask = np.zeros((1, OBJECT_POINTERS), np.float32)
        object_pointers[0, : len(pointers)] = np.stack(pointers)
        object_pointer_mask[0, : len(pointers)] = 1
        return {
            "memories": memories,
            "memory_mask": memory_mask,
            "object_pointers": object_pointers,
            "object_pointer_mask": object_pointer_mask,
        }

    def step(self, frame: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        """Segment one frame, prompted on the first frame and tracked on later ones, and add its memory; return the
        mask logits."""
        prompted = prompt is not None
        encoded = self.run("image_encoder", {"image": frame})
        features = {name: encoded[name] for name in ("high_res_features_0", "high_res_features_1")}
        embeddings = encoded["image_embeddings"]
        if not prompted:
            bank = {"image_embeddings": embeddings, **self.bank()}
            embeddings = self.run("memory_attention", bank)["conditioned_embeddings"]
        decoder_prompt = prompt if prompted else NO_PROMPT
        decoded = self.run("mask_decoder", {"image_embeddings": embeddings, **features, **decoder_prompt.inputs()})
        mask, pointer = select_mask(decoded, multimask=len(decoder_prompt.points) <= 1)
        high_res = upsample(mask)
        # A prompted frame's memory takes the thresholded mask; a tracked frame's its probabilities.
        probabilities = (high_res > 0).astype(np.float32) if prompted else sigmoid(high_res)
        memory = self.run("memory_encoder", {"image_embeddings": encoded["image_embeddings"], "mask": probabilities})
        self.memories.append(memory["memory"])
        self.pointers.append(pointer)
        return mask


def first_frame_prompt(mask: np.ndarray, kind: str, rng: random.Random) -> Prompt:
    """The prompt of one kind for the object mask covers on the first frame."""
    height, width = mask.shape

    def to_frame(x, y):
        return x * IMAGE_SIZE / width, y * IMAGE_SIZE / height

    def pick(cells):
        """One random (x, y) pixel of the cells."""
        return cells[rng.randrange(len(cells))][::-1]

    if kind == "click":
        return Prompt((to_frame(*pick(np.argwhere(mask))),), ("foreground",))
    if kind == "click_and_background":
        points = to_frame(*pick(np.argwhere(mask))), to_frame(*pick(np.argwhere(~mask)))
        return Prompt(points, ("foreground", "background"))
    rows, columns = np.nonzero(mask)
    corners = to_frame(columns.min(), rows.min()), to_frame(columns.max() + 1, rows.max() + 1)
    return Prompt(corners, ("box_top_left", "box_bottom_right"))


@dataclass
class CalibrationSet:
    """One component's calibration samples and its inputs on the check clip, which compile.py hands to the process
    that quantizes the component."""

    samples: list[dict[str, np.ndarray]]
    checks: list[dict[str, np.ndarray]]

    def save(self, path: Path) -> None:
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path: Path) -> CalibrationSet:
        with path.open("rb") as stream:
            return pickle.load(stream)


def tracked_clips(
    graphs: dict[str, Path], dataset: Path, clips: int, kept_frames: int
) -> list[dict[str, list[dict[str, np.ndarray]]]]:
    """Each clip's inputs to every graph, from tracking one object through each of clips seeded cal sequences.

    Clip i is prompted with PROMPTS[i % 3]. Each graph keeps kept_frames of a clip's inputs, spread over the clip
    in frame order, so the tracked clips fit in memory.
    """
    import onnxruntime as ort

    sessions = {
        component: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for component, path in graphs.items()
    }
    rng = random.Random(SEED)
    recorded = []
    with DavisCal(dataset) as cal:
        sequences = cal.sequences()
        rng.shuffle(sequences)
        if len(sequences) < clips:
            raise ValueError(f"the dataset has {len(sequences)} sequences to track, {clips} are required")
        for index, sequence in enumerate(sequences[:clips]):
            kind = PROMPTS[index % len(PROMPTS)]
            frames = cal.frames(sequence)
            prompt = first_frame_prompt(largest_object(cal.object_ids(sequence, frames[0])), kind, rng)
            tracker = Tracker(sessions)
            for position, frame in enumerate(frames):
                tracker.step(preprocess(cal.image(sequence, frame)), prompt if position == 0 else None)
            recorded.append({component: spread(inputs, kept_frames) for component, inputs in tracker.recorded.items()})
            print(f"Tracked calibration clip {index + 1}/{clips}: {sequence}, {kind}", flush=True)
    return recorded


def spread(items: list, count: int) -> list:
    """count of the items, evenly spaced from the first to the last."""
    return [items[index] for index in np.linspace(0, len(items) - 1, count).round().astype(int)]


def calibration_sets(
    graphs: dict[str, Path], dataset: Path, clips: int, sample_counts: dict[str, int], check_frames: int
) -> dict[str, CalibrationSet]:
    """Each component's CalibrationSet: sample_counts[component] of the first clips - 1 clips' inputs, spread evenly
    over them rather than taken from the first clips, and check_frames of the last clip's inputs, spread over it."""
    kept_frames = max(check_frames, -(-max(sample_counts.values()) // (clips - 1)))
    *calibrating, checking = tracked_clips(graphs, dataset, clips, kept_frames)
    sets = {}
    for component in COMPONENTS:
        inputs = [frame for clip in calibrating for frame in clip[component]]
        count = sample_counts[component]
        if count > len(inputs):
            raise ValueError(f"{component} needs {count} calibration samples; {len(inputs)} were tracked")
        sets[component] = CalibrationSet(spread(inputs, count), spread(checking[component], check_frames))
    return sets
