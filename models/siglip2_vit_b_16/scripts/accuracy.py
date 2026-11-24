"""Measure zero-shot ImageNet top-1 and top-5 with the two compiled encoders.

At evaluation startup, encode the upstream ImageNet class prompts with the compiled text encoder.
Normalize each prompt embedding, average within each class and normalize again. Keep the resulting
1,000 class vectors in memory and rank their cosine similarities to each compiled image embedding.
Text tokenization and embedding-table lookup run on CPU using the packaged tokenizer and lookup weights.
Generated class vectors are never persisted.

Recipe: Big Vision commit 0127fb6b337ee2a27bf4e54dea79cff176527356, configs/proj/image_text/common.py
get_disclf(224) and evaluators/proj/image_text/discriminative_classifier.py defaults: canonicalized
clip_paper templates, first class alias, normalized prompt ensemble and cosine ranking. The release
does not publish the exact evaluation config for its paper score; reproducing 78.2% is unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import string
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import CLASS_NAMES, MEAN, RESCALE, STD, VALIDATION_SPLIT, validation_labels
from model_report import log
from PIL import Image

# Big Vision commit 0127fb6b337ee2a27bf4e54dea79cff176527356:
# evaluators/proj/image_text/prompt_engineering_constants.py, clip_paper (80 CLIP templates plus the bare class name).
TEMPLATES = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "{}",
)

COMPONENTS = {"image_encoder", "text_encoder"}
EMBED_DIM = 768
TEXT_LENGTH = 64

# Images decode on worker threads while the MPK runs; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8
TIMEOUT_MS = 120000
# The stage's fingerprint covers this script but not the check file, so the script names the file's SHA-256: a new
# file needs a new value here, which reruns the stage.
TEXT_CHECK = Path(__file__).with_name("text_encoder_check.npz")
TEXT_CHECK_SHA256 = "273bfe20216fb3c1ef681f453335b37b066f687744dec2893154384e3c1305e5"
# BF16 activations keep the text embeddings close to the ONNX's: on every check prompt on a DevKit, the w16a16 build
# scores 0.9999 or more and the w8a16 build, with INT8 weights, 0.995 or more. A missing, corrupt or mismatched text
# encoder falls far below.
MIN_TEXT_COSINE = 0.99


def preprocess_image(content: bytes) -> np.ndarray:
    """Big Vision get_disclf(224): bilinear resize(256), central_crop(224), value_range(-1,1).

    Its TensorFlow resize uses half-pixel centers without antialiasing and casts back to uint8.
    Pillow's bilinear downsampling applies an antialiasing filter, so use NumPy for this step.
    """
    with Image.open(io.BytesIO(content)) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float32)
    y, x = [
        np.maximum((np.arange(256, dtype=np.float32) + 0.5) * (length / 256) - 0.5, 0) for length in image.shape[:2]
    ]
    y0, x0 = y.astype(np.int32), x.astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, image.shape[0] - 1), np.minimum(x0 + 1, image.shape[1] - 1)
    dx, dy = (x - x0.astype(np.float32))[None, :, None], (y - y0.astype(np.float32))[:, None, None]
    top = image[y0[:, None], x0] + (image[y0[:, None], x1] - image[y0[:, None], x0]) * dx
    bottom = image[y1[:, None], x0] + (image[y1[:, None], x1] - image[y1[:, None], x0]) * dx
    image = np.clip(top + (bottom - top) * dy, 0, 255).astype(np.uint8)[16:240, 16:240]
    return (image.astype(np.float32) * RESCALE - MEAN) / STD


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized and normalized the image, or looked the prompt's tokens up; Neat still quantizes and
    # tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values if "=" in value)
    if len(mpks) != len(values) or set(mpks) != COMPONENTS:
        raise SystemExit("--mpk must name exactly image_encoder=<path> and text_encoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("invalid embedding: values and norms must be finite and norms positive")
    return values / norms


def class_scores(outputs: list[pyneat.Tensor], classes: np.ndarray) -> np.ndarray:
    if len(outputs) != 1:
        raise ValueError("expected one image embedding output")
    embedding = outputs[0].to_numpy(copy=False).reshape(-1)
    if embedding.shape != (EMBED_DIM,):
        raise ValueError(f"expected an image embedding of {EMBED_DIM} values, found {embedding.shape}")
    return classes @ normalize(embedding)


def load_text_inputs(resources: Path):
    """Read the CPU tokenizer and lookup weights shipped with the compiled encoders."""
    from tokenizers import Tokenizer

    directory = resources / "text_encoder"
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=TEXT_LENGTH)
    tokenizer.enable_padding(length=TEXT_LENGTH, pad_id=0, pad_token="<pad>")
    table = np.load(directory / "token_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if table.shape != (256000, EMBED_DIM) or table.dtype != np.float32:
        raise ValueError(f"unexpected token embedding table: {table.shape}, {table.dtype}")
    check = load_text_check()
    actual_ids = np.asarray([item.ids for item in tokenizer.encode_batch(check["prompts"].tolist())])
    if not np.array_equal(actual_ids, check["token_ids"]):
        raise ValueError("packaged tokenizer does not reproduce the text probe's token IDs")
    return tokenizer, table


def class_prompts(name: str) -> list[str]:
    # Big Vision canonicalizes names and templates before formatting, and uses the first class alias.
    def canonicalize(text, keep):
        punctuation = str.maketrans("", "", string.punctuation)
        text = keep.join(part.translate(punctuation) for part in text.replace("_", " ").split(keep))
        return re.sub(r"\s+", " ", text.lower()).strip()

    name = canonicalize(name, ",").split(",")[0]
    return [canonicalize(template, "{}").format(name) for template in TEMPLATES]


def build_class_embeddings(mpk: Path, tokenizer, table: np.ndarray, class_names=CLASS_NAMES) -> np.ndarray:
    """Prepare every class once using the compiled text encoder; retain only the class vectors."""
    model = pyneat.Model(str(mpk), tensor_options())
    runner = None
    classes = []
    started = time.monotonic()
    try:
        for index, name in enumerate(class_names, start=1):
            features = []
            for encoded in tokenizer.encode_batch(class_prompts(name)):
                ids = np.asarray(encoded.ids, np.int32)
                embeddings = table[ids][None]
                tensor = pyneat.Tensor.from_numpy(
                    embeddings, copy=True, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC
                )
                if runner is None:
                    runner = model.build([tensor])
                (output,) = runner.run([tensor], timeout_ms=TIMEOUT_MS)
                feature = output.to_numpy(copy=False).reshape(EMBED_DIM)
                features.append(normalize(feature))
            classes.append(normalize(np.mean(features, axis=0)))
            if index == 1 or index % 100 == 0 or index == len(class_names):
                log(f"text encoder: {index}/{len(class_names)} classes, {len(TEMPLATES)} templates/class")
    finally:
        if runner is not None:
            runner.close()
    log(f"Prepared class vectors in {time.monotonic() - started:.1f} s")
    return np.stack(classes).astype(np.float32)


def load_text_check(path: Path = TEXT_CHECK, sha256: str = TEXT_CHECK_SHA256) -> dict[str, np.ndarray]:
    """The check's arrays, from a file with the SHA-256 the script names."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"{path.name} does not have TEXT_CHECK_SHA256; regenerate it with surgery.py --text-check")
    with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def prompt_embeds(check: dict[str, np.ndarray], index: int) -> np.ndarray:
    """The CPU step before the text encoder: each of the prompt's token IDs replaced by its token embedding row."""
    rows = dict(zip(check["row_ids"].tolist(), check["rows"]))
    return np.stack([rows[token] for token in check["token_ids"][index].tolist()])


def check_text_encoder(mpk: Path) -> list[float]:
    """Embed each check prompt with the text encoder's MPK; fail unless every embedding matches the FP32 ONNX's."""
    check = load_text_check()
    model = pyneat.Model(str(mpk), tensor_options())
    runner = None
    cosines = []
    for index, (prompt, expected) in enumerate(zip(check["prompts"].tolist(), check["text_embeds"])):
        # Tagged HWC like performance.py's prompt: Neat sizes the text input as height 1, width 64 and depth 768.
        tensor = pyneat.Tensor.from_numpy(
            prompt_embeds(check, index)[None],
            copy=True,
            memory=pyneat.TensorMemory.EV74,
            layout=pyneat.TensorLayout.HWC,
        )
        if runner is None:
            runner = model.build([tensor])
        (output,) = runner.run([tensor], timeout_ms=TIMEOUT_MS)
        embedding = output.to_numpy(copy=False).reshape(EMBED_DIM)
        cosines.append(float(np.dot(embedding, expected) / (np.linalg.norm(embedding) * np.linalg.norm(expected))))
        log(f"text encoder: {prompt!r} has cosine similarity {cosines[-1]:.5f} with the FP32 ONNX")
    runner.close()
    # A NaN fails as well.
    if not all(cosine >= MIN_TEXT_COSINE for cosine in cosines):
        found = ", ".join(f"{cosine:.5f}" for cosine in cosines)
        raise ValueError(
            f"the text encoder strays from the FP32 ONNX: cosine similarities {found}, "
            f"each must be at least {MIN_TEXT_COSINE}"
        )
    return cosines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=<path>, for each component")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument(
        "--resources",
        type=Path,
        required=True,
        help="package root containing text_encoder/tokenizer.json and token_embeddings.npy",
    )
    parser.add_argument("--limit", type=int, help="measure the first images only")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)
    mpks = parse_mpks(args.mpk)
    check_text_encoder(mpks["text_encoder"])

    labels = validation_labels(dataset)
    available = sorted(labels)
    names = available[: args.limit] if args.limit else available

    tokenizer, table = load_text_inputs(args.resources)
    classes = build_class_embeddings(mpks["text_encoder"], tokenizer, table)
    del tokenizer, table
    model = pyneat.Model(str(mpks["image_encoder"]), tensor_options())
    runner = None
    top1 = top5 = 0
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        logits = class_scores(runner.run([tensor], timeout_ms=TIMEOUT_MS), classes)
        if not np.isfinite(logits).all():
            raise ValueError(f"non-finite logits for {name}")
        ranked = np.argsort(-logits, kind="stable")[:5]
        top1 += int(ranked[0] == labels[name])
        top5 += int(labels[name] in ranked)
        if count == 1 or count % 500 == 0 or count == len(names):
            log(f"{count}/{len(names)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")

    runner.close()
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": {
                    "top1_accuracy": top1 / len(names),
                    "top5_accuracy": top5 / len(names),
                },
                "split": VALIDATION_SPLIT,
                "evaluated_samples": len(names),
                "available_samples": len(available),
                "details": {
                    "inference_seconds": round(time.monotonic() - started, 1),
                    "classes": len(CLASS_NAMES),
                    "templates_per_class": len(TEMPLATES),
                    "text_reference": "compiled text encoder",
                },
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
