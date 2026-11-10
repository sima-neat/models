"""Measure CLIP ViT-B/16 zero-shot ImageNet top-1 and top-5 accuracy with its MPKs on a Modalix board.

Surgery bakes the 1,000 ImageNet class embeddings (OpenAI's class names and 80 prompt templates, run through
the text encoder in FP32) into the image encoder's logits output, so zero-shot classification runs the image encoder
alone. The board has no tokenizer, so the script first checks the text encoder on the eight prompts in
text_probe.npz, which surgery.py --text-probe writes: their token IDs, only the token-embedding rows those IDs use,
and the converted ONNX's FP32 end-of-text rows. It looks each prompt's tokens up on the CPU, as an application does,
runs the text encoder, and stops when an end-of-text row's cosine similarity with its FP32 row is below
MIN_TEXT_COSINE.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import CLASSES, VALIDATION_SPLIT, preprocess_image, validation_labels

COMPONENTS = {"image_encoder", "text_encoder"}
EMBED_DIM = 512
TEXT_SHAPE = (1, 77, EMBED_DIM)
# The accuracy fingerprint hashes this script but not the file beside it, so the script pins the file's checksum.
TEXT_PROBE = Path(__file__).with_name("text_probe.npz")
TEXT_PROBE_SHA256 = "0c2b26bbee9ee7587a0b39df3e2e354eef3b2a434c6f7d97bad7398a36df5aac"
# BF16 activations keep the text encoder's end-of-text rows close to FP32: on every probe prompt on a DevKit, the
# w16a16 build scores 0.9998 or more and the w8a16 build, with INT8 weights, 0.9975 or more. A broken text encoder
# scores far lower; token rows shifted by one score 0.27 to 0.56.
MIN_TEXT_COSINE = 0.99
# Images decode on worker threads while the MPK runs; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized, cropped and normalized the image; Neat still quantizes and tessellates.
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


def class_logits(outputs: list[pyneat.Tensor]) -> np.ndarray:
    """The logits output, told apart from the image embedding by its size."""
    arrays = [output.to_numpy(copy=False).reshape(-1) for output in outputs]
    sizes = sorted(array.size for array in arrays)
    if sizes != [EMBED_DIM, CLASSES]:
        raise ValueError(f"expected the image embedding and {CLASSES} logits, found output sizes {sizes}")
    return next(array for array in arrays if array.size == CLASSES)


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def load_text_probe(path: Path = TEXT_PROBE) -> dict[str, np.ndarray]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != TEXT_PROBE_SHA256:
        raise ValueError(f"{path} does not match TEXT_PROBE_SHA256")
    with np.load(io.BytesIO(content)) as saved:
        return {name: saved[name] for name in saved.files}


def token_embeddings(probe: dict[str, np.ndarray], index: int) -> np.ndarray:
    """A prompt's text encoder input: each of its token IDs replaced by that token's row of the embedding table."""
    table = dict(zip(probe["row_ids"].tolist(), probe["rows"]))
    return np.stack([table[token] for token in probe["token_ids"][index].tolist()]).reshape(TEXT_SHAPE)


def check_text_encoder(mpk: Path, probe: dict[str, np.ndarray]) -> None:
    """Run the text encoder on each probe prompt; stop unless every end-of-text row is within MIN_TEXT_COSINE of
    its FP32 row."""
    model = pyneat.Model(str(mpk), tensor_options())
    runner = None
    low = []
    for index, (prompt, ids) in enumerate(zip(probe["prompts"].tolist(), probe["token_ids"])):
        # Neat sizes the input as height 1, width 77 and depth 512, so the tensor declares HWC axes.
        tensor = pyneat.Tensor.from_numpy(
            token_embeddings(probe, index), copy=True, memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC
        )
        if runner is None:
            runner = model.build([tensor])
        (output,) = runner.run([tensor], timeout_ms=120000)
        # The first end-of-text token: its ID is the highest in CLIP's vocabulary.
        row = output.to_numpy().reshape(TEXT_SHAPE[1:])[int(np.argmax(ids))]
        expected = probe["text_embeds"][index]
        cosine = float(np.dot(row, expected) / (np.linalg.norm(row) * np.linalg.norm(expected)))
        log(f"text encoder, {prompt!r}: cosine similarity {cosine:.5f} with FP32")
        # A NaN fails the comparison too.
        if not cosine >= MIN_TEXT_COSINE:
            low.append(f"{prompt!r} ({cosine:.4f})")
    runner.close()
    if low:
        raise ValueError(f"text encoder end-of-text rows below cosine {MIN_TEXT_COSINE} with FP32: {', '.join(low)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=<path>, for each component")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    labels = validation_labels(dataset)
    available = sorted(labels)
    names = available[: args.limit] if args.limit else available

    mpks = parse_mpks(args.mpk)
    check_text_encoder(mpks["text_encoder"], load_text_probe())
    model = pyneat.Model(str(mpks["image_encoder"]), tensor_options())
    runner = None
    top1 = top5 = 0
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        logits = class_logits(runner.run([tensor], timeout_ms=120000))
        if not np.isfinite(logits).all():
            raise ValueError(f"non-finite logits for {name}")
        ranked = np.argsort(-logits, kind="stable")[:5]
        top1 += int(ranked[0] == labels[name])
        top5 += int(labels[name] in ranked)
        if count == 1 or count % 500 == 0 or count == len(names):
            log(f"{count}/{len(names)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")

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
                "details": {"inference_seconds": round(time.monotonic() - started, 1)},
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
