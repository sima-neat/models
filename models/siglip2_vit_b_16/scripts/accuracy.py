"""Measure SigLIP 2 ViT-B/16 zero-shot ImageNet top-1 and top-5 accuracy with its MPKs on a Modalix board.

Surgery bakes the 1,000 ImageNet class embeddings (OpenAI's class names and 7 selected prompt templates, run
through the text encoder in FP32) into the image encoder's logits output, so zero-shot classification runs the image
encoder alone. The text encoder is checked first, as an application runs it. The board has no tokenizer, so
text_encoder_check.npz holds 7 tokenized prompts, the token embedding rows they use, and each prompt's text embedding
from the FP32 ONNX (surgery.py --text-check writes it). The CPU looks each prompt's rows up, the text encoder's MPK
embeds it, and the stage fails when any embedding's cosine similarity with the ONNX's is below MIN_TEXT_COSINE.
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
EMBED_DIM = 768
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


def class_logits(outputs: list[pyneat.Tensor]) -> np.ndarray:
    """The logits output, told apart from the image embedding by its size."""
    arrays = [output.to_numpy(copy=False).reshape(-1) for output in outputs]
    sizes = sorted(array.size for array in arrays)
    if sizes != [EMBED_DIM, CLASSES]:
        raise ValueError(f"expected the image embedding and {CLASSES} logits, found output sizes {sizes}")
    return next(array for array in arrays if array.size == CLASSES)


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


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
        tensor = pyneat.Tensor.from_numpy(prompt_embeds(check, index)[None], copy=True,
                                          memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.HWC)
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
        raise ValueError(f"the text encoder strays from the FP32 ONNX: cosine similarities {found}, "
                         f"each must be at least {MIN_TEXT_COSINE}")
    return cosines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=<path>, for each component")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)
    mpks = parse_mpks(args.mpk)
    check_text_encoder(mpks["text_encoder"])

    labels = validation_labels(dataset)
    available = sorted(labels)
    names = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(mpks["image_encoder"]), tensor_options())
    runner = None
    top1 = top5 = 0
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        logits = class_logits(runner.run([tensor], timeout_ms=TIMEOUT_MS))
        if not np.isfinite(logits).all():
            raise ValueError(f"non-finite logits for {name}")
        ranked = np.argsort(-logits, kind="stable")[:5]
        top1 += int(ranked[0] == labels[name])
        top5 += int(labels[name] in ranked)
        if count == 1 or count % 500 == 0 or count == len(names):
            log(f"{count}/{len(names)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {
                "top1_accuracy": top1 / len(names),
                "top5_accuracy": top5 / len(names),
            },
            "split": VALIDATION_SPLIT, "evaluated_samples": len(names), "available_samples": len(available),
            "details": {"inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
