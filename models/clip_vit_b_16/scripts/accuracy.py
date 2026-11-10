"""Measure zero-shot ImageNet top-1 and top-5 with the two compiled encoders.

At evaluation startup, encode the upstream ImageNet class prompts with the compiled text encoder.
Normalize each prompt embedding, average within each class and normalize again. Keep the resulting
1,000 class vectors in memory and rank their cosine similarities to each compiled image embedding.
Text tokenization and embedding-table lookup run on CPU. Source assets are fetched from the pinned
upstream checkpoint, or read from --checkpoint-dir; generated class vectors are never persisted.
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
from calibration import CLASS_NAMES, TEMPLATES, VALIDATION_SPLIT, preprocess_image, validation_labels

COMPONENTS = {"image_encoder", "text_encoder"}
EMBED_DIM = 512
TEXT_SHAPE = (1, 77, EMBED_DIM)
SOURCE_REPOSITORY = "openai/clip-vit-base-patch16"
SOURCE_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
WEIGHTS_FILE = "pytorch_model.bin"
TOKEN_EMBEDDING = "text_model.embeddings.token_embedding.weight"
TEXT_LENGTH = 77

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


def load_text_inputs(checkpoint_dir: Path | None = None):
    """Load the tokenizer and CPU lookup table, without loading a CPU transformer."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    def asset(name):
        return (
            checkpoint_dir / name
            if checkpoint_dir is not None
            else Path(hf_hub_download(SOURCE_REPOSITORY, name, revision=SOURCE_REVISION))
        )

    tokenizer = Tokenizer.from_file(str(asset("tokenizer.json")))
    tokenizer.enable_truncation(max_length=TEXT_LENGTH)
    tokenizer.enable_padding(length=TEXT_LENGTH, pad_id=49407, pad_token="<|endoftext|>")
    import torch

    weights = torch.load(asset(WEIGHTS_FILE), map_location="cpu", weights_only=True)
    table = weights[TOKEN_EMBEDDING].numpy()
    if table.shape != (49408, EMBED_DIM) or table.dtype != np.float32:
        raise ValueError(f"unexpected token embedding table: {table.shape}, {table.dtype}")
    return tokenizer, table


def class_prompts(name: str) -> list[str]:
    return [template.format(name) for template in TEMPLATES]


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
                (output,) = runner.run([tensor], timeout_ms=120000)
                feature = output.to_numpy(copy=False).reshape(TEXT_LENGTH, EMBED_DIM)[int(ids.argmax())]
                features.append(normalize(feature))
            classes.append(normalize(np.mean(features, axis=0)))
            if index == 1 or index % 100 == 0 or index == len(class_names):
                log(f"text encoder: {index}/{len(class_names)} classes, {len(TEMPLATES)} templates/class")
    finally:
        if runner is not None:
            runner.close()
    log(f"Prepared class vectors in {time.monotonic() - started:.1f} s")
    return np.stack(classes).astype(np.float32)


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
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="local pinned tokenizer.json and checkpoint; default: HF cache/download"
    )
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    labels = validation_labels(dataset)
    available = sorted(labels)
    names = available[: args.limit] if args.limit else available

    mpks = parse_mpks(args.mpk)
    check_text_encoder(mpks["text_encoder"], load_text_probe())
    tokenizer, table = load_text_inputs(args.checkpoint_dir)
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
        logits = class_scores(runner.run([tensor], timeout_ms=120000), classes)
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
