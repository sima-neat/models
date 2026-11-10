"""Verified calibration images and their paired prompt lists.

The exact research selection is reproduced from official COCO image URLs when
--dataset is omitted. A supplied directory must contain the same verified bytes.
This direct-download route was explicitly requested because Artifactory access
is unavailable; it never substitutes validation images for calibration.

The BF16 simulator processes 32 independent prompt rows per call. A cache bound
to the graph, tokenizer, lookup table, prompts and compiler versions shares the
vectors across model sizes. Set YOLO_WORLD_PROMPT_CACHE to share its directory.
"""
import hashlib
import importlib.metadata
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
from calibration_samples import SAMPLES

IMAGE_SIZE = 640
FILL = 114
PROMPT_ROWS = 32
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def letterbox_geometry(width, height):
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized = round(width * gain), round(height * gain)
    dx, dy = (IMAGE_SIZE - resized[0]) / 2, (IMAGE_SIZE - resized[1]) / 2
    return gain, resized, (round(dx - .1), round(dy - .1), round(dx + .1), round(dy + .1))


def preprocess_image(content):
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("cannot decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, size, (left, top, right, bottom) = letterbox_geometry(image.shape[1], image.shape[0])
    if size != (image.shape[1], image.shape[0]):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(FILL, FILL, FILL))
    return image.astype(np.float32) / np.float32(255)


def verified_images(directory=None, download_missing=False):
    supplied = directory is not None
    directory = Path(directory) if supplied else Path.home() / ".cache/yolo-world/cal128"
    directory.mkdir(parents=True, exist_ok=True)

    def one(record):
        candidates = [directory / record["file_name"], directory / "train2017" / record["file_name"]]
        path = next((p for p in candidates if p.is_file()), candidates[0])
        if not path.exists():
            if supplied and not download_missing:
                raise ValueError(f"calibration selection missing {record['file_name']}")
            url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/" + record["file_name"]
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"calibration image checksum mismatch: {record['file_name']}")
            with tempfile.NamedTemporaryFile(dir=directory, prefix=path.name + ".", delete=False) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            temporary.replace(path)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"calibration image checksum mismatch: {path}")
        return preprocess_image(content)

    with ThreadPoolExecutor(4) as pool:
        return np.stack(list(pool.map(one, SAMPLES)))


def tokenize(prompts, tokenizer_json, table):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    tokenizer.no_truncation()
    tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
    ids = [encoding.ids for encoding in tokenizer.encode_batch(prompts)]
    if any(len(row) != 77 for row in ids):
        raise ValueError("a prompt exceeds CLIP's 77-token context")
    ids = np.asarray(ids, np.int64)
    selectors = np.zeros((len(ids), 1, 96, 1), np.float32)
    selectors[np.arange(len(ids)), 0, ids.argmax(axis=1), 0] = 1
    return table[ids][:, None, :, :].astype(np.float32), selectors


def prompt_cache_key(prompts, graph, tokenizer_json, table):
    versions = {}
    for package in ("sima-frontend", "sima-mlc", "sima-kernels", "jax", "jaxlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    def file_hash(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    identity = {"version": 1, "rows": PROMPT_ROWS, "precision": "bf16-sima", "matmul_precision": "highest",
                "graph": file_hash(graph), "tokenizer": file_hash(tokenizer_json), "versions": versions,
                "table": hashlib.sha256(memoryview(np.ascontiguousarray(table)).cast("B")).hexdigest(),
                "prompts": prompts}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def execute_prompt_rows(model, prompts, tokenizer_json, table):
    """Run fixed groups of independent prompt rows through the public BF16 simulator."""
    import jax
    from afe.apis.defines import InputName

    lookup, selectors = tokenize(prompts, tokenizer_json, table)
    vectors = []
    with jax.default_matmul_precision("highest"):
        for start in range(0, len(prompts), PROMPT_ROWS):
            count = min(PROMPT_ROWS, len(prompts) - start)
            # Repeat the last prompt to retain one static shape, including the final partial group.
            indices = np.minimum(np.arange(start, start + PROMPT_ROWS), len(prompts) - 1)
            inputs = {InputName("token_embeddings"): np.ascontiguousarray(lookup[indices].transpose(1, 0, 2, 3)),
                      InputName("eos_selector"): np.ascontiguousarray(selectors[indices].transpose(1, 0, 2, 3))}
            (result,) = model.execute(inputs=inputs, use_jax=True)
            vectors.append(np.asarray(result, np.float32).reshape(PROMPT_ROWS, 512)[:count])
            LOGGER.info("Calibration prompt embeddings: %d/%d", start + count, len(prompts))
    vectors = np.concatenate(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(abs(norms - 1) > .02):
        raise ValueError("invalid BF16 calibration prompt embeddings")
    # Preserve the original per-prompt FP32 normalization order.
    return np.stack([vector / np.linalg.norm(vector) for vector in vectors])


def cached_prompt_vectors(prompts, graph, tokenizer_json, table, create_model, cache_dir=None):
    """S/M/L share the text tower; one process creates this content-bound cache."""
    key = prompt_cache_key(prompts, graph, tokenizer_json, table)
    root = Path(cache_dir or os.environ.get("YOLO_WORLD_PROMPT_CACHE", Path.home() / ".cache/yolo-world/prompts"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / (key + ".npy")
    LOGGER.info("Calibration prompt cache: %s", path)
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            vectors = np.load(path, allow_pickle=False)
            if (vectors.shape != (len(prompts), 512) or vectors.dtype != np.float32
                    or not np.isfinite(vectors).all() or np.any(abs(np.linalg.norm(vectors, axis=1) - 1) > 1e-5)):
                raise ValueError(f"invalid calibration prompt cache: {path}")
            LOGGER.info("Reusing %d cached BF16 prompt embeddings", len(prompts))
            return vectors
        vectors = execute_prompt_rows(create_model(), prompts, tokenizer_json, table)
        with tempfile.NamedTemporaryFile(dir=root, prefix=key + ".", delete=False) as stream:
            np.save(stream, vectors, allow_pickle=False)
            temporary = Path(stream.name)
        temporary.replace(path)
        return vectors


def calibration_inputs(directory, text_graph, tokenizer_json, table, create_text_model):
    from afe.apis.defines import InputName

    images = verified_images(directory)
    prompts = sorted({prompt for record in SAMPLES for prompt in record["prompts"]})
    vectors = dict(zip(prompts, cached_prompt_vectors(prompts, text_graph, tokenizer_json, table,
                                                     create_text_model), strict=True))
    samples = []
    for image, record in zip(images, SAMPLES, strict=True):
        embeddings = np.zeros((1, 80, 512), np.float32)
        mask = np.zeros((1, 80), np.float32)
        for i, prompt in enumerate(record["prompts"]):
            embeddings[0, i] = vectors[prompt]
        mask[0, :record["active_count"]] = 1
        samples.append({InputName("images"): image[None], InputName("text_embeddings"): embeddings,
                        InputName("prompt_mask"): mask})
    return samples
