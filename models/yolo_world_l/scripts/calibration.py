"""Verified calibration images and their paired prompt lists.

The exact research selection is reproduced from official COCO image URLs when
--dataset is omitted. A supplied directory must contain the same verified bytes.
This direct-download route was explicitly requested because Artifactory access
is unavailable; it never substitutes validation images for calibration.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
from calibration_samples import SAMPLES

IMAGE_SIZE = 640
FILL = 114


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


def calibration_inputs(directory, text_model, tokenizer_json, table):
    from afe.apis.defines import InputName

    images = verified_images(directory)
    prompts = sorted({prompt for record in SAMPLES for prompt in record["prompts"]})
    lookup, selectors = tokenize(prompts, tokenizer_json, table)
    vectors = {}
    for i, prompt in enumerate(prompts):
        (result,) = text_model.execute(inputs={InputName("token_embeddings"): lookup[i:i+1],
                                              InputName("eos_selector"): selectors[i:i+1]}, use_jax=True)
        vector = np.asarray(result, np.float32).reshape(512)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) < 0.9:
            raise ValueError(f"invalid text embedding for {prompt!r}")
        vectors[prompt] = vector / np.linalg.norm(vector)
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
