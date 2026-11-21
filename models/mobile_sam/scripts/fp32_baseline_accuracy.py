"""Score MobileSAM's FP32 ONNX chain with the same COCO box prompts and mask scoring as accuracy.py.

The chain prepares each image and resizes each mask on the CPU, as segment_anything does (pipeline.preprocess and
pipeline.image_mask), where accuracy.py has Neat do both on the board.

Run on a host with the accuracy dependencies, onnx, onnxruntime, onnxsim, torch and timm; no PyNeat or registry
credentials are needed. --onnx is the converted image_encoder.onnx that compile.py receives, with the weights that
model.yaml's conversion.outputs names beside it; surgery.py builds the same FP32 component graphs from them that
compile.py quantizes. This is the masked
six-slot reference, not an independent upstream SamPredictor oracle. No score is published until measured.
Complete predictions and their provenance are saved before scoring; --score-only retries scoring without
running the model. This does not resume interrupted inference or produce partial-dataset reference scores.

    PYTHONPATH=tools python models/mobile_sam/scripts/fp32_baseline_accuracy.py \
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import accuracy
import numpy as np
from model_registry.conversion import conversion_outputs
from pipeline import COMPONENTS, INPUT_SHAPES, image_mask, mask_layout, preprocess, tensor_name
from surgery import export_components

MODEL_DIR = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "FP32 ONNX Runtime CPU; each non-crowd object's ground-truth box, mask 0, "
    "six-slot prompts with inactive-key masking; segment_anything's CPU preprocessing and mask resizing; "
    "the same prompts and COCO scoring as accuracy.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_digest(ground_truth, image_ids: list[int], image_dir: Path) -> str:
    digest = hashlib.sha256()
    for record in ground_truth.loadImgs(image_ids):
        identity = [record["id"], record["file_name"], sha256(image_dir / record["file_name"])]
        digest.update((json.dumps(identity, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, document: dict, *, overwrite: bool = True) -> None:
    """Do not leave an apparently complete prediction/report file after a failed write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(document, stream, allow_nan=False)
            stream.write("\n")
            stream.close()
            if overwrite:
                temporary.replace(path)
            else:
                os.link(temporary, path)  # Atomic create-if-absent on the same filesystem; never clobber predictions.
        finally:
            temporary.unlink(missing_ok=True)


def validate_predictions(ground_truth, image_ids: list[int], predictions: list[dict], prompts: int) -> None:
    """Check the selected non-crowd image/category counts, including empty masks."""
    annotations = ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=image_ids, iscrowd=False))
    expected = Counter((item["image_id"], item["category_id"]) for item in annotations)
    actual = Counter((item["image_id"], item["category_id"]) for item in predictions)
    if not predictions or prompts != len(predictions) or actual != expected:
        raise ValueError("prediction counts do not match the selected non-crowd image/category counts")
    if any(not math.isfinite(item["score"]) for item in predictions):
        raise ValueError("prediction scores must be finite")


class FloatRunners:
    """One ONNX Runtime CPU session per component; the encoder is reused for every prompt of its image."""

    def __init__(self, graphs: dict[str, Path]) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        self.sessions = {
            name: onnxruntime.InferenceSession(str(graphs[name]), options, providers=["CPUExecutionProvider"])
            for name in COMPONENTS
        }
        for name, session in self.sessions.items():
            expected = {tensor_name(name, key): list(shape) for key, shape in INPUT_SHAPES[name].items()}
            actual = {item.name: item.shape for item in session.get_inputs()}
            if actual != expected:
                raise ValueError(f"unexpected {name} ONNX inputs: {actual}")

    def run(self, component: str, values: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.sessions[component].run(None, {tensor_name(component, key): value for key, value in values.items()})

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """An RGB image's embedding, and the image size its masks are resized to."""
        pixels = preprocess(image)
        (embedding,) = self.run("image_encoder", {"image": np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])})
        return embedding, image.shape[:2]

    def decode(
        self, embedding: tuple[np.ndarray, tuple[int, int]], prompt: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float]:
        """The first mask, resized to the image and thresholded, and its predicted IoU."""
        features, size = embedding
        masks, scores = mask_layout(*self.run("prompt_decoder", {"image_embeddings": features, **prompt}))
        return image_mask(masks[0], size), float(scores[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="converted image_encoder.onnx, weights.pth beside")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="FP32 reference JSON, not a DevKit benchmark report")
    parser.add_argument(
        "--predictions", type=Path, help="complete prediction/provenance JSON; default: <output stem>.predictions.json"
    )
    parser.add_argument(
        "--score-only", action="store_true", help="score existing complete predictions, with no inference"
    )
    parser.add_argument("--limit", type=int, help="seeded subset for local diagnostics, never the published reference")
    args = parser.parse_args(argv)
    dataset_id, separator, directory = args.dataset.partition("=")
    if dataset_id != "coco-2017-val" or not separator or not directory.strip():
        parser.error("--dataset must name coco-2017-val=<extracted directory>")
    if args.limit is not None and not 0 < args.limit <= accuracy.VAL2017_IMAGES:
        parser.error(f"--limit must be between 1 and {accuracy.VAL2017_IMAGES}")
    dataset = Path(directory)
    weights = args.onnx.with_name(conversion_outputs(MODEL_DIR)[1])
    predictions_path = args.predictions or args.output.with_suffix(".predictions.json")
    if predictions_path.resolve() == args.output.resolve():
        parser.error("--predictions and --output must be different files")
    if predictions_path.exists() and not args.score_only:
        parser.error("predictions already exist; use --score-only or choose a new --predictions path")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = accuracy.COCO(str(dataset / accuracy.ANNOTATIONS))
    available = sorted(ground_truth.getImgIds())
    if len(available) != accuracy.VAL2017_IMAGES:
        parser.error(f"expected {accuracy.VAL2017_IMAGES} validation images, found {len(available)}")
    image_ids = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available
    provenance = {
        "onnx_sha256": sha256(args.onnx),
        "weights_sha256": sha256(weights),
        "dataset": dataset_id,
        "annotations_sha256": sha256(dataset / accuracy.ANNOTATIONS),
        "selected_images_sha256": image_digest(ground_truth, image_ids, dataset / accuracy.IMAGES),
        "image_ids": image_ids,
        "available_samples": len(available),
        "scripts_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "fp32_baseline_accuracy.py",
                "accuracy.py",
                "pipeline.py",
                "surgery.py",
                "sam_decoder.py",
                "sam_modules.py",
                "tiny_vit_4d.py",
                "tiny_vit_sam.py",
            )
        },
        "conditions": CONDITIONS,
    }
    if args.score_only:
        saved = json.loads(predictions_path.read_text(encoding="utf-8"))
        if saved["provenance"] != provenance:
            raise ValueError("prediction provenance does not match the source, dataset, evaluated IDs and scripts")
        results, prompts = saved["predictions"], saved["prompts"]
    else:
        with tempfile.TemporaryDirectory(prefix="mobile-sam-fp32-") as temporary:
            graphs = export_components(args.onnx, weights, Path(temporary) / "components")
            runners = FloatRunners(graphs)
            results, prompts = accuracy.evaluate(
                ground_truth, image_ids, dataset / accuracy.IMAGES, runners.encode, runners.decode
            )
        del runners  # The scoring phase does not need to retain three ORT sessions and their allocations.
    validate_predictions(ground_truth, image_ids, results, prompts)
    if not args.score_only:
        write_json(
            predictions_path, {"provenance": provenance, "prompts": prompts, "predictions": results}, overwrite=False
        )
    metrics = accuracy.score(ground_truth, results, image_ids)
    document = {
        "onnx_sha256": provenance["onnx_sha256"],
        "dataset": dataset_id,
        "evaluated_samples": len(image_ids),
        "available_samples": len(available),
        "prompts": prompts,
        "metrics": metrics,
        "conditions": CONDITIONS,
    }
    write_json(args.output, document)
    print(json.dumps(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
