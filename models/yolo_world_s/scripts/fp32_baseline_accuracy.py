"""Score FP32 ONNX with exactly accuracy.py's COCO inputs and decoding.

Supply the same frozen BF16 CLIP prompt archive used by the compiled detector,
so this baseline isolates detector quantization. ONNX Runtime CUDA disables
TF32. The public compiler API remains the execution path for quantized GPU work.
"""
import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import accuracy
import numpy as np
import onnxruntime as ort
from faster_coco_eval import COCO, COCOeval_faster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="BF16 CLIP embeddings, mask and category IDs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                        default="CPUExecutionProvider")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.provider == "CUDAExecutionProvider":
        # Reuse the CUDA/cuDNN libraries already installed with PyTorch.
        import torch  # noqa: F401
        ort.preload_dlls()
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        parser.error("expected coco-2017-val")
    dataset = Path(directory)
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(dataset / accuracy.ANNOTATIONS))
    available = sorted(truth.getImgIds())
    ids = available[:args.limit] if args.limit else available
    categories = np.array(sorted(truth.getCatIds()))
    with np.load(args.prompts) as archive:
        embeddings, mask = archive["text_embeddings"].copy(), archive["prompt_mask"].copy()
        np.testing.assert_array_equal(archive["category_ids"], categories)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    provider = (args.provider, {"use_tf32": "0"}) if args.provider == "CUDAExecutionProvider" else args.provider
    if args.provider not in ort.get_available_providers():
        raise RuntimeError(f"requested provider {args.provider} unavailable")
    session = ort.InferenceSession(str(args.onnx), settings, providers=[provider])
    rows = []
    start = time.monotonic()
    records = truth.loadImgs(ids)
    images = accuracy.prefetched(
        lambda row: accuracy.preprocess_image((dataset / accuracy.IMAGES / row["file_name"]).read_bytes()), records)
    for count, (record, image) in enumerate(zip(records, images, strict=True), start=1):
        outputs = session.run(None, {"images": image.transpose(2, 0, 1)[None], "text_embeddings": embeddings,
                                    "prompt_mask": mask})
        outputs = [value.transpose(0, 2, 3, 1) for value in outputs]
        rows.append(accuracy.decode(outputs, record["id"], record["width"], record["height"], categories))
        if count == 1 or count % 250 == 0:
            print(f"{count}/{len(ids)}, {(time.monotonic()-start)/count:.3f} s/image", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        results = truth.loadRes(np.concatenate(rows))
    scorer = COCOeval_faster(truth, results, "bbox")
    scorer.params.imgIds = ids
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"evaluated_samples": len(ids), "available_samples": len(available),
              "metrics": accuracy.coco_stats(scorer), "seconds": time.monotonic()-start,
              "onnx_sha256": hashlib.file_digest(args.onnx.open("rb"), "sha256").hexdigest(),
              "prompts_sha256": hashlib.file_digest(args.prompts.open("rb"), "sha256").hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
