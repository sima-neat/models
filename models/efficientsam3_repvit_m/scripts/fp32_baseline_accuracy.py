"""Score the FP32 ONNX from surgery.export_model, before prepare_model replaces decoder functions.

Use requirements-accuracy.txt and onnxruntime (or onnxruntime-gpu >= 1.21 for CUDA). The model's accuracy.py supplies
preprocessing, prompt tokenization, query selection, mask resizing and COCO scoring. Default: the same 100
lowest-ID images and all 80 prompts as DevKit accuracy. The registered FP32 reference is the measured output
of this runner at the deployment grid, using CUDA with TF32 disabled.

    PYTHONPATH=tools python models/efficientsam3_repvit_m/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --resources <package-root> --dataset coco-2017-val=<directory> \\
        --output fp32-baseline.json --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import accuracy
import numpy as np


def onnx_pipeline(session, names: list[str], tokenizer) -> Callable:
    """Encode each prompt once, then return inference in the same output layout as the board pipeline."""
    import onnxruntime as ort

    options = ort.RunOptions()
    options.only_execute_path_to_fetches = True
    feeds = {
        "token_one_hot": np.zeros((1, accuracy.CONTEXT, accuracy.VOCABULARY), np.float32),
        "pixel_values": np.zeros((1, 3, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE), np.float32),
        "text": np.zeros((1, accuracy.CONTEXT + 1, accuracy.FEATURES), np.float32),
    }
    texts = []
    for name in names:
        tokens = accuracy.tokens(name, tokenizer)
        features, = session.run(["text_features"], {**feeds, "token_one_hot": accuracy.one_hot(tokens)}, options)
        texts.append(accuracy.packed(features, tokens))

    def infer(image):
        pixels = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        for text in texts:
            detections, masks = session.run(
                ["detections", "mask_logits"], {**feeds, "pixel_values": pixels, "text": text}, options
            )
            yield detections, masks[0]

    return infer


def main(argv: list[str] | None = None) -> int:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True, help="package root containing tokenizer/")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=accuracy.EVALUATED_IMAGES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val" or args.limit <= 0:
        parser.error("use coco-2017-val and a positive image limit")
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        parser.error(f"{provider} is unavailable in this ONNX Runtime installation")
    if args.device == "cuda":
        ort.preload_dlls()
    providers = [(provider, {"use_tf32": "0"})] if args.device == "cuda" else [provider]
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"ONNX Runtime did not activate {provider}")
    tokenizer = accuracy.load_tokenizer(args.resources)
    result = accuracy.evaluate(lambda names: onnx_pipeline(session, names, tokenizer), Path(directory), args.limit)
    metrics = result["metrics"]
    output = {"dataset": dataset_id, "images": result["evaluated_samples"], "metrics": metrics}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
