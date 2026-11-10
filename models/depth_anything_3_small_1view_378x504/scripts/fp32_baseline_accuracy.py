"""Score surgery.py's FP32 ONNX with accuracy.py's NYU preprocessing and alignment.

Use --dataset nyu-depth-v2-default=<directory> for the registered JPEG/mm-PNG archive.
The measured reference uses --original-nyu-test <directory>: the official 654-image
test split extracted as test/rgb_NNNN.png and test/depth_NNNN.npy (float32 meters),
with test_indices_one_based in provenance.json. These are the original lossless
RGB and depth values from NYU's labeled MAT and official splits.mat.
"""

import argparse
import json
import time
from pathlib import Path

import accuracy
import numpy as np
import onnxruntime as ort


def original_frames(directory: Path) -> list[Path]:
    provenance = json.loads((directory / "provenance.json").read_text())
    images = sorted((directory / "test").glob("rgb_*.png"), key=lambda path: int(path.stem.removeprefix("rgb_")))
    ids = [int(path.stem.removeprefix("rgb_")) for path in images]
    if len(ids) != accuracy.FRAMES or len(set(ids)) != accuracy.FRAMES or ids != provenance["test_indices_one_based"]:
        raise ValueError(f"expected all {accuracy.FRAMES} official test IDs from provenance.json")
    return images


def evaluate_original(images, infer):
    rows = []
    started = time.monotonic()
    for count, frame in enumerate(images, start=1):
        truth = np.load(frame.with_name(frame.stem.replace("rgb_", "depth_", 1) + ".npy"))
        if truth.dtype != np.float32 or truth.shape != (480, 640):
            raise ValueError(f"expected original 480x640 float32 depth in meters: {frame}")
        rows.append(accuracy.score(infer(accuracy.preprocess_image(frame.read_bytes())), truth))
        if count == 1 or count % 100 == 0 or count == len(images):
            print(f"{count}/{len(images)} images; {(time.monotonic() - started) / count:.2f} s/image", flush=True)
    return {name: float(np.mean([row[name] for row in rows])) for name in ("abs_rel", "delta1")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help=f"{accuracy.DATASET}=<directory>")
    source.add_argument("--original-nyu-test", type=Path, help="directory of original test PNG/float32-NPY pairs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.original_nyu_test is not None:
        images = original_frames(args.original_nyu_test)
    else:
        dataset_id, directory = args.dataset.split("=", 1)
        if dataset_id != accuracy.DATASET:
            parser.error(f"--dataset must name {accuracy.DATASET}")
        images = accuracy.frames(Path(directory))
    images = accuracy.selected_frames(images, args.limit)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])

    def infer(image):
        inputs = {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None])}
        return accuracy.depth_output(session.run(["depth", "depth_conf"], inputs))

    evaluate = evaluate_original if args.original_nyu_test is not None else accuracy.evaluate
    result = {"samples": len(images), "metrics": evaluate(images, infer)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
