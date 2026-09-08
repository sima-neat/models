"""Convert verified RF-DETR checkpoints and compare ONNX with unmodified PyTorch."""

from __future__ import annotations

import argparse
import json
import types
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from PIL import Image

PACKAGE_VERSION = "1.9.4"
TRANSFORMERS_VERSION = "5.16.1"
INTERFACE = {"input": [1, 3, 512, 512], "dets": [1, 300, 4], "labels": [1, 300, 91]}
OUTPUTS = ("dets", "labels")
TOLERANCE = 1e-3


def validate_export(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise RuntimeError("RF-DETR export must use ONNX opset 17")
    if [value.name for value in model.graph.input] != ["input"]:
        raise RuntimeError("RF-DETR export must have one input named input")
    if [value.name for value in model.graph.output] != list(OUTPUTS):
        raise RuntimeError("RF-DETR export must return dets then labels")
    for value in (*model.graph.input, *model.graph.output):
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise RuntimeError(f"unexpected RF-DETR tensor {value.name}: dtype or shape does not match")


def compare_outputs(sample: str, expected: list[np.ndarray], actual: list[np.ndarray]) -> list[dict[str, Any]]:
    if len(expected) != len(OUTPUTS) or len(actual) != len(OUTPUTS):
        raise RuntimeError("RF-DETR comparison requires dets and labels")
    records = []
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        same_shape = reference.shape == candidate.shape == tuple(INTERFACE[name])
        finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
        record: dict[str, Any] = {
            "sample": sample,
            "output": name,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": finite,
            "rtol": TOLERANCE,
            "atol": TOLERANCE,
            "passed": bool(same_shape and finite and np.allclose(candidate, reference, rtol=TOLERANCE, atol=TOLERANCE)),
        }
        if same_shape and finite:
            error = candidate.astype(np.float64) - reference.astype(np.float64)
            record.update(
                max_abs_error=float(np.max(np.abs(error))),
                mean_abs_error=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))),
            )
        records.append(record)
    return records


def _images(source_dir: Path, output_dir: Path) -> list[tuple[str, Path]]:
    real_image = source_dir / "validation.png"
    if not real_image.is_file():
        raise FileNotFoundError("verified validation.png is required beside the checkpoint")
    rng = np.random.RandomState(123)
    images = []
    for name, shape in (("seed123-native512", (512, 512, 3)), ("seed123-resized", (375, 640, 3))):
        path = output_dir / f"{name}.png"
        Image.fromarray(rng.randint(0, 256, size=shape, dtype=np.uint8)).save(path)
        images.append((name, path))
    return [*images, ("coco-sample", real_image)]


def _hugging_face(source: Path, output: Path, images: list[tuple[str, Path]], report: dict[str, Any]):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Hugging Face converter requires transformers=={TRANSFORMERS_VERSION}")
    if source.name != "model.safetensors":
        raise ValueError("the pinned Hugging Face checkpoint must be named model.safetensors")
    processor = AutoImageProcessor.from_pretrained(source.parent, local_files_only=True)
    model, loading = AutoModelForObjectDetection.from_pretrained(
        source.parent,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Hugging Face checkpoint does not match the model: {loading}")
    model = model.cpu().float().eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, value):
            result = self.detector(pixel_values=value, return_dict=True)
            return result.pred_boxes, result.logits

    reference = Wrapper(model).eval()
    samples = []
    for name, path in images:
        with Image.open(path) as image:
            sample = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        samples.append((name, sample.cpu().float()))
    prepared = deepcopy(reference)
    adapted = []
    for name, module in prepared.named_modules():
        if module.__class__.__name__ == "RfDetrDinov2Embeddings":
            dummy = torch.zeros(1, 1025, module.position_embeddings.shape[-1])
            with torch.no_grad():
                fixed = module.interpolate_pos_encoding(dummy, 512, 512).detach().clone()
            module.register_buffer("fixed_512_position_encoding", fixed)

            def fixed_encoding(self, embeddings, height, width):
                return self.fixed_512_position_encoding

            module.interpolate_pos_encoding = types.MethodType(fixed_encoding, module)
            adapted.append(name)
    if len(adapted) != 1:
        raise RuntimeError(f"expected one RF-DETR positional encoding module, found {adapted}")
    report["preparation"] = {"fixed_512_position_encoding": adapted, "reference": "unmodified PyTorch"}
    with torch.inference_mode():
        torch.onnx.export(
            prepared,
            (samples[0][1],),
            str(output),
            input_names=["input"],
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    return reference, samples


def _roboflow(source: Path, output: Path, images: list[tuple[str, Path]], report: dict[str, Any]):
    from rfdetr import RFDETRSmall
    from rfdetr.export.main import make_infer_image

    if version("rfdetr") != PACKAGE_VERSION:
        raise RuntimeError(f"Roboflow converter requires rfdetr=={PACKAGE_VERSION}")
    detector = RFDETRSmall(pretrain_weights=str(source), resolution=512, device="cpu")
    reference = deepcopy(detector.model.model).cpu().float().eval()
    samples = [(name, make_infer_image(str(path), (512, 512), 1, "cpu")) for name, path in images]
    path = Path(
        detector.export(
            output_dir=str(output.parent),
            infer_dir=str(images[0][1]),
            shape=(512, 512),
            batch_size=1,
            dynamic_batch=False,
            opset_version=17,
            format="onnx",
            fp16=False,
            verbose=False,
        )
    ).resolve()
    if path.parent != output.parent:
        raise RuntimeError("Roboflow exporter wrote outside the requested output directory")
    if path != output:
        path.rename(output)
    report["preparation"] = {"exporter": f"rfdetr=={PACKAGE_VERSION}", "reference": "unmodified PyTorch"}
    return reference, samples


def export_model(checkpoint: Path, output_dir: Path, report: dict[str, Any] | None = None) -> Path:
    import onnxruntime as ort
    import torch

    report = report if report is not None else {}
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    images = _images(checkpoint.parent, output_dir)
    output = output_dir / "model.onnx"
    if checkpoint.suffix == ".safetensors":
        report["route"] = "hugging_face_experimental"
        reference, samples = _hugging_face(checkpoint, output, images, report)
    else:
        if checkpoint.suffix != ".pth":
            raise ValueError("RF-DETR conversion requires a .pth or .safetensors checkpoint")
        report["route"] = "roboflow"
        reference, samples = _roboflow(checkpoint, output, images, report)
    validate_export(output)
    report["onnx"] = {"file": output.name, "opset": 17, "interface": INTERFACE, "checked": True}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    records = report.setdefault("comparisons", [])
    saved: dict[str, np.ndarray] = {}
    for index, (name, sample) in enumerate(samples):
        value = sample.numpy()
        if value.shape != tuple(INTERFACE["input"]) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid preprocessed input for {name}")
        with torch.inference_mode():
            expected = reference(sample)
        if isinstance(expected, dict):
            expected = (expected["pred_boxes"], expected["pred_logits"])
        expected = [tensor.detach().cpu().numpy() for tensor in expected]
        saved[f"input_{index}"] = value
        for output_name, tensor in zip(OUTPUTS, expected, strict=True):
            saved[f"{output_name}_{index}"] = tensor
        actual = session.run(list(OUTPUTS), {"input": value})
        records.extend(compare_outputs(name, expected, actual))
    np.savez(output_dir / "reference.npz", **saved)
    report["reference"] = {"file": "reference.npz", "samples": [name for name, _ in samples]}
    if not all(record["passed"] for record in records):
        raise RuntimeError("ONNX differs from unmodified PyTorch; proposal order is not altered or ignored")
    return output


def convert(source: Path, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema_version": 1, "status": "failed", "comparisons": []}
    try:
        output = export_model(source, output_dir, report)
        report["status"] = "passed"
        return output
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Verified local checkpoint entrypoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for conversion artifacts")
    args = parser.parse_args()
    print(convert(args.source, args.output_dir))


if __name__ == "__main__":
    main()
