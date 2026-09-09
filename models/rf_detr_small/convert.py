"""Export RF-DETR and smoke-check ONNX against the original PyTorch model."""

import argparse
import json
import types
from copy import deepcopy
from pathlib import Path

import jsonschema
import numpy as np
import onnx
import yaml
from model_registry.fetch import Fetcher
from model_registry.model import find_schema
from PIL import Image

INTERFACE = {"input": [1, 3, 512, 512], "dets": [1, 300, 4], "labels": [1, 300, 91]}
OUTPUTS = ("dets", "labels")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("RF-DETR export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in model.graph.input] != ["input"] or [value.name for value in model.graph.output] != list(
        OUTPUTS
    ):
        raise ValueError("Expected input, dets and labels in that order")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        if reference.shape != candidate.shape or list(reference.shape) != INTERFACE[name]:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"Non-finite output: {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def convert(source, output_dir):
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    torch.manual_seed(123)
    image = Image.fromarray(np.random.RandomState(123).randint(0, 256, (512, 512, 3), dtype=np.uint8))
    if source.suffix == ".safetensors":
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        processor = AutoImageProcessor.from_pretrained(source.parent, local_files_only=True)
        detector, loading = AutoModelForObjectDetection.from_pretrained(
            source.parent,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint does not match the model: {loading}")

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, value):
                result = self.model(pixel_values=value, return_dict=True)
                return result.pred_boxes, result.logits

        reference = Wrapper(detector).cpu().float().eval()
        sample = processor(images=image, return_tensors="pt")["pixel_values"]
        prepared = deepcopy(reference)
        embeddings = [m for m in prepared.modules() if m.__class__.__name__ == "RfDetrDinov2Embeddings"]
        if len(embeddings) != 1:
            raise ValueError("Expected one positional-encoding module")
        module = embeddings[0]
        with torch.no_grad():
            dummy = torch.zeros(1, 1025, module.position_embeddings.shape[-1])
            fixed = module.interpolate_pos_encoding(dummy, 512, 512)
        module.register_buffer("fixed_position_encoding", fixed.detach().clone())
        module.interpolate_pos_encoding = types.MethodType(lambda self, *args: self.fixed_position_encoding, module)
        with torch.inference_mode():
            torch.onnx.export(
                prepared,
                (sample,),
                str(output),
                input_names=["input"],
                output_names=list(OUTPUTS),
                opset_version=17,
                dynamo=False,
            )
    else:
        from rfdetr import RFDETRSmall
        from rfdetr.export.main import make_infer_image

        if source.suffix != ".pth":
            raise ValueError("Expected a .pth or .safetensors checkpoint")
        detector = RFDETRSmall(pretrain_weights=str(source), resolution=512, device="cpu")
        reference = deepcopy(detector.model.model).cpu().float().eval()
        image_path = output_dir / "synthetic.png"
        image.save(image_path)
        sample = make_infer_image(str(image_path), (512, 512), 1, "cpu")
        exported = Path(
            detector.export(
                output_dir=str(output_dir),
                infer_dir=str(image_path),
                shape=(512, 512),
                batch_size=1,
                dynamic_batch=False,
                opset_version=17,
                format="onnx",
                fp16=False,
                verbose=False,
            )
        )
        if exported.resolve() != output.resolve():
            exported.rename(output)
        image_path.unlink()

    validate_export(output)
    with torch.inference_mode():
        expected = reference(sample)
    if isinstance(expected, dict):
        expected = (expected["pred_boxes"], expected["pred_logits"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    compare_outputs([value.numpy() for value in expected], session.run(list(OUTPUTS), {"input": sample.numpy()}))
    print(f"Synthetic-input PyTorch/ONNX smoke check passed: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = yaml.safe_load(args.config.read_text())["source"]
    schema = json.loads(find_schema(Path(__file__).parent).read_text())
    jsonschema.validate(source, {"$ref": "#/$defs/source", "$defs": schema["$defs"]})
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
