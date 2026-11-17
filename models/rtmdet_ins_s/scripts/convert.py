"""Export RTMDet-Ins S from its OpenMMLab checkpoint and check ONNX against PyTorch.

The export is MMDetection's tensor-mode forward pass, `model._forward`: the network up to its raw outputs, before
the CPU post-processing (`predict_by_feat`) that selects detections and builds their masks. Its input is the
letterboxed image normalized as MMDetection's DetDataPreprocessor does, in BGR order.
"""

import argparse
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth"
# MMDetection's own configuration for this checkpoint, shipped inside the pinned mmdet wheel.
CONFIG = "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py"
MMDET = "3.3.0"
IMAGE_SIZE = 640
CLASS_COUNT = 80
# Each prior's dynamic mask head: three 1x1 convolutions over 8 mask features plus 2 relative coordinates.
KERNEL_PARAMETERS = 169
MASK_FEATURES = 8
STRIDES = (8, 16, 32)
LEVELS = [IMAGE_SIZE // stride for stride in STRIDES]
INTERFACE = {
    "input": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    **{f"cls_score_{i}": [1, CLASS_COUNT, size, size] for i, size in enumerate(LEVELS)},
    **{f"bbox_pred_{i}": [1, 4, size, size] for i, size in enumerate(LEVELS)},
    **{f"kernel_pred_{i}": [1, KERNEL_PARAMETERS, size, size] for i, size in enumerate(LEVELS)},
    "mask_feat": [1, MASK_FEATURES, LEVELS[0], LEVELS[0]],
}
OUTPUTS = [name for name in INTERFACE if name != "input"]
SEEDS = range(123, 127)


class _CompiledOpsStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve `mmcv.ops` and its submodules as empty placeholders.

    mmcv-lite has no compiled ops and full mmcv has no wheel for this environment. MMDetection imports them at
    module level, but building RTMDet-Ins and running its tensor-mode forward pass uses none of them.
    """

    class _Module(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {"__init__": lambda self, *args, **kwargs: None})

    def find_spec(self, fullname, path, target=None):
        if fullname == "mmcv.ops" or fullname.startswith("mmcv.ops."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = self._Module(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected the input and outputs {list(INTERFACE)}, found {[value.name for value in values]}")
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
        # Box distances are in input pixels. Rounding through the network moves a few of them by up to 0.023 px on
        # random inputs, so they keep a sub-pixel absolute bound; every other output stays within 1e-3.
        atol = 0.05 if name.startswith("bbox_pred") else 1e-3
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=atol, err_msg=name)


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"input": INTERFACE["input"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded export")
    onnx.save(model, path)


def build_upstream(checkpoint):
    """MMDetection's RTMDet-Ins S from its packaged configuration, with the checkpoint's weights loaded strictly."""
    sys.meta_path.insert(0, _CompiledOpsStub())
    import mmdet
    import torch
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmengine.config import Config

    if mmdet.__version__ != MMDET:
        raise RuntimeError(f"Expected MMDetection {MMDET}, found {mmdet.__version__}")
    register_all_modules()
    config = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim" / "configs" / CONFIG))
    # The backbone's ImageNet initialization would download weights the checkpoint replaces.
    config.model.backbone.init_cfg = None
    model = MODELS.build(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.cpu().float().eval()


def convert(source, output_dir):
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    upstream = build_upstream(source)

    class TensorMode(torch.nn.Module):
        """MMDetection's tensor-mode forward pass, with its nested outputs flattened in OUTPUTS order."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, value):
            scores, boxes, kernels, mask_features = self.model._forward(value)
            return (*scores, *boxes, *kernels, mask_features)

    reference = TensorMode(upstream).eval()

    def synthetic(seed):
        # Normalized pixels: (0..255 - mean) / std spans about -2.2 to 2.7.
        values = np.random.RandomState(seed).uniform(-2.2, 2.7, INTERFACE["input"]).astype(np.float32)
        return torch.from_numpy(values)

    with torch.inference_mode():
        torch.onnx.export(
            reference,
            (synthetic(SEEDS[0]),),
            str(output),
            input_names=["input"],
            output_names=OUTPUTS,
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = synthetic(seed)
        with torch.inference_mode():
            expected = [value.numpy() for value in reference(sample)]
        compare_outputs(expected, session.run(OUTPUTS, {"input": sample.numpy()}))
    print(f"Synthetic-input PyTorch/ONNX check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
