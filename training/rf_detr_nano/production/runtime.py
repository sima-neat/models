"""Public PyNeat local qualification route for the frozen QAT tensor ABI.

The source-faithful ordered image program is mandatory. There is deliberately
no fallback to the old affine preprocessor, public FLOAT32 handoffs, inferred
packed-buffer offsets, or a different proposal-selection rule.

The NumPy CPU bridge/decoder is an explicit oracle implementation. Native
TopK/Gather and BoxDecode integration/performance are a separate dependency;
this module never reports these helpers as optimized kernel execution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from contract import SHAPES, input_grid, output_grids, verify_packages
from numerics import detections, top_proposals


def image_options(pyneat, contract: dict, width: int, height: int):
    options = pyneat.ModelOptions()
    program_type = getattr(pyneat, "PreprocessProgram", None)
    operation = getattr(pyneat, "PreprocessOperation", None)
    if program_type is None or operation is None or not hasattr(options.preprocess, "program"):
        raise RuntimeError(
            "this Core/PyNeat lacks the required ordered PreprocessProgram; legacy preprocessing is not equivalent"
        )
    program = program_type()
    program.stages = [getattr(operation, name) for name in contract["preprocessing"]["stages"]]
    program.rescale_factor = float(
        np.array(contract["preprocessing"]["rescale_float32_bits"], dtype=np.uint32).view(np.float32)
    )
    program.mean = contract["preprocessing"]["mean"]
    program.stddev = contract["preprocessing"]["std"]
    grid = input_grid(contract)
    program.quantization_step = float(grid.step)
    program.quantization_zero_point = grid.zero_point
    program.validate()
    options.preprocess.program = program
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.resize.width = options.preprocess.resize.height = 384
    options.preprocess.resize.scaling_type = contract["preprocessing"]["resize"]
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = contract["preprocessing"]["mean"]
    options.preprocess.normalize.stddev = contract["preprocessing"]["std"]
    return options


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"could not decode RGB image: {path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_tensor(pyneat, image: np.ndarray):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("the image route requires UINT8 HWC RGB")
    return pyneat.Tensor.from_numpy(
        image, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def proposal_inputs(pyneat, outputs: list):
    if len(outputs) != 3 or [tuple(value.shape) for value in outputs] != SHAPES["backbone"]["outputs"]:
        raise RuntimeError("backbone public outputs do not expose the declared typed QAT ABI")
    features, scores, boxes = outputs
    score_codes, box_codes = scores.to_numpy(), boxes.to_numpy()
    if features.dtype != scores.dtype or score_codes.dtype != np.int8 or box_codes.dtype != np.int8:
        raise RuntimeError(
            "raw INT8 handoff is unavailable; refusing FLOAT32 dequantize/requantize or packed-offset guessing"
        )
    selected, _ = top_proposals(score_codes, box_codes)
    proposals = pyneat.Tensor.from_numpy(selected, copy=True, memory=pyneat.TensorMemory.EV74)
    # Preserve decoder input order: selected boxes FIRST, encoder memory SECOND.
    # cvu() is the existing public safe ownership/copy path. Do not claim this
    # is the campaign's aligned zero-copy native bridge.
    return [proposals, features.cvu()]


def decode_outputs(outputs: list, grids: tuple, *, width: int, height: int, image_id: int) -> np.ndarray:
    if len(outputs) != 3:
        raise RuntimeError("decoder must return deltas, logits and box references")
    return detections([value.to_numpy() for value in outputs], grids, width=width, height=height, image_id=image_id)


class Pipeline:
    """Sequential correctness route; variable shapes rebuild safely, never drop."""

    def __init__(self, packages: dict[str, Path], contract: dict, capacity: tuple[int, int]):
        import pyneat

        verify_packages(contract, packages)
        self.pyneat = pyneat
        self.models = {
            "backbone": pyneat.Model(str(packages["backbone"]), image_options(pyneat, contract, *capacity)),
            "decoder": pyneat.Model(str(packages["decoder"])),
        }
        self.grids = output_grids(contract)
        self.capacity = capacity
        self.shape = None
        self.runners = []

    def close(self) -> None:
        errors = []
        for runner in reversed(self.runners):
            try:
                runner.close()
            except Exception as error:
                errors.append(error)
        self.runners = []
        self.shape = None
        if errors:
            raise RuntimeError("QAT runner closure failed") from errors[0]

    def __call__(self, image: np.ndarray, image_id: int) -> np.ndarray:
        height, width = image.shape[:2]
        if width > self.capacity[0] or height > self.capacity[1]:
            raise ValueError("image exceeds the declared model input capacity")
        tensor = image_tensor(self.pyneat, image)
        if self.shape != image.shape:
            self.close()
            backbone = self.models["backbone"].build([tensor])
            self.runners.append(backbone)
            try:
                outputs = backbone.run([tensor], timeout_ms=30000)
                inputs = proposal_inputs(self.pyneat, outputs)
                decoder = self.models["decoder"].build(inputs)
                self.runners.append(decoder)
                self.shape = image.shape
            except Exception:
                self.close()
                raise
        else:
            outputs = self.runners[0].run([tensor], timeout_ms=30000)
            inputs = proposal_inputs(self.pyneat, outputs)
        # Keep source outputs and input owners alive through actual completion.
        decoded = self.runners[1].run(inputs, timeout_ms=30000)
        return decode_outputs(decoded, self.grids, width=width, height=height, image_id=image_id)
