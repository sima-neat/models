# DepthART Base strict-INT8 QAT source recipe

This directory contains the model-specific part of the DepthART Base QAT flow.
The reusable fake-quantization, exact `q8 + p64` recurrence, lifecycle, export,
and validation APIs live in [`sima-neat/qat`](https://github.com/sima-neat/qat/pull/6).

## Apply the equivalent architecture

Start from upstream commit `0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c` and apply:

```bash
git apply /path/to/training/depthart_base/depthart_dynamic_p64_source.patch
```

The patch changes how the selective scan is expressed, not the real-valued
function. It uses grouped convolutions for static projections, evaluates the
affine recurrence with an associative tree, and removes only the exact first
chunk identity `a * 0 + b = b`. Source tests compare the new path with the
original recurrence before quantization.

## Add QAT in one call

```python
from sima_qat import (
    enable_depthart_dynamic_p64_fake_quant,
    freeze_depthart_dynamic_p64,
    prepare_depthart_dynamic_p64,
    write_depthart_dynamic_p64_compile_profile,
)

model = load_depthart_base().cuda()
prepare_depthart_dynamic_p64(model, safety_margin=1.05, observe=True)

enable_depthart_dynamic_p64_fake_quant(model, False)
with torch.no_grad():
    for sample in calibration_loader:
        model(sample["image"].cuda())

freeze_depthart_dynamic_p64(model)
enable_depthart_dynamic_p64_fake_quant(model, True)
train_with_task_and_distillation_losses(model, training_loader)

# Export ordinary ONNX Q/DQ + Mul/Add and its hash-bound compiler profile.
torch.onnx.export(
    model.eval(),
    (torch.zeros(1, 3, 448, 576, device="cuda"),),
    "depthart.onnx",
    input_names=["image"],
    output_names=["depth"],
    opset_version=17,
)
write_depthart_dynamic_p64_compile_profile(
    model, "depthart.onnx", "depthart.depthart_p64.json"
)
```

`qat.yaml` pins the source, data split, exact QAT revision, quantization policy,
losses, export hashes, and acceptance gates. The private NYUv2/Cityscapes data
is intentionally not distributed from this repository.

## Loss policy

Use standardized output and spatial-gradient distillation from a frozen FP32
teacher, intermediate feature preservation at encoder stages 0/2/4/6, and
paired NYUv2 affine inverse-depth supervision. Final refinement may add the
metric-aligned AbsRel/log-RMSE/delta1 surrogate exposed by the QAT runner.
The recurrence fake quantizers use exact target integer math in the forward
pass and STE gradients in the backward pass.

The 100-image host isolation result (`delta1=0.919875`, `AbsRel=0.088424`) covers
strict INT8 recurrence with the remainder in FP32. It is not a silicon claim;
promotion requires the compiled precision audit and all 654 board samples.
