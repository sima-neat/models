# OSTrack ViT-B no-CE INT8 QAT qualification

This directory contains the reproducible promotion tooling for the batch-1,
no-candidate-elimination OSTrack deployment graph.  It does not contain model
weights, datasets, compiler outputs, or signed download URLs.

`export_qat_catalog.py` reconstructs the exact QAT topology from an immutable
OSTrack source bundle, loads a selected checkpoint strictly, and emits the
complete target-facing quantization catalogue.  The catalogue is an input to
the ModelSDK Quantization Workspace promotion step; it is not itself a `.sima`
model or a compiled artifact.

The promotion flow must preserve these values through all four evidence
levels:

1. PyTorch fake-quantized execution.
2. ONNX/QDQ preview.
3. serialized ModelSDK `.sima` execution.
4. compiled MLA simulator and Modalix board execution.

Independent AFE recalibration is not an acceptable substitute for the selected
QAT catalogue.

## Qualified Modalix deployment

The accepted Modalix deployment is the accuracy-first `r57` variant. It keeps
the representable backbone QAT paths quantized, executes semantic attention and
the 15 score/size/offset head affines in BF16, and contains exactly one MLA ELF.
It is therefore **mixed precision**, not full INT8. The content-bound silicon
results and artifact digests are recorded in `silicon_qualification.json`.

On the fixed 32-record COCO17/TrackingNet development set, Modalix `.1.246`
matched serialized AFE execution per sample, reached IoU `0.8690026236` and
Success@0.5 `0.96875`, and stayed within 0.02 IoU of both the BF16 and PyTorch
QAT references. Warmup-excluded mean latency was `17.064 ms` (`58.60 FPS`).
