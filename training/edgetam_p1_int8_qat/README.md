# EdgeTAM P1 INT8 QAT qualification

This directory promotes the selected target-facing EdgeTAM QAT state into the
three deployable P1 components used by Neat:

1. image-prompt/frame-zero processing;
2. recurrent `s1_p1` tracking; and
3. the mask-and-pixel-feature memory encoder used to overwrite the one memory
   slot after every frame.

The current contract is deliberately named P1.  It must not be described as
the full upstream `s7_p16` product recurrence contract.

`export_qat_catalog.py` reconstructs the fused official PyTorch model from a
content-bound qualification bundle, loads the selected state strictly, and
emits every asymmetric A8 and per-output W8 value.  ModelSDK promotion must
bind this catalogue exactly through the typed Quantization Workspace.  The
catalogue is not itself compiler or board evidence.  Independent post-training
recalibration does not preserve the qualified QAT model and is therefore
rejected.
