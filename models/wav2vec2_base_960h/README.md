# Wav2Vec2 Base 960h

![Wav2Vec2 Base 960h model hero](hero.png)

## Overview

Wav2Vec2 Base 960h transcribes English speech into text using a connectionist temporal classification (CTC) model. It is useful for processing speech recordings and for prototyping voice interfaces. The package runs the convolutional feature frontend on the CPU and the transformer encoder on the accelerator, then decodes character scores on the CPU. This fixed-shape build processes overlapping five-second windows and uses greedy decoding; transcription quality depends on recording conditions and window boundaries.

## Inputs

The application supplies mono, 16 kHz speech as floating-point PCM samples.

Preprocessing:

1. Divide the recording into 80,000-sample windows at 64,000-sample intervals, and pad the final window with zeros.
2. Normalize each padded window to zero mean and unit variance using `sqrt(variance + 1e-7)`.
3. Run the included `frontend.onnx` CPU graph with float32 `input_values` shaped `[1, 80000]`.
4. Send its float32 output, shaped `[1, 249, 768]`, to the accelerator component.

## Outputs

| Output | Size | Holds |
| --- | --- | --- |
| `logits` | `[1, 249, 32]` | Per-frame scores for 32 CTC vocabulary tokens. |

Postprocessing:

1. Limit the final padded window to `max(0, floor((samples - 400) / 320) + 1)` output frames.
2. Discard the first 25 frames of each window after the first and the last 24 frames of each window before the last.
3. Select the highest-scoring token in each remaining frame and collapse consecutive repeated IDs, including across windows.
4. Remove the `<pad>` blank token, map IDs with the included `vocab.json`, and replace `|` with a space.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
