# Parakeet TDT 0.6B v3

![Parakeet TDT 0.6B v3 model hero](hero.png)

## Overview

Parakeet TDT 0.6B v3 turns speech into text, for captions, transcripts of recordings, and voice interfaces that can wait a few seconds. It recognizes 25 European languages without being told which, and returns punctuated text. Every 2 seconds it reads a 14-second window of audio, 10 seconds before and 2 seconds after the new audio, and keeps the text for the new 2 seconds only. The package holds one compiled encoder and two small CPU models that turn its output into text, so the first text arrives about 4 seconds after speech starts.

## Inputs

One 14-second window of 16 kHz mono audio as 128-bin log-mel features: 10 seconds of history, 2 seconds of new audio, and 2 seconds of lookahead.

Preprocessing:

1. Apply preemphasis 0.97, then a 512-point transform with a 400-sample symmetric Hann window and a 160-sample hop, using the filters and window in `frontend.npz`.
2. Take the log of the 128 mel energies, and normalize each bin over the window's real frames.
3. Pad the window to 1,400 frames and mark the padding in the mask inputs, for the shorter windows at the start and end of a recording.
4. Move the window forward by 2 seconds for each call.

## Outputs

| Output | Size | Holds |
| --- | --- | --- |
| `encoded` | 175×640 | One row of 640 values per 80 ms of the window, for the decoder |

Postprocessing:

1. Keep the 25 rows of `encoded` that cover the new 2 seconds.
2. For each kept row, run `joiner_cpu_int8.onnx`, and read `token_logits` and `duration_logits`.
3. After each token other than blank, token 8192, run `predictor_cpu_int8.onnx` on it with `hidden` and `cell`, and keep `prediction`, `next_hidden`, and `next_cell`.
4. Move forward by the predicted number of rows, and carry what is left over into the next window.
5. Turn the tokens into text with `tokenizer.json`.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
