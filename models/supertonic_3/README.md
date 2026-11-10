# Supertonic 3

![Supertonic 3 model hero](hero.png)

## Overview

Supertonic 3 is Supertone's text-to-speech model. It speaks a sentence of up
to about 180 characters, in one of 31 languages, in one of ten voices, as up to
13.4 seconds of 44.1 kHz audio. On the accelerator, a vector estimator turns
noise into speech features over eight steps, and a vocoder turns them into
sound. Your application runs two small text models and the update between
steps on the CPU.

## Inputs and outputs

The ZIP package holds both components, `supertonic_runtime_data.npz`, and
Supertone's text models, voices and license under `supertonic-3/`.

**Input.**

1. Normalize the text as Supertone's code does: Unicode NFKD, no emoji, plain quotes and dashes, single spaces, and a final period if it has no ending punctuation.
2. Wrap it in its language tag, such as `<en>Hello.</en>`.
3. Map each character to a number with `supertonic-3/onnx/unicode_indexer.json`, up to 192; it has no precomposed letters such as `é`.
4. Load `style_ttl` and `style_dp` for a voice, such as `F1`, from `supertonic-3/voice_styles/`.
5. Run `supertonic-3/onnx/duration_predictor.onnx` for the speech length in seconds, divided by the speed.
6. Run `supertonic-3/onnx/text_encoder.onnx` for the text embedding.

**Between the components.**

1. Make ceil(length × 44,100 / 3,072) frames of seeded random noise, at most 192, and pad the frames, text embedding and masks to 192.
2. Take rows (frames − 1) and (characters − 1) of `rope_sin_cos_by_length` in `supertonic_runtime_data.npz` as `rope_tables`.
3. For step k = 0 to 7, set `time_sinusoidal` to the sines, then cosines, of (k / 8) × 1000 × 10000^(−i/31), i = 0 to 31.
4. Run the vector estimator with two input sets: the text embedding, `style_ttl` and `conditional_style_key`; then `unconditional_text`, `unconditional_style_value` and `unconditional_style_key` from the runtime data.
5. Add (4 × the first `velocity` − 3 × the second) / 8 to the frames, and multiply by `latent_mask`.

**Outputs.**

1. Pass the final frames to the vocoder as `latent`.
2. Read its `wav_frames` in order.
3. Keep the first (length × 44,100) samples as the audio.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
