# Supertonic 3

![Supertonic 3 model hero](hero.png)

## Overview

Supertonic 3 is Supertone's text-to-speech model. It reads a sentence of up to
about 180 characters in one of 31 languages and speaks it in one of ten voices,
five female and five male, as up to 13.4 seconds of 44.1 kHz audio. Two
components run on the accelerator: a vector estimator, which turns noise into
speech features over eight steps, and a vocoder, which turns those features
into sound. Your application runs two small text models on the CPU, and a short
update between the steps.

## Inputs and outputs

The ZIP package holds everything below: both components,
`supertonic_runtime_data.npz`, and Supertone's text models, voices and license
under `supertonic-3/`.

**Input.**

1. Wrap the text in its language tag, such as `<en>Hello.</en>`.
2. Map each character to a number with `supertonic-3/onnx/unicode_indexer.json`, up to 192 characters.
3. Load `style_ttl` and `style_dp` for a voice, such as `F1`, from `supertonic-3/voice_styles/`.
4. Run `supertonic-3/onnx/duration_predictor.onnx` for the speech length in seconds, and divide it by the speaking speed.
5. Run `supertonic-3/onnx/text_encoder.onnx` for the text embedding.

**Between the components.**

1. Make ceil(length × 44,100 / 3,072) frames of seeded random noise, at most 192, and pad the frames, text embedding and masks to 192.
2. Take rows (frames − 1) and (characters − 1) of `rope_sin_cos_by_length` in `supertonic_runtime_data.npz` as `rope_tables`.
3. For each step k from 0 to 7, set `time_sinusoidal` to the sines, then the cosines, of (k / 8) × 1000 × 10000^(−i/31), for i from 0 to 31.
4. Run the vector estimator with two input sets: the text embedding, `style_ttl` and `conditional_style_key`; then `unconditional_text`, `unconditional_style_value` and `unconditional_style_key` from the runtime data.
5. Add (4 × the first `velocity` − 3 × the second) / 8 to the frames, and multiply by `latent_mask`.

**Outputs.**

1. Pass the final frames to the vocoder as `latent`.
2. Read its `wav_frames` in order, 512 samples each.
3. Keep the first (length × 44,100) samples as the audio.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
