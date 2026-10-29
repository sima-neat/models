# Supertonic 3

![Supertonic 3 model hero](hero.png)

## Overview

Supertonic 3 is Supertone's multilingual text-to-speech model. It turns a
sentence of up to about 180 characters into up to 13.4 seconds of 44.1 kHz
speech in one of ten voices. Two components run on the accelerator: a vector
estimator that refines a noisy latent over eight steps, and a vocoder that turns
the final latent into a waveform. Your application runs the text encoder and
duration predictor on the CPU, and the guidance and update between steps.

## Inputs and outputs

The package holds everything the pipeline reads: both compiled components,
`supertonic_runtime_data.npz`, and Supertone's CPU files and license under
`supertonic-3/`, from the same
[Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
revision the components were compiled from.

**Input.** Wrap the text in its language tag, such as `<en>Hello.</en>`, and map each
character to an index with `supertonic-3/onnx/unicode_indexer.json`, at most 192.
Read `style_ttl` and `style_dp` from a file in `supertonic-3/voice_styles/`. On
the CPU, run `supertonic-3/onnx/duration_predictor.onnx` for the duration in
seconds, divided by the speaking speed, and `supertonic-3/onnx/text_encoder.onnx`
for the text embedding.

**Between the components.** Make a latent of ceil(duration × 44100 / 3072)
frames, at most 192, from seeded standard normal noise. Pad the latent, text
embedding, and masks to 192. From `supertonic_runtime_data.npz`, written by
the vector estimator's graph surgery, stack rows (latent length − 1) and
(text length − 1) of `rope_sin_cos_by_length` as `rope_tables`. For each step k
of 8:

1. Set `time_sinusoidal` to the sines, then the cosines, of (k / 8) × 1000 × 10000^(−i/31) for i = 0 to 31.
2. Run the vector estimator once with two input sets stacked: first the text embedding, the voice's `style_ttl`, and `conditional_style_key`; then `unconditional_text`, `unconditional_style_value`, and `unconditional_style_key` from the runtime data.
3. Add (4 × the first `velocity` − 3 × the second) / 8 to the latent, then multiply by `latent_mask`.

**Outputs.** Pass the final latent to the vocoder as `latent`. Its `wav_frames`
hold 1,152 frames of 512 samples in time order. Read them in order and keep
duration × 44100 samples.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
