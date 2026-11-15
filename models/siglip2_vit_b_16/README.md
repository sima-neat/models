# SigLIP 2 ViT-B/16

![SigLIP 2 ViT-B/16 model hero](hero.png)

## Overview

SigLIP 2 ViT-B/16 matches images with text. Use it to search images with a
written query, or to sort images into classes that you describe in words,
without training. Its two encoders from Google, one for images and one for
text, each return an embedding, a list of numbers. A matching image and
description have a high cosine similarity. Your application tokenizes the text
on the CPU and compares the embeddings itself. Encode class descriptions once
and reuse their embeddings for every image.

## Inputs and outputs

**Image input.** One 224×224 RGB image for the image encoder, normalized with the mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

- With Neat: pass your frame as an image, with stretch resizing and this mean and standard deviation.
- Yourself: resize straight to 224×224 with bilinear resampling, without cropping or keeping the aspect ratio. Scale to 0–1, normalize, and pass a 224×224×3 (height, width, channel) float tensor with Neat's resize, color conversion, and normalization off.

**Text input.** One prompt for the text encoder, prepared on the CPU as the float tensor `token_embeds` of 1×64×768 (height, width, channel).

1. Lowercase the prompt and tokenize it with the packaged Gemma tokenizer `text_encoder/tokenizer.json`, which appends the end-of-sequence token. Cut longer prompts to 64 tokens and pad shorter ones to 64 with the padding token (ID 0).
2. Replace each token ID with its 768-value row from `text_encoder/token_embeddings.npy`.

**Outputs.** One tensor per encoder. Neat has no decoder for them.

| Tensor | Size | Holds |
|---|---|---|
| `image_embeds` | 768 values | the image's embedding |
| `text_embeds` | 768 values | the prompt's embedding |

1. Divide `image_embeds` and `text_embeds` by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
2. To classify, encode one description per class and select the most similar class. For several descriptions per class, normalize each embedding, average them, and normalize the average.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
