# CLIP ViT-B/16

![CLIP ViT-B/16 model hero](hero.png)

## Overview

CLIP ViT-B/16 matches images with text. Use it to search images with a written
query, or to sort images into classes that you describe in words, without
training. Its two encoders from OpenAI, one for images and one for text, each
return an embedding, a list of numbers. A matching image and description have a
high cosine similarity. Your application tokenizes the text on the CPU and
compares the embeddings itself. Encode class descriptions once and reuse their
embeddings for every image.

## Inputs and outputs

### Inputs

**Image:** One 224×224 RGB image for the image encoder, normalized with the mean (0.48145466, 0.4578275, 0.40821073) and standard deviation (0.26862954, 0.26130258, 0.27577711).

- With Neat: pass your frame as an image, with stretch resizing and this mean and standard deviation.
- Yourself: resize so the shorter side is 224 pixels with bicubic resampling, and crop the center 224×224. Scale to 0–1, normalize, and pass a 224×224×3 (height, width, channel) float tensor with Neat's resize, color conversion, and normalization off.

Neat's stretch squeezes a frame that is not square. Prefer the crop for such
frames.

**Text:** One prompt for the text encoder, prepared on the CPU as the float tensor `token_embeds` of 1×77×512 (height, width, channel).

1. Tokenize the prompt with the packaged `text_encoder/tokenizer.json`. Cut longer prompts to 77 tokens and pad shorter ones to 77 with the end-of-text token (ID 49407).
2. Replace each token ID with its 512-value row from `text_encoder/token_embeddings.npy`.

### Outputs

One tensor per encoder. Neat has no decoder for them.

| Tensor | Size | Holds |
|---|---|---|
| `image_embeds` | 512 values | the image's embedding |
| `text_embeds` | 77 rows of 512 values | one row per token |

1. Take the row of `text_embeds` at the first end-of-text token.
2. Divide `image_embeds` and that row by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
3. To classify, encode one description per class and select the most similar class. For several descriptions per class, normalize each embedding, average them, and normalize the average.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
