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

## Inputs

| Input | Size | Holds |
|---|---|---|
| image | 224×224×3 (height, width, channel) | a normalized RGB image, for the image encoder |
| `token_embeds` | 1×64×768 | the prompt's token embeddings, for the text encoder |

Image preprocessing:

1. Resize to 256×256 with bilinear resampling and no antialiasing, without keeping the aspect ratio.
2. Crop the center 224×224.
3. Divide by 255.
4. Subtract the mean (0.5, 0.5, 0.5).
5. Divide by the standard deviation (0.5, 0.5, 0.5).

Text preprocessing:

1. Lowercase the prompt.
2. Tokenize it with the packaged `text_encoder/tokenizer.json`, which appends the end-of-sequence token.
3. Cut it to 64 tokens, or pad it to 64 with the padding token (ID 0).
4. Replace each token ID with its 768-value row from `text_encoder/token_embeddings.npy`.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `image_embeds` | 768 values | the image's embedding |
| `text_embeds` | 768 values | the prompt's embedding |

Postprocessing:

1. Divide `image_embeds` and `text_embeds` by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
2. To classify, encode one description per class and select the most similar class.
3. For several descriptions per class, normalize each embedding, average them, and normalize the average.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
