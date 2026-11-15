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

## Inputs

| Input | Size | Holds |
|---|---|---|
| image | 224×224×3 (height, width, channel) | a normalized RGB image, for the image encoder |
| `token_embeds` | 1×77×512 | the prompt's token embeddings, for the text encoder |

Image preprocessing:

1. Resize with bicubic resampling so the shorter side is 224.
2. Crop the center 224×224.
3. Divide by 255.
4. Subtract the mean (0.48145466, 0.4578275, 0.40821073).
5. Divide by the standard deviation (0.26862954, 0.26130258, 0.27577711).

Text preprocessing:

1. Tokenize the prompt with the packaged `text_encoder/tokenizer.json`.
2. Cut it to 77 tokens, or pad it to 77 with the end-of-text token (ID 49407).
3. Replace each token ID with its 512-value row from `text_encoder/token_embeddings.npy`.

## Outputs

| Output | Size | Holds |
|---|---|---|
| `image_embeds` | 512 values | the image's embedding |
| `text_embeds` | 77 rows of 512 values | one row per token |

Postprocessing:

1. Take the row of `text_embeds` at the first end-of-text token.
2. Divide `image_embeds` and that row by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
3. To classify, encode one description per class and select the most similar class.
4. For several descriptions per class, normalize each embedding, average them, and normalize the average.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
