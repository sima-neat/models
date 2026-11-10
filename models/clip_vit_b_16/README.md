# CLIP ViT-B/16

![CLIP ViT-B/16 model hero](hero.png)

## Overview

CLIP ViT-B/16 is OpenAI's image-text model. It embeds images and text in one
512-value space, so an image and a caption that match get similar embeddings.
It runs as two components: an image encoder, which takes a 224×224 RGB image,
and a text encoder, which takes a 77-token prompt. Each component returns
embeddings that an application can use for image-text search or zero-shot
classification. Encode a set of class descriptions once and reuse those text
embeddings for each image, or encode new descriptions as the application needs
them.

## Inputs and outputs

**Image input.** Resize the image so its shorter side is 224 pixels, with
bicubic resampling, and crop the center 224×224. Scale the RGB values to 0–1,
subtract the mean (0.48145466, 0.4578275, 0.40821073) and divide by the
standard deviation (0.26862954, 0.26130258, 0.27577711). Pass this image as a
224×224×3 (height, width, channel) float tensor, with Neat's resize, color
conversion, and normalization turned off.

**Text input.** Tokenize the prompt with CLIP's tokenizer (`tokenizer.json` in
the source checkpoint), padded to 77 tokens with the end-of-text token. On the
CPU, replace each token ID with its 512-value row of the checkpoint's token
embedding table, `text_model.embeddings.token_embedding.weight`. This gives the
77×512 float tensor `token_embeds`; pass it as a 1×77×512 tensor with height,
width, channel (HWC) axes.

**Outputs.** `image_embeds` is the image's 512-value embedding.
`text_embeds` holds one 512-value row per token.

1. Take the row of `text_embeds` at the first end-of-text token (ID 49407).
2. Divide `image_embeds` and that row by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
3. For zero-shot classification, encode descriptions for each candidate class and select the class with the highest similarity. To ensemble several descriptions of one class, normalize each text embedding, average them, then normalize the average.

The encoders have no fixed class list. ImageNet evaluation uses all 1,000
classes in the standard order of sorted WordNet IDs.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
