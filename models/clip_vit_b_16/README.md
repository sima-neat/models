# CLIP ViT-B/16

![CLIP ViT-B/16 model hero](hero.png)

## Overview

CLIP ViT-B/16 is OpenAI's image-text model. Its separate image and text
encoders map 224×224 RGB images and 77-token prompts to 512-value embeddings
in a shared space. Use them for image-text search or zero-shot classification.
Encode class descriptions once and reuse their embeddings for each image,
or encode new descriptions as needed. Matching images and descriptions have
higher cosine similarity.

## Inputs and outputs

**Image input.** Resize the image so its shorter side is 224 pixels, with
bicubic resampling, and crop the center 224×224. Scale the RGB values to 0–1,
subtract the mean (0.48145466, 0.4578275, 0.40821073) and divide by the
standard deviation (0.26862954, 0.26130258, 0.27577711). Pass this image as a
224×224×3 (height, width, channel) float tensor, with Neat's resize, color
conversion, and normalization turned off.

**Text input.** Use the packaged `text_encoder/tokenizer.json` to tokenize the
prompt, padded to 77 tokens with the end-of-text token. On CPU, replace each ID
with its 512-value row from `text_encoder/token_embeddings.npy`. Pass the
resulting float tensor `token_embeds` as 1×77×512 with HWC axes.

**Outputs.** `image_embeds` is the image's 512-value embedding.
`text_embeds` holds one 512-value row per token.

1. Take the row of `text_embeds` at the first end-of-text token (ID 49407).
2. Divide `image_embeds` and that row by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
3. For classification, select the class with the highest similarity. For multiple descriptions per class, normalize each embedding, average them, then normalize the average.

The encoders have no fixed class list. ImageNet evaluation uses all 1,000
classes in the standard order of sorted WordNet IDs.

**Speed measurements.** The image route letterboxes into 1280×720, then Neat
stretches the entire padded frame to 224×224 and normalizes it. Accuracy uses
the resize/crop route described above. Text timing starts from prepared
embeddings; CPU tokenization and lookup are excluded. Results are separate
per encoder. The package ships `licenses/LICENSE`.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
