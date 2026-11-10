# SigLIP 2 ViT-B/16

![SigLIP 2 ViT-B/16 model hero](hero.png)

## Overview

SigLIP 2 ViT-B/16 is Google's image-text model, trained with a sigmoid loss.
It embeds images and text in one 768-value space, so an image and a caption
that match get similar embeddings. It runs as two components: an image
encoder, which takes a 224×224 RGB image, and a text encoder, which takes a
64-token prompt. Each component returns embeddings that an application can use
for image-text search or zero-shot classification. Encode a set of class
descriptions once and reuse those text embeddings for each image, or encode new
descriptions as the application needs them.

## Inputs and outputs

**Image input.** Resize the image straight to 224×224 with bilinear
resampling, without keeping its aspect ratio or cropping. Scale the RGB values
to 0–1, subtract 0.5, and divide by 0.5. Pass this image
as a 224×224×3 (height, width, channel) float tensor, with Neat's resize, color
conversion, and normalization turned off.

**Text input.** Lowercase the prompt and tokenize it with SigLIP 2's Gemma
tokenizer (`tokenizer.json` in the source checkpoint), which appends the
end-of-sequence token, then pad it to 64 tokens with the padding token. On the
CPU, replace each token ID with its 768-value row of the checkpoint's token
embedding table, `text_model.embeddings.token_embedding.weight`. This gives the
64×768 float tensor `token_embeds`; pass it as a 1×64×768 tensor with height,
width, channel (HWC) axes.

**Outputs.** `image_embeds` is the image's 768-value embedding, and
`text_embeds` is the prompt's 768-value embedding.

1. Divide `image_embeds` and `text_embeds` by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
2. For zero-shot classification, encode descriptions for each candidate class and select the class with the highest similarity. To ensemble several descriptions of one class, normalize each text embedding, average them, then normalize the average.

The encoders have no fixed class list. ImageNet evaluation uses all 1,000
classes in the standard order of sorted WordNet IDs.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
