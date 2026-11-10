# SigLIP 2 ViT-B/16

![SigLIP 2 ViT-B/16 model hero](hero.png)

## Overview

SigLIP 2 ViT-B/16 is Google's image-text model, trained with a sigmoid loss.
It embeds images and text in one 768-value space, so an image and a caption
that match get similar embeddings. It runs as two components: an image
encoder, which takes a 224×224 RGB image, and a text encoder, which takes a
64-token prompt. The image encoder also returns zero-shot scores for the 1,000
ImageNet classes, built from OpenAI's class names and 7 prompt templates, so an
application can classify those classes without the text encoder. Use the text
encoder to compare images with prompts of your own.

## Inputs and outputs

**Image input.** Resize the image straight to 224×224 with bilinear
resampling, without keeping its aspect ratio or cropping. Scale the RGB values
to 0–1, subtract 0.5, and divide by 0.5. The accuracy script passes this image
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
`text_embeds` is the prompt's. `logits` holds 1,000 ImageNet class scores: the
model's logit scale times the cosine similarity between the image and each
class, plus its logit bias.

1. Divide `image_embeds` and `text_embeds` by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
2. To classify ImageNet classes, apply a sigmoid to each of the `logits` for that class's probability, and take the highest.

Class index i is the i-th ImageNet-1k WordNet ID in sorted order, which is the
standard ImageNet-1k class order.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
