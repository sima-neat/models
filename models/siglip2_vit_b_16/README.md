# SigLIP 2 ViT-B/16

![SigLIP 2 ViT-B/16 model hero](hero.png)

## Overview

SigLIP 2 ViT-B/16 is Google's image-text model, trained with a sigmoid loss.
Its separate image and text encoders map 224×224 RGB images and 64-token prompts
to 768-value embeddings in a shared space. Use them for image-text search or
zero-shot classification. Encode class descriptions once and reuse their
embeddings for each image, or encode new descriptions as needed.

## Inputs and outputs

**Image input.** Resize the image straight to 224×224 with bilinear
resampling, without keeping its aspect ratio or cropping. Scale the RGB values
to 0–1, subtract 0.5, and divide by 0.5. Pass this image
as a 224×224×3 (height, width, channel) float tensor, with Neat's resize, color
conversion, and normalization turned off.

ImageNet accuracy uses Big Vision's evaluation geometry: bilinear resize to
256×256, then center-crop 224×224. The checkpoint's processor and compilation
fidelity checks use the direct 224×224 resize above.

**Text input.** Lowercase the prompt and tokenize it with the packaged Gemma
tokenizer `text_encoder/tokenizer.json`. It appends end-of-sequence; pad to 64
tokens with the padding token. On CPU, replace each ID with its 768-value row
from `text_encoder/token_embeddings.npy`. Pass the resulting float tensor
`token_embeds` as 1×64×768 with HWC axes.

**Outputs.** `image_embeds` is the image's 768-value embedding, and
`text_embeds` is the prompt's 768-value embedding.

1. Divide `image_embeds` and `text_embeds` by their lengths and take their dot product, the cosine similarity. A higher value means a better match.
2. For zero-shot classification, encode descriptions for each candidate class and select the class with the highest similarity. To ensemble several descriptions of one class, normalize each text embedding, average them, then normalize the average.

The encoders have no fixed class list. ImageNet evaluation uses all 1,000
classes in the standard order of sorted WordNet IDs.

**Speed measurements.** The image route letterboxes into 1280×720, then Neat
stretches the entire padded frame to 224×224 and normalizes it. Accuracy uses
the resize/crop route described above. Text timing starts from prepared
embeddings; CPU tokenization and lookup are excluded. Results are separate
per encoder. The package ships `licenses/LICENSE`.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
