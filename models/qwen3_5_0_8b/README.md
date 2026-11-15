# Qwen3.5-0.8B

![Qwen3.5-0.8B model hero](hero.png)

## Overview

Qwen3.5-0.8B answers questions about an image, describes what it shows, and
holds a text conversation. Use it to caption images, answer questions about a
camera frame, or chat about a picture. Your application sends a prompt, with or
without an image, and gets the answer back as text, complete or streamed as it
is generated. It is the 0.8B size of the Qwen3.5 family from Alibaba Cloud.

*Image resolution and maximum context length can be changed when recompiling the model, with context lengths up to 8,192 tokens.*

## Inputs

One text prompt, an optional image, and the earlier messages when you continue a conversation.

- `prompt`: the question or instruction, with any source text it refers to. Let the GenAI API apply the bundled chat template.
- `image`: for an image request, an image decoded as RGB. The GenAI API applies the bundled processor settings for the 448×448 input. Do not normalize the pixels yourself.
- Earlier messages: pass them to continue a conversation. Start a new conversation when they should no longer influence the answer.

## Outputs

The answer as text.

- `text`: the generated answer, as one complete response.
- Streaming: read incremental tokens as they arrive instead of waiting for the complete response.

## Building applications

Run the package with the [Neat GenAI Model API](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
