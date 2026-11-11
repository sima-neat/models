# Qwen3-VL-2B Instruct

![Qwen3-VL-2B Instruct model hero](hero.png)

## Overview

Qwen3-VL-2B Instruct is a vision-language model from Alibaba Cloud with approximately 2 billion parameters. It combines text and image understanding to answer questions about pictures, describe visual content, and hold text-based conversations. This package runs locally on SiMa.ai Modalix and contains the language model and image encoder together. It accepts text-only or image-plus-text requests and generates responses that can be displayed as complete answers or streamed to the application.

*Image resolution and maximum context length can be changed when recompiling the model.*

## Inputs and outputs

1. Supply the question as `prompt`, with prior messages when continuing a conversation.
2. For an image request, decode `image` as RGB. Let the GenAI API apply the bundled processor settings for the 448 × 448 input; do not manually normalize pixels.
3. Read the generated `text` as a complete response or incremental tokens. Start a new conversation when previous messages should no longer influence the answer.

## Building applications

Run the package with the [Neat GenAI Model API](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
