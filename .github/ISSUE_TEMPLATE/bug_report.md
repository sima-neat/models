---
name: Bug Report
about: Report a model that fails to download, compile, run, or match its published results
title: "[Bug] "
labels: bug
assignees: ''
---

## What happened?

Describe the observed behavior and the result you expected. Include the exact error message and how often the problem occurs.

## How can we reproduce it?

Name the model ID and build profile, the stage that fails (download, convert, `compile`, `performance`, or `accuracy`), and the command you ran. For a documentation or model browser problem, link the affected page.

## Environment

- Model Compiler version, or the `sima-cli models` version for a download
- Palette SDK version and the host or DevKit used
- Neat package version, when a benchmark fails

## Impact and workaround

Which workflow is affected? Include any workaround or last known working version, if known.

## Additional diagnostics (optional)

Include only details that help explain the failure, such as the MPK manifest fields, input shapes, the measured and published values, redacted logs, or a stack trace. Use public or synthetic inputs where possible. Redact credentials, private URLs, and customer data.
