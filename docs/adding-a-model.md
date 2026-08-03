# Adding a model

1. Create `models/<model-id>/model.yaml`. The directory and `id` must match and
   use lowercase snake case.
2. Pin every remote input to immutable content. For Hugging Face, use a full
   repository commit and the file's SHA-256:

   ```yaml
   source:
     uri: hf://models/owner/repository@0123456789abcdef0123456789abcdef01234567/path/model.onnx
     sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
     size_bytes: 123456
   ```

   `models`, `datasets`, and `spaces` are valid Hugging Face repository types.
   Branches such as `main` are rejected because they are mutable.
3. Describe the ONNX interface and preprocessing explicitly. Do not rely on a
   model card changing independently of this repository.
4. Declare surgery as `type: none` or point at a model-local hook. Keep generic
   transformations in shared tooling.
5. Add a model-specific compile recipe and declare it on each variant. Recipes
   use this command-line contract:

   ```text
   compile.py --model <verified-source> --output-dir <new-directory> --variant <variant-id>
   ```

   The recipe may perform model-specific surgery, quantization, and compilation,
   and must leave exactly one `*_mpk.tar.gz` in the output directory. It must not
   download an unpinned model or silently select a different compiler.
6. Declare targets and recipe parameters explicitly. Do not add a compiler pin
   to a model variant: all variants use the repository-wide Model Compiler
   version in `deps/manifest.json`. Unknown fields are rejected so misspellings
   cannot silently change a build.
7. Validate, fetch, and inspect the resulting CI matrix:

   ```bash
   model-registry validate models/<model-id>
   model-registry fetch models/<model-id>
   model-registry matrix --repository . --changed-path models/<model-id>/model.yaml
   model-registry resolve-compiler --repository .
   ```

The fetch command translates `hf://` identifiers to the Hugging Face resolve
endpoint, streams into a temporary file, verifies size and SHA-256, and atomically
places the content in a checksum-addressed cache. The default cache is
`~/.cache/model-registry`; set `MODEL_REGISTRY_CACHE` or pass `--cache-dir` to
change it. `HF_TOKEN` is used when present and is never printed.

Do not commit downloaded models, datasets, build directories, credentials,
signed URLs, or generated compiler output.

Changes below `models/<model-id>/` select only that model. Changes to the schema,
shared registry/runtime tools, packaging metadata, or the compile workflow
select the full catalog. Deleted model directories are not compiled. Manual
workflow dispatch can select a comma-separated list or the full catalog.

Do not hand-author distribution metadata. After compilation, `model-registry
package` verifies the MPK and compile provenance, creates each model's
`metadata.json`, and creates the root selectable `metadata.json` plus
`catalog.json`. The standard Vulcan publisher calculates the URL-encoded branch
key and 12-character commit folder, and the promotion workflow updates
`latest.tag` only after the manifest is available.
