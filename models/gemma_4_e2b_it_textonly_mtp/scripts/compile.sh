#!/usr/bin/env bash
# Compile the INT4 target and INT8 MTP assistant as one deployment.
# Usage: compile.sh <checkpoint_directory> <output_directory> --draft_model_path <directory> [llima-compile options...]
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 <checkpoint_directory> <output_directory> --draft_model_path <directory> [llima-compile options...]" >&2
  exit 2
fi
target="$1"
output="$2"
shift 2
draft=""
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft_model_path) draft="$2"; shift 2 ;;
    *) arguments+=("$1"); shift ;;
  esac
done
[[ -n "$draft" ]] || { echo "--draft_model_path is required" >&2; exit 2; }
download_commands="$("${MODEL_COMPILER_PYTHON:-python3}" - "$0" <<'PY'
import os
import shlex
import sys
from pathlib import Path
from urllib.parse import urlsplit

from model_registry.model import load_model

model = load_model(Path(sys.argv[1]).resolve().parents[1])
profile_id = os.environ.get("BUILD_PROFILE_ID", model["build_profiles"][0]["id"])
profile = next(item for item in model["build_profiles"] if item["id"] == profile_id)
parameters = profile["components"][0]["parameters"]
if parameters.get("draft_precision") != "a16w8":
    raise SystemExit("Gemma 4 MTP requires draft_precision: a16w8")
source = model["components"][0]["source"]
files = {item["path"]: item for item in source["files"]}
for variable, config in (("MODEL_PATH", source["entrypoint"]),
                         ("DRAFT_PATH", parameters["draft_model_path"] + "/config.json")):
    url = files[config]["uri"].replace("hf://models/", "https://huggingface.co/", 1).replace("@", "/resolve/", 1)
    if urlsplit(url).hostname != "huggingface.co":
        raise SystemExit("Gemma 4 MTP requires pinned Hugging Face checkpoint sources")
    repository, revision_path = urlsplit(url).path.lstrip("/").split("/resolve/", 1)
    revision = revision_path.split("/", 1)[0]
    print(f'# hf download {shlex.quote(repository)} --revision {shlex.quote(revision)} --local-dir "${{{variable}}}"')
PY
)"
arguments+=(--jobs 20)
configuration="$(mktemp "${TMPDIR:-/tmp}/gemma4-mtp-XXXXXXXX.py")"
trap 'rm -f "$configuration"' EXIT
cat > "$configuration" <<'PY'
def get_layer_configuration(model_properties, layer):
    # Prequantized target weights retain their encoded precision.
    return {"precision": "A_BF16_W_INT8" if model_properties["is_draft_model"] else "BF16"}
PY
llima-compile "$target" -o "$output" --draft_model_path "$draft" -c "$configuration" "${arguments[@]}"

# Save a standalone recipe, including the assistant precision configuration.
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
  printf 'MODEL_PATH=%q\n' "$(basename "$target")"
  printf 'DRAFT_PATH=%q\n' "$(basename "$draft")"
  printf 'OUTPUT_PATH=%q\n\n' "$(basename "${target%-Safetensors}")-MTP-compiled"
  printf '%s\n\n' "$download_commands"
  cat <<'SH'
configuration="$(mktemp "${TMPDIR:-/tmp}/gemma4-mtp-XXXXXXXX.py")"
trap 'rm -f "$configuration"' EXIT
cat > "$configuration" <<'PY'
SH
  cat "$configuration"
  printf 'PY\n\n'
  printf 'llima-compile "${MODEL_PATH}" -o "${OUTPUT_PATH}" --draft_model_path "${DRAFT_PATH}" -c "$configuration"'
  printf ' %q' "${arguments[@]}"
  printf ' "$@"\n'
} > "$output/compile.sh"
