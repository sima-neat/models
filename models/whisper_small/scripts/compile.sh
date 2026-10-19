#!/usr/bin/env bash
# Compile the original Whisper checkpoint as A16W8, including every encoder and decoder part.
# Usage: compile.sh <checkpoint_directory> <output_directory> [Whisper options...]
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <checkpoint_directory> <output_directory> [Whisper options...]" >&2
  exit 2
fi
checkpoint="$1"
output="$2"
arguments=(--num_processes 20 "${@:3}")
compiler="$(cat <<'PY'
import argparse
import logging
from pathlib import Path

from afe.apis.error_handling_variables import enable_verbose_error_messages
from sima_lmm.model import FileGenMode, FileGenPrecision, WhisperModel

parser = argparse.ArgumentParser(description="Compile Whisper for Modalix")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--num_processes", type=int, default=20)
parser.add_argument("--use_future_token_mask", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--enable_log_probe", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
enable_verbose_error_messages()
model = WhisperModel.from_hf_cache(
    hf_cache_path=args.checkpoint,
    model_name=args.checkpoint.name,
    sima_path=args.output / "sima_files",
    use_future_token_mask=args.use_future_token_mask,
    enable_log_probe=args.enable_log_probe,
)
model.gen_files(
    FileGenMode.ALL,
    precision={"all": {"precision": FileGenPrecision.A_BF16_W_INT8}},
    log_level=logging.INFO,
    num_processes=args.num_processes,
    part="all",
    resume=True,
)
PY
)"
python3 -c "$compiler" "$checkpoint" "$output" "${arguments[@]}"

# Save a self-contained recipe with the executed options alongside the deployment.
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
  printf 'MODEL_PATH=%q\nOUTPUT_PATH=%q\n\n' "$(basename "$checkpoint")" whisper-small-a16w8
  printf '# hf download openai/whisper-small --revision 973afd24965f72e36ca33b3055d56a652f456b4d --local-dir "${MODEL_PATH}"\n\n'
  printf '%s\n' "compiler=\$(cat <<'PY'" "$compiler" 'PY' ')' ''
  printf 'python3 -c "$compiler" "${MODEL_PATH}" "${OUTPUT_PATH}"'
  printf ' %q' "${arguments[@]}"
  printf '\n'
} > "$output/compile.sh"
chmod +x "$output/compile.sh"
