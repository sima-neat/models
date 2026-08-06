#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$recipe_dir/../.." && pwd)"
venv="$repository_root/.venv-yolo26-qat"
uv_bin="${UV_BIN:-uv}"

if [[ -z "${SIMA_QAT_WHEEL:-}" && -z "${SIMA_QAT_SITE_PACKAGES:-}" ]]; then
  echo "Set SIMA_QAT_WHEEL or SIMA_QAT_SITE_PACKAGES." >&2
  exit 2
fi
if [[ -n "${SIMA_QAT_WHEEL:-}" && -n "${SIMA_QAT_SITE_PACKAGES:-}" ]]; then
  echo "Set only one SiMa QAT source." >&2
  exit 2
fi

"$uv_bin" venv --python 3.10 "$venv"
"$uv_bin" pip install \
  --python "$venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "$recipe_dir/requirements.in"

if [[ -n "${SIMA_QAT_WHEEL:-}" ]]; then
  "$uv_bin" pip install --python "$venv/bin/python" --no-deps "$SIMA_QAT_WHEEL"
else
  site="$($venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "$SIMA_QAT_SITE_PACKAGES" > "$site/sima_qat_sdk.pth"
fi

"$venv/bin/python" - <<'PY'
import torch
import torchvision
import ultralytics
from sima_qat.qat_api import sima_prepare_qat_model

assert torch.__version__.split("+")[0] == "2.3.1", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.18.1", torchvision.__version__
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("sima_prepare_qat_model", sima_prepare_qat_model)
PY
