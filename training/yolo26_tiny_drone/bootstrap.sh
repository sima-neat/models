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

PYTHONPATH="$recipe_dir${PYTHONPATH:+:$PYTHONPATH}" "$venv/bin/python" - <<'PY'
import torch
import torchvision
import ultralytics
from model import lower_attention_for_mla, validate_attention_qat_scaffolding
from provenance import activate_verified_swml_qat_runtime

swml_qat_runtime = activate_verified_swml_qat_runtime()
from sima_qat.qat_api import sima_prepare_qat_model
from ultralytics.nn.modules.block import Attention

assert torch.__version__.split("+")[0] == "2.3.1", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.18.1", torchvision.__version__
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("swml_qat_package_sha256", swml_qat_runtime.package_sha256)
print("sima_prepare_qat_model", sima_prepare_qat_model)

attention = Attention(128, num_heads=2).train()
attention, lowered_count = lower_attention_for_mla(attention)
assert lowered_count == 1, lowered_count
example = (torch.rand(1, 128, 8, 8),)
prepared = sima_prepare_qat_model(attention, example, torch.device("cpu"))
attention_evidence = validate_attention_qat_scaffolding(
    prepared,
    expected_matmuls=4,
    expected_softmaxes=2,
)
print("attention_qat", attention_evidence)
PY
