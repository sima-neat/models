#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_dir/../../../.." && pwd)"
venv_path="${QAT_VENV_PATH:-$repository_root/.venv-qat}"
tools_path="${QAT_TOOLS_PATH:-$repository_root/.qat-tools}"
uv_path="${UV_BIN:-}"

if [[ -z "$uv_path" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv_path="$(command -v uv)"
  else
    mkdir -p "$tools_path"
    curl -LsSf https://astral.sh/uv/0.8.3/install.sh | UV_INSTALL_DIR="$tools_path" sh
    uv_path="$tools_path/uv"
  fi
fi

"$uv_path" python install 3.10.18
"$uv_path" venv --clear --python 3.10.18 "$venv_path"
"$uv_path" pip sync --python "$venv_path/bin/python" "$script_dir/requirements.lock"

if [[ -n "${SIMA_QAT_WHEEL:-}" ]]; then
  test -f "$SIMA_QAT_WHEEL"
  "$uv_path" pip install --python "$venv_path/bin/python" --no-deps "$SIMA_QAT_WHEEL"
elif [[ -n "${SIMA_QAT_SITE_PACKAGES:-}" ]]; then
  test -d "$SIMA_QAT_SITE_PACKAGES/sima_qat"
  "$venv_path/bin/python" - "$SIMA_QAT_SITE_PACKAGES" <<'PY'
from pathlib import Path
import site
import sys

source = Path(sys.argv[1]).resolve()
destination = Path(site.getsitepackages()[0]) / "sima_qat_sdk.pth"
destination.write_text(str(source) + "\n", encoding="utf-8")
PY
else
  echo "Set SIMA_QAT_WHEEL or SIMA_QAT_SITE_PACKAGES to the SDK 2.0 SiMa QAT package." >&2
  exit 2
fi

"$venv_path/bin/python" - <<'PY'
import importlib.metadata
import sys

import onnx
import sima_qat
import torch
import torchvision

qat_version = importlib.metadata.version('swml-qat')
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+")[0] == "2.3.1", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.18.1", torchvision.__version__
assert onnx.__version__ == "1.17.0", onnx.__version__
assert qat_version == "1.0.0.dev0+develop.20", qat_version
print(f"Python: {sys.version.split()[0]}")
print(f"SiMa QAT: {qat_version}")
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"SiMa QAT module: {sima_qat.__file__}")
PY
