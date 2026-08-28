#!/usr/bin/env bash

is_model_compiler_python() {
  local candidate="$1"
  local candidate_bin=""

  [[ -x "${candidate}" ]] || return 1
  candidate_bin="$(cd "$(dirname "${candidate}")" && pwd)"
  [[ -x "${candidate_bin}/mla-masm" ]] || return 1
  "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
import afe  # noqa: F401

raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

resolve_model_compiler_python() {
  local candidate=""
  local configured_root=""

  if [[ -n "${MODEL_COMPILER_PYTHON:-}" ]]; then
    if is_model_compiler_python "${MODEL_COMPILER_PYTHON}"; then
      printf '%s\n' "${MODEL_COMPILER_PYTHON}"
      return
    fi
    echo "MODEL_COMPILER_PYTHON is not a Python 3.10+ Model Compiler environment: ${MODEL_COMPILER_PYTHON}" >&2
    return 1
  fi

  if [[ -n "${MODEL_COMPILER_HOME:-}" ]]; then
    candidate="${MODEL_COMPILER_HOME%/}/bin/python"
    if is_model_compiler_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
    echo "MODEL_COMPILER_HOME does not contain a valid Model Compiler environment: ${MODEL_COMPILER_HOME}" >&2
    return 1
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidate="${VIRTUAL_ENV%/}/bin/python"
    if is_model_compiler_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  fi

  for configured_root in "${NEAT_SDK_EXTENSIONS_DIR:-}" "${SIMA_SDK_EXTENSIONS_DIR:-}"; do
    [[ -n "${configured_root}" ]] || continue
    candidate="${configured_root%/}/model-compiler/bin/python"
    if is_model_compiler_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  for candidate in \
    "/sdk-extensions/model-compiler/bin/python" \
    "/sdk-add-on/model-compiler/bin/python" \
    "${HOME}/sima-sdk-extensions/model-compiler/bin/python" \
    "${HOME}/sdk-extensions/model-compiler/bin/python" \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"; do
    if is_model_compiler_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}
