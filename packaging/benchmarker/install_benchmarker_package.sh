#!/usr/bin/env bash

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPENDENCIES_FILE="${PACKAGE_DIR}/dependencies.json"
SIMA_CLI_BIN="${BENCHMARKER_SIMA_CLI:-sima-cli}"

if [[ ! -f "${DEPENDENCIES_FILE}" ]]; then
  echo "Missing benchmarker dependency declaration: ${DEPENDENCIES_FILE}" >&2
  exit 1
fi

dependency="$(python3 - "${DEPENDENCIES_FILE}" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
core = value["dependencies"]["neat-core"]
print("\t".join((core["version"], core["install_target"], value["platform_version"])))
PY
)"

IFS=$'\t' read -r NEAT_VERSION CORE_TARGET PLATFORM_VERSION <<< "${dependency}"
if [[ -z "${NEAT_VERSION}" || "${CORE_TARGET}" != "core@v${NEAT_VERSION}" || -z "${PLATFORM_VERSION}" ]]; then
  echo "Invalid benchmarker dependency declaration" >&2
  exit 1
fi

if [[ "${BENCHMARKER_SKIP_PLATFORM_CHECK:-0}" != "1" ]]; then
  actual_platform="$(python3 - <<'PY'
import pathlib

path = pathlib.Path("/etc/buildinfo")
if path.is_file():
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "DISTRO_VERSION":
            print(value.strip())
            break
PY
)"
  if [[ "${actual_platform}" != "${PLATFORM_VERSION}" ]]; then
    echo "Platform version mismatch: benchmarker requires ${PLATFORM_VERSION}, found ${actual_platform:-unknown}" >&2
    exit 1
  fi
fi

detect_pyneat_python() {
  local candidate
  if [[ -n "${BENCHMARKER_PYNEAT_VENV:-}" ]]; then
    candidate="${BENCHMARKER_PYNEAT_VENV}/bin/python"
    [[ -x "${candidate}" ]] && printf '%s\n' "${candidate}"
    return
  fi
  for candidate in "${HOME}/pyneat/bin/python" "/media/nvme/pyneat/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
}

pyneat_matches() {
  local python_bin="$1"
  "${python_bin}" - "${NEAT_VERSION}" <<'PY' >/dev/null 2>&1
import sys
import pyneat

raise SystemExit(0 if pyneat.__version__.split("+", 1)[0] == sys.argv[1] else 1)
PY
}

PYNEAT_PYTHON="$(detect_pyneat_python || true)"
if [[ "${BENCHMARKER_SKIP_CORE_INSTALL:-0}" != "1" ]] && \
   { [[ -z "${PYNEAT_PYTHON}" ]] || ! pyneat_matches "${PYNEAT_PYTHON}"; }; then
  command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1 || {
    echo "sima-cli is required to install ${CORE_TARGET}" >&2
    exit 1
  }
  echo "Installing benchmarker dependency: ${CORE_TARGET}"
  "${SIMA_CLI_BIN}" neat install "${CORE_TARGET}"
  PYNEAT_PYTHON="$(detect_pyneat_python || true)"
fi

if [[ -z "${PYNEAT_PYTHON}" ]] || ! pyneat_matches "${PYNEAT_PYTHON}"; then
  echo "pyneat ${NEAT_VERSION} is not available after installing ${CORE_TARGET}" >&2
  exit 1
fi

wheel_count="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_benchmarker-*.whl' | wc -l | tr -d ' ')"
wheel_path="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_benchmarker-*.whl' | sort | head -n 1)"
if [[ "${wheel_count}" != "1" || -z "${wheel_path}" ]]; then
  echo "Expected exactly one benchmarker wheel, found ${wheel_count}" >&2
  exit 1
fi

"${PYNEAT_PYTHON}" -m pip install --no-deps --force-reinstall "${wheel_path}"
"${PYNEAT_PYTHON}" - "${NEAT_VERSION}" "${PLATFORM_VERSION}" <<'PY'
import json
import pathlib
import sys

import benchmarker
import pyneat

expected_neat, expected_platform = sys.argv[1:]
actual_neat = pyneat.__version__
if actual_neat.split("+", 1)[0] != expected_neat:
    raise SystemExit(f"Neat version mismatch: expected {expected_neat}, found {actual_neat}")

record = {
    "schema_version": 1,
    "benchmarker_version": benchmarker.tool_version(),
    "neat_version": actual_neat,
    "platform_version": expected_platform,
    "executable": str(pathlib.Path(sys.executable).with_name("benchmarker")),
}
destination = pathlib.Path(sys.prefix) / "share" / "benchmarker" / "install.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Installed benchmarker {record['benchmarker_version']}: {record['executable']}")
print(f"Using pyneat {actual_neat}")
PY
