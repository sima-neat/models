#!/usr/bin/env bash
# Fail when the board-stage tooling this workflow runs differs from the measured source revision's.
#
# Usage: require_source_board_tooling.sh <source_checkout>
#
# A dispatch runs the benchmarking workflow from the branch tip, while publication computes the stage
# keys from the source revision. If the branch has since changed tooling those keys hash, the run would
# publish under a key that does not describe it; the newer push measures the build with its own keys.
# Run from the trusted checkout with model-registry installed.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <source_checkout>" >&2
  exit 2
fi

python - "$1" <<'PY'
import sys
from pathlib import Path

from model_registry.stage_keys import ACCURACY_TOOLING, LLIMA_PERFORMANCE_TOOLING, PERFORMANCE_TOOLING

source = Path(sys.argv[1])


def content(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


changed = sorted(
    relative
    for relative in {*PERFORMANCE_TOOLING, *ACCURACY_TOOLING, *LLIMA_PERFORMANCE_TOOLING}
    if content(Path(relative)) != content(source / relative)
)
if changed:
    sys.exit(
        "The branch changed board-stage tooling after the measured revision: "
        + ", ".join(changed)
        + ". The newer push measures this build with its own stage keys."
    )
PY
