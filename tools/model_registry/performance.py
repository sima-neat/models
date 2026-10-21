"""Validate that declared benchmarks have a model-owned runnable script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ValidationError


def validate_benchmarks(model: dict[str, Any], model_dir: str | Path) -> None:
    if model.get("benchmark"):
        script = Path(model_dir) / "scripts/performance.py"
        if not script.is_file():
            raise ValidationError(f"performance script is missing: {script}")
