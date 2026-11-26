"""The registry's public endpoint for each Vulcan environment, from ``config/registry.json``.

The file holds only what may be published. The IAM API, bucket, KMS key, and roles of each environment stay in its
GitHub environment variables, which the workflows pass to each command.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import ValidationError

CONFIG_FILE = Path("config") / "registry.json"


def public_url(root: Path, environment: str) -> str:
    """The public, unsigned registry endpoint of ``environment`` (``staging`` or ``production``)."""
    path = root / CONFIG_FILE
    try:
        environments = json.loads(path.read_text(encoding="utf-8"))["environments"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValidationError(f"{path}: cannot read the registry environments: {exc}") from exc
    if environment not in environments:
        raise ValidationError(f"{path}: no registry environment {environment!r}; expected {sorted(environments)}")
    return environments[environment]["public_url"]
