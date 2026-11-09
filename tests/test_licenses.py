from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
LICENSES_FILE = "artifact-browser/licenses.json"
LICENSES = json.loads((ROOT / LICENSES_FILE).read_text(encoding="utf-8"))


def test_licenses_match_their_schema() -> None:
    schema = json.loads((ROOT / "schemas" / "licenses.schema.json").read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(LICENSES)


def test_every_model_declares_a_listed_license() -> None:
    unlisted = {}
    for manifest in sorted((ROOT / "models").glob("*/model.yaml")):
        license_id = yaml.safe_load(manifest.read_text(encoding="utf-8"))["license"]["spdx"]
        if license_id not in LICENSES:
            unlisted[manifest.parent.name] = license_id

    assert not unlisted, f"add these licenses to {LICENSES_FILE}: {unlisted}"
