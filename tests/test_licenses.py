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


def _license_files(resource: dict) -> list[str]:
    """The license files one resource installs: the file it pins, or those in its file set."""
    names = [resource["install_path"], *resource["source"].get("files", {})]
    return [name for name in names if Path(name).name.startswith("LICENSE")]


def test_every_downloadable_onnx_package_ships_its_license() -> None:
    # LLiMa packages ship their deployment folder and take no resources (docs/metadata-schema.md).
    missing = []
    for manifest in sorted((ROOT / "models").glob("*/model.yaml")):
        model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if model["compilation"]["format"] != "onnx" or not LICENSES[model["license"]["spdx"]]["downloadable"]:
            continue
        if not any(_license_files(resource) for resource in model.get("resources", {}).values()):
            missing.append(manifest.parent.name)

    assert not missing, f"pin the upstream license as a `license` resource in model.yaml: {missing}"
