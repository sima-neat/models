from __future__ import annotations

import copy
from pathlib import Path

import pytest
from model_registry.errors import ValidationError
from model_registry.fidelity_config import (
    load_recipe_fidelity_config,
    validate_recipe_fidelity_result,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "models" / "rf_detr_small" / "tests" / "correctness.yaml"
SCHEMA_PATH = ROOT / "schemas" / "recipe-fidelity-v1.schema.json"


def _report(config: dict) -> dict:
    return {
        "component_id": config["result_component"],
        "status": config["status"],
        "dataset": {
            key: config["dataset"][key]
            for key in ("id", "variant", "sha256")
        },
        "selection": config["selection"],
        "thresholds": config["thresholds"],
        "samples": config["selection"]["samples"],
        "metrics": [
            {
                "name": name,
                "cosine_similarity": 0.99,
                "normalized_rmse": 0.01,
            }
            for name in config["outputs"]
        ],
    }


def test_fidelity_result_matches_its_declared_contract() -> None:
    config = load_recipe_fidelity_config(CONFIG_PATH, SCHEMA_PATH)

    validate_recipe_fidelity_result(config, _report(config), "fidelity.json")


def test_fidelity_result_rejects_failed_thresholds() -> None:
    config = load_recipe_fidelity_config(CONFIG_PATH, SCHEMA_PATH)
    report = copy.deepcopy(_report(config))
    report["metrics"][0]["cosine_similarity"] = 0.1

    with pytest.raises(ValidationError, match="thresholds failed"):
        validate_recipe_fidelity_result(config, report, "fidelity.json")
