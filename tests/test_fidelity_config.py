from __future__ import annotations

import copy
from pathlib import Path

import pytest
from model_registry.errors import ValidationError
from model_registry.fidelity_config import find_fidelity_config, validate_fidelity_result
from model_registry.model import load_and_validate_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "rf_detr_small"


def _config() -> dict:
    model = load_and_validate_model(MODEL_DIR)
    return find_fidelity_config(
        MODEL_DIR, model, build_profile_id="w8a8_w16a16", component_id="decoder"
    )


def _report(config: dict) -> dict:
    return {
        "component_id": config["result_component"],
        "status": config["status"],
        "dataset": {
            key: config["dataset"][key]
            for key in ("id", "variant", "sha256")
        },
        "selection": config["selection"],
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
    config = _config()

    validate_fidelity_result(config, _report(config), "fidelity.json")


def test_diagnostic_fidelity_accepts_low_similarity() -> None:
    config = _config()
    report = copy.deepcopy(_report(config))
    report["metrics"][0]["cosine_similarity"] = 0.1

    report["metrics"][0]["normalized_rmse"] = 2.0

    validate_fidelity_result(config, report, "fidelity.json")


def test_fidelity_result_still_requires_the_declared_samples() -> None:
    config = _config()
    report = _report(config)
    report["samples"] -= 1

    with pytest.raises(ValidationError, match="samples does not match"):
        validate_fidelity_result(config, report, "fidelity.json")
