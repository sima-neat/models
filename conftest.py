"""Fixtures for each model's tests under models/<id>/tests/.

A test that takes `converted`, `compiled`, `accuracy` or `performance` checks that real stage output and
runs only where it exists: scripts/run_model_tests.sh passes its path with the option of the same name.
Without the option pytest deselects the test, so a plain run checks only what needs no stage output.

A test parametrized with `profile_id` is skipped for a build profile its model.yaml leaves out, such as one
commented out until it builds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
from model_registry.errors import ValidationError
from model_registry.model import MODEL_FILE, load_model
from model_registry.model_tests import STAGE_GROUP, TESTS_FILE, load_model_tests, skip_reason

STAGE_OUTPUTS = {
    "converted": "converted ONNX model the conversion tests check",
    "compiled": "one build's compile output directory, <root>/<model>/<build profile>",
    "accuracy": "model-report.json the accuracy tests check",
    "performance": "model-report.json the performance tests check",
}
# Compiler and DevKit modules the model scripts import but the tests never call. Where one is not
# installed, a loaded script gets a stand-in for it.
COMPILER_AND_BOARD_MODULES = (
    "afe",
    "afe.apis",
    "afe.apis.defines",
    "afe.apis.loaded_net",
    "afe.apis.model",
    "afe.core",
    "afe.core.configs",
    "afe.core.utils",
    "afe.ir",
    "afe.ir.tensor_type",
    "afe.load",
    "afe.load.importers",
    "afe.load.importers.general_importer",
    "sima_utils",
    "sima_utils.data",
    "sima_utils.data.data_generator",
    "onnxruntime",
    "pyneat",
    "faster_coco_eval",
    "faster_coco_eval.core",
)
MISSING_MODULES = [
    name for name in COMPILER_AND_BOARD_MODULES if importlib.util.find_spec(name.partition(".")[0]) is None
]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("model stage outputs")
    for name, help_text in STAGE_OUTPUTS.items():
        group.addoption(f"--{name}", type=Path, help=help_text)
    group.addoption("--build-profile", help="the build profile an accuracy or performance report measured")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep the tests whose stage outputs were given, or only those needing none; skip the files tests.yaml skips."""
    given = {name for name in STAGE_OUTPUTS if config.getoption(name)}
    selected, deselected = [], []
    for item in items:
        needed = _stage_outputs(item)
        runs = bool(needed) and needed <= given if given else not needed
        (selected if runs else deselected).append(item)
        reason = _skip_reason(item.path) if runs else None
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"skipped in {TESTS_FILE}: {reason}"))
        profile_id = _undeclared_profile(item) if runs else None
        if profile_id:
            item.add_marker(pytest.mark.skip(reason=f"{MODEL_FILE} does not declare build profile {profile_id}"))
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _stage_outputs(item: pytest.Item) -> set[str]:
    """The stage-output fixtures of this file the test takes; a test's own fixture of the same name is not one."""
    definitions = getattr(item, "_fixtureinfo", None)
    definitions = definitions.name2fixturedefs if definitions else {}
    return {
        name
        for name in STAGE_OUTPUTS
        if name in definitions and definitions[name][-1].func.__module__ == _stage_outputs.__module__
    }


def _skip_reason(path: Path) -> str | None:
    """The tests.yaml skip_reason of models/<id>/tests/<group>/test_<stage>.py, when its tests are skipped."""
    stage = path.stem.removeprefix("test_")
    if path.parent.parent.name != "tests" or STAGE_GROUP.get(stage) != path.parent.name:
        return None
    try:
        return skip_reason(load_model_tests(path.parents[2]), stage)
    except ValidationError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _undeclared_profile(item: pytest.Item) -> str | None:
    """The `profile_id` a test is parametrized with, when the model.yaml of its models/<id> folder leaves it out."""
    callspec = getattr(item, "callspec", None)
    profile_id = callspec.params.get("profile_id") if callspec else None
    model_dir = next(
        (parent for parent in item.path.parents if parent.parent.name == "models" and (parent / MODEL_FILE).is_file()),
        None,
    )
    if profile_id is None or model_dir is None:
        return None
    return None if profile_id in _declared_profiles(model_dir) else profile_id


@cache
def _declared_profiles(model_dir: Path) -> frozenset[str]:
    try:
        return frozenset(profile["id"] for profile in load_model(model_dir)["build_profiles"])
    except ValidationError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _given(request: pytest.FixtureRequest, name: str) -> Path:
    path = request.config.getoption(name)
    if path is None:
        raise pytest.UsageError(f"--{name} is required by {request.node.nodeid}")
    return path


@pytest.fixture
def model_dir(request: pytest.FixtureRequest) -> Path:
    """The models/<id> folder of the requesting test."""
    for parent in request.path.parents:
        if parent.parent.name == "models" and (parent / "model.yaml").is_file():
            return parent
    raise pytest.UsageError(f"{request.path} is not inside a models/<id> folder")


@pytest.fixture
def load_script(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModuleType]:
    """Import one of the model's scripts by its path under scripts/, such as "compile" or "calibration"."""
    for name in MISSING_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())

    def load(name: str) -> ModuleType:
        path = model_dir / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{model_dir.name}_{name.replace('/', '_')}", path)
        assert spec and spec.loader, f"cannot load {path}"
        module = importlib.util.module_from_spec(spec)
        # A script's dataclasses look their module up while it runs.
        monkeypatch.setitem(sys.modules, spec.name, module)
        # Scripts import their siblings by name, so another model's sibling must not stand in for this one's.
        shadowed = {item.stem: sys.modules.pop(item.stem, None) for item in path.parent.glob("*.py")}
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
            for sibling, previous in shadowed.items():
                if previous is None:
                    sys.modules.pop(sibling, None)
                else:
                    sys.modules[sibling] = previous
        return module

    return load


@pytest.fixture
def converted(request: pytest.FixtureRequest) -> Path:
    """The converted ONNX model."""
    return _given(request, "converted")


@pytest.fixture
def compiled(request: pytest.FixtureRequest) -> Path:
    """One build's compile output: compile-outputs.json and a folder per component holding its MPK."""
    return _given(request, "compiled")


@pytest.fixture
def accuracy(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The build profile's DevKit accuracy report, as published (schemas/model-report.schema.json)."""
    return json.loads(_given(request, "accuracy").read_text(encoding="utf-8"))


@pytest.fixture
def performance(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The build profile's DevKit performance report, as published (schemas/model-report.schema.json)."""
    return json.loads(_given(request, "performance").read_text(encoding="utf-8"))


@pytest.fixture
def build_profile(request: pytest.FixtureRequest) -> str:
    """The build profile the given compile output or DevKit report belongs to."""
    if request.config.getoption("accuracy") or request.config.getoption("performance"):
        profile = request.config.getoption("build_profile")
        if profile is None:
            raise pytest.UsageError("--build-profile is required with an accuracy or performance report")
        return profile
    if request.config.getoption("compiled"):
        outputs = json.loads((_given(request, "compiled") / "compile-outputs.json").read_text(encoding="utf-8"))
        return outputs["build_profile_id"]
    raise pytest.UsageError(f"build_profile needs --compiled, --accuracy or --performance: {request.node.nodeid}")
