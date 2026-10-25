"""Load and validate models/<id>/model.yaml.

model.yaml says what a model is and what each stage must produce. load_model() expands it into the shape the
registry works with: each build profile compiles every component in one compile invocation named after the
profile, and packages them as the `default` package.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from .benchmark_declarations import resolve_accuracy_requirements, validate_benchmark_declarations
from .errors import SourcePolicyError, ValidationError
from .model_tests import TESTS_DIR, TESTS_SCHEMA, validate_model_tests
from .source import SourceSpec
from .taxonomy import find_taxonomy, load_taxonomy, validate_model_categorization

MODEL_FILE = "model.yaml"
SCHEMA_RELATIVE_PATH = Path("schemas/model.schema.json")
SCRIPTS_DIR = "scripts"
# A model folder holds its manifest, its documentation, model-owned code in scripts/, and its tests in tests/.
MODEL_FOLDER_ENTRIES = {MODEL_FILE, "README.md", "hero.png", SCRIPTS_DIR, TESTS_DIR}
# vulcan-model-convert.yml runs these files for a model that declares `conversion`.
CONVERSION_FILES = (f"{SCRIPTS_DIR}/convert.py", f"{SCRIPTS_DIR}/requirements-convert.txt")
# The one component of a model that does not list compilation.components.
SINGLE_COMPONENT = "model"
DEFAULT_PACKAGE = "default"
# vulcan-model-convert.yml names the one conversion a model declares.
DEFAULT_CONVERSION = "default"
RECIPES = {"onnx": "scripts/compile.py", "huggingface": "scripts/compile.sh"}


def component_for(model: dict[str, Any], component_id: str) -> dict[str, Any]:
    try:
        return next(component for component in model["components"] if component["id"] == component_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown component: {component_id}") from exc


def build_profile_for(model: dict[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        return next(profile for profile in model["build_profiles"] if profile["id"] == profile_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown build profile: {profile_id}") from exc


def profile_component_for(
    model: dict[str, Any], profile_id: str, component_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = build_profile_for(model, profile_id)
    try:
        binding = next(item for item in profile["components"] if item["id"] == component_id)
    except StopIteration as exc:
        raise ValidationError(f"build profile {profile_id!r} does not include component {component_id!r}") from exc
    return profile, binding


def mpk_filename(model: dict[str, Any], profile_id: str, component_id: str) -> str:
    """Name a compiled MPK by model and variant ID, adding the component when the variant has several."""
    profile = build_profile_for(model, profile_id)
    name = f"{model['id']}_{profile_id}"
    if len(profile["components"]) > 1:
        name = f"{name}_{component_id}"
    return f"{name}_mpk.tar.gz"


def effective_runner_class(profile: dict[str, Any], binding: dict[str, Any]) -> str:
    return str(binding.get("runner_class", profile.get("runner_class", "default")))


def effective_runner_market(profile: dict[str, Any], binding: dict[str, Any]) -> str:
    return binding.get("runner_market", profile.get("runner_market", "spot"))


def compile_invocations_for(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """A build profile's one compile invocation, which compiles every component."""
    return profile["compile_invocations"]


def source_members(source: dict[str, Any]) -> list[tuple[str, SourceSpec]]:
    if source["type"] == "file":
        spec = SourceSpec.from_mapping(source)
        return [(spec.filename, spec)]
    return [(member["path"], SourceSpec.from_mapping(member)) for member in source["files"]]


def model_file(model_dir: str | Path) -> Path:
    directory = Path(model_dir).expanduser().resolve()
    path = directory / MODEL_FILE
    if not path.is_file():
        raise ValidationError(f"{directory}: missing {MODEL_FILE}")
    return path


def _read_model(model_dir: str | Path) -> dict[str, Any]:
    path = model_file(model_dir)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a YAML object")
    return value


def load_model(model_dir: str | Path) -> dict[str, Any]:
    return normalize_model(_read_model(model_dir), model_file(model_dir).parent)


def normalize_model(value: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Expand model.yaml into the registry's shape, without fetching sources or importing execution SDKs."""
    authored = deepcopy(value)
    try:
        compilation = authored.pop("compilation")
        component_ids = compilation.get("components", [SINGLE_COMPONENT])
        source = _expand_source(compilation["source"])
        model = {
            **authored,
            "id": directory.name,
            "maturity": "experimental",  # Vulcan's catalog writer requires a maturity.
            "architecture": {"id": authored["architecture"]},
            "components": [
                {
                    "id": component_id,
                    "source": source,
                    "interface": {"format": compilation["format"]},
                    "role": "inference",
                    "recipe": RECIPES[compilation["format"]],
                }
                for component_id in component_ids
            ],
            "calibrations": authored.get("calibrations", []),
            "build_profiles": [],
        }
        if "conversion" in authored:
            model["conversion"] = {DEFAULT_CONVERSION: {"source": _expand_source(authored["conversion"]["source"])}}
        if "upstream" not in model and "repository" in compilation["source"]:
            model["upstream"] = {
                "repository": f"https://huggingface.co/{compilation['source']['repository']}",
                "revision": compilation["source"]["revision"],
            }
        if "upstream" in model:
            model["upstream"].setdefault("model_card", model["upstream"]["repository"])

        for profile_id, profile in authored["build_profiles"].items():
            calibration = profile.get("calibration_dataset")
            bindings = [{"id": component_id, **binding} for component_id, binding in profile["components"].items()]
            # The compile recipe calibrates its first component; later components calibrate on its outputs.
            if calibration is not None and bindings:
                bindings[0]["calibration"] = calibration
            model["build_profiles"].append({
                "id": profile_id,
                "target": "modalix",
                **({"runner_class": profile["runner_class"]} if "runner_class" in profile else {}),
                **({"runner_market": profile["runner_market"]} if "runner_market" in profile else {}),
                "components": bindings,
                # One invocation compiles every component, so its ID is the profile ID. Vulcan's catalog and
                # promotion contracts still name each build's compile_invocation_id.
                "compile_invocations": [{"id": profile_id, "components": list(component_ids)}],
            })

        model["compositions"] = [
            {"id": DEFAULT_PACKAGE, "type": "single", "component": SINGLE_COMPONENT,
             "entrypoint": compilation.get("entrypoint", "default")}
            if component_ids == [SINGLE_COMPONENT]
            else {"id": DEFAULT_PACKAGE, "type": "declarative"}
        ]
        model["packages"] = [{
            "id": DEFAULT_PACKAGE, "default": True, "components": list(component_ids),
            "build_profiles": [profile["id"] for profile in model["build_profiles"]], "composition": DEFAULT_PACKAGE,
        }]
        if "accuracy" in model:
            # Each dataset goes by its Dataset Registry source ID; validate_model() adds its required_metrics.
            pins = model["accuracy"]["datasets"]
            datasets = {dataset_name(pin): pin for pin in pins}
            if len(datasets) != len(pins):
                raise ValidationError(f"{directory}: accuracy lists a dataset twice")
            model["accuracy"] = {**model["accuracy"], "datasets": datasets}
        return model
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValidationError(f"{directory}: invalid model declaration: {exc}") from exc


def dataset_name(pin: dict[str, Any]) -> str:
    """The Dataset Registry's source ID for a pinned dataset variant, such as coco-2017-val."""
    return f"{pin['id']}-{pin['variant']}"


def _expand_source(source: dict[str, Any]) -> dict[str, Any]:
    """Expand a source into a file or a file set whose members each have a path and a URI."""
    if "files" not in source:
        return {"type": "file", **source}
    repository = source.get("repository")
    revision = source.get("revision")
    files = []
    for path, specification in source["files"].items():
        member = {"path": path, **specification}
        if "uri" not in member:
            if not repository or not revision:
                raise ValidationError(f"source member {path!r} requires a URI or repository and revision")
            member["uri"] = f"hf://models/{repository}@{revision}/{path}"
        files.append(member)
    return {"type": "file_set", "entrypoint": source["main_file"], "files": files}


def find_schema(model_dir: str | Path, schema_path: str | Path | None = None) -> Path:
    if schema_path is not None:
        candidate = Path(schema_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValidationError(f"schema not found: {candidate}")
        return candidate

    configured_root = os.environ.get("MODEL_REGISTRY_ROOT", "").strip()
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve() / SCHEMA_RELATIVE_PATH)

    directory = Path(model_dir).expanduser().resolve()
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (directory, *directory.parents))
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"{directory}: could not locate {SCHEMA_RELATIVE_PATH}; run inside the repository or set MODEL_REGISTRY_ROOT"
    )


def _format_path(parts: list[Any]) -> str:
    rendered = ""
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + str(part)
    return rendered or "<root>"


def _error_path(error: Any) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = next((name for name in error.validator_value if name not in error.instance), None)
        if missing is not None:
            parts.append(missing)
    return _format_path(parts)


def component_pipelines(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Each declared pipeline, keyed by model ID."""
    return {model_id: model["pipeline"] for model_id, model in sorted(models.items()) if "pipeline" in model}


def validate_model(
    model_dir: str | Path,
    authored: dict[str, Any],
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate model.yaml as written and return its expanded form."""
    directory = Path(model_dir).expanduser().resolve()
    schema_file = find_schema(directory, schema_path)
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_file}: could not load schema: {exc}") from exc
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(authored), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ValidationError(f"{directory}: {details}")

    for profile_id, profile in authored["build_profiles"].items():
        markets = {effective_runner_market(profile, binding) for binding in profile["components"].values()}
        if len(markets) != 1:
            raise ValidationError(
                f"{directory}: {profile_id} has conflicting runner_market requests in grouped compile"
            )

    model = normalize_model(authored, directory)
    if "upstream" not in model:
        raise ValidationError(f"{directory}: upstream is required when the source has no Hugging Face repository")
    try:
        validate_model_categorization(model, load_taxonomy(find_taxonomy(directory, taxonomy_path)))
    except ValidationError as exc:
        raise ValidationError(f"{directory}: {exc}") from exc

    sources = [authored["compilation"]["source"]]
    if "conversion" in authored:
        sources.append(authored["conversion"]["source"])
    for source in sources:
        if "files" in source and source["main_file"] not in source["files"]:
            raise ValidationError(f"{directory}: source main_file {source['main_file']!r} is not one of its files")
    try:
        for component in model["components"][:1]:
            source_members(component["source"])
        for conversion in model.get("conversion", {}).values():
            source_members(conversion["source"])
    except (KeyError, TypeError, ValueError, SourcePolicyError) as exc:
        raise ValidationError(f"{directory}: source policy: {exc}") from exc

    component_ids = [component["id"] for component in model["components"]]
    if authored["compilation"]["format"] == "huggingface":
        if len(component_ids) != 1 or authored["compilation"]["source"].get("main_file") != "config.json":
            raise ValidationError(
                f"{directory}: a Hugging Face model is one component compiled from a checkpoint file set "
                "whose main_file is config.json"
            )
    available_calibrations = {item["id"] for item in model["calibrations"]}
    for profile_id, profile in authored["build_profiles"].items():
        if sorted(profile["components"]) != sorted(component_ids):
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} must set the parameters of exactly the components "
                f"{', '.join(component_ids)}"
            )
        calibration = profile.get("calibration_dataset")
        if calibration is not None and calibration not in available_calibrations:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} names unknown calibration_dataset {calibration!r}"
            )

    pipeline_components = [stage["component"] for stage in model.get("pipeline", []) if "component" in stage]
    if "pipeline" in model and sorted(pipeline_components) != sorted(component_ids):
        raise ValidationError(f"{directory}: pipeline must list each component exactly once")

    _check_folder_layout(directory)
    referenced_paths = [model["components"][0]["recipe"]]
    if "conversion" in model:
        referenced_paths.extend(CONVERSION_FILES)
    for relative in referenced_paths:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            raise ValidationError(f"{directory}: referenced file does not exist inside model directory: {relative}")
    # The repository that holds the schema holds the dataset requirements.
    resolve_accuracy_requirements(model, schema_file.parent.parent, directory)
    validate_benchmark_declarations(model, directory)
    validate_model_tests(directory, model, schema_file.parent / TESTS_SCHEMA)
    return model


def _only_python_caches(path: Path) -> bool:
    """True for a directory git left behind holding only ignored bytecode caches."""
    files = (item for item in path.rglob("*") if item.is_file()) if path.is_dir() else None
    return files is not None and all("__pycache__" in item.relative_to(path).parts for item in files)


def _check_folder_layout(directory: Path) -> None:
    unexpected = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name not in MODEL_FOLDER_ENTRIES and not entry.name.startswith(".") and not _only_python_caches(entry)
    )
    if unexpected:
        raise ValidationError(
            f"{directory}: unexpected model folder entries: {', '.join(unexpected)}; "
            f"keep model code in {SCRIPTS_DIR}/, tests in {TESTS_DIR}/ and stage settings in {MODEL_FILE}"
        )


def load_and_validate_model(
    model_dir: str | Path,
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    return validate_model(model_dir, _read_model(model_dir), schema_path, taxonomy_path)


def load_models(repository: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate every models/<id>/model.yaml, keyed by model ID."""
    root = Path(repository).expanduser().resolve()
    model_root = root / "models"
    if not model_root.is_dir():
        raise ValidationError(f"models directory not found: {model_root}")
    return {
        definition.parent.name: load_and_validate_model(definition.parent, root / "schemas" / "model.schema.json")
        for definition in sorted(model_root.glob("*/model.yaml"))
    }
