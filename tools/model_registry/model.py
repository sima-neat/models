from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from .accuracy_config import DEVKIT_ACCURACY_ENGINE
from .artifact_metadata import validate_artifact_metadata
from .errors import SourcePolicyError, ValidationError
from .model_tests import TESTS_DIR, TESTS_SCHEMA, validate_model_tests
from .source import SourceSpec, source_identity
from .taxonomy import find_taxonomy, load_taxonomy, validate_model_categorization

MODEL_FILE = "model.yaml"
SCHEMA_RELATIVE_PATH = Path("schemas/model.schema.json")
SCRIPTS_DIR = "scripts"
# A model folder holds its manifest, its documentation, model-owned code in scripts/, and its tests in tests/.
MODEL_FOLDER_ENTRIES = {MODEL_FILE, "README.md", "hero.png", SCRIPTS_DIR, TESTS_DIR}
STAGES = ("fidelity", "accuracy", "benchmark")
# vulcan-model-convert.yml runs these files for a model that declares `conversion`.
CONVERSION_FILES = (f"{SCRIPTS_DIR}/convert.py", f"{SCRIPTS_DIR}/requirements-convert.txt")
# A devkit_script accuracy entry runs this script on the DevKit.
ACCURACY_SCRIPT = f"{SCRIPTS_DIR}/accuracy.py"
ACCURACY_REQUIREMENTS = f"{SCRIPTS_DIR}/requirements-accuracy.txt"
# A benchmark entry runs this script on the DevKit.
PERFORMANCE_SCRIPT = f"{SCRIPTS_DIR}/performance.py"


def _by_id(items: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        if item_id in indexed:
            raise ValidationError(f"duplicate {kind} ids: {item_id}")
        indexed[item_id] = item
    return indexed


def _check_file_set(directory: Path, source: dict[str, Any], label: str) -> None:
    if source["type"] != "file_set":
        return
    member_paths = [member["path"] for member in source["files"]]
    if len(member_paths) != len(set(member_paths)):
        raise ValidationError(f"{directory}: {label} source contains duplicate paths")
    if source["entrypoint"] not in set(member_paths):
        raise ValidationError(f"{directory}: {label} source entrypoint is not a declared file")


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


def profile_component_requires_calibration(model: dict[str, Any], profile_id: str, component_id: str) -> bool:
    _, binding = profile_component_for(model, profile_id, component_id)
    return "calibration" in binding


def effective_runner_class(profile: dict[str, Any], binding: dict[str, Any]) -> str:
    return str(binding.get("runner_class", profile.get("runner_class", "default")))


def compile_invocations_for(profile: dict[str, Any]) -> list[dict[str, Any]]:
    declared = profile.get("compile_invocations")
    if declared is not None:
        return declared
    return [
        {"id": binding["id"], "components": [binding["id"]]}
        for binding in profile["components"]
        if binding.get("status", "supported") == "supported"
    ]


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


def load_model(model_dir: str | Path) -> dict[str, Any]:
    path = model_file(model_dir)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a YAML object")
    return value


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


def composition_document(declaration: dict[str, Any]) -> dict[str, Any]:
    """Return a declarative composition as the standalone document packages ship."""
    return {key: value for key, value in declaration.items() if key not in {"id", "type"}}


def _load_declarative_composition(
    directory: Path,
    declaration: dict[str, Any],
    components: dict[str, dict[str, Any]],
    schema_directory: Path,
) -> set[str]:
    path = f"{directory}: composition {declaration['id']!r}"
    document = composition_document(declaration)
    try:
        schema = json.loads((schema_directory / "composition.schema.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: could not load the composition schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ValidationError(f"{path}: {details}")

    instances = _by_id(document["components"], f"component instance in composition {declaration['id']}")
    ports: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for instance_id, instance in instances.items():
        component_id = instance["component"]
        if component_id not in components:
            raise ValidationError(f"{path}: instance {instance_id!r} references unknown component {component_id!r}")
        entrypoints = _by_id(components[component_id]["interface"]["entrypoints"], f"entrypoint in {component_id}")
        entrypoint_id = instance["entrypoint"]
        if entrypoint_id not in entrypoints:
            raise ValidationError(f"{path}: instance {instance_id!r} references unknown entrypoint {entrypoint_id!r}")
        entrypoint = entrypoints[entrypoint_id]
        ports[instance_id] = (
            {tensor["name"]: tensor for tensor in entrypoint["inputs"]},
            {tensor["name"]: tensor for tensor in entrypoint["outputs"]},
        )

    destinations: set[tuple[str, str]] = set()
    graph = {instance_id: set() for instance_id in instances}
    for connection in document["connections"]:
        source = connection["from"]
        destination = connection["to"]
        if source["component"] not in ports or source["output"] not in ports[source["component"]][1]:
            raise ValidationError(f"{path}: connection references unknown output {source}")
        if destination["component"] not in ports or destination["input"] not in ports[destination["component"]][0]:
            raise ValidationError(f"{path}: connection references unknown input {destination}")
        destination_key = (destination["component"], destination["input"])
        if destination_key in destinations:
            raise ValidationError(
                f"{path}: input {destination_key[0]}.{destination_key[1]} is connected more than once"
            )
        destinations.add(destination_key)
        source_tensor = ports[source["component"]][1][source["output"]]
        destination_tensor = ports[destination["component"]][0][destination["input"]]
        if source_tensor["dtype"] != destination_tensor["dtype"]:
            raise ValidationError(f"{path}: connection {source} -> {destination} has incompatible dtypes")
        source_shape = source_tensor["shape"]
        destination_shape = destination_tensor["shape"]
        if len(source_shape) != len(destination_shape) or any(
            isinstance(left, int) and isinstance(right, int) and left != right
            for left, right in zip(source_shape, destination_shape, strict=True)
        ):
            raise ValidationError(f"{path}: connection {source} -> {destination} has incompatible shapes")
        graph[source["component"]].add(destination["component"])

    input_names: set[str] = set()
    for public_input in document["inputs"]:
        if public_input["name"] in input_names:
            raise ValidationError(f"{path}: duplicate public input {public_input['name']!r}")
        input_names.add(public_input["name"])
        destination = public_input["to"]
        if destination["component"] not in ports or destination["input"] not in ports[destination["component"]][0]:
            raise ValidationError(f"{path}: public input references unknown input {destination}")
        destination_key = (destination["component"], destination["input"])
        if destination_key in destinations:
            raise ValidationError(
                f"{path}: input {destination_key[0]}.{destination_key[1]} is connected more than once"
            )
        destinations.add(destination_key)

    required_destinations = {
        (instance_id, input_name)
        for instance_id, (inputs, _) in ports.items()
        for input_name in inputs
    }
    if destinations != required_destinations:
        missing = ", ".join(f"{instance}.{port}" for instance, port in sorted(required_destinations - destinations))
        raise ValidationError(f"{path}: composition has unconnected component inputs: {missing}")

    output_names: set[str] = set()
    for public_output in document["outputs"]:
        if public_output["name"] in output_names:
            raise ValidationError(f"{path}: duplicate public output {public_output['name']!r}")
        output_names.add(public_output["name"])
        source = public_output["from"]
        if source["component"] not in ports or source["output"] not in ports[source["component"]][1]:
            raise ValidationError(f"{path}: public output references unknown output {source}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit_instance(instance_id: str) -> None:
        if instance_id in visiting:
            raise ValidationError(f"{path}: composition connections contain a cycle at {instance_id!r}")
        if instance_id in visited:
            return
        visiting.add(instance_id)
        for downstream in graph[instance_id]:
            visit_instance(downstream)
        visiting.remove(instance_id)
        visited.add(instance_id)

    for instance_id in instances:
        visit_instance(instance_id)
    return {instance["component"] for instance in instances.values()}


def validate_model(
    model_dir: str | Path,
    model: dict[str, Any],
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> None:
    directory = Path(model_dir).expanduser().resolve()
    schema_file = find_schema(directory, schema_path)
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_file}: could not load schema: {exc}") from exc

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(model), key=lambda item: tuple(str(part) for part in item.absolute_path))
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ValidationError(f"{directory}: {details}")

    taxonomy = load_taxonomy(find_taxonomy(directory, taxonomy_path))
    try:
        validate_model_categorization(model, taxonomy)
    except ValidationError as exc:
        raise ValidationError(f"{directory}: {exc}") from exc

    model_id = str(model["id"])
    if directory.name != model_id:
        raise ValidationError(f"{directory}: id must match directory name; found {model_id!r}")

    try:
        for component in model["components"]:
            source_members(component["source"])
        for resource in model["resources"]:
            source_members(resource["source"])
        for calibration in model["calibrations"]:
            if calibration["status"] == "ready" and "source" in calibration:
                source_members(calibration["source"])
        for conversion in model.get("conversion", {}).values():
            source_members(conversion["source"])
    except (KeyError, TypeError, ValueError, SourcePolicyError) as exc:
        raise ValidationError(f"{directory}: source policy: {exc}") from exc

    components = _by_id(model["components"], "component")
    resources = _by_id(model["resources"], "resource")
    calibrations = _by_id(model["calibrations"], "calibration")
    profiles = _by_id(model["build_profiles"], "build profile")
    compositions = _by_id(model["compositions"], "composition")
    packages = _by_id(model["packages"], "package")

    for conversion_id, conversion in model.get("conversion", {}).items():
        _check_file_set(directory, conversion["source"], f"conversion {conversion_id!r}")

    for component_id, component in components.items():
        _check_file_set(directory, component["source"], f"component {component_id!r}")

        dependency_ids = set(component.get("depends_on", ()))
        unknown_dependencies = dependency_ids - set(components)
        if unknown_dependencies:
            raise ValidationError(
                f"{directory}: component {component_id!r} references unknown dependencies: "
                f"{', '.join(sorted(unknown_dependencies))}"
            )
        if component_id in dependency_ids:
            raise ValidationError(f"{directory}: component {component_id!r} cannot depend on itself")

        interface = component["interface"]
        entrypoints = _by_id(interface["entrypoints"], f"entrypoint in component {component_id}")
        shape_profiles = _by_id(interface["shape_profiles"], f"shape profile in component {component_id}")
        symbols: dict[str, set[str]] = {}
        for entrypoint_id, entrypoint in entrypoints.items():
            inputs = {tensor["name"] for tensor in entrypoint["inputs"]}
            outputs = {tensor["name"] for tensor in entrypoint["outputs"]}
            if len(inputs) != len(entrypoint["inputs"]) or len(outputs) != len(entrypoint["outputs"]):
                raise ValidationError(
                    f"{directory}: component {component_id!r} entrypoint {entrypoint_id!r} has duplicate tensor names"
                )
            symbols[entrypoint_id] = {
                dimension
                for tensor in (*entrypoint["inputs"], *entrypoint["outputs"])
                for dimension in tensor["shape"]
                if isinstance(dimension, str)
            }
            state_ids: set[str] = set()
            for state in entrypoint.get("state", ()):
                if state["id"] in state_ids:
                    raise ValidationError(
                        f"{directory}: component {component_id!r} entrypoint {entrypoint_id!r} "
                        f"duplicates state {state['id']!r}"
                    )
                state_ids.add(state["id"])
                if state["input"] not in inputs or state["output"] not in outputs:
                    raise ValidationError(
                        f"{directory}: component {component_id!r} state {state['id']!r} has unresolved bindings"
                    )
        for shape_profile_id, shape_profile in shape_profiles.items():
            entrypoint_id = shape_profile["entrypoint"]
            if entrypoint_id not in entrypoints:
                raise ValidationError(
                    f"{directory}: component {component_id!r} shape profile {shape_profile_id!r} "
                    f"references unknown entrypoint {entrypoint_id!r}"
                )
            unknown_symbols = set(shape_profile["bindings"]) - symbols[entrypoint_id]
            if unknown_symbols:
                raise ValidationError(
                    f"{directory}: component {component_id!r} shape profile {shape_profile_id!r} "
                    f"references unknown dimensions: {', '.join(sorted(unknown_symbols))}"
                )

    def visit(component_id: str, path: tuple[str, ...]) -> None:
        if component_id in path:
            cycle = " -> ".join((*path, component_id))
            raise ValidationError(f"{directory}: component dependency cycle: {cycle}")
        for dependency_id in components[component_id].get("depends_on", ()):
            visit(dependency_id, (*path, component_id))

    for component_id in components:
        visit(component_id, ())

    for profile_id, profile in profiles.items():
        bindings = _by_id(profile["components"], f"component binding in build profile {profile_id}")
        unknown_components = set(bindings) - set(components)
        if unknown_components:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} references unknown components: "
                f"{', '.join(sorted(unknown_components))}"
            )
        for component_id, binding in bindings.items():
            calibration_id = binding.get("calibration")
            if calibration_id is not None and calibration_id not in calibrations:
                raise ValidationError(
                    f"{directory}: build profile {profile_id!r} component {component_id!r} "
                    f"references unknown calibration {calibration_id!r}"
                )
            shape_profile_id = binding.get("shape_profile")
            available_shapes = {item["id"] for item in components[component_id]["interface"]["shape_profiles"]}
            if shape_profile_id is not None and shape_profile_id not in available_shapes:
                raise ValidationError(
                    f"{directory}: build profile {profile_id!r} component {component_id!r} "
                    f"references unknown shape profile {shape_profile_id!r}"
                )

        if "compile_invocations" not in profile:
            continue
        invocations = _by_id(profile["compile_invocations"], f"compile invocation in build profile {profile_id}")
        invocation_components = [
            component_id
            for invocation in invocations.values()
            for component_id in invocation["components"]
        ]
        unknown_invocation_components = set(invocation_components) - set(bindings)
        if unknown_invocation_components:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} compile invocations reference unknown components: "
                f"{', '.join(sorted(unknown_invocation_components))}"
            )
        unsupported_invocation_components = {
            component_id
            for component_id in invocation_components
            if bindings[component_id].get("status", "supported") == "unsupported"
        }
        if unsupported_invocation_components:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} compile invocations include unsupported components: "
                f"{', '.join(sorted(unsupported_invocation_components))}"
            )
        duplicate_invocation_components = {
            component_id
            for component_id in invocation_components
            if invocation_components.count(component_id) > 1
        }
        if duplicate_invocation_components:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} compile invocations duplicate components: "
                f"{', '.join(sorted(duplicate_invocation_components))}"
            )
        supported_components = {
            component_id
            for component_id, binding in bindings.items()
            if binding.get("status", "supported") == "supported"
        }
        omitted_components = supported_components - set(invocation_components)
        if omitted_components:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} compile invocations omit supported components: "
                f"{', '.join(sorted(omitted_components))}"
            )

        for invocation_id, invocation in invocations.items():
            grouped_components = [components[component_id] for component_id in invocation["components"]]
            source_identities = {source_identity(component["source"]) for component in grouped_components}
            if len(source_identities) != 1:
                raise ValidationError(
                    f"{directory}: build profile {profile_id!r} compile invocation {invocation_id!r} "
                    "components must share one source content identity"
                )
            runner_classes = {
                effective_runner_class(profile, bindings[component_id])
                for component_id in invocation["components"]
            }
            if len(runner_classes) != 1:
                raise ValidationError(
                    f"{directory}: build profile {profile_id!r} compile invocation {invocation_id!r} "
                    "components must share one effective runner class"
                )
            calibration_ids = {
                bindings[component_id]["calibration"]
                for component_id in invocation["components"]
                if "calibration" in bindings[component_id]
            }
            primary_calibration = bindings[invocation["components"][0]].get("calibration")
            if len(calibration_ids) > 1 or (calibration_ids and primary_calibration not in calibration_ids):
                raise ValidationError(
                    f"{directory}: build profile {profile_id!r} compile invocation {invocation_id!r} "
                    "must select one shared calibration on its first component"
                )

    composition_components: dict[str, set[str]] = {}
    for composition_id, composition in compositions.items():
        if composition["type"] == "single":
            component_id = composition["component"]
            if component_id not in components:
                raise ValidationError(
                    f"{directory}: composition {composition_id!r} references unknown component {component_id!r}"
                )
            entrypoints = {item["id"] for item in components[component_id]["interface"]["entrypoints"]}
            if composition["entrypoint"] not in entrypoints:
                raise ValidationError(
                    f"{directory}: composition {composition_id!r} references unknown entrypoint "
                    f"{composition['entrypoint']!r}"
                )
            composition_components[composition_id] = {component_id}
        else:
            composition_components[composition_id] = _load_declarative_composition(
                directory, composition, components, schema_file.parent
            )

    default_packages = [item["id"] for item in model["packages"] if item["default"]]
    if len(default_packages) != 1:
        raise ValidationError(f"{directory}: exactly one package must set default: true")
    for package_id, package in packages.items():
        for field, available in (
            ("components", components),
            ("resources", resources),
            ("build_profiles", profiles),
        ):
            unknown = set(package[field]) - set(available)
            if unknown:
                raise ValidationError(
                    f"{directory}: package {package_id!r} references unknown {field}: {', '.join(sorted(unknown))}"
                )
        if package["composition"] not in compositions:
            raise ValidationError(
                f"{directory}: package {package_id!r} references unknown composition {package['composition']!r}"
            )
        missing_composition_components = composition_components[package["composition"]] - set(package["components"])
        if missing_composition_components:
            raise ValidationError(
                f"{directory}: package {package_id!r} omits components required by composition "
                f"{package['composition']!r}: {', '.join(sorted(missing_composition_components))}"
            )
        for profile_id in package["build_profiles"]:
            bindings = {item["id"]: item for item in profiles[profile_id]["components"]}
            missing = set(package["components"]) - set(bindings)
            unsupported = {
                component_id
                for component_id in package["components"]
                if bindings.get(component_id, {}).get("status", "supported") == "unsupported"
            }
            if missing or unsupported:
                unavailable = sorted(missing | unsupported)
                raise ValidationError(
                    f"{directory}: package {package_id!r} is incomplete for build profile {profile_id!r}: "
                    f"{', '.join(unavailable)}"
                )
            # Each compile invocation publishes the packages it completes.
            if not any(
                set(package["components"]) <= set(invocation["components"])
                for invocation in compile_invocations_for(profiles[profile_id])
            ):
                raise ValidationError(
                    f"{directory}: package {package_id!r} must compile {', '.join(package['components'])} "
                    f"in one compile invocation of build profile {profile_id!r}"
                )

    _check_folder_layout(directory)
    referenced_paths = [component["recipe"] for component in model["components"]]
    referenced_paths.extend(
        component["surgery"]["entrypoint"].split(":", 1)[0]
        for component in model["components"]
        if component["surgery"]["type"] == "python"
    )
    if model.get("conversion"):
        referenced_paths.extend(CONVERSION_FILES)
    if any(entry.get("engine") == DEVKIT_ACCURACY_ENGINE for entry in model["accuracy"]):
        referenced_paths.append(ACCURACY_SCRIPT)
    for relative in referenced_paths:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            raise ValidationError(f"{directory}: referenced file does not exist inside model directory: {relative}")

    for stage in STAGES:
        _by_id(model[stage], f"{stage} entry")
        coverage_seen: set[tuple[str | None, str, str, str]] = set()
        for entry in model[stage]:
            scope = entry["scope"]
            reference_ids = components if scope["type"] == "component" else packages
            reference_field = scope["type"]
            scoped_id = scope[reference_field]
            if scoped_id not in reference_ids:
                raise ValidationError(
                    f"{directory}: {stage} entry {entry['id']!r} references unknown {reference_field} "
                    f"{scoped_id!r}"
                )
            unknown_profiles = set(entry["build_profiles"]) - set(profiles)
            if unknown_profiles:
                raise ValidationError(
                    f"{directory}: {stage} entry {entry['id']!r} references unknown build profiles: "
                    f"{', '.join(sorted(unknown_profiles))}"
                )
            for profile_id in entry["build_profiles"]:
                coverage = (entry.get("engine"), scope["type"], scoped_id, profile_id)
                if coverage in coverage_seen:
                    raise ValidationError(
                        f"{directory}: more than one {stage} entry covers "
                        f"{scope['type']} {scoped_id!r} in build profile {profile_id!r}"
                    )
                coverage_seen.add(coverage)
                if scope["type"] == "component":
                    bindings = {item["id"]: item for item in profiles[profile_id]["components"]}
                    binding = bindings.get(scoped_id)
                    if binding is None or binding.get("status", "supported") == "unsupported":
                        raise ValidationError(
                            f"{directory}: {stage} entry {entry['id']!r} targets unavailable component "
                            f"{scoped_id!r} in build profile {profile_id!r}"
                        )
                elif profile_id not in packages[scoped_id]["build_profiles"]:
                    raise ValidationError(
                        f"{directory}: {stage} entry {entry['id']!r} targets package {scoped_id!r} "
                        f"outside its build profiles"
                    )

    for entry in model["fidelity"]:
        label = f"{directory}: fidelity {entry['id']!r}"
        if entry["scope"]["type"] != "package":
            raise ValidationError(f"{label}: fidelity requires package scope")
        package = packages[entry["scope"]["package"]]
        result_component = entry["result_component"]
        if result_component not in package["components"]:
            raise ValidationError(f"{label}: result_component {result_component!r} is not in the tested package")
        output_names = {
            output["name"]
            for entrypoint in components[result_component]["interface"]["entrypoints"]
            for output in entrypoint["outputs"]
        }
        unknown_outputs = set(entry["outputs"]) - output_names
        if unknown_outputs:
            raise ValidationError(
                f"{label}: fidelity outputs are not exposed by {result_component!r}: "
                f"{', '.join(sorted(unknown_outputs))}"
            )

    devkit_profiles: set[str] = set()
    for entry in model["accuracy"]:
        if entry.get("engine") == DEVKIT_ACCURACY_ENGINE:
            _validate_devkit_accuracy(entry, packages, profiles, directory, devkit_profiles)

    for entry in model["benchmark"]:
        metric_names = {metric if isinstance(metric, str) else metric["name"] for metric in entry["metrics"]}
        missing = set(entry.get("required_metrics", ())) - metric_names
        if missing:
            raise ValidationError(
                f"{directory}: benchmark {entry['id']!r} required_metrics are not declared in metrics: "
                f"{', '.join(sorted(missing))}"
            )

    validate_model_tests(directory, model, schema_file.parent / TESTS_SCHEMA)


def _validate_devkit_accuracy(
    entry: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    directory: Path,
    seen_profiles: set[str],
) -> None:
    label = f"{directory}: accuracy {entry['id']!r}"
    scope = entry["scope"]
    scoped = packages[scope["package"]]["components"] if scope["type"] == "package" else [scope["component"]]
    if entry["result_component"] not in scoped:
        raise ValidationError(f"{label}: result_component {entry['result_component']!r} is not in its scope")
    # The accuracy stage runs with the MPKs of one compile invocation.
    for profile_id in entry["build_profiles"]:
        if not any(set(scoped) <= set(item["components"]) for item in compile_invocations_for(profiles[profile_id])):
            raise ValidationError(
                f"{label}: build profile {profile_id!r} must compile {', '.join(scoped)} in one compile invocation"
            )
    missing = set(entry.get("required_metrics", ())) - set(entry["metrics"])
    if missing:
        raise ValidationError(f"{label}: required_metrics are not declared in metrics: {', '.join(sorted(missing))}")
    undeclared = set(entry.get("reference", {}).get("metrics", {})) - set(entry["metrics"])
    if undeclared:
        raise ValidationError(
            f"{label}: reference metrics are not declared in metrics: {', '.join(sorted(undeclared))}"
        )
    # The accuracy stage runs scripts/accuracy.py once per build profile.
    repeated = seen_profiles & set(entry["build_profiles"])
    if repeated:
        raise ValidationError(
            f"{label}: more than one {DEVKIT_ACCURACY_ENGINE} entry covers build profile {sorted(repeated)[0]!r}"
        )
    seen_profiles.update(entry["build_profiles"])


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
    model = load_model(model_dir)
    validate_model(model_dir, model, schema_path, taxonomy_path)
    return model


def load_models(repository: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate every models/<id>/model.yaml, keyed by model ID."""
    root = Path(repository).expanduser().resolve()
    model_root = root / "models"
    discovered: dict[str, dict[str, Any]] = {}
    if not model_root.is_dir():
        raise ValidationError(f"models directory not found: {model_root}")

    for definition in sorted(model_root.glob("*/model.yaml")):
        model = load_and_validate_model(definition.parent, root / "schemas" / "model.schema.json")
        validate_artifact_metadata(model)
        model_id = str(model["id"])
        if model_id in discovered:
            raise ValidationError(f"duplicate model id {model_id!r}: {definition}")
        discovered[model_id] = model
    return discovered
