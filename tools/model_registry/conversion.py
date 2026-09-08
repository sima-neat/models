"""Run a model-owned converter in an isolated Python environment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .errors import ModelRegistryError, ValidationError
from .fetch import Fetcher, sha256_file
from .model import find_schema, source_members


def _local_file(directory: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValidationError(f"Expected a model-local filename: {name}")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"Missing regular model-local file: {path}")
    return path


def validate_conversion(model_dir: Path, config: str = "convert.yaml") -> dict[str, Any]:
    directory = model_dir.expanduser().resolve()
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", directory.name):
        raise ValidationError("Conversion model directory must use a snake_case model ID")
    config_path = _local_file(directory, config)
    try:
        declaration = yaml.safe_load(config_path.read_text())
        schema_dir = find_schema(directory).parent
        schema = json.loads((schema_dir / "convert.schema.json").read_text())
        model_schema = json.loads((schema_dir / "model.schema.json").read_text())
        registry = Registry().with_resource(model_schema["$id"], Resource.from_contents(model_schema))
        errors = list(Draft202012Validator(schema, registry=registry).iter_errors(declaration))
        if errors:
            raise ValidationError(f"{config_path}: {errors[0].message}")
        source = declaration["source"]
        members = source_members(source)
        if source["type"] == "file_set":
            paths = [name for name, _ in members]
            if any("." in name.split("/") or name.split("/")[0] == "source-set.json" for name in paths):
                raise ValidationError("Conversion source paths must be canonical and cannot use source-set.json")
            if len(paths) != len(set(name.casefold() for name in paths)):
                raise ValidationError("Conversion source paths must be unique, including case")
            if source["entrypoint"] not in paths:
                raise ValidationError("Conversion entrypoint must be a declared source member")
        _local_file(directory, "convert.py")
        files = [
            config_path,
            _local_file(directory, declaration["environment"]["requirements"]),
            *(_local_file(directory, path.name) for path in sorted(directory.glob("*.py"))),
        ]
        hashes = {path.name: sha256_file(path)[0] for path in files}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        return {
            "model_id": directory.name,
            "fingerprint": fingerprint,
            "config": config,
            "environment": {
                **declaration["environment"],
                "requirements_sha256": hashes[declaration["environment"]["requirements"]],
            },
            "source": source,
            "files": hashes,
        }
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValidationError(f"Cannot validate conversion {config_path}: {exc}") from exc


def _isolated_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(("PYTHON", "PIP_")) or name == "VIRTUAL_ENV":
            environment.pop(name, None)
    environment.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
    return environment


def convert_model(
    model_dir: Path,
    *,
    work_dir: Path,
    cache_dir: Path,
    python: str = "python3.12",
    config: str = "convert.yaml",
) -> dict[str, Any]:
    plan = validate_conversion(model_dir, config)
    directory = model_dir.expanduser().resolve()
    work = work_dir.expanduser().absolute()
    if work.exists() or work.is_symlink():
        raise ValidationError(f"Conversion work directory must be new: {work}")
    work.mkdir(parents=True)
    cache = cache_dir.expanduser().resolve()
    environment = _isolated_environment()
    log_path = work / "conversion.log"
    try:
        with log_path.open("w") as log:

            def run(command: list[str], *, capture: bool = False) -> str:
                log.write(f"Running {command[0]} {command[1:]}\n")
                log.flush()
                result = subprocess.run(
                    command,
                    env=environment,
                    cwd=directory,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE if capture else log,
                    stderr=log,
                )
                return result.stdout if capture else ""

            runtime = json.loads(
                run(
                    [
                        python,
                        "-E",
                        "-s",
                        "-c",
                        "import json,platform,sys; print(json.dumps("
                        "{'python':platform.python_version(),'system':platform.system(),"
                        "'machine':platform.machine(),'os':platform.platform()}))",
                    ],
                    capture=True,
                )
            )
            if ".".join(runtime["python"].split(".")[:2]) != plan["environment"]["python"]:
                raise ValidationError(f"Requested Python {plan['environment']['python']}, found {runtime['python']}")
            source, _, source_sha256, _ = Fetcher(cache / "sources").fetch_source(plan["source"])
            lock = directory / plan["environment"]["requirements"]
            cache_key = hashlib.sha256(
                json.dumps({**runtime, "lock": sha256_file(lock)[0]}, sort_keys=True).encode()
            ).hexdigest()
            for name, folder in (
                ("PIP_CACHE_DIR", cache / "wheels" / cache_key),
                ("HF_HOME", cache / "huggingface"),
                ("TORCH_HOME", cache / "torch"),
                ("TMPDIR", work / "tmp"),
            ):
                folder.mkdir(parents=True, exist_ok=True)
                environment[name] = str(folder)
            run([python, "-E", "-s", "-m", "venv", str(work / "venv")])
            interpreter = str(work / "venv" / "bin" / "python")
            environment["VIRTUAL_ENV"] = str(work / "venv")
            environment["PATH"] = str(work / "venv" / "bin") + os.pathsep + environment.get("PATH", "")
            run(
                [
                    interpreter,
                    "-E",
                    "-s",
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--cache-dir",
                    environment["PIP_CACHE_DIR"],
                    "-r",
                    str(lock),
                ]
            )
            output = work / "export"
            run(
                [
                    interpreter,
                    "-E",
                    "-s",
                    str(directory / "convert.py"),
                    "--source",
                    str(source),
                    "--output-dir",
                    str(output),
                ]
            )
            if output.is_symlink() or not output.is_dir():
                raise ValidationError("Converter output directory must be a regular directory")
            for path in output.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValidationError(f"Conversion output must contain regular files: {path.name}")
            graph = output / "model.onnx"
            if not graph.is_file() or not graph.stat().st_size:
                raise ValidationError("Converter did not create a nonempty model.onnx")
            validation = json.loads((output / "validation.json").read_text())
            if not isinstance(validation, dict) or validation.get("status") != "passed":
                raise ValidationError("Converter validation.json must report status passed")
            provenance = {
                **plan,
                "runtime": runtime,
                "source_identity": source_sha256,
                "dependencies": run(
                    [interpreter, "-E", "-s", "-m", "pip", "--isolated", "freeze"], capture=True
                ).splitlines(),
            }
            if os.environ.get("GITHUB_SHA"):
                provenance["revision"] = os.environ["GITHUB_SHA"]
            if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
                provenance["workflow_run"] = (
                    f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                )
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
            return {**plan, "output_dir": str(output), "provenance": provenance, "source_path": str(source)}
    except (OSError, ValueError, subprocess.CalledProcessError, ModelRegistryError) as exc:
        (work / "failure.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise ModelRegistryError(f"Conversion failed: {exc}. Diagnostics: {work}") from exc
