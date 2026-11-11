"""Ensure a benchmark runner has the selected Core artifact and its runtime dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError

PRODUCTION_ARTIFACTS = "https://artifacts.neat.sima.ai/core"
ENVIRONMENTS = {
    "production": "prod", "prod": "prod", "prd": "prod",
    "staging": "stg", "stg": "stg",
    "development": "dev", "dev": "dev",
}
INSTALL_ENVIRONMENTS = {"prod": "production", "stg": "staging", "dev": "dev"}
DEPENDENCY_PACKAGES = {"runner": "simaai-runner", "scheduler": "simaai-scheduler"}
DEPENDENCIES_FILE = "resolved-deps-manifest.json"
DEBIAN_VERSION = r"[0-9][A-Za-z0-9.+:~\-]*"


def _read_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _target_parts(target: str) -> tuple[str, str]:
    value = target.removeprefix("core@").strip("/")
    if ":" in value:
        branch, tag = value.rsplit(":", 1)
        return branch, tag
    if value == "latest" or re.fullmatch(r"[a-fA-F0-9]{7,40}", value):
        return "main", value
    return value, "latest"


def installed_matches(
    status: Any, metadata: Any, branch: str, tag: str, environment: str, dependencies: Any,
) -> bool:
    """Only a complete match with the selected artifact can authorize skipping installation."""
    try:
        resources = metadata["resources"]
        core = status["components"]["core"]
        channel = core["channel"]
        installed_tag = core["tag"]
        installed_environment = core["provenance"]["vulcanEnvironment"]
    except (KeyError, TypeError):
        return False
    if not all(isinstance(value, str) and value.strip() for value in (
        channel, installed_tag, installed_environment,
    )) or not isinstance(resources, list) or not all(isinstance(item, str) for item in resources):
        return False
    if (channel.strip() not in {branch, branch.replace("/", "-").replace(" ", "-")}
            or installed_tag.strip() != tag
            or ENVIRONMENTS.get(installed_environment.strip()) != environment):
        return False
    # A Core tag can be republished with newer dependencies. Tag equality alone is insufficient.
    names = [item.rsplit("/", 1)[-1] for item in resources]
    if not isinstance(dependencies, dict):
        return False
    # Stable package filenames carry no version; use the artifact's checksum-pinned dependency manifest.
    for component, package in DEPENDENCY_PACKAGES.items():
        candidates = [name for name in names if name == package + ".deb" or name.startswith(package + "_")]
        if candidates != [package + ".deb"]:
            return False
        dependency = dependencies.get(component)
        version = dependency.get("debian_version") if isinstance(dependency, dict) else None
        if not isinstance(version, str) or re.fullmatch(DEBIAN_VERSION, version) is None:
            return False
        try:
            if status["components"][component]["version"] != version:
                return False
        except (KeyError, TypeError):
            return False
    return True


def _published_dependencies(metadata: Any, index: str) -> Any:
    """Read Runner/Scheduler versions from the dependency manifest pinned by the selected artifact."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("resources"), list):
        return None
    resources = metadata["resources"]
    if not all(package + ".deb" in resources for package in DEPENDENCY_PACKAGES.values()):
        return None
    checksums = metadata.get("resources-checksum")
    expected = checksums.get(DEPENDENCIES_FILE) if isinstance(checksums, dict) else None
    if resources.count(DEPENDENCIES_FILE) != 1 or not isinstance(expected, str):
        return None
    content = _read_text(f"{index}/{DEPENDENCIES_FILE}")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected:
        return None
    return json.loads(content)


def ensure_neat_core(
    repository: Path,
    *,
    environment: str = "production",
    artifacts_base_url: str | None = None,
) -> None:
    """Resolve one artifact, verify Core/Runner/Scheduler, install if needed, then report the runner."""
    normalized_environment = ENVIRONMENTS.get(environment)
    if normalized_environment is None:
        raise ValidationError(f"unknown Core artifact environment: {environment}")
    target = resolve_benchmark_runtime(repository)["neat_core"]["install_target"]
    branch, tag = _target_parts(target)
    explicit_base = artifacts_base_url or os.environ.get("NEAT_ARTIFACTS_BASE_URL")
    # As in Apps, a non-production environment needs its own explicitly configured index.
    base = explicit_base or (PRODUCTION_ARTIFACTS if normalized_environment == "prod" else None)
    if normalized_environment != "prod" and base and base.rstrip("/") == PRODUCTION_ARTIFACTS:
        base = None
    artifact_branch = quote(quote(branch, safe=""), safe="")
    index = f"{base.rstrip('/')}/{artifact_branch}" if base else None
    if tag == "latest":
        if index is None:
            raise ValidationError("resolving floating Core targets outside production requires NEAT_ARTIFACTS_BASE_URL")
        try:
            tag = _read_text(f"{index}/latest.tag").strip()
        except (OSError, ValueError) as exc:
            raise ValidationError(f"could not resolve the published Core tag for {target}: {exc}") from exc
        if tag == "latest" or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) is None:
            raise ValidationError(f"invalid published Core tag for {target}: {tag!r}")

    matches = False
    if index is not None:
        try:
            status = subprocess.run(["neat", "--json"], check=True, capture_output=True, text=True, timeout=30)
            selected_index = f"{index}/{tag}"
            metadata = json.loads(_read_text(f"{selected_index}/metadata-minimal.json"))
            dependencies = _published_dependencies(metadata, selected_index)
            matches = installed_matches(
                json.loads(status.stdout), metadata, branch, tag, normalized_environment, dependencies,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            # Missing/unreadable status or metadata never authorizes using an unverified installed build.
            matches = False
    try:
        if matches:
            print(f"NEAT core already installed ({branch}/{tag}). Skipping install.", flush=True)
        else:
            subprocess.run([
                "sima-cli", "neat", "install", "--env", INSTALL_ENVIRONMENTS[normalized_environment],
                "--type", "minimal", f"core@{branch}:{tag}",
            ], check=True)
        python = os.environ.get("PYNEAT_PYTHON", str(Path.home() / "pyneat/bin/python"))
        subprocess.run([python, "-c", 'import pyneat; print("Core:", pyneat.__version__, pyneat.__file__)'], check=True)
        subprocess.run(["dpkg-query", "-W", "*neat*", "simaai-*"], check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(f"Neat Core benchmark setup failed: {exc}") from exc
