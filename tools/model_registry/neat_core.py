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
STATUS_TIMEOUT_S = 30


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


def _component(status: Any, name: str) -> dict:
    """Return one component of the `neat --json` status, or an empty one when the status lacks it."""
    components = status.get("components") if isinstance(status, dict) else None
    component = components.get(name) if isinstance(components, dict) else None
    return component if isinstance(component, dict) else {}


def _shown(value: Any) -> str:
    """Show an installed value in the public log only when it is a non-empty string."""
    return value if isinstance(value, str) and value else "missing"


def installed_mismatch(
    status: Any, metadata: Any, branch: str, tag: str, environment: str, dependencies: Any,
) -> str | None:
    """Name the first difference between the installed Core and the selected artifact, or None on a complete match.

    Only a complete match can authorize skipping installation.
    """
    resources = metadata.get("resources") if isinstance(metadata, dict) else None
    if not isinstance(resources, list) or not all(isinstance(item, str) for item in resources):
        return "the published Core metadata has no readable resource list"

    core = _component(status, "core")
    provenance = core.get("provenance") if isinstance(core.get("provenance"), dict) else {}
    channel, installed_tag, installed_environment = (
        value.strip() if isinstance(value, str) else None
        for value in (core.get("channel"), core.get("tag"), provenance.get("vulcanEnvironment"))
    )
    if channel not in {branch, branch.replace("/", "-").replace(" ", "-")}:
        # `neat` assumes a channel when the installed package names none; the tag tells the two cases apart.
        assumed = " (assumed)" if core.get("channelAssumed") is True else ""
        return (f"installed Core channel is {_shown(channel)}{assumed} and tag is {_shown(installed_tag)}, "
                f"pinned {branch}")
    if installed_tag != tag:
        return f"installed Core tag is {_shown(installed_tag)}, pinned {tag}"
    if ENVIRONMENTS.get(installed_environment) != environment:
        return f"installed Core environment is {_shown(installed_environment)}, pinned {environment}"

    # A Core tag can be republished with newer dependencies. Tag equality alone is insufficient.
    if not isinstance(dependencies, dict):
        return "the published dependency manifest could not be verified"
    names = [item.rsplit("/", 1)[-1] for item in resources]
    # Stable package filenames carry no version; use the artifact's checksum-pinned dependency manifest.
    for component, package in DEPENDENCY_PACKAGES.items():
        candidates = [name for name in names if name == package + ".deb" or name.startswith(package + "_")]
        if candidates != [package + ".deb"]:
            return f"the published Core metadata does not list exactly one {package} package"
        dependency = dependencies.get(component)
        version = dependency.get("debian_version") if isinstance(dependency, dict) else None
        if not isinstance(version, str) or re.fullmatch(DEBIAN_VERSION, version) is None:
            return f"the published dependency manifest has no valid {component} version"
        installed_version = _component(status, component).get("version")
        if installed_version != version:
            return f"installed {package} is {_shown(installed_version)}, pinned {version}"
    return None


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


def _check_installed_core(selected_index: str, branch: str, tag: str, environment: str) -> str | None:
    """Return why the runner cannot reuse its installed Core for the selected artifact, or None when it can.

    The reason goes to a public CI log, so it names only versions, tags, channels, and environments.
    """
    try:
        result = subprocess.run(
            ["neat", "--json"], check=True, capture_output=True, text=True, timeout=STATUS_TIMEOUT_S,
        )
        status = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return f"neat --json timed out after {STATUS_TIMEOUT_S} s"
    except subprocess.SubprocessError:
        return "neat --json failed"
    except OSError:
        return "neat --json could not start"
    except ValueError:
        return "neat --json returned unreadable output"

    try:
        metadata = json.loads(_read_text(f"{selected_index}/metadata-minimal.json"))
    except (OSError, ValueError):
        return "the published Core metadata could not be read"
    try:
        dependencies = _published_dependencies(metadata, selected_index)
    except (OSError, ValueError):
        # installed_mismatch reports a manifest that could not be read as unverified.
        dependencies = None
    return installed_mismatch(status, metadata, branch, tag, environment, dependencies)


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
    base = PRODUCTION_ARTIFACTS if normalized_environment == "prod" else explicit_base
    if normalized_environment != "prod" and base and base.rstrip("/") == PRODUCTION_ARTIFACTS:
        base = None
    # Vulcan stores an encoded ref key; URL transport escapes it again to preserve slashes in the ref.
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

    if index is None:
        reason = "no NEAT_ARTIFACTS_BASE_URL to compare the installed Core with"
    else:
        reason = _check_installed_core(f"{index}/{tag}", branch, tag, normalized_environment)
    try:
        if reason is None:
            print(f"NEAT core already installed ({branch}/{tag}). Skipping install.", flush=True)
        else:
            print(f"Installing NEAT core ({branch}/{tag}): {reason}.", flush=True)
            subprocess.run([
                "sima-cli", "neat", "install", "--env", INSTALL_ENVIRONMENTS[normalized_environment],
                "--type", "minimal", f"core@{branch}:{tag}",
            ], check=True)
        python = os.environ.get("PYNEAT_PYTHON", str(Path.home() / "pyneat/bin/python"))
        subprocess.run([python, "-c", 'import pyneat; print("Core:", pyneat.__version__, pyneat.__file__)'], check=True)
        subprocess.run(["dpkg-query", "-W", "*neat*", "simaai-*"], check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(f"Neat Core benchmark setup failed: {exc}") from exc
