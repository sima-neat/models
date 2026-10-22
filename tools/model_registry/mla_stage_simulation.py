"""MLA stage simulation: the Model Compiler simulator's operation counts and latency for each compiled MLA stage.

After compile.py, run_model_compilation.sh runs ``model-registry simulate-mla-stages``, which runs
``mla-msim --dry-run`` on every MLA ELF of every component's MPK and writes mla-stage-simulation.json beside
compile-outputs.json. The package run publishes it, and the model browser compares each stage's simulated
operations with the MLA stage latency Core measures on a DevKit. Stages are numbered as Core numbers them:
the MPK manifest's MLA and A65 plugins in execution order, from zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .errors import ValidationError

SIMULATION_FILE = "mla-stage-simulation.json"
SCHEMA_VERSION = 1
MLA_MSIM = "mla-msim"
# Clock cycles mla-msim simulates before it stops; at the MLA's 1 GHz this is 2 s, far above any stage's latency.
CYCLE_LIMIT = 2_000_000_000
# A dry run of the largest stage takes about 2 s.
TIMEOUT_S = 600
# Core counts these processors as stages; only MLA stages run on the MLA.
STAGE_PROCESSORS = {"MLA", "A65"}


def simulate_mla_stages(mpk: Path) -> dict[str, Any]:
    """One MPK's simulated MLA stages, with status "ok" or why they were not simulated."""
    msim = shutil.which(MLA_MSIM)
    if msim is None:
        return {"status": f"{MLA_MSIM} not found; activate the Model Compiler", "stages": []}
    try:
        with tempfile.TemporaryDirectory() as scratch:
            stages = [
                {"stage_index": stage_index, **_simulate(msim, elf)}
                for stage_index, elf in _extract_mla_stages(mpk, Path(scratch))
            ]
    except subprocess.CalledProcessError as error:
        return {"status": f"{MLA_MSIM} exited with {error.returncode}: {error.stderr.strip()[-500:]}", "stages": []}
    except (OSError, ValueError, tarfile.TarError, subprocess.TimeoutExpired) as error:
        return {"status": f"{MLA_MSIM} failed: {error}", "stages": []}
    return {"status": "ok", "stages": stages}


def write_mla_stage_simulation(output_dir: Path) -> Path:
    """Simulate every component compile-outputs.json records and write mla-stage-simulation.json beside it."""
    outputs_path = output_dir / "compile-outputs.json"
    try:
        outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"{outputs_path}: could not load compile outputs: {error}") from error
    components = {}
    for output in outputs["outputs"]:
        if "mpk_path" not in output:
            raise ValidationError(f"{outputs_path}: {output['component_id']} has no MPK to simulate")
        components[output["component_id"]] = simulate_mla_stages(output_dir / output["mpk_path"])
    path = output_dir / SIMULATION_FILE
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": outputs["model_id"],
        "build_profile_id": outputs["build_profile_id"],
        "components": components,
    }, indent=2) + "\n", encoding="utf-8")
    return path


def _extract_mla_stages(mpk: Path, directory: Path) -> list[tuple[int, Path]]:
    """Each MLA stage's index and its ELF, extracted into ``directory``."""
    with tarfile.open(mpk, "r:gz") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        manifests = [name for name in members if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise ValueError(f"{mpk.name} must hold one *_mpk.json manifest, not {len(manifests)}")
        plugins = json.load(archive.extractfile(members[manifests[0]])).get("plugins", [])
        stages = [plugin for plugin in _execution_order(plugins) if plugin.get("processor", "").upper() in
                  STAGE_PROCESSORS]
        extracted = []
        for stage_index, plugin in enumerate(stages):
            if plugin["processor"].upper() != "MLA":
                continue
            name = Path((plugin.get("resources") or {}).get("executable") or "").name
            if name not in members:
                raise ValueError(f"{mpk.name} has no MLA ELF {name!r} for {plugin.get('name')}")
            elf = directory / name
            elf.write_bytes(archive.extractfile(members[name]).read())
            extracted.append((stage_index, elf))
    return extracted


def _execution_order(plugins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plugins by their sequence, as Core orders them; one without a sequence follows those with one."""
    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, plugin = item
        sequence = plugin.get("sequence")
        return (sequence if isinstance(sequence, int) and sequence >= 0 else 1_000_000 + index, index)

    return [plugin for _, plugin in sorted(enumerate(plugins), key=key)]


def _simulate(msim: str, elf: Path) -> dict[str, Any]:
    """Run mla-msim on one ELF and read its operation counts and simulated latency."""
    stats_path = elf.with_suffix(".stats")
    subprocess.run(
        [msim, str(elf), "--dry-run", f"--limit={CYCLE_LIMIT}", f"--stats={stats_path}"],
        check=True, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    stats = _read_stats(stats_path)
    latency_us = float(stats["Latency [ms]"]) * 1000
    # At 1 GHz a cycle is 1 ns; a run that reached the limit stopped early, so its latency and counts are partial.
    if latency_us * 1000 >= CYCLE_LIMIT:
        raise ValueError(f"{elf.name} reached the {CYCLE_LIMIT}-cycle simulation limit")
    return {
        "effective_ops": int(stats["Effective arithmetic ops"]),
        "total_ops": int(stats["Total arithmetic ops"]),
        "simulated_latency_us": latency_us,
    }


def _read_stats(path: Path) -> dict[str, str]:
    """The ``Key: value`` lines of an mla-msim --stats file."""
    stats = dict(line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
    missing = {"Latency [ms]", "Effective arithmetic ops", "Total arithmetic ops"} - set(stats)
    if missing:
        raise ValueError(f"{path.name} lacks {', '.join(sorted(missing))}")
    return stats
