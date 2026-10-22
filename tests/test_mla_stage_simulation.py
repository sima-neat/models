from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from model_registry.cli import main
from model_registry.errors import ValidationError
from model_registry.mla_stage_simulation import simulate_mla_stages, write_mla_stage_simulation

ROOT = Path(__file__).resolve().parents[1]
# mla-msim --dry-run --stats output for the resnet_50 w8a8 MPK, from the installed Model Compiler.
RESNET_STATS = ROOT / "tests/fixtures/mla-msim-resnet-50-w8a8.stats"
RESNET_STAGE = {"effective_ops": 8213719016, "total_ops": 10445905152, "simulated_latency_us": 678.432}
# Listed out of order: Core orders plugins by sequence, and one without a sequence runs last.
PLUGINS = [
    {"name": "MLA_1", "processor": "MLA", "resources": {"executable": "model_stage2_mla.elf"}},
    {"name": "postprocess", "processor": "A65", "sequence": 3},
    {"name": "MLA_0", "processor": "MLA", "sequence": 2, "resources": {"executable": "model_stage1_mla.elf"}},
    {"name": "quantize", "processor": "EV74", "sequence": 1, "resources": {"executable": "kernel_name_tbd"}},
]


def _mpk(path: Path, plugins: list[dict] = PLUGINS) -> Path:
    members = {"model_mpk.json": json.dumps({"plugins": plugins}).encode()}
    members.update({plugin["resources"]["executable"]: b"elf" for plugin in plugins if plugin["processor"] == "MLA"})
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def mla_msim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake mla-msim on PATH: it logs its arguments and writes ``stats`` to its --stats file, or fails."""
    bin_dir, log = tmp_path / "bin", tmp_path / "mla-msim.log"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))

    def install(stats: str = RESNET_STATS.read_text(encoding="utf-8"), exit_code: int = 0) -> Path:
        script = bin_dir / "mla-msim"
        script.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
            f"if {exit_code}:\n"
            "    sys.stderr.write('ELF load failed\\n')\n"
            f"    sys.exit({exit_code})\n"
            "stats = next(arg for arg in sys.argv if arg.startswith('--stats='))[len('--stats='):]\n"
            f"open(stats, 'w').write({stats!r})\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return log

    return install


def test_each_mla_stage_is_simulated_under_cores_stage_index(tmp_path: Path, mla_msim) -> None:
    log = mla_msim()

    simulation = simulate_mla_stages(_mpk(tmp_path / "model.tar.gz"))

    assert simulation == {"status": "ok", "stages": [{"stage_index": 0, **RESNET_STAGE},
                                                      {"stage_index": 2, **RESNET_STAGE}]}
    runs = log.read_text(encoding="utf-8").splitlines()
    assert [run.split()[0].rsplit("/", 1)[1] for run in runs] == ["model_stage1_mla.elf", "model_stage2_mla.elf"]
    assert all(" --dry-run --limit=2000000000 --stats=" in run for run in runs)


def test_a_missing_simulator_is_recorded_as_the_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    assert simulate_mla_stages(_mpk(tmp_path / "model.tar.gz")) == {
        "status": "mla-msim not found; activate the Model Compiler", "stages": []}


@pytest.mark.parametrize(("simulator", "status"), [
    ({"exit_code": 3}, "mla-msim exited with 3: ELF load failed"),
    ({"stats": "Latency [ms]: 2000\nTotal arithmetic ops: 1\nEffective arithmetic ops: 1\n"},
     "mla-msim failed: model_stage1_mla.elf reached the 2000000000-cycle simulation limit"),
    ({"stats": "Latency [ms]: 0.5\n"},
     "mla-msim failed: model_stage1_mla.stats lacks Effective arithmetic ops, Total arithmetic ops"),
])
def test_a_failed_simulation_is_recorded_as_the_status(tmp_path: Path, mla_msim, simulator: dict,
                                                       status: str) -> None:
    mla_msim(**simulator)

    assert simulate_mla_stages(_mpk(tmp_path / "model.tar.gz")) == {"status": status, "stages": []}


def test_an_mpk_without_its_mla_elf_is_recorded_as_the_status(tmp_path: Path, mla_msim) -> None:
    mla_msim()
    mpk = tmp_path / "model.tar.gz"
    with tarfile.open(mpk, "w:gz") as archive:
        data = json.dumps({"plugins": PLUGINS}).encode()
        info = tarfile.TarInfo("model_mpk.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    assert simulate_mla_stages(mpk) == {
        "status": "mla-msim failed: model.tar.gz has no MLA ELF 'model_stage1_mla.elf' for MLA_0", "stages": []}


def test_simulate_mla_stages_writes_every_component_beside_the_compile_outputs(tmp_path: Path, mla_msim,
                                                                              capsys) -> None:
    mla_msim()
    for component in ("backbone", "decoder"):
        (tmp_path / component).mkdir()
        _mpk(tmp_path / component / "model.tar.gz")
    (tmp_path / "compile-outputs.json").write_text(json.dumps({
        "model_id": "rf_detr_medium", "build_profile_id": "w8a8_w16a16",
        "outputs": [{"component_id": component, "mpk_path": f"{component}/model.tar.gz"}
                    for component in ("backbone", "decoder")],
    }), encoding="utf-8")

    assert main(["simulate-mla-stages", str(tmp_path)]) == 0

    path = tmp_path / "mla-stage-simulation.json"
    assert capsys.readouterr().out.strip() == str(path)
    simulation = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas/mla-stage-simulation.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(simulation)
    assert simulation["model_id"] == "rf_detr_medium"
    assert set(simulation["components"]) == {"backbone", "decoder"}
    assert all(component["status"] == "ok" for component in simulation["components"].values())


def test_a_compile_output_without_an_mpk_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "compile-outputs.json").write_text(json.dumps({
        "model_id": "llama", "build_profile_id": "default", "outputs": [{"component_id": "model"}],
    }), encoding="utf-8")

    with pytest.raises(ValidationError, match="model has no MPK to simulate"):
        write_mla_stage_simulation(tmp_path)
