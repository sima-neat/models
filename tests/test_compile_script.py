"""The registry formats every ONNX compile.py shares: tools/model_registry/compile_script.py."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator
from model_registry.compile_script import (
    CompileArguments,
    component_parameters,
    fidelity_report,
    parse_arguments,
    read_mpk_manifest,
)
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model" / "model.yaml"
FIDELITY_SCHEMA = ROOT / "schemas" / "numerical-fidelity.schema.json"
FIDELITY = {
    "status": "diagnostic",
    "dataset": {"id": "imagenet-2012", "variant": "cal", "sha256": "e" * 64, "size_bytes": 1},
    "selection": {"method": "seeded_random_without_replacement", "seed": 42, "offset": 20, "samples": 2},
    "outputs": ["logits", "boxes"],
}


def _model_dir(tmp_path: Path, fidelity: dict[str, Any] | None = None) -> Path:
    """The fixture model, whose one profile sets parameters, with numerical_fidelity when given."""
    model = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    model["build_profiles"]["fixture"]["components"]["model"]["parameters"] = {
        "precision": "w8a8",
        "calibration_samples": 20,
    }
    if fidelity is not None:
        model["numerical_fidelity"] = fidelity
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return model_dir


def _write_mpk(path: Path, members: list[tarfile.TarInfo], files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as stream:
        for member in members:
            stream.addfile(member)
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    return path


def test_parse_arguments_resolves_each_path_the_runner_passes(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path, FIDELITY)
    (tmp_path / "model.onnx").touch()
    (tmp_path / "cal.tar.gz").touch()
    (tmp_path / "fidelity").mkdir()
    argv = [
        "--model", str(tmp_path / "work" / ".." / "model.onnx"),
        "--output-dir", str(tmp_path / "out"),
        "--build-profile", "fixture",
        "--compile-invocation", "fixture",
        "--dataset", str(tmp_path / "cal.tar.gz"),
        "--fidelity-dataset", str(tmp_path / "fidelity"),
    ]

    arguments = parse_arguments("Compile the fixture.", model_dir, argv)

    root = tmp_path.resolve()
    assert arguments == CompileArguments(
        model=root / "model.onnx",
        output_dir=root / "out",
        build_profile="fixture",
        dataset=root / "cal.tar.gz",
        fidelity_dataset=root / "fidelity",
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("valid", None),
        ("other_invocation", "a build profile compiles in one invocation named after it"),
        ("checkpoint", "--model must be the verified ONNX source, not "),
        ("missing_model", "model does not exist: "),
        ("missing_dataset", "dataset does not exist: "),
        ("fidelity_not_passed", "model.yaml declares numerical_fidelity; pass --fidelity-dataset"),
        ("missing_fidelity_dataset", "fidelity dataset does not exist: "),
        ("fidelity_undeclared", None),
    ],
)
def test_parse_arguments_rejects_an_invocation_the_runner_would_not_make(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, message: str | None
) -> None:
    model_dir = _model_dir(tmp_path, None if case == "fidelity_undeclared" else FIDELITY)
    for name in ("model.onnx", "weights.pth", "cal.tar.gz", "fidelity.tar.gz"):
        (tmp_path / name).touch()
    if case == "missing_model":
        (tmp_path / "model.onnx").unlink()
    options = {
        "--model": tmp_path / ("weights.pth" if case == "checkpoint" else "model.onnx"),
        "--output-dir": tmp_path / "out",
        "--build-profile": "fixture",
        "--compile-invocation": "other" if case == "other_invocation" else "fixture",
        "--dataset": tmp_path / ("absent.tar.gz" if case == "missing_dataset" else "cal.tar.gz"),
        "--fidelity-dataset": tmp_path / ("absent.tar.gz" if case == "missing_fidelity_dataset" else "fidelity.tar.gz"),
    }
    if case in {"fidelity_not_passed", "fidelity_undeclared"}:
        del options["--fidelity-dataset"]
    argv = [str(item) for option in options.items() for item in option]

    if message is None:
        parsed = parse_arguments("Compile the fixture.", model_dir, argv)
        expected = None if case == "fidelity_undeclared" else (tmp_path / "fidelity.tar.gz").resolve()
        assert parsed.fidelity_dataset == expected
    else:
        with pytest.raises(SystemExit) as stopped:
            parse_arguments("Compile the fixture.", model_dir, argv)
        assert stopped.value.code == 2
        assert message in capsys.readouterr().err


def test_parse_arguments_accepts_a_model_onnx_link_to_a_file_without_the_suffix(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    blob = tmp_path / "blobs" / ("e" * 64)
    blob.parent.mkdir()
    blob.touch()
    (tmp_path / "model.onnx").symlink_to(blob)
    argv = ["--model", str(tmp_path / "model.onnx"), "--output-dir", str(tmp_path / "out"),
            "--build-profile", "fixture", "--compile-invocation", "fixture"]  # fmt: skip

    arguments = parse_arguments("Compile the fixture.", model_dir, argv)

    assert arguments.model == blob.resolve()


def test_component_parameters_reads_one_component_of_a_declared_profile(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)

    parameters = component_parameters(model_dir, "fixture", "model")

    assert parameters == {"precision": "w8a8", "calibration_samples": 20}
    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        component_parameters(model_dir, "w32a32", "model")
    with pytest.raises(ValidationError, match="build profile 'fixture' does not include component 'decoder'"):
        component_parameters(model_dir, "fixture", "decoder")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("valid", None),
        ("absolute", "unsafe MPK member: /etc/model.elf"),
        ("parent", "unsafe MPK member: ../model.elf"),
        ("symlink", "unsafe MPK member: link.elf"),
        ("hardlink", "unsafe MPK member: copy.elf"),
        ("no_manifest", "expected one MPK manifest, found []"),
        ("two_manifests", "expected one MPK manifest, found ['model_mpk.json', 'second_mpk.json']"),
    ],
)
def test_read_mpk_manifest_returns_the_members_and_the_one_manifest(
    tmp_path: Path, case: str, message: str | None
) -> None:
    manifest = {"plugins": [{"processor": "MLA", "sequence": 0}]}
    files = {"model.elf": b"elf", "model_mpk.json": json.dumps(manifest).encode()}
    members = []
    if case in {"absolute", "parent"}:
        files[{"absolute": "/etc/model.elf", "parent": "../model.elf"}[case]] = b"elf"
    if case in {"symlink", "hardlink"}:
        link = tarfile.TarInfo({"symlink": "link.elf", "hardlink": "copy.elf"}[case])
        link.type = tarfile.SYMTYPE if case == "symlink" else tarfile.LNKTYPE
        link.linkname = "model.elf"
        members.append(link)
    if case == "no_manifest":
        del files["model_mpk.json"]
    if case == "two_manifests":
        files["second_mpk.json"] = b"{}"
    archive = _write_mpk(tmp_path / "model_mpk.tar.gz", members, files)

    if message is None:
        assert read_mpk_manifest(archive) == (["model.elf", "model_mpk.json"], manifest)
    else:
        with pytest.raises(RuntimeError) as raised:
            read_mpk_manifest(archive)
        assert str(raised.value) == message


def test_fidelity_report_sums_each_output_over_every_sample() -> None:
    compared = [
        {"logits": (np.array([3.0, 4.0]), np.array([3.0, 4.0])), "boxes": (np.array([[1.0, 1.0]]), np.zeros((1, 2)))},
        {"logits": (np.array([0.0, 0.0]), np.array([0.0, 2.0])), "boxes": (np.zeros((1, 2)), np.zeros((1, 2)))},
    ]

    report = fidelity_report(
        "fixture", "w8a8", FIDELITY, compared, reference="ONNX source (ONNX Runtime)", candidate="Quantized graph"
    )

    Draft202012Validator(json.loads(FIDELITY_SCHEMA.read_text(encoding="utf-8"))).validate(report)
    assert {key: value for key, value in report.items() if key != "metrics"} == {
        "schema_version": 1,
        "model_id": "fixture",
        "build_profile_id": "w8a8",
        "status": "diagnostic",
        "reference": "ONNX source (ONNX Runtime)",
        "candidate": "Quantized graph",
        "dataset": {"id": "imagenet-2012", "variant": "cal", "sha256": "e" * 64},
        "selection": FIDELITY["selection"],
        "samples": 2,
    }
    logits, boxes = report["metrics"]
    # logits: expected (3, 4, 0, 0), actual (3, 4, 0, 2). Error energy 4 over 4 values, expected energy 25.
    assert logits == {
        "name": "logits",
        "cosine_similarity": pytest.approx(25 / np.sqrt(25 * 29)),
        "normalized_rmse": pytest.approx(np.sqrt(4 / 4) / np.sqrt(25 / 4)),
        "mean_absolute_error": pytest.approx(2 / 4),
        "max_absolute_error": 2.0,
    }
    # boxes: the candidate is all zeros, so it has no direction to agree with. The largest error is in sample 0.
    assert boxes == {
        "name": "boxes",
        "cosine_similarity": 0.0,
        "normalized_rmse": pytest.approx(1.0),
        "mean_absolute_error": pytest.approx(2 / 4),
        "max_absolute_error": 1.0,
    }


@pytest.mark.parametrize(
    ("compared", "error", "message"),
    [
        ([{"logits": (np.zeros(2), np.zeros(2))}], None, None),
        ([], ValueError, "a fidelity report needs at least one compared sample"),
        (
            [{"logits": (np.ones(2), np.ones(2))}, {"boxes": (np.ones(2), np.ones(2))}],
            ValueError,
            "fidelity sample 1 compares ['boxes'], the first sample ['logits']",
        ),
        ([{"logits": (np.ones((1, 2)), np.ones(2))}], RuntimeError, "logits of fidelity sample 0 has shape (2,)"),
        ([{"logits": (np.ones(2), np.array([1.0, np.nan]))}], RuntimeError, "contains non-finite values"),
    ],
)
def test_fidelity_report_rejects_samples_it_cannot_compare(
    compared: list[dict[str, tuple[np.ndarray, np.ndarray]]], error: type[Exception] | None, message: str | None
) -> None:
    def report() -> dict[str, Any]:
        return fidelity_report("fixture", "w8a8", FIDELITY, compared, reference="ONNX", candidate="AFE")

    if error is None:
        # Two all-zero outputs agree exactly.
        assert report()["metrics"][0]["cosine_similarity"] == 1.0
    else:
        with pytest.raises(error) as raised:
            report()
        assert message in str(raised.value)
