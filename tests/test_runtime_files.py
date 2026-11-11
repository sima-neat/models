from __future__ import annotations

import pytest
from model_registry.errors import ValidationError
from model_registry.fetch import sha256_file


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/outside",
        "a/../b",
        "a//b",
        "",
        "./a",
        "a\\b",
        "compile-outputs.json",
        "model/compile-result.json",
        "x_mpk.tar.gz",
    ],
)
def test_runtime_files_reject_unsafe_or_reserved_paths(tmp_path, name):
    from model_registry.runtime_files import validate_runtime_files

    with pytest.raises(ValidationError, match="unsafe runtime file"):
        validate_runtime_files([{"path": name, "sha256": "0" * 64}], tmp_path)


@pytest.mark.parametrize("problem", ["missing", "checksum", "duplicate", "symlink"])
def test_runtime_files_reject_invalid_payloads(tmp_path, problem):
    from model_registry.runtime_files import validate_runtime_files

    path = tmp_path / "tokenizer.json"
    path.write_text("{}")
    files = [{"path": path.name, "sha256": sha256_file(path)[0]}]
    if problem == "missing":
        path.unlink()
    elif problem == "checksum":
        path.write_text("changed")
    elif problem == "duplicate":
        files *= 2
    else:
        target = tmp_path / "original"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValidationError):
        validate_runtime_files(files, tmp_path)


@pytest.mark.parametrize("problem", ["checksum", "missing", "duplicate", "symlink"])
def test_restore_rejects_invalid_runtime_files_without_installing_them(tmp_path, problem):
    import zipfile

    from model_registry.runtime_files import restore_runtime_files

    archive = tmp_path / "package.zip"
    name = "model/tokenizer.json"
    files = [{"path": name, "sha256": "0" * 64}]
    with zipfile.ZipFile(archive, "w") as package:
        if problem == "symlink":
            member = zipfile.ZipInfo(name)
            member.external_attr = 0o120777 << 16
            package.writestr(member, "../../outside")
        elif problem != "missing":
            package.writestr(name, "{}")
            if problem == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    package.writestr(name, "{}")
    output = tmp_path / "restored"
    output.mkdir()
    with pytest.raises(ValidationError):
        restore_runtime_files(archive, files, output)
    assert not (output / name).exists()
