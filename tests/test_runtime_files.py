from __future__ import annotations

import pytest
from model_registry.errors import ValidationError
from model_registry.fetch import sha256_file
from model_registry.runtime_files import validate_runtime_files


def test_runtime_files_allow_aliases_above_the_trusted_root(tmp_path):
    actual = tmp_path / "actual" / "profile"
    actual.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(actual.parent, target_is_directory=True)
    asset = actual / "tokenizer.json"
    asset.write_bytes(b"{}")
    files = [{"path": asset.name, "sha256": sha256_file(asset)[0]}]

    assert validate_runtime_files(files, alias / "profile") == files


def test_runtime_files_reject_symlink_directories_below_the_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tokenizer.json").write_bytes(b"{}")
    root = tmp_path / "profile"
    root.mkdir()
    (root / "text_encoder").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValidationError, match="unsafe runtime file"):
        validate_runtime_files([{"path": "text_encoder/tokenizer.json", "sha256": "0" * 64}], root)


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
    with pytest.raises(ValidationError, match="unsafe runtime file"):
        validate_runtime_files([{"path": name, "sha256": "0" * 64}], tmp_path)


@pytest.mark.parametrize("problem", ["missing", "checksum", "duplicate", "symlink"])
def test_runtime_files_reject_invalid_payloads(tmp_path, problem):
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


def runtime_profile(root):
    import json

    root.mkdir()
    asset = root / "derived/constants.json"
    asset.parent.mkdir()
    asset.write_text('{"example": 1}')
    (root / "compile-outputs.json").write_text(json.dumps({
        "runtime_files": [{"path": "derived/constants.json", "sha256": sha256_file(asset)[0]}],
    }))
    return asset


def test_copy_preserves_pinned_resources_and_is_idempotent(tmp_path):
    from model_registry.runtime_files import copy_runtime_files

    source, output = tmp_path / "profile", tmp_path / "resources"
    asset = runtime_profile(source)
    output.mkdir()
    (output / "pinned.json").write_text("pinned")
    assert copy_runtime_files(source, output) == output
    assert (output / "derived/constants.json").read_bytes() == asset.read_bytes()
    assert (output / "pinned.json").read_text() == "pinned"
    copy_runtime_files(source, output, copy_function=lambda *args: pytest.fail("must reuse identical files"))


@pytest.mark.parametrize("problem", ["corrupt-source", "collision", "symlink", "bad-copy"])
def test_copy_rejects_invalid_inputs_and_never_overwrites_resources(tmp_path, problem):
    from model_registry.runtime_files import copy_runtime_files

    source, output = tmp_path / "profile", tmp_path / "resources"
    asset = runtime_profile(source)
    target = output / "derived/constants.json"
    if problem == "corrupt-source":
        asset.write_text("corrupt")
    elif problem == "collision":
        target.parent.mkdir(parents=True)
        target.write_text("existing pinned resource")
    elif problem == "symlink":
        output.mkdir()
        (output / "derived").symlink_to(source, target_is_directory=True)
    copy = (lambda *args: None) if problem == "bad-copy" else __import__("shutil").copy2
    with pytest.raises(ValidationError):
        copy_runtime_files(source, output, copy_function=copy)
    if problem == "collision":
        assert target.read_text() == "existing pinned resource"
    assert not output.exists() or not (output / "compile-outputs.json").exists()


def test_copy_with_no_supporting_files_does_not_create_an_output(tmp_path):
    from model_registry.runtime_files import copy_runtime_files

    assert copy_runtime_files(tmp_path / "empty", tmp_path / "output") is None
    assert not (tmp_path / "output").exists()
