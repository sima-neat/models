from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "target-branch-checks.yml"
MAX_FILE_BYTES = 2 * 1024 * 1024
# Test fixtures that must be committed although .gitignore excludes them or they exceed MAX_FILE_BYTES.
ALLOWED_FILES: frozenset[str] = frozenset()


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def committed_file_problems(root: Path, allowed: frozenset[str] = ALLOWED_FILES) -> list[str]:
    def tracked(*options: str) -> list[str]:
        output = subprocess.run(["git", "ls-files", "-z", *options], cwd=root, check=True, capture_output=True).stdout
        return [path for path in output.decode().split("\0") if path and path not in allowed]

    problems = [
        f"{path}: excluded by .gitignore"
        for path in tracked("--cached", "--ignored", "--exclude-per-directory=.gitignore")
    ]
    sizes = {path: (root / path).stat().st_size for path in tracked() if (root / path).is_file()}
    return problems + [
        f"{path}: {size} bytes, over {MAX_FILE_BYTES}" for path, size in sizes.items() if size > MAX_FILE_BYTES
    ]


def test_pull_requests_run_only_the_unprivileged_checks() -> None:
    workflow = load(WORKFLOW)

    assert set(workflow["on"]) == {"pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert "${{ secrets." not in WORKFLOW.read_text(encoding="utf-8")
    assert all("secrets" not in job for job in workflow["jobs"].values())
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert {name: job["uses"] for name, job in workflow["jobs"].items()} == {
        "package": "./.github/workflows/model-registry-package.yml",
        "portal": "./.github/workflows/developer-docs.yml",
    }


def test_only_target_branch_checks_runs_for_pull_requests() -> None:
    triggered = {
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if {"pull_request", "pull_request_target"} & set(load(path)["on"])
    }

    assert triggered == {"target-branch-checks.yml", "pr-sanitization.yml"}


def test_checked_workflows_run_only_on_hosted_runners() -> None:
    for name in ("model-registry-package.yml", "developer-docs.yml"):
        jobs = load(WORKFLOWS / name)["jobs"]
        assert all(job["runs-on"] == "ubuntu-latest" for job in jobs.values()), name


def test_pull_requests_lint_python_with_ruff() -> None:
    runs = [step.get("run") for step in load(WORKFLOWS / "model-registry-package.yml")["jobs"]["build"]["steps"]]

    assert runs.index("python -m ruff check .") > runs.index("python -m pip install -e '.[dev]'")


def test_no_committed_file_is_ignored_or_over_the_size_limit() -> None:
    problems = committed_file_problems(ROOT)

    assert not problems, "remove these files or add test fixtures to ALLOWED_FILES:\n" + "\n".join(problems)


def test_committed_file_check_rejects_forced_binaries_and_large_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.onnx\n")
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "fixture.onnx").write_bytes(b"onnx")
    (tmp_path / "large.bin").write_bytes(bytes(MAX_FILE_BYTES + 1))
    (tmp_path / "limit.bin").write_bytes(bytes(MAX_FILE_BYTES))
    subprocess.run(["git", "-C", str(tmp_path), "add", "--force", "."], check=True)

    assert committed_file_problems(tmp_path, frozenset({"fixture.onnx"})) == [
        "model.onnx: excluded by .gitignore",
        f"large.bin: {MAX_FILE_BYTES + 1} bytes, over {MAX_FILE_BYTES}",
    ]
