from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    failures = []
    for document in markdown_files:
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            resolved = (document.parent / path).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not failures, "broken local Markdown links:\n" + "\n".join(failures)


def test_developer_index_covers_the_contribution_path() -> None:
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for guide in ("adding-a-model.md", "metadata-schema.md", "local-development.md", "contributing.md"):
        assert f"]({guide})" in text


def test_local_guide_uses_supported_validation_commands() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    for command in (
        "model-registry validate ",
        "model-registry validate-catalog ",
        "model-registry fetch ",
        "model-registry fetch-calibration ",
        "model-registry matrix ",
        "model-registry resolve-compiler ",
        "model-registry verify-mpk ",
    ):
        assert command in text


def test_bounded_runner_documents_dynamic_image_geometry() -> None:
    text = (ROOT / "docs" / "benchmarking.md").read_text(encoding="utf-8")

    assert "original_width: 1280" in text
    assert "original_height: 720" in text
    assert "For image inputs with dynamic spatial dimensions" in text
    assert "Input geometry overrides are not supported" not in text
