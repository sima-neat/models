from __future__ import annotations

from pathlib import Path

import pytest
from model_registry.documentation import REQUIRED_SECTIONS, load_model_documentation
from model_registry.errors import ValidationError


def _readme(overview: str, *, title: str = "Fixture Model") -> str:
    sections = []
    for heading in REQUIRED_SECTIONS:
        body = overview if heading == "Overview" else f"Content for {heading.lower()}."
        sections.append(f"## {heading}\n\n{body}")
    return f"# {title}\n\n" + "\n\n".join(sections) + "\n"


def _overview(words: int = 50) -> str:
    return " ".join(f"word{index}" for index in range(words))


def test_repository_model_readmes_follow_the_documentation_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    for manifest in sorted((root / "models").glob("*/model.yaml")):
        model_dir = manifest.parent
        display_name = next(
            line.split(":", 1)[1].strip().strip('"\'')
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.startswith("display_name:")
        )
        documentation = load_model_documentation(model_dir, display_name, repository=root)
        assert documentation["source_path"] == f"models/{model_dir.name}/README.md"
        assert len(documentation["readme_sha256"]) == 64


@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        (_readme(_overview(), title="Wrong title"), "H1 must exactly match"),
        (_readme(_overview()).replace("## Usage", "## How to use"), "H2 sections must be"),
        (_readme(_overview(49)), "50 to 120 words"),
        (_readme(_overview(121)), "50 to 120 words"),
        (_readme("\n\n".join((_overview(20), _overview(20), _overview(20)))), "one or two prose paragraphs"),
        (_readme("- " + _overview()), "prose paragraphs only"),
        (_readme(_overview()).rstrip("\n"), "end with a newline"),
    ],
)
def test_model_readme_contract_rejects_invalid_content(tmp_path: Path, markdown: str, message: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)
