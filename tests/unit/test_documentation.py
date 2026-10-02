from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from model_registry.documentation import REQUIRED_SECTIONS, load_model_documentation
from model_registry.errors import ValidationError


def _readme(overview: str, *, title: str = "Fixture Model") -> str:
    sections = []
    for heading in REQUIRED_SECTIONS:
        body = overview if heading == "Overview" else f"Content for {heading.lower()}."
        sections.append(f"## {heading}\n\n{body}")
    return f"# {title}\n\n![{title} model hero](hero.png)\n\n" + "\n\n".join(sections) + "\n"


def _overview(words: int = 50) -> str:
    return " ".join(f"word{index}" for index in range(words))


def _png(width: int = 1200, height: int = 480, *, filter_byte: int = 0) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(bytes([filter_byte]) + b"\x22\x33\x44" * width for _ in range(height))
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(rows)),
        chunk(b"IEND", b""),
    ))


def _indexed_png(palette: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(b"\x00" + b"\x00" * 1200 for _ in range(480))
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", 1200, 480, 8, 3, 0, 0, 0)),
        chunk(b"PLTE", palette),
        chunk(b"IDAT", zlib.compress(rows)),
        chunk(b"IEND", b""),
    ))


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
        assert documentation["hero"]["source_path"] == f"models/{model_dir.name}/hero.png"


@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        (_readme(_overview(), title="Wrong title"), "H1 must exactly match"),
        (_readme(_overview()).replace("(hero.png)", "(https://example.invalid/hero.png)"), "must begin with H1"),
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
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match=message):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_rejects_wrong_hero_dimensions(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png(600, 240))

    with pytest.raises(ValidationError, match="must be 1200x480"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload[:24], "truncated PNG chunk"),
        (lambda payload: payload[:29] + bytes([payload[29] ^ 1]) + payload[30:], "invalid PNG checksum"),
        (lambda payload: payload[:-12], "invalid PNG structure"),
        (lambda payload: payload + b"trailing", "invalid PNG structure"),
    ],
)
def test_model_readme_contract_rejects_corrupt_hero_png(tmp_path: Path, mutate, message: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(mutate(_png()))

    with pytest.raises(ValidationError, match=message):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_rejects_undecodable_hero_pixels(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png(filter_byte=5))

    with pytest.raises(ValidationError, match="invalid PNG filter"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize("palette", [b"\x00", b"", b"\x00\x00\x00" * 257])
def test_model_readme_contract_rejects_invalid_indexed_palette(tmp_path: Path, palette: bytes) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_indexed_png(palette))

    with pytest.raises(ValidationError, match="invalid PNG palette"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_accepts_valid_indexed_palette(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_indexed_png(b"\x00\x00\x00"))

    documentation = load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)

    assert documentation["hero"]["media_type"] == "image/png"
