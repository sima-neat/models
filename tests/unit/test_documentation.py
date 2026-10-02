from __future__ import annotations

import re
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


def _png(
    width: int = 1200, height: int = 480, *, filter_byte: int = 0,
    color_type: int = 2, ancillary: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    channels = {2: 3, 6: 4}[color_type]
    rows = b"".join(bytes([filter_byte]) + b"\x22" * (width * channels) for _ in range(height))
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
        *(chunk(kind, data) for kind, data in ancillary),
        chunk(b"IDAT", zlib.compress(rows)),
        chunk(b"IEND", b""),
    ))


def _indexed_png(palette: bytes, *, pixel: int = 0, filter_type: int = 0) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    if filter_type == 1:
        encoded_row = bytes([pixel]) + b"\x00" * 1199
    else:
        encoded_row = bytes([pixel]) * 1200
    rows = b"".join(bytes([filter_type]) + encoded_row for _ in range(480))
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


def test_repository_model_readme_shell_blocks_do_not_use_redirection_shaped_placeholders() -> None:
    root = Path(__file__).resolve().parents[2]
    placeholder = re.compile(r"<[A-Za-z0-9_-]+>")
    for readme in sorted((root / "models").glob("*/README.md")):
        markdown = readme.read_text(encoding="utf-8")
        for block in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", markdown, flags=re.DOTALL):
            assert not placeholder.search(block), f"shell-unsafe placeholder in {readme}"


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_png(color_type=6, ancillary=((b"tRNS", b"\x00" * 6),)), "invalid PNG transparency chunk"),
        (
            _png(ancillary=((b"tRNS", b"\x00" * 6), (b"PLTE", b"\x00\x00\x00"))),
            "invalid PNG transparency chunk",
        ),
        (_png(ancillary=((b"tIME", b"\x00" * 7),)), "invalid PNG time chunk"),
    ],
)
def test_model_readme_contract_rejects_invalid_ancillary_chunks(
    tmp_path: Path, payload: bytes, message: str,
) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(payload)

    with pytest.raises(ValidationError, match=message):
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


def test_model_readme_contract_rejects_undefined_palette_index(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview()), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_indexed_png(b"\x00\x00\x00", pixel=1, filter_type=1))

    with pytest.raises(ValidationError, match="undefined PNG palette entry"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_rejects_pipe_table_without_leading_delimiter(tmp_path: Path) -> None:
    overview = "\n".join((
        "Name | Value",
        "--- | ---",
        "model | " + _overview(48),
    ))
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(overview), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="prose paragraphs only"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_rejects_setext_heading_in_overview(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "README.md").write_text(_readme(_overview() + "\n---"), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize("indent", ("    ", "\t", " \t", "  \t", "   \t"))
def test_model_readme_contract_rejects_indented_code_in_overview(tmp_path: Path, indent: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    overview = _overview() + f"\n\n{indent}model-registry validate-catalog"
    (model_dir / "README.md").write_text(_readme(overview), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="prose paragraphs only"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_rejects_setext_h2_outside_overview(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace(
        "Content for usage.", "Content for usage.\n\nUnexpected section\n---\n\nUnexpected content.",
    )
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize(
    "heading",
    ("<em>Unexpected</em>", "[Unexpected](https://example.invalid)", "**Unexpected**", "![Unexpected](hero.png)"),
)
def test_model_readme_contract_recognizes_inline_markup_in_setext_h2(tmp_path: Path, heading: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace(
        "Content for usage.", f"Content for usage.\n\n{heading}\n---\n\nUnexpected content.",
    )
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize(
    "heading",
    ("# Extra title", "##", "### Setup", "   ###", "   ### Setup", "#### Setup", "Unexpected section\n==="),
)
def test_model_readme_contract_rejects_nonstandard_headings(tmp_path: Path, heading: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace("Content for usage.", f"Content for usage.\n\n{heading}")
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize(
    "heading", ("> ### Setup", "- ### Setup", "1. ### Setup", "- > ### Setup", "> Unexpected\n> ---"),
)
def test_model_readme_contract_rejects_headings_in_markdown_containers(tmp_path: Path, heading: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace("Content for usage.", f"Content for usage.\n\n{heading}")
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


def test_model_readme_contract_allows_h3_in_reproduction_section(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace(
        "Content for reproducing the artifact.", "   ### Conversion\n\nRun the supported conversion command.",
    )
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    documentation = load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)

    assert documentation["source_path"] == "models/fixture/README.md"


@pytest.mark.parametrize("heading", ("###", "### #", "### ##", "   ### ###"))
def test_model_readme_contract_rejects_empty_h3_in_reproduction_section(tmp_path: Path, heading: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace(
        "Content for reproducing the artifact.", f"{heading}\n\nRun the supported conversion command.",
    )
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="H2 sections must be"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize("indent", ("", "   "))
def test_model_readme_contract_rejects_link_reference_definition_in_overview(tmp_path: Path, indent: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    overview = _overview() + f"\n\n{indent}[guide]: https://example.invalid/guide"
    (model_dir / "README.md").write_text(_readme(overview), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="prose paragraphs only"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize(
    "html",
    ("<!--\nmaintainer note\n-->", "<div\nclass=note>", "<?xml\nversion=1?>", "<![CDATA[\nmaintainer note\n]]>"),
)
def test_model_readme_contract_rejects_multiline_html_in_overview(tmp_path: Path, html: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    overview = _overview() + f"\n\n{html}"
    (model_dir / "README.md").write_text(_readme(overview), encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    with pytest.raises(ValidationError, match="prose paragraphs only"):
        load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)


@pytest.mark.parametrize("marker", ("```", "~~~~"))
def test_model_readme_contract_ignores_h2_inside_fenced_code(tmp_path: Path, marker: str) -> None:
    model_dir = tmp_path / "models" / "fixture"
    model_dir.mkdir(parents=True)
    markdown = _readme(_overview()).replace(
        "Content for usage.", f"{marker}bash\n## Setup\necho setup\nUnexpected\n---\n{marker}",
    )
    (model_dir / "README.md").write_text(markdown, encoding="utf-8")
    (model_dir / "hero.png").write_bytes(_png())

    documentation = load_model_documentation(model_dir, "Fixture Model", repository=tmp_path)

    assert documentation["source_path"] == "models/fixture/README.md"
