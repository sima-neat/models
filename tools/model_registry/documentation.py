from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Any

from .errors import ValidationError

MAX_README_BYTES = 128 * 1024
MAX_HERO_BYTES = 2 * 1024 * 1024
HERO_FILENAME = "hero.png"
HERO_WIDTH = 1200
HERO_HEIGHT = 480
REQUIRED_SECTIONS = (
    "Overview",
    "Usage",
    "Model interface",
    "Available builds",
    "Quality and performance",
    "Source and license",
    "Reproducing the artifact",
)
_H2_RE = re.compile(r"^## ([^#].*?)\s*$")
_OVERVIEW_FORBIDDEN_RE = re.compile(r"^(?:\s*(?:[-*+] |\d+[.)] |```|~~~|#|> |\||!\[)|\s*<[^>]+>)")


def _load_hero(model_dir: Path, display_name: str, repository: Path) -> dict[str, Any]:
    hero = model_dir / HERO_FILENAME
    try:
        payload = hero.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read model hero image {hero}: {exc}") from exc
    if not payload or len(payload) > MAX_HERO_BYTES:
        raise ValidationError(f"model hero image must contain between 1 and {MAX_HERO_BYTES} bytes: {hero}")
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise ValidationError(f"model hero image must be a PNG: {hero}")
    width, height = struct.unpack(">II", payload[16:24])
    if (width, height) != (HERO_WIDTH, HERO_HEIGHT):
        raise ValidationError(f"model hero image must be {HERO_WIDTH}x{HERO_HEIGHT}, found {width}x{height}: {hero}")
    try:
        source_path = hero.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValidationError(f"model hero image must be inside the repository: {hero}") from exc
    return {
        "artifact_name": "hero",
        "markdown_path": HERO_FILENAME,
        "alt_text": f"{display_name} model hero",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "media_type": "image/png",
        "source_path": source_path,
        "path": hero,
    }


def load_model_documentation(model_dir: Path, display_name: str, *, repository: Path) -> dict[str, Any]:
    readme = model_dir / "README.md"
    try:
        payload = readme.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read model README {readme}: {exc}") from exc
    if not payload or len(payload) > MAX_README_BYTES:
        raise ValidationError(f"model README must contain between 1 and {MAX_README_BYTES} bytes: {readme}")
    try:
        markdown = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"model README must be UTF-8: {readme}") from exc
    if not markdown.endswith("\n"):
        raise ValidationError(f"model README must end with a newline: {readme}")

    lines = markdown.splitlines()
    if not lines or lines[0] != f"# {display_name}":
        raise ValidationError(f"model README H1 must exactly match display_name {display_name!r}: {readme}")
    hero = _load_hero(model_dir, display_name, repository)
    expected_hero = f"![{hero['alt_text']}]({hero['markdown_path']})"
    if len(lines) < 5 or lines[1] != "" or lines[2] != expected_hero or lines[3] != "" or lines[4] != "## Overview":
        raise ValidationError(
            f"model README must begin with H1, blank line, {expected_hero}, blank line, and ## Overview: {readme}"
        )

    sections = [(index, match.group(1)) for index, line in enumerate(lines) if (match := _H2_RE.fullmatch(line))]
    names = tuple(name for _, name in sections)
    if names != REQUIRED_SECTIONS:
        raise ValidationError(
            f"model README H2 sections must be {', '.join(REQUIRED_SECTIONS)} in order: {readme}"
        )
    overview_start = sections[0][0] + 1
    overview_end = sections[1][0]
    overview = "\n".join(lines[overview_start:overview_end]).strip()
    if not overview:
        raise ValidationError(f"model README Overview must not be empty: {readme}")
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", overview) if paragraph.strip()]
    if not 1 <= len(paragraphs) <= 2:
        raise ValidationError(
            f"model README Overview must contain one or two prose paragraphs, found {len(paragraphs)}: {readme}"
        )
    word_count = len(re.findall(r"\S+", overview))
    if not 50 <= word_count <= 120:
        raise ValidationError(f"model README Overview must contain 50 to 120 words, found {word_count}: {readme}")
    if any(_OVERVIEW_FORBIDDEN_RE.match(line) for line in overview.splitlines()):
        raise ValidationError(f"model README Overview must contain prose paragraphs only: {readme}")

    try:
        source_path = readme.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValidationError(f"model README must be inside the repository: {readme}") from exc
    return {
        "format": "markdown",
        "overview_markdown": overview,
        "readme_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source_path": source_path,
        "path": readme,
        "hero": hero,
    }
