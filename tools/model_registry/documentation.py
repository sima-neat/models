from __future__ import annotations

import hashlib
import re
import struct
import zlib
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

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
_ADAM7_PASSES = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
                 (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def _png_scanline_lengths(width: int, height: int, bits_per_pixel: int, interlace: int) -> list[int]:
    if interlace == 0:
        return [1 + (width * bits_per_pixel + 7) // 8] * height
    lengths = []
    for x_start, y_start, x_step, y_step in _ADAM7_PASSES:
        pass_width = 0 if width <= x_start else (width - x_start + x_step - 1) // x_step
        pass_height = 0 if height <= y_start else (height - y_start + y_step - 1) // y_step
        if pass_width and pass_height:
            lengths.extend([1 + (pass_width * bits_per_pixel + 7) // 8] * pass_height)
    return lengths


def _validate_png(payload: bytes, hero: Path) -> tuple[int, int]:
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValidationError(f"model hero image must be a PNG: {hero}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise ValidationError(f"model hero image contains a truncated PNG chunk: {hero}")
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        end = offset + 12 + length
        if end > len(payload):
            raise ValidationError(f"model hero image contains a truncated PNG chunk: {hero}")
        kind = payload[offset + 4:offset + 8]
        if len(kind) != 4 or not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in kind):
            raise ValidationError(f"model hero image has an invalid PNG structure: {hero}")
        data = payload[offset + 8:offset + 8 + length]
        checksum = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != checksum:
            raise ValidationError(f"model hero image contains an invalid PNG checksum: {hero}")
        chunks.append((kind, data))
        offset = end
        if kind == b"IEND":
            break
    if offset != len(payload) or not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise ValidationError(f"model hero image has an invalid PNG structure: {hero}")
    if sum(kind == b"IHDR" for kind, _ in chunks) != 1 or sum(kind == b"IEND" for kind, _ in chunks) != 1:
        raise ValidationError(f"model hero image has an invalid PNG structure: {hero}")
    if any(kind[0] & 0x20 == 0 and kind not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"} for kind, _ in chunks):
        raise ValidationError(f"model hero image contains an unsupported critical PNG chunk: {hero}")
    ihdr = chunks[0][1]
    if len(ihdr) != 13 or chunks[-1][1]:
        raise ValidationError(f"model hero image has an invalid PNG structure: {hero}")
    width, height, depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (width, height) != (HERO_WIDTH, HERO_HEIGHT):
        raise ValidationError(
            f"model hero image must be {HERO_WIDTH}x{HERO_HEIGHT}, found {width}x{height}: {hero}"
        )
    if (
        color_type not in _PNG_DEPTHS or depth not in _PNG_DEPTHS[color_type]
        or compression != 0 or filtering != 0 or interlace not in {0, 1}
    ):
        raise ValidationError(f"model hero image uses an unsupported PNG format: {hero}")
    idat_indexes = [index for index, (kind, _) in enumerate(chunks) if kind == b"IDAT"]
    if not idat_indexes or idat_indexes != list(range(idat_indexes[0], idat_indexes[-1] + 1)):
        raise ValidationError(f"model hero image has an invalid PNG data stream: {hero}")
    palette_indexes = [index for index, (kind, _) in enumerate(chunks) if kind == b"PLTE"]
    if (
        len(palette_indexes) > 1
        or (palette_indexes and palette_indexes[0] > idat_indexes[0])
        or (color_type == 3 and not palette_indexes)
        or (color_type in {0, 4} and palette_indexes)
    ):
        raise ValidationError(f"model hero image has an invalid PNG palette: {hero}")
    compressed = b"".join(chunks[index][1] for index in idat_indexes)
    scanline_lengths = _png_scanline_lengths(width, height, _PNG_CHANNELS[color_type] * depth, interlace)
    expected_size = sum(scanline_lengths)
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(compressed, expected_size + 1)
        if decoder.unconsumed_tail or len(decoded) > expected_size:
            raise ValidationError(f"model hero image has an invalid PNG data stream: {hero}")
        decoded += decoder.flush(expected_size + 1 - len(decoded))
    except zlib.error as exc:
        raise ValidationError(f"model hero image has an invalid PNG data stream: {hero}") from exc
    if not decoder.eof or decoder.unused_data or len(decoded) != expected_size:
        raise ValidationError(f"model hero image has an invalid PNG data stream: {hero}")
    scanline_offset = 0
    for scanline_length in scanline_lengths:
        if decoded[scanline_offset] > 4:
            raise ValidationError(f"model hero image has an invalid PNG filter: {hero}")
        scanline_offset += scanline_length
    return width, height


def _load_hero(model_dir: Path, display_name: str, repository: Path) -> dict[str, Any]:
    hero = model_dir / HERO_FILENAME
    try:
        payload = hero.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read model hero image {hero}: {exc}") from exc
    if not payload or len(payload) > MAX_HERO_BYTES:
        raise ValidationError(f"model hero image must contain between 1 and {MAX_HERO_BYTES} bytes: {hero}")
    _validate_png(payload, hero)
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
