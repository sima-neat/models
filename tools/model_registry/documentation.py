from __future__ import annotations

import hashlib
import re
import stat
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
REQUIRED_SECTIONS = ("Overview", "Inputs and outputs", "Building applications")
# The model page shows the rest: metrics, benchmarks, variants, versions, license, and source links.
MAX_README_WORDS = 400
BANNED_TERMS = (
    "MPK", "A65", "MLA", "AFE", "ELF", "tessellation", "mAP", "AP50", "FPS",
    "latency", "benchmark", "sima-cli", "model-registry", "Vulcan",
)
BUILDING_APPLICATIONS = (
    "Run the package with the [Neat Model API]"
    "(https://developer.sima.ai/software/develop-apps/development-workflow/model). "
    "For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples)."
)
GENAI_BUILDING_APPLICATIONS = (
    "Run the package with the [Neat GenAI Model API]"
    "(https://developer.sima.ai/software/develop-apps/development-workflow/genai-model). "
    "For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples)."
)
_H2_RE = re.compile(r"^## ([^#].*?)\s*$")
_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_SETEXT_HEADING_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_THEMATIC_BREAK_RE = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
_PARAGRAPH_EXCLUDED_RE = re.compile(r"^ {0,3}(?:[-+*][ \t]+|\d{1,9}[.)][ \t]+|>|#{1,6}(?:[ \t]+|$))")
_LIST_ITEM_RE = re.compile(r"^( {0,3})(?:[-+*]|\d{1,9}[.)])([ \t]+|$)")
_RAW_HTML_BLOCK_RE = re.compile(r"^ {0,3}<")
_AUTOLINK_RE = re.compile(
    r"^ {0,3}<(?:"
    r"[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\x00-\x20]*"
    r"|[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
    r")>[ \t]*$"
)
_OVERVIEW_FORBIDDEN_RE = re.compile(
    r"^(?: {0,3}\[[^]\n]+\]:|\s*(?:[-*+](?:\s+|$)|\d+[.)](?:\s+|$)|```|~~~|#|>\s?|\||!\[))"
)
_IMAGE_TOKEN_RE = re.compile(r"(?<!\\)(?:\\\\)*!\[[^]\n]*\]")
_INLINE_HTML_RE = re.compile(
    r"(?<!\\)(?:\\\\)*(?:<!--|<\?|<![A-Z]|<!\[CDATA\[|</?[A-Za-z][A-Za-z0-9-]*(?:\s|/?>))"
)
_INLINE_HTML_HEADING_RE = re.compile(r"(?<!\\)(?:\\\\)*</?h[1-6](?:[ \t]|/?>)", re.IGNORECASE)
# A lowercase term also matches its capitalized and inflected forms; an acronym matches only itself or its plural.
_IO_LABEL_RE = re.compile(r"^\*\*([A-Za-z][A-Za-z ]*)\.\*\*(?:\s|$)")
_TABLE_LAYOUT_RE = re.compile(r"[|:-]+")
_BANNED_TERM_RE = re.compile("|".join(
    rf"(?<![\w-])(?i:{re.escape(term)})" if term.islower() else rf"(?<![\w-]){re.escape(term)}s?(?![\w-])"
    for term in BANNED_TERMS
))

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
_ADAM7_PASSES = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
                 (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def _png_pass_dimensions(width: int, height: int, interlace: int) -> list[tuple[int, int]]:
    if interlace == 0:
        return [(width, height)]
    dimensions = []
    for x_start, y_start, x_step, y_step in _ADAM7_PASSES:
        pass_width = 0 if width <= x_start else (width - x_start + x_step - 1) // x_step
        pass_height = 0 if height <= y_start else (height - y_start + y_step - 1) // y_step
        if pass_width and pass_height:
            dimensions.append((pass_width, pass_height))
    return dimensions


def _png_scanline_lengths(width: int, height: int, bits_per_pixel: int, interlace: int) -> list[int]:
    return [
        1 + (pass_width * bits_per_pixel + 7) // 8
        for pass_width, pass_height in _png_pass_dimensions(width, height, interlace)
        for _ in range(pass_height)
    ]


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
    return (left, above, upper_left)[distances.index(min(distances))]


def _validate_indexed_pixels(
    decoded: bytes, width: int, height: int, depth: int, interlace: int, palette_entries: int, hero: Path
) -> None:
    offset = 0
    mask = (1 << depth) - 1
    for pass_width, pass_height in _png_pass_dimensions(width, height, interlace):
        row_bytes = (pass_width * depth + 7) // 8
        previous = bytearray(row_bytes)
        for _ in range(pass_height):
            filter_type = decoded[offset]
            filtered = decoded[offset + 1:offset + 1 + row_bytes]
            reconstructed = bytearray(row_bytes)
            for index, value in enumerate(filtered):
                left = reconstructed[index - 1] if index else 0
                above = previous[index]
                upper_left = previous[index - 1] if index else 0
                predictor = (0, left, above, (left + above) // 2, _paeth(left, above, upper_left))[filter_type]
                reconstructed[index] = (value + predictor) & 0xFF
            for pixel in range(pass_width):
                bit_offset = pixel * depth
                palette_index = (reconstructed[bit_offset // 8] >> (8 - depth - bit_offset % 8)) & mask
                if palette_index >= palette_entries:
                    raise ValidationError(f"model hero image references an undefined PNG palette entry: {hero}")
            previous = reconstructed
            offset += row_bytes + 1


def _valid_png_time(payload: bytes) -> bool:
    if len(payload) != 7:
        return False
    year = struct.unpack(">H", payload[:2])[0]
    month, day, hour, minute, second = payload[2:]
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_days = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return (
        year > 0 and 1 <= month <= 12 and 1 <= day <= month_days[month - 1]
        and hour <= 23 and minute <= 59 and second <= 60
    )


def _validate_ancillary_chunks(
    chunks: list[tuple[bytes, bytes]], color_type: int, depth: int,
    idat_index: int, palette_index: int | None, palette_entries: int, hero: Path,
) -> None:
    ancillary = [(index, kind, data) for index, (kind, data) in enumerate(chunks) if kind[0] & 0x20]
    unsupported = [kind for _, kind, _ in ancillary if kind not in {b"tIME", b"tRNS"}]
    if unsupported:
        raise ValidationError(f"model hero image contains an unsupported PNG ancillary chunk: {hero}")
    for kind in (b"tIME", b"tRNS"):
        if sum(chunk_kind == kind for _, chunk_kind, _ in ancillary) > 1:
            raise ValidationError(f"model hero image contains a duplicate PNG ancillary chunk: {hero}")
    time_payload = next((data for _, kind, data in ancillary if kind == b"tIME"), None)
    if time_payload is not None and not _valid_png_time(time_payload):
        raise ValidationError(f"model hero image contains an invalid PNG time chunk: {hero}")
    transparency = next(((index, data) for index, kind, data in ancillary if kind == b"tRNS"), None)
    if transparency is None:
        return
    transparency_index, payload = transparency
    if (
        transparency_index > idat_index or color_type not in {0, 2, 3}
        or (palette_index is not None and palette_index > transparency_index)
    ):
        raise ValidationError(f"model hero image contains an invalid PNG transparency chunk: {hero}")
    maximum_sample = (1 << depth) - 1
    if color_type == 0:
        valid = len(payload) == 2 and struct.unpack(">H", payload)[0] <= maximum_sample
    elif color_type == 2:
        valid = len(payload) == 6 and all(
            struct.unpack(">H", payload[index:index + 2])[0] <= maximum_sample for index in range(0, 6, 2)
        )
    else:
        valid = (
            palette_index is not None and palette_index < transparency_index
            and 1 <= len(payload) <= palette_entries
        )
    if not valid:
        raise ValidationError(f"model hero image contains an invalid PNG transparency chunk: {hero}")


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
        if (
            len(kind) != 4
            or not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in kind)
            or not 65 <= kind[2] <= 90
        ):
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
    palette = chunks[palette_indexes[0]][1] if len(palette_indexes) == 1 else b""
    palette_entries = len(palette) // 3
    if (
        len(palette_indexes) > 1
        or (palette_indexes and palette_indexes[0] > idat_indexes[0])
        or (color_type == 3 and not palette_indexes)
        or (color_type in {0, 4} and palette_indexes)
        or (palette_indexes and (not palette or len(palette) % 3 != 0 or palette_entries > 256))
        or (color_type == 3 and palette_entries > 2 ** depth)
    ):
        raise ValidationError(f"model hero image has an invalid PNG palette: {hero}")
    _validate_ancillary_chunks(
        chunks, color_type, depth, idat_indexes[0],
        palette_indexes[0] if palette_indexes else None, palette_entries, hero,
    )
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
    if color_type == 3:
        _validate_indexed_pixels(decoded, width, height, depth, interlace, palette_entries, hero)
    return width, height


def _is_markdown_table(lines: list[str]) -> bool:
    for index, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        headers = line.strip().strip("|").split("|")
        divider = lines[index + 1].strip().strip("|").split("|")
        if len(headers) > 1 and len(divider) == len(headers) and all(
            re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in divider
        ):
            return True
    return False


def _is_setext_heading(lines: list[str]) -> bool:
    return any(
        lines[index - 1].strip() and re.fullmatch(r"\s*(?:=+|-+)\s*", line)
        for index, line in enumerate(lines[1:], start=1)
    )


def _is_indented_code(line: str) -> bool:
    column = 0
    for character in line:
        if character == " ":
            column += 1
        elif character == "\t":
            column += 4 - column % 4
        else:
            break
    return column >= 4


def _strip_markdown_containers(line: str) -> tuple[str, bool]:
    nested = False
    while True:
        if match := re.match(r"^ {0,3}>[ \t]?", line):
            nested = True
            line = line[match.end():]
            continue
        if match := _LIST_ITEM_RE.match(line):
            nested = True
            line, _ = _list_item_content(line, match)
            continue
        return line, nested


def _list_item_content(line: str, match: re.Match[str]) -> tuple[str, int]:
    marker_width = _column_width(line[:match.start(2)])
    padding_width = _column_width(line[:match.end()]) - marker_width
    effective_padding = padding_width if 1 <= padding_width <= 4 else 1
    content = " " * max(0, padding_width - effective_padding) + line[match.end():]
    return content, marker_width + effective_padding


def _markdown_container_stack(line: str) -> tuple[str, list[tuple[str, int]]]:
    containers: list[tuple[str, int]] = []
    while True:
        if match := re.match(r"^ {0,3}>[ \t]?", line):
            containers.append(("blockquote", 0))
            line = line[match.end():]
            continue
        if match := _LIST_ITEM_RE.match(line):
            line, indent = _list_item_content(line, match)
            containers.append(("list", indent))
            continue
        return line, containers


def _strip_expected_containers(line: str, containers: list[tuple[str, int]]) -> tuple[str, bool]:
    if not line.strip():
        return line, True
    for kind, indent in containers:
        if kind == "blockquote":
            if match := re.match(r"^ {0,3}>[ \t]?", line):
                line = line[match.end():]
            else:
                return line, False
        elif _leading_columns(line) < indent:
            return line, False
        else:
            line = _strip_columns(line, indent)
    return line, True


def _column_width(text: str) -> int:
    column = 0
    for character in text:
        column = column + (4 - column % 4) if character == "\t" else column + 1
    return column


def _leading_columns(line: str) -> int:
    return _column_width(line[:len(line) - len(line.lstrip(" \t"))])


def _strip_columns(line: str, columns: int) -> str:
    offset = 0
    column = 0
    while offset < len(line) and column < columns and line[offset] in " \t":
        character = line[offset]
        column = column + (4 - column % 4) if character == "\t" else column + 1
        offset += 1
    return " " * max(0, column - columns) + line[offset:]


def _markdown_h2_sections(lines: list[str]) -> tuple[list[tuple[int, str]], bool]:
    sections = []
    has_invalid_heading = False
    fence_character = None
    fence_length = 0
    fence_outer_indent = 0
    fence_containers: list[tuple[str, int]] = []
    previous_paragraph_context: tuple[tuple[int, ...], tuple[str, ...]] | None = None
    current_section = None
    list_indents: list[int] = []
    for index, line in enumerate(lines):
        if fence_character is not None:
            outer_matches = (
                not line.strip() or not fence_outer_indent or _leading_columns(line) >= fence_outer_indent
            )
            container_line = _strip_columns(line, fence_outer_indent) if fence_outer_indent and outer_matches else line
            markdown_line, containers_match = _strip_expected_containers(container_line, fence_containers)
            if outer_matches and containers_match:
                if re.fullmatch(
                    rf" {{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*", markdown_line,
                ):
                    fence_character = None
                previous_paragraph_context = None
                continue
            fence_character = None
            fence_outer_indent = 0
            fence_containers = []
        if line.strip():
            indentation = _leading_columns(line)
            while list_indents and indentation < list_indents[-1]:
                list_indents.pop()
        existing_list_indent = list_indents[-1] if list_indents else 0
        container_line = _strip_columns(line, list_indents[-1]) if list_indents else line
        if list_item := _LIST_ITEM_RE.match(container_line):
            base_indent = list_indents[-1] if list_indents else 0
            _, item_indent = _list_item_content(container_line, list_item)
            content_indent = base_indent + item_indent
            if not list_indents or content_indent > list_indents[-1]:
                list_indents.append(content_indent)
        nested_line, nested = _strip_markdown_containers(container_line)
        markdown_line = nested_line if nested else container_line
        _, markdown_containers = _markdown_container_stack(container_line)
        paragraph_context = (
            tuple(list_indents),
            tuple(kind for kind, _ in markdown_containers if kind == "blockquote"),
        )
        fence = _FENCE_OPEN_RE.fullmatch(markdown_line)
        if fence is not None:
            marker, info = fence.groups()
            if marker[0] == "~" or "`" not in info:
                fence_character = marker[0]
                fence_length = len(marker)
                fence_outer_indent = existing_list_indent
                _, fence_containers = _markdown_container_stack(container_line)
                previous_paragraph_context = None
                continue
        if (
            _RAW_HTML_BLOCK_RE.match(markdown_line) and not _AUTOLINK_RE.fullmatch(markdown_line)
            or current_section != "Overview"
            and not _is_indented_code(markdown_line)
            and _INLINE_HTML_HEADING_RE.search(markdown_line)
        ):
            has_invalid_heading = True
            previous_paragraph_context = None
            continue
        if nested and (
            _ATX_HEADING_RE.fullmatch(nested_line) or _SETEXT_HEADING_RE.fullmatch(nested_line)
        ):
            has_invalid_heading = True
            previous_paragraph_context = None
            continue
        if _SETEXT_HEADING_RE.fullmatch(container_line) and previous_paragraph_context == paragraph_context:
            has_invalid_heading = True
            previous_paragraph_context = None
            continue
        if match := _H2_RE.fullmatch(container_line):
            if list_indents:
                has_invalid_heading = True
                previous_paragraph_context = None
                continue
            current_section = match.group(1)
            sections.append((index, current_section))
            previous_paragraph_context = None
            continue
        if heading := _ATX_HEADING_RE.fullmatch(container_line):
            if not (index == 0 and len(heading.group(1)) == 1):
                has_invalid_heading = True
            previous_paragraph_context = None
            continue
        paragraph_line = nested_line if nested else container_line
        stripped = paragraph_line.strip()
        is_paragraph_text = bool(stripped) and not (
            _is_indented_code(paragraph_line) or _PARAGRAPH_EXCLUDED_RE.match(paragraph_line)
            or _THEMATIC_BREAK_RE.fullmatch(paragraph_line)
        )
        previous_paragraph_context = paragraph_context if is_paragraph_text else None
    return sections, has_invalid_heading


def _repository_regular_file(path: Path, repository: Path, description: str) -> str:
    repository = repository.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(repository)
    except ValueError as exc:
        raise ValidationError(f"{description} must be inside the repository: {path}") from exc
    current = repository
    try:
        for part in relative.parts:
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValidationError(f"{description} path must not contain symlinks: {path}")
        if not stat.S_ISREG(mode):
            raise ValidationError(f"{description} must be a regular file: {path}")
    except OSError as exc:
        raise ValidationError(f"could not inspect {description} {path}: {exc}") from exc
    return relative.as_posix()


def _load_hero(model_dir: Path, display_name: str, repository: Path) -> dict[str, Any]:
    hero = model_dir / HERO_FILENAME
    source_path = _repository_regular_file(hero, repository, "model hero image")
    try:
        payload = hero.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read model hero image {hero}: {exc}") from exc
    if not payload or len(payload) > MAX_HERO_BYTES:
        raise ValidationError(f"model hero image must contain between 1 and {MAX_HERO_BYTES} bytes: {hero}")
    _validate_png(payload, hero)
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


def load_model_documentation(
    model_dir: Path, display_name: str, *, repository: Path, is_genai: bool = False
) -> dict[str, Any]:
    readme = model_dir / "README.md"
    source_path = _repository_regular_file(readme, repository, "model README")
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
    hero_path = model_dir / HERO_FILENAME
    if not hero_path.exists() and not hero_path.is_symlink():
        raise ValidationError(f"model hero image is required: {hero_path}")
    hero = _load_hero(model_dir, display_name, repository)
    expected_hero = f"![{hero['alt_text']}]({hero['markdown_path']})"
    valid_preamble = (
        len(lines) >= 5 and lines[1] == "" and lines[2] == expected_hero
        and lines[3] == "" and lines[4] == "## Overview"
    )
    if not valid_preamble:
        raise ValidationError(
            f"model README must begin with H1, blank line, {expected_hero}, blank line, and ## Overview: {readme}"
        )

    sections, has_invalid_heading = _markdown_h2_sections(lines)
    names = tuple(name for _, name in sections)
    if names != REQUIRED_SECTIONS or has_invalid_heading:
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
    overview_lines = overview.splitlines()
    if (
        any(
            _is_indented_code(line) or _OVERVIEW_FORBIDDEN_RE.match(line)
            or _IMAGE_TOKEN_RE.search(line)
            for line in overview_lines
        )
        or _INLINE_HTML_RE.search(overview)
        or _is_markdown_table(overview_lines)
        or _is_setext_heading(overview_lines)
        or any(_THEMATIC_BREAK_RE.fullmatch(line) for line in overview_lines)
    ):
        raise ValidationError(f"model README Overview must contain prose paragraphs only: {readme}")

    # Compare words, so the paragraph may wrap anywhere.
    building_applications = " ".join("\n".join(lines[sections[-1][0] + 1:]).split())
    expected_applications = GENAI_BUILDING_APPLICATIONS if is_genai else BUILDING_APPLICATIONS
    if building_applications != expected_applications:
        raise ValidationError(
            f"model README Building applications must be exactly {expected_applications!r}: {readme}"
        )
    for number, line in enumerate(lines, start=1):
        if term := _BANNED_TERM_RE.search(line):
            raise ValidationError(
                f"model README line {number} uses {term.group(0)!r}; write it in plain words: {readme}"
            )
    io_lines = lines[sections[1][0] + 1:sections[2][0]]
    labels = [match.group(1) for line in io_lines if (match := _IO_LABEL_RE.match(line))]
    inputs = [index for index, label in enumerate(labels) if label.lower().endswith(("input", "inputs"))]
    outputs = [index for index, label in enumerate(labels) if label in ("Outputs", "Output")]
    if not inputs or not outputs or inputs[0] > outputs[-1]:
        raise ValidationError(
            f"model README Inputs and outputs must have a bold input label, then **Outputs.** or **Output.**: {readme}"
        )
    # A table's pipes and divider row are layout, not words.
    readme_words = len([
        word for word in re.findall(r"\S+", "\n".join(lines[4:])) if not _TABLE_LAYOUT_RE.fullmatch(word)
    ])
    if readme_words > MAX_README_WORDS:
        raise ValidationError(
            f"model README must contain at most {MAX_README_WORDS} words after the hero, found {readme_words}: {readme}"
        )

    documentation = {
        "format": "markdown",
        "overview_markdown": overview,
        "readme_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source_path": source_path,
        "path": readme,
        "hero": hero,
    }
    return documentation
