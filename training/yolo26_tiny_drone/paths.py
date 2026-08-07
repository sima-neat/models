"""Path conventions shared by the tiny-drone training stages."""

import re
from pathlib import Path

_NUMBER = re.compile(r"(\d+)")


def natural_path_key(path: Path) -> tuple[str | int, ...]:
    """Order numeric frame components chronologically without requiring zero padding."""
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NUMBER.split(path.as_posix()))


def ultralytics_label_path(image: Path) -> Path:
    """Return the label path Ultralytics will derive for an image path."""
    parts = list(image.parts)
    try:
        images_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image path must contain an 'images' directory for Ultralytics: {image}") from error
    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")
