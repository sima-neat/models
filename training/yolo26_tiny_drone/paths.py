"""Path conventions shared by the tiny-drone training stages."""

from pathlib import Path


def ultralytics_label_path(image: Path) -> Path:
    """Return the label path Ultralytics will derive for an image path."""
    parts = list(image.parts)
    try:
        images_index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"image path must contain an 'images' directory for Ultralytics: {image}") from error
    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")
