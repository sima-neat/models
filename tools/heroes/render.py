#!/usr/bin/env python3
"""Render every model's hero.png from tools/heroes/specs.py.

    python tools/heroes/render.py                 # write models/<id>/hero.png for every spec
    python tools/heroes/render.py yolo26n         # write selected models
    python tools/heroes/render.py --check         # fail if a committed hero differs from its spec
    python tools/heroes/render.py --sheet out.png # also write a contact sheet of all heroes
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import replace
from pathlib import Path

from draw import (
    BACKDROPS,
    GROUND,
    HEIGHT,
    INK,
    LABEL_FONT_SIZE,
    OBJECTS,
    SCALE,
    WHITE,
    WIDTH,
    Box,
    Canvas,
    Family,
    Item,
    draw_detection,
    draw_skeleton,
    font,
    mix,
    pad,
    text_width,
)
from PIL import Image, ImageDraw, ImageFilter
from specs import FAMILIES, FAMILY_BACKDROPS, SPECS, HeroSpec

REPOSITORY = Path(__file__).resolve().parents[2]


def scene(spec: HeroSpec, family: Family, override: dict[int, str] | None = None) -> tuple[Canvas, list[Box]]:
    """Draw the backdrop and every item; return the canvas and each item's bounding box."""
    c = Canvas(family.tint)
    BACKDROPS[spec.backdrop or FAMILY_BACKDROPS[spec.family]](c, family, spec.variant)
    boxes = []
    for index, item in enumerate(spec.items):
        c.override = (override or {}).get(index)
        boxes.append(OBJECTS[item.kind](c, item))
        c.override = None
    return c, boxes


def detection(spec: HeroSpec, family: Family, chips: list[Box]) -> Image.Image:
    c, boxes = scene(spec, family)
    if spec.inset is not None:
        _magnify(c, spec.items[spec.inset], boxes[spec.inset], family)
    for item, box in zip(spec.items, boxes):
        if item.label:
            chips.append(draw_detection(c, pad(box, 10 if item.size < 0.6 else 6), item.label, family.accent))
    return c.finish()


def _magnify(c: Canvas, item: Item, box: Box, family: Family) -> None:
    """Show one small object enlarged in a panel, connected to where it sits in the frame."""
    panel = (760, 96, 1140, 404)
    zoom = 5.5
    cy = (box[1] + box[3]) / 2
    for corner in ((box[2] + 10, box[1] - 10), (box[2] + 10, box[3] + 10)):
        target = (panel[0], panel[1] + 12) if corner[1] < cy else (panel[0], panel[3] - 12)
        c.line([corner, target], mix(family.accent, WHITE, 0.45), 3, round_caps=False)
    c.rect(panel, mix(family.tint, WHITE, 0.55), radius=16, outline=mix(family.accent, WHITE, 0.45), width=3)
    px, py = (panel[0] + panel[2]) / 2, (panel[1] + panel[3]) / 2
    big = replace(item, x=px, ground=py + (item.ground - cy) * zoom, size=item.size * zoom)
    OBJECTS[item.kind](c, big)


def keypoints(spec: HeroSpec, family: Family, chips: list[Box]) -> Image.Image:
    c, boxes = scene(spec, family)
    for item, box in zip(spec.items, boxes):
        if item.kind == "person" and item.label:
            draw_skeleton(c, item, family.accent)
            chips.append(draw_detection(c, pad(box, 6), item.label, family.accent, stroke=3))
    return c.finish()


def classification(spec: HeroSpec, family: Family, chips: list[Box]) -> Image.Image:
    c = Canvas(family.tint)
    BACKDROPS["plain"](c, family, spec.variant)
    frame = (110, 48, 500, 432)
    c.rect(frame, WHITE, radius=20)
    c.rect((frame[0] + 16, frame[1] + 16, frame[2] - 16, frame[3] - 16), family.ground, radius=12)
    c.rect((frame[0] + 16, 332, frame[2] - 16, frame[3] - 16), mix(family.ground, family.shade, 0.6),
           radius=12, corners=(False, False, True, True))
    for item in spec.items:
        OBJECTS[item.kind](c, item)
    left, right, top = 590, 1110, 118
    for rank, (label, score) in enumerate(spec.scores):
        y = top + rank * 112
        c.text((left, y), label, INK if rank == 0 else mix(INK, WHITE, 0.3))
        value = f"{score:.2f}"
        c.text((right - text_width(value), y), value, INK if rank == 0 else mix(INK, WHITE, 0.3))
        chips.append((left, y - LABEL_FONT_SIZE, right, y))
        c.rect((left, y + 18, right, y + 36), family.shade, radius=9)
        c.rect((left, y + 18, left + max(18, (right - left) * score), y + 36),
               family.accent if rank == 0 else mix(family.accent, WHITE, 0.55), radius=9)
    return c.finish()


def _panel(c: Canvas, picture: Image.Image, box: Box, border: str) -> None:
    """Paste a full-size rendered scene, scaled down, into box on canvas c."""
    x0, y0, x1, y1 = (round(v * SCALE) for v in box)
    c.image.paste(picture.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))
    c.rect(box, "", radius=2, outline=border, width=3)


def depth(spec: HeroSpec, family: Family, chips: list[Box]) -> Image.Image:
    """The input image beside its depth map: near is the accent, far fades to white."""
    near, far = family.accent, mix(family.accent, WHITE, 0.88)
    d = Canvas(far)
    for y in range(GROUND, HEIGHT, 2):
        d.rect((0, y, WIDTH, y + 2), mix(far, near, 0.35 + 0.5 * (y - GROUND) / (HEIGHT - GROUND)))
    for item in spec.items:
        d.override = mix(near, far, float(item.look.get("depth", "0.5")))
        OBJECTS[item.kind](d, item)
    d.override = None

    c = Canvas(family.tint)
    BACKDROPS["plain"](c, family, "")
    left, right = (50, 70, 540, 266), (660, 70, 1150, 266)
    _panel(c, scene(spec, family)[0].finish(), left, mix(family.accent, WHITE, 0.6))
    _panel(c, d.finish(), right, family.accent)
    c.line([(562, 168), (628, 168)], family.accent, 6)
    c.poly([(640, 168), (620, 154), (620, 182)], family.accent)
    baseline = 382
    c.text((right[0], baseline), "near", INK)
    far_x = right[2] - text_width("far")
    c.text((far_x, baseline), "far", INK)
    bar = (right[0] + text_width("near") + 24, baseline - 26, far_x - 24, baseline - 8)
    steps = 48
    for i in range(steps):
        x0 = bar[0] + (bar[2] - bar[0]) * i / steps
        c.rect((x0, bar[1], x0 + (bar[2] - bar[0]) / steps + 0.5, bar[3]), mix(near, far, i / (steps - 1)))
    chips.append((right[0], baseline - LABEL_FONT_SIZE, right[2], baseline))
    return c.finish()


def segmentation(spec: HeroSpec, family: Family, chips: list[Box]) -> Image.Image:
    """Masks on the objects that match a text prompt; other objects stay unmarked."""
    c, _ = scene(spec, family)
    m = Canvas("#000000")
    m.override = WHITE
    for item in spec.items:
        if item.mask:
            OBJECTS[item.kind](m, item)
    mask = m.image.convert("L")
    fill = Image.new("RGB", c.image.size, family.accent)
    c.image = Image.composite(Image.blend(c.image, fill, 0.45), c.image, mask)
    edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3 * SCALE + 1 - (3 * SCALE) % 2))
    c.image = Image.composite(fill, c.image, edge)
    c.draw = ImageDraw.Draw(c.image)
    text = f"\u201c{spec.prompt}\u201d"
    chip = (32, 28, 32 + text_width(text) + 48, 28 + LABEL_FONT_SIZE + 24)
    c.rect(chip, WHITE, radius=(chip[3] - chip[1]) / 2, outline=family.accent, width=4)
    c.text((chip[0] + 24, chip[3] - 20), text, family.accent)
    chips.append(chip)
    return c.finish()


TEMPLATES = {"classification": classification, "detection": detection, "keypoints": keypoints, "depth": depth,
             "segmentation": segmentation}


def render(spec: HeroSpec) -> tuple[Image.Image, list[Box]]:
    """Render one hero; also return every text chip's box so layout can be checked."""
    chips: list[Box] = []
    image = TEMPLATES[spec.template](spec, FAMILIES[spec.family], chips)
    return image, chips


def encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def contact_sheet(model_ids: list[str], path: Path) -> None:
    columns, thumb_w, thumb_h, caption = 3, 400, 160, 34
    rows = (len(model_ids) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (thumb_w + 16) + 16, rows * (thumb_h + caption + 16) + 16), "#F8FAFC")
    draw = ImageDraw.Draw(sheet)
    for index, model_id in enumerate(model_ids):
        x = 16 + (index % columns) * (thumb_w + 16)
        y = 16 + (index // columns) * (thumb_h + caption + 16)
        sheet.paste(render(SPECS[model_id])[0].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + thumb_h + 26), model_id, fill=INK, font=font(7), anchor="ls")
    sheet.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model IDs to render (default: every spec)")
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    parser.add_argument("--sheet", type=Path, help="also write a contact sheet PNG of the selected heroes")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.models) - set(SPECS))
    if unknown:
        parser.error(f"no hero spec for: {', '.join(unknown)}")
    model_ids = args.models or sorted(SPECS)
    stale = []
    for model_id in model_ids:
        target = REPOSITORY / "models" / model_id / "hero.png"
        payload = encode(render(SPECS[model_id])[0])
        if args.check:
            if not target.is_file() or target.read_bytes() != payload:
                stale.append(model_id)
        elif target.parent.is_dir():
            target.write_bytes(payload)
        else:
            parser.error(f"model directory does not exist: {target.parent}")
    if args.sheet:
        contact_sheet(model_ids, args.sheet)
    if stale:
        print("hero.png differs from its spec; rerun tools/heroes/render.py for: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
