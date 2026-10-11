"""Drawing primitives, scene objects, backdrops, and task templates for model hero images.

Everything is drawn at SCALE times the final size and downsampled once, which anti-aliases
shapes and text without depending on Pillow's per-primitive smoothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 480
SCALE = 3
LABEL_FONT_SIZE = 48  # renders about 11 CSS px on the narrowest catalog card
GROUND = 404

INK = "#1E293B"
WHITE = "#FFFFFF"

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Family:
    """Accent and backdrop shared by every model of one architecture family."""

    name: str
    accent: str
    tint: str  # backdrop base, a pale tint of the accent
    shade: str  # backdrop motif color, one step darker than tint
    ground: str


@dataclass(frozen=True)
class Item:
    """One object in a scene: what it is, where it stands, and how it is labeled."""

    kind: str
    x: float
    size: float = 1.0
    label: str | None = None
    look: dict[str, str] = field(default_factory=dict)
    ground: float = GROUND
    pose: str = "stand"
    mask: bool = False  # promptable segmentation: this object matches the prompt


class Canvas:
    """An ImageDraw wrapper that takes final-pixel coordinates and draws at SCALE."""

    def __init__(self, background: str):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), background)
        self.draw = ImageDraw.Draw(self.image)
        self.override: str | None = None  # paints every shape one color, for masks and depth

    def _c(self, color: str) -> str:
        return self.override or color

    @staticmethod
    def _p(points):
        return [(x * SCALE, y * SCALE) for x, y in points]

    def rect(self, box: Box, fill: str, radius: float = 0, outline: str | None = None, width: float = 0,
             corners: tuple[bool, bool, bool, bool] | None = None) -> None:
        x0, y0, x1, y1 = (v * SCALE for v in box)
        self.draw.rounded_rectangle(
            (x0, y0, x1, y1), radius=radius * SCALE, fill=self._c(fill) if fill else None,
            outline=self._c(outline) if outline else None, width=round(width * SCALE), corners=corners,
        )

    def ellipse(self, box: Box, fill: str | None, outline: str | None = None, width: float = 0) -> None:
        self.draw.ellipse(tuple(v * SCALE for v in box), fill=self._c(fill) if fill else None,
                          outline=self._c(outline) if outline else None, width=round(width * SCALE))

    def circle(self, cx: float, cy: float, r: float, fill: str | None, outline: str | None = None,
               width: float = 0) -> None:
        self.ellipse((cx - r, cy - r, cx + r, cy + r), fill, outline, width)

    def poly(self, points, fill: str) -> None:
        self.draw.polygon(self._p(points), fill=self._c(fill))

    def line(self, points, color: str, width: float, round_caps: bool = True) -> None:
        self.draw.line(self._p(points), fill=self._c(color), width=round(width * SCALE), joint="curve")
        if round_caps:
            for x, y in (points[0], points[-1]):
                self.circle(x, y, width / 2, color)

    def arc(self, box: Box, start: float, end: float, color: str, width: float) -> None:
        self.draw.arc(tuple(v * SCALE for v in box), start, end, fill=self._c(color), width=round(width * SCALE))

    def chord(self, box: Box, start: float, end: float, fill: str) -> None:
        self.draw.chord(tuple(v * SCALE for v in box), start, end, fill=self._c(fill))

    def text(self, xy: tuple[float, float], text: str, color: str, size: int = LABEL_FONT_SIZE) -> None:
        self.draw.text((xy[0] * SCALE, xy[1] * SCALE), text, fill=color, font=font(size), anchor="ls")

    def finish(self) -> Image.Image:
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow bundles Aileron Regular, so rendering needs no system font.
    return ImageFont.load_default(size=size * SCALE)


def text_width(text: str, size: int = LABEL_FONT_SIZE) -> float:
    return font(size).getlength(text) / SCALE


def mix(a: str, b: str, t: float) -> str:
    """Blend hex color a toward b by t in [0, 1]."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


# Muted object palette shared by all families, so only the overlay carries the family accent.
SKIN = "#D6B69C"
HAIR = "#4B3B33"
SLATE = "#64748B"
SLATE_DARK = "#334155"
SLATE_LIGHT = "#94A3B8"
STONE = "#CBD5E1"
CLAY = "#C9805B"
SAGE = "#7FA48B"
SAND = "#E4CFA3"
CREAM = "#F4EEE4"
DENIM = "#4F6D8F"
WOOD = "#B08463"
GLASS = "#DCE8F2"
TIRE = "#2B3442"


def look(item: Item, role: str, default: str) -> str:
    return item.look.get(role, default)


# --- People -------------------------------------------------------------------------------------

# Joint offsets (dx, dy) from the person's ground point at size 1.0, height about 228 px.
POSES: dict[str, dict[str, tuple[float, float]]] = {
    "stand": {
        "head": (0, -204), "neck": (0, -176), "l_shoulder": (-27, -166), "r_shoulder": (27, -166),
        "l_elbow": (-35, -124), "r_elbow": (35, -124), "l_wrist": (-36, -84), "r_wrist": (36, -84),
        "l_hip": (-15, -98), "r_hip": (15, -98), "l_knee": (-16, -52), "r_knee": (16, -52),
        "l_ankle": (-17, -8), "r_ankle": (17, -8),
    },
    "walk": {
        "head": (0, -204), "neck": (0, -176), "l_shoulder": (-27, -166), "r_shoulder": (27, -166),
        "l_elbow": (-38, -126), "r_elbow": (36, -124), "l_wrist": (-32, -86), "r_wrist": (44, -88),
        "l_hip": (-15, -98), "r_hip": (15, -98), "l_knee": (-15, -52), "r_knee": (21, -58),
        "l_ankle": (-16, -8), "r_ankle": (24, -20),
    },
    "wave": {
        "head": (0, -204), "neck": (0, -176), "l_shoulder": (-27, -166), "r_shoulder": (27, -166),
        "l_elbow": (-35, -124), "r_elbow": (60, -190), "l_wrist": (-36, -84), "r_wrist": (66, -236),
        "l_hip": (-15, -98), "r_hip": (15, -98), "l_knee": (-16, -52), "r_knee": (16, -52),
        "l_ankle": (-17, -8), "r_ankle": (17, -8),
    },
    "reach": {
        "head": (0, -204), "neck": (0, -176), "l_shoulder": (-27, -166), "r_shoulder": (27, -166),
        "l_elbow": (-35, -124), "r_elbow": (56, -136), "l_wrist": (-36, -84), "r_wrist": (84, -118),
        "l_hip": (-15, -98), "r_hip": (15, -98), "l_knee": (-16, -52), "r_knee": (18, -52),
        "l_ankle": (-18, -8), "r_ankle": (20, -8),
    },
}

SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]


def joints(item: Item) -> dict[str, tuple[float, float]]:
    s = item.size
    return {name: (item.x + dx * s, item.ground + dy * s) for name, (dx, dy) in POSES[item.pose].items()}


def person(c: Canvas, item: Item) -> Box:
    s, j = item.size, joints(item)
    shirt, pants = look(item, "shirt", CLAY), look(item, "pants", SLATE_DARK)

    def mid(a, b):
        return ((j[a][0] + j[b][0]) / 2, (j[a][1] + j[b][1]) / 2)

    for side in ("l", "r"):
        c.line([j[f"{side}_hip"], j[f"{side}_knee"], j[f"{side}_ankle"]], pants, 24 * s)
        ax, ay = j[f"{side}_ankle"]
        c.rect((ax - 15 * s, ay - 4 * s, ax + 15 * s, ay + 8 * s), TIRE, radius=5 * s)
    (lsx, lsy), (rsx, rsy), (lhx, lhy), (rhx, rhy) = j["l_shoulder"], j["r_shoulder"], j["l_hip"], j["r_hip"]
    c.rect((lhx - 12 * s, lhy - 14 * s, rhx + 12 * s, lhy + 10 * s), pants, radius=8 * s)
    c.poly([(lsx - 2 * s, lsy - 10 * s), (rsx + 2 * s, rsy - 10 * s), (rhx + 12 * s, rhy), (lhx - 12 * s, lhy)], shirt)
    for side in ("l", "r"):
        c.circle(*j[f"{side}_shoulder"], 10 * s, shirt)
        c.line([j[f"{side}_shoulder"], j[f"{side}_elbow"], j[f"{side}_wrist"]], shirt, 16 * s)
        c.circle(*j[f"{side}_wrist"], 8 * s, SKIN)
    if "tether" in item.look:
        tx, ty = (float(v) for v in item.look["tether"].split(","))
        c.line([j["r_wrist"], (tx, ty)], SLATE, 3, round_caps=False)
    hx, hy = j["head"]
    c.rect((hx - 9 * s, hy + 12 * s, hx + 9 * s, j["neck"][1] + 4 * s), SKIN, radius=4 * s)
    c.circle(hx, hy, 24 * s, SKIN)
    c.chord((hx - 25 * s, hy - 26 * s, hx + 25 * s, hy + 18 * s), 180, 360, look(item, "hair", HAIR))
    xs = [p[0] for p in j.values()]
    return (min(xs) - 18 * s, hy - 26 * s, max(xs) + 18 * s, item.ground + 8 * s)


# --- Vehicles -----------------------------------------------------------------------------------

def wheel(c: Canvas, cx: float, cy: float, r: float) -> None:
    c.circle(cx, cy, r, TIRE)
    c.circle(cx, cy, r * 0.45, STONE)


def car(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    body = look(item, "body", DENIM)
    c.poly([(x - 78 * s, g - 74 * s), (x - 48 * s, g - 122 * s), (x + 50 * s, g - 122 * s),
            (x + 88 * s, g - 74 * s)], body)
    c.poly([(x - 64 * s, g - 78 * s), (x - 42 * s, g - 112 * s), (x - 4 * s, g - 112 * s),
            (x - 4 * s, g - 78 * s)], GLASS)
    c.poly([(x + 6 * s, g - 78 * s), (x + 6 * s, g - 112 * s), (x + 44 * s, g - 112 * s),
            (x + 72 * s, g - 78 * s)], GLASS)
    c.rect((x - 132 * s, g - 80 * s, x + 132 * s, g - 22 * s), body, radius=22 * s)
    c.rect((x + 112 * s, g - 64 * s, x + 128 * s, g - 52 * s), SAND, radius=4 * s)
    c.rect((x - 128 * s, g - 62 * s, x - 116 * s, g - 50 * s), CLAY, radius=4 * s)
    wheel(c, x - 76 * s, g - 22 * s, 24 * s)
    wheel(c, x + 76 * s, g - 22 * s, 24 * s)
    return (x - 132 * s, g - 122 * s, x + 132 * s, g)


def bus(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    body = look(item, "body", SAND)
    c.rect((x - 200 * s, g - 196 * s, x + 200 * s, g - 24 * s), body, radius=20 * s)
    c.rect((x - 200 * s, g - 70 * s, x + 200 * s, g - 50 * s), look(item, "stripe", CLAY))
    for i in range(5):
        wx = x - 186 * s + i * 66 * s
        c.rect((wx, g - 178 * s, wx + 54 * s, g - 112 * s), GLASS, radius=6 * s)
    c.rect((x + 146 * s, g - 178 * s, x + 188 * s, g - 30 * s), GLASS, radius=6 * s)
    c.rect((x + 192 * s, g - 60 * s, x + 202 * s, g - 46 * s), CREAM, radius=3 * s)
    wheel(c, x - 128 * s, g - 22 * s, 26 * s)
    wheel(c, x + 96 * s, g - 22 * s, 26 * s)
    return (x - 200 * s, g - 196 * s, x + 202 * s, g)


def truck(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.rect((x - 170 * s, g - 190 * s, x + 50 * s, g - 34 * s), look(item, "cargo", CREAM), radius=8 * s)
    c.rect((x - 170 * s, g - 190 * s, x + 50 * s, g - 34 * s), "", radius=8 * s, outline=STONE, width=3 * s)
    cab = look(item, "body", SAGE)
    c.poly([(x + 58 * s, g - 34 * s), (x + 58 * s, g - 140 * s), (x + 128 * s, g - 140 * s),
            (x + 168 * s, g - 92 * s), (x + 168 * s, g - 34 * s)], cab)
    c.poly([(x + 74 * s, g - 96 * s), (x + 74 * s, g - 126 * s), (x + 122 * s, g - 126 * s),
            (x + 148 * s, g - 96 * s)], GLASS)
    c.rect((x - 170 * s, g - 48 * s, x + 168 * s, g - 30 * s), SLATE_DARK, radius=4 * s)
    for wx in (-120, -66, 118):
        wheel(c, x + wx * s, g - 24 * s, 24 * s)
    return (x - 170 * s, g - 190 * s, x + 168 * s, g)


def bicycle(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    frame = look(item, "frame", CLAY)
    r = 36 * s
    back, front = (x - 56 * s, g - r), (x + 56 * s, g - r)
    for cx, cy in (back, front):
        c.circle(cx, cy, r, None, outline=TIRE, width=7 * s)
    crank, seat, bar = (x - 6 * s, g - r), (x - 22 * s, g - 92 * s), (x + 40 * s, g - 96 * s)
    c.line([back, crank, bar, front], frame, 7 * s)
    c.line([back, seat, crank], frame, 7 * s)
    c.line([seat, (x + 30 * s, g - 92 * s)], frame, 7 * s)
    c.line([(seat[0] - 14 * s, seat[1] - 6 * s), (seat[0] + 12 * s, seat[1] - 6 * s)], TIRE, 8 * s)
    c.line([bar, (bar[0] + 12 * s, bar[1] - 12 * s)], TIRE, 7 * s)
    return (x - 56 * s - r, g - 112 * s, x + 56 * s + r, g)


def boat(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.poly([(x - 120 * s, g - 46 * s), (x + 130 * s, g - 46 * s), (x + 96 * s, g), (x - 96 * s, g)],
           look(item, "hull", DENIM))
    c.rect((x - 120 * s, g - 52 * s, x + 130 * s, g - 42 * s), CREAM, radius=3 * s)
    c.line([(x, g - 50 * s), (x, g - 230 * s)], SLATE_DARK, 6 * s)
    c.poly([(x + 8 * s, g - 222 * s), (x + 8 * s, g - 66 * s), (x + 112 * s, g - 66 * s)], CREAM)
    c.poly([(x - 8 * s, g - 190 * s), (x - 8 * s, g - 66 * s), (x - 86 * s, g - 66 * s)], look(item, "sail", SAND))
    return (x - 120 * s, g - 230 * s, x + 130 * s, g)


def drone(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    body = look(item, "body", SLATE_DARK)
    for dx in (-1, 1):
        c.line([(x, y), (x + dx * 30 * s, y - 10 * s)], body, 5 * s)
        c.rect((x + dx * 30 * s - 18 * s, y - 16 * s, x + dx * 30 * s + 18 * s, y - 12 * s), SLATE, radius=2 * s)
    c.rect((x - 13 * s, y - 7 * s, x + 13 * s, y + 7 * s), body, radius=5 * s)
    c.line([(x - 8 * s, y + 7 * s), (x - 12 * s, y + 14 * s)], body, 3 * s)
    c.line([(x + 8 * s, y + 7 * s), (x + 12 * s, y + 14 * s)], body, 3 * s)
    return (x - 48 * s, y - 18 * s, x + 48 * s, y + 16 * s)


# --- Street furniture ---------------------------------------------------------------------------

def traffic_light(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.rect((x - 6 * s, g - 210 * s, x + 6 * s, g), SLATE_DARK, radius=3 * s)
    c.rect((x - 26 * s, g - 300 * s, x + 26 * s, g - 150 * s), SLATE_DARK, radius=12 * s)
    lit = look(item, "lit", "green")
    for i, name in enumerate(("red", "amber", "green")):
        on = {"red": "#E5484D", "amber": "#F2A93B", "green": "#3FA36B"}[name]
        c.circle(x, g - 274 * s + i * 48 * s, 15 * s, on if name == lit else "#4B5668")
    return (x - 26 * s, g - 300 * s, x + 26 * s, g)


def bench(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    wood = look(item, "wood", WOOD)
    c.rect((x - 90 * s, g - 92 * s, x + 90 * s, g - 78 * s), wood, radius=4 * s)
    c.rect((x - 90 * s, g - 70 * s, x + 90 * s, g - 56 * s), wood, radius=4 * s)
    c.rect((x - 96 * s, g - 50 * s, x + 96 * s, g - 38 * s), wood, radius=4 * s)
    for dx in (-76, 70):
        c.rect((x + dx * s, g - 38 * s, x + (dx + 8) * s, g), SLATE_DARK, radius=2 * s)
    return (x - 96 * s, g - 92 * s, x + 96 * s, g)


# --- Animals ------------------------------------------------------------------------------------

def dog(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    fur, dark = look(item, "fur", "#B98B5E"), look(item, "dark", "#7A5638")
    for dx in (-44, -26, 30, 46):
        c.rect((x + (dx - 6) * s, g - 50 * s, x + (dx + 6) * s, g - 2 * s), fur if dx in (-44, 30) else dark,
               radius=5 * s)
    c.line([(x - 54 * s, g - 70 * s), (x - 78 * s, g - 96 * s)], fur, 9 * s)
    c.rect((x - 62 * s, g - 88 * s, x + 58 * s, g - 42 * s), fur, radius=22 * s)
    c.line([(x + 46 * s, g - 70 * s), (x + 62 * s, g - 100 * s)], fur, 26 * s)
    c.rect((x + 46 * s, g - 128 * s, x + 96 * s, g - 92 * s), fur, radius=16 * s)
    c.rect((x + 84 * s, g - 112 * s, x + 108 * s, g - 94 * s), fur, radius=8 * s)
    c.circle(x + 107 * s, g - 107 * s, 5 * s, INK)
    c.poly([(x + 52 * s, g - 124 * s), (x + 70 * s, g - 126 * s), (x + 58 * s, g - 92 * s)], dark)
    c.circle(x + 80 * s, g - 115 * s, 3.5 * s, INK)
    return (x - 82 * s, g - 130 * s, x + 112 * s, g)


def cat(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    fur, stripe = look(item, "fur", "#B7A48E"), look(item, "stripe", "#8C7660")
    c.line([(x + 50 * s, g - 12 * s), (x + 92 * s, g - 20 * s), (x + 104 * s, g - 70 * s)], fur, 16 * s)
    c.ellipse((x - 62 * s, g - 150 * s, x + 62 * s, g), fur)
    for i in range(3):
        y = g - 118 * s + i * 30 * s
        c.line([(x - 52 * s, y), (x - 30 * s, y + 6 * s)], stripe, 9 * s)
        c.line([(x + 52 * s, y), (x + 30 * s, y + 6 * s)], stripe, 9 * s)
    c.rect((x - 34 * s, g - 30 * s, x - 8 * s, g), mix(fur, WHITE, 0.25), radius=12 * s)
    c.rect((x + 8 * s, g - 30 * s, x + 34 * s, g), mix(fur, WHITE, 0.25), radius=12 * s)
    hy = g - 176 * s
    c.poly([(x - 50 * s, hy - 14 * s), (x - 46 * s, hy - 70 * s), (x - 12 * s, hy - 36 * s)], fur)
    c.poly([(x + 50 * s, hy - 14 * s), (x + 46 * s, hy - 70 * s), (x + 12 * s, hy - 36 * s)], fur)
    c.poly([(x - 42 * s, hy - 22 * s), (x - 40 * s, hy - 54 * s), (x - 22 * s, hy - 36 * s)], "#E3B7A4")
    c.poly([(x + 42 * s, hy - 22 * s), (x + 40 * s, hy - 54 * s), (x + 22 * s, hy - 36 * s)], "#E3B7A4")
    c.ellipse((x - 56 * s, hy - 46 * s, x + 56 * s, hy + 44 * s), fur)
    for dx in (-1, 1):
        c.line([(x + dx * 8 * s, hy - 44 * s), (x + dx * 12 * s, hy - 26 * s)], stripe, 6 * s)
        c.ellipse((x + dx * 22 * s - 9 * s, hy - 12 * s, x + dx * 22 * s + 9 * s, hy + 6 * s), "#5E7F4A")
        c.ellipse((x + dx * 22 * s - 3 * s, hy - 10 * s, x + dx * 22 * s + 3 * s, hy + 4 * s), INK)
    c.poly([(x - 7 * s, hy + 14 * s), (x + 7 * s, hy + 14 * s), (x, hy + 22 * s)], "#C98F86")
    for dy in (16, 24):
        c.line([(x - 18 * s, hy + dy * s), (x - 52 * s, hy + (dy - 6) * s)], CREAM, 2 * s, round_caps=False)
        c.line([(x + 18 * s, hy + dy * s), (x + 52 * s, hy + (dy - 6) * s)], CREAM, 2 * s, round_caps=False)
    return (x - 62 * s, hy - 70 * s, x + 112 * s, g)


def horse(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    coat, dark = look(item, "coat", "#9A6B4C"), look(item, "dark", "#5E3F2D")
    for dx in (-62, -44, 46, 64):
        c.line([(x + dx * s, g - 92 * s), (x + dx * s, g - 6 * s)], coat if dx in (-62, 46) else dark, 13 * s)
        c.rect((x + (dx - 8) * s, g - 10 * s, x + (dx + 8) * s, g), INK, radius=3 * s)
    c.line([(x - 84 * s, g - 132 * s), (x - 98 * s, g - 70 * s)], dark, 12 * s)
    c.rect((x - 88 * s, g - 156 * s, x + 84 * s, g - 86 * s), coat, radius=34 * s)
    c.poly([(x + 44 * s, g - 150 * s), (x + 74 * s, g - 230 * s), (x + 108 * s, g - 214 * s),
            (x + 84 * s, g - 126 * s)], coat)
    c.poly([(x + 76 * s, g - 232 * s), (x + 128 * s, g - 186 * s), (x + 116 * s, g - 172 * s),
            (x + 70 * s, g - 204 * s)], coat)
    c.line([(x + 46 * s, g - 150 * s), (x + 74 * s, g - 230 * s)], dark, 10 * s)
    c.poly([(x + 78 * s, g - 230 * s), (x + 84 * s, g - 250 * s), (x + 92 * s, g - 226 * s)], coat)
    c.circle(x + 96 * s, g - 212 * s, 4 * s, INK)
    return (x - 104 * s, g - 252 * s, x + 130 * s, g)


def sheep(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    wool, face = look(item, "wool", CREAM), look(item, "face", "#3F3A36")
    for dx in (-30, -14, 18, 34):
        c.rect((x + (dx - 5) * s, g - 40 * s, x + (dx + 5) * s, g), face, radius=4 * s)
    for cx, cy, r in ((-40, -66, 26), (-12, -80, 30), (20, -78, 28), (40, -60, 24), (0, -52, 30), (-30, -48, 22),
                      (28, -48, 22)):
        c.circle(x + cx * s, g + cy * s, r * s, wool)
    c.ellipse((x + 44 * s, g - 92 * s, x + 80 * s, g - 50 * s), face)
    c.ellipse((x + 38 * s, g - 90 * s, x + 54 * s, g - 78 * s), face)
    c.circle(x + 68 * s, g - 78 * s, 3.5 * s, CREAM)
    return (x - 66 * s, g - 110 * s, x + 82 * s, g)


def bird(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    color = look(item, "body", SLATE_DARK)
    c.line([(x - 40 * s, y - 14 * s), (x - 18 * s, y - 20 * s), (x, y)], color, 7 * s)
    c.line([(x, y), (x + 18 * s, y - 20 * s), (x + 40 * s, y - 14 * s)], color, 7 * s)
    c.ellipse((x - 9 * s, y - 6 * s, x + 9 * s, y + 8 * s), color)
    return (x - 46 * s, y - 26 * s, x + 46 * s, y + 12 * s)


def kite(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    a, b = look(item, "a", CLAY), look(item, "b", SAND)
    c.poly([(x, y - 70 * s), (x + 48 * s, y), (x, y + 18 * s)], a)
    c.poly([(x, y - 70 * s), (x - 48 * s, y), (x, y + 18 * s)], b)
    c.line([(x, y + 18 * s), (x + 10 * s, y + 52 * s), (x - 8 * s, y + 84 * s)], SLATE, 3 * s)
    for t, dy in ((10, 46), (-6, 76)):
        c.poly([(x + t * s - 7 * s, y + dy * s), (x + t * s + 7 * s, y + dy * s - 6 * s),
                (x + t * s + 5 * s, y + dy * s + 6 * s)], a)
    return (x - 50 * s, y - 72 * s, x + 50 * s, y + 88 * s)


def umbrella(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.line([(x, g - 168 * s), (x + 6 * s, g)], SLATE_DARK, 6 * s)
    c.chord((x - 110 * s, g - 220 * s, x + 110 * s, g - 120 * s), 180, 360, look(item, "canopy", CLAY))
    c.poly([(x - 36 * s, g - 170 * s), (x, g - 220 * s), (x + 36 * s, g - 170 * s)], look(item, "stripe", CREAM))
    return (x - 110 * s, g - 222 * s, x + 110 * s, g)


# --- Indoor -------------------------------------------------------------------------------------

def tv(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.rect((x - 110 * s, g - 64 * s, x + 110 * s, g), look(item, "stand", WOOD), radius=6 * s)
    c.rect((x - 96 * s, g - 40 * s, x - 6 * s, g - 12 * s), mix(WOOD, INK, 0.25), radius=3 * s)
    c.rect((x + 6 * s, g - 40 * s, x + 96 * s, g - 12 * s), mix(WOOD, INK, 0.25), radius=3 * s)
    c.rect((x - 10 * s, g - 86 * s, x + 10 * s, g - 64 * s), SLATE_DARK)
    c.rect((x - 128 * s, g - 236 * s, x + 128 * s, g - 86 * s), SLATE_DARK, radius=8 * s)
    c.rect((x - 118 * s, g - 226 * s, x + 118 * s, g - 96 * s), look(item, "screen", "#5B7A99"), radius=4 * s)
    c.poly([(x - 118 * s, g - 150 * s), (x - 40 * s, g - 196 * s), (x + 20 * s, g - 150 * s),
            (x + 60 * s, g - 176 * s), (x + 118 * s, g - 132 * s), (x + 118 * s, g - 96 * s),
            (x - 118 * s, g - 96 * s)], "#7E9AB5")
    return (x - 128 * s, g - 236 * s, x + 128 * s, g - 64 * s)


def couch(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    fabric = look(item, "fabric", SAGE)
    dark = mix(fabric, INK, 0.18)
    c.rect((x - 170 * s, g - 168 * s, x + 170 * s, g - 70 * s), dark, radius=24 * s)
    c.rect((x - 146 * s, g - 150 * s, x - 4 * s, g - 74 * s), fabric, radius=16 * s)
    c.rect((x + 4 * s, g - 150 * s, x + 146 * s, g - 74 * s), fabric, radius=16 * s)
    c.rect((x - 170 * s, g - 92 * s, x + 170 * s, g - 24 * s), fabric, radius=14 * s)
    c.rect((x - 190 * s, g - 118 * s, x - 146 * s, g - 24 * s), dark, radius=18 * s)
    c.rect((x + 146 * s, g - 118 * s, x + 190 * s, g - 24 * s), dark, radius=18 * s)
    for dx in (-160, 152):
        c.rect((x + dx * s, g - 24 * s, x + (dx + 8) * s, g), WOOD, radius=2 * s)
    return (x - 190 * s, g - 168 * s, x + 190 * s, g)


def potted_plant(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    leaf = look(item, "leaf", "#5F8F6B")
    light = mix(leaf, WHITE, 0.25)
    for angle, length, color in ((-62, 120, leaf), (-100, 140, light), (-80, 150, leaf), (-118, 116, leaf),
                                 (-40, 96, light), (-140, 92, light)):
        rad = math.radians(angle)
        tip = (x + math.cos(rad) * length * s, g - 66 * s + math.sin(rad) * length * s)
        c.line([(x, g - 66 * s), tip], color, 22 * s)
    c.poly([(x - 40 * s, g - 72 * s), (x + 40 * s, g - 72 * s), (x + 30 * s, g), (x - 30 * s, g)],
           look(item, "pot", CLAY))
    c.rect((x - 44 * s, g - 78 * s, x + 44 * s, g - 64 * s), mix(look(item, "pot", CLAY), INK, 0.15), radius=4 * s)
    return (x - 100 * s, g - 220 * s, x + 100 * s, g)


def table(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    wood = look(item, "wood", WOOD)
    w = item.look.get("width")
    half = float(w) / 2 if w else 150
    c.rect((x - half * s, g - 120 * s, x + half * s, g - 104 * s), wood, radius=4 * s)
    for dx in (-half + 14, half - 26):
        c.rect((x + dx * s, g - 104 * s, x + (dx + 12) * s, g), mix(wood, INK, 0.2), radius=3 * s)
    return (x - half * s, g - 120 * s, x + half * s, g)


def chair(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    wood = look(item, "wood", "#8E6A50")
    flip = -1 if item.look.get("facing") == "left" else 1  # the side the seat faces
    back, front = x - flip * 40 * s, x + flip * 40 * s
    c.rect((back - 6 * s, g - 190 * s, back + 6 * s, g), wood, radius=4 * s)
    c.rect((back - 7 * s, g - 186 * s, back + 7 * s, g - 120 * s), mix(wood, INK, 0.15), radius=4 * s)
    c.rect((x - 48 * s, g - 96 * s, x + 48 * s, g - 82 * s), wood, radius=4 * s)
    c.rect((front - 5 * s, g - 86 * s, front + 5 * s, g), wood, radius=3 * s)
    return (x - 48 * s, g - 190 * s, x + 48 * s, g)


def laptop(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    c.poly([(x - 78 * s, g - 118 * s), (x + 62 * s, g - 118 * s), (x + 70 * s, g - 12 * s), (x - 70 * s, g - 12 * s)],
           SLATE_DARK)
    c.poly([(x - 68 * s, g - 108 * s), (x + 54 * s, g - 108 * s), (x + 60 * s, g - 22 * s), (x - 60 * s, g - 22 * s)],
           look(item, "screen", "#6F8FAF"))
    c.rect((x - 100 * s, g - 12 * s, x + 100 * s, g), SLATE, radius=4 * s)
    return (x - 100 * s, g - 118 * s, x + 100 * s, g)


def cup(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    color = look(item, "body", CREAM)
    c.circle(x + 26 * s, g - 34 * s, 16 * s, None, outline=color, width=8 * s)
    c.rect((x - 28 * s, g - 64 * s, x + 28 * s, g), color, radius=10 * s)
    c.rect((x - 28 * s, g - 64 * s, x + 28 * s, g - 54 * s), mix(color, INK, 0.1), radius=4 * s)
    return (x - 28 * s, g - 64 * s, x + 46 * s, g)


def bottle(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    glass = look(item, "glass", "#6E9C84")
    c.rect((x - 9 * s, g - 150 * s, x + 9 * s, g - 110 * s), glass, radius=4 * s)
    c.rect((x - 11 * s, g - 160 * s, x + 11 * s, g - 146 * s), CLAY, radius=3 * s)
    c.rect((x - 26 * s, g - 118 * s, x + 26 * s, g), glass, radius=16 * s)
    c.rect((x - 26 * s, g - 78 * s, x + 26 * s, g - 40 * s), CREAM)
    return (x - 26 * s, g - 160 * s, x + 26 * s, g)


def bowl(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    color = look(item, "body", "#7F9DBB")
    c.chord((x - 62 * s, g - 92 * s, x + 62 * s, g + 4 * s), 0, 180, color)
    c.rect((x - 66 * s, g - 50 * s, x + 66 * s, g - 42 * s), mix(color, INK, 0.15), radius=4 * s)
    c.circle(x - 20 * s, g - 54 * s, 14 * s, "#D9534F")
    c.circle(x + 10 * s, g - 58 * s, 16 * s, "#F2A93B")
    c.circle(x + 34 * s, g - 52 * s, 12 * s, "#7FA48B")
    return (x - 66 * s, g - 74 * s, x + 66 * s, g)


def refrigerator(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    body = look(item, "body", "#E6EAEE")
    c.rect((x - 74 * s, g - 300 * s, x + 74 * s, g), body, radius=12 * s)
    c.rect((x - 74 * s, g - 300 * s, x + 74 * s, g), "", radius=12 * s, outline=SLATE_LIGHT, width=3 * s)
    c.line([(x - 74 * s, g - 196 * s), (x + 74 * s, g - 196 * s)], SLATE_LIGHT, 3 * s, round_caps=False)
    c.rect((x + 50 * s, g - 270 * s, x + 58 * s, g - 214 * s), SLATE, radius=4 * s)
    c.rect((x + 50 * s, g - 180 * s, x + 58 * s, g - 100 * s), SLATE, radius=4 * s)
    return (x - 74 * s, g - 300 * s, x + 74 * s, g)


def clock(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    c.circle(x, y, 44 * s, look(item, "rim", SLATE_DARK))
    c.circle(x, y, 36 * s, CREAM)
    for i in range(12):
        a = math.radians(i * 30)
        c.circle(x + math.cos(a) * 28 * s, y + math.sin(a) * 28 * s, 2.5 * s, SLATE)
    c.line([(x, y), (x, y - 24 * s)], INK, 5 * s)
    c.line([(x, y), (x + 17 * s, y + 6 * s)], INK, 5 * s)
    return (x - 44 * s, y - 44 * s, x + 44 * s, y + 44 * s)


def counter(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    half = float(item.look.get("width", 260)) / 2
    top = look(item, "top", "#E2E8F0")
    c.rect((x - half * s, g - 110 * s, x + half * s, g), look(item, "body", "#B6C3D1"), radius=4 * s)
    c.rect((x - half * s - 8 * s, g - 122 * s, x + half * s + 8 * s, g - 106 * s), top, radius=4 * s)
    for i in range(int(half // 80)):
        dx = -half + 18 + i * 160
        c.rect((x + dx * s, g - 94 * s, x + (dx + 140) * s, g - 14 * s), "", radius=6 * s,
               outline=mix(look(item, "body", "#B6C3D1"), INK, 0.15), width=3 * s)
    return (x - half * s, g - 122 * s, x + half * s, g)


# --- Background scenery (never labeled) ---------------------------------------------------------

def tree(c: Canvas, item: Item) -> Box:
    s, x, g = item.size, item.x, item.ground
    crown = look(item, "crown", "#9DBCA5")
    c.rect((x - 8 * s, g - 110 * s, x + 8 * s, g), look(item, "trunk", "#9C8574"), radius=4 * s)
    c.circle(x, g - 150 * s, 56 * s, crown)
    c.circle(x - 34 * s, g - 118 * s, 38 * s, crown)
    c.circle(x + 36 * s, g - 120 * s, 40 * s, crown)
    return (x - 72 * s, g - 206 * s, x + 76 * s, g)


def window(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    frame = look(item, "frame", CREAM)
    w, h = float(item.look.get("w", 180)) * s, float(item.look.get("h", 150)) * s
    c.rect((x - w / 2, y - h, x + w / 2, y), frame, radius=6 * s)
    c.rect((x - w / 2 + 12 * s, y - h + 12 * s, x + w / 2 - 12 * s, y - 12 * s), look(item, "sky", "#CFE3F2"))
    c.rect((x - 4 * s, y - h + 12 * s, x + 4 * s, y - 12 * s), frame)
    c.rect((x - w / 2 + 12 * s, y - h / 2 - 4 * s, x + w / 2 - 12 * s, y - h / 2 + 4 * s), frame)
    return (x - w / 2, y - h, x + w / 2, y)


def cloud(c: Canvas, item: Item) -> Box:
    s, x, y = item.size, item.x, item.ground
    color = look(item, "color", WHITE)
    c.rect((x - 70 * s, y - 24 * s, x + 70 * s, y), color, radius=12 * s)
    c.circle(x - 24 * s, y - 26 * s, 30 * s, color)
    c.circle(x + 18 * s, y - 34 * s, 36 * s, color)
    return (x - 70 * s, y - 70 * s, x + 70 * s, y)


OBJECTS: dict[str, Callable[[Canvas, Item], Box]] = {
    "person": person, "car": car, "bus": bus, "truck": truck, "bicycle": bicycle, "boat": boat, "drone": drone,
    "traffic_light": traffic_light, "bench": bench, "dog": dog, "cat": cat, "horse": horse, "sheep": sheep,
    "bird": bird, "kite": kite, "umbrella": umbrella, "tv": tv, "couch": couch, "potted_plant": potted_plant,
    "table": table, "chair": chair, "laptop": laptop, "cup": cup, "bottle": bottle, "bowl": bowl,
    "refrigerator": refrigerator, "clock": clock, "counter": counter, "tree": tree, "window": window,
    "cloud": cloud,
}


# --- Backdrops ----------------------------------------------------------------------------------
# A family keeps one backdrop, so its models read as a set; scenes vary inside it.

def backdrop_street(c: Canvas, family: Family, variant: str) -> None:
    for x0, w, h in ((-20, 150, 210), (150, 110, 160), (280, 170, 250), (470, 130, 190), (620, 160, 230),
                     (800, 120, 170), (940, 170, 240), (1130, 120, 190)):
        c.rect((x0, GROUND - h, x0 + w, GROUND), family.shade, radius=4)
        for row in range(int((h - 40) // 44)):
            for col in range(int((w - 24) // 36)):
                wx, wy = x0 + 18 + col * 36, GROUND - h + 22 + row * 44
                c.rect((wx, wy, wx + 18, wy + 22), mix(family.shade, family.tint, 0.55), radius=2)
    c.rect((0, GROUND, WIDTH, HEIGHT), family.ground)
    c.rect((0, GROUND, WIDTH, GROUND + 8), mix(family.ground, INK, 0.08))
    if variant == "crosswalk":
        for i in range(9):
            x = 330 + i * 64
            c.poly([(x, GROUND + 20), (x + 36, GROUND + 20), (x + 30, HEIGHT), (x - 6, HEIGHT)], WHITE)


def backdrop_room(c: Canvas, family: Family, variant: str) -> None:
    c.rect((0, GROUND, WIDTH, HEIGHT), family.ground)
    for i in range(0, WIDTH, 150):
        c.line([(i, GROUND), (i - 40, HEIGHT)], mix(family.ground, INK, 0.07), 3, round_caps=False)
    c.rect((0, GROUND - 14, WIDTH, GROUND), mix(family.tint, INK, 0.08))
    if variant == "kitchen":
        for row in range(3):
            for col in range(0, WIDTH, 80):
                x = col + (40 if row % 2 else 0)
                c.rect((x + 2, 150 + row * 44, x + 76, 190 + row * 44), family.shade, radius=3)


def backdrop_outdoor(c: Canvas, family: Family, variant: str) -> None:
    c.circle(560, 96, 40, mix(family.shade, WHITE, 0.2))
    c.ellipse((-200, GROUND - 120, 520, GROUND + 160), family.shade)
    c.ellipse((360, GROUND - 90, 1400, GROUND + 200), mix(family.shade, family.tint, 0.4))
    if variant == "shore":
        c.rect((0, GROUND - 40, WIDTH, GROUND + 20), "#BFD6E6")
        for x in range(0, WIDTH, 120):
            c.line([(x + 20, GROUND - 22), (x + 70, GROUND - 22)], "#D8E7F1", 4)
        c.rect((0, GROUND + 20, WIDTH, HEIGHT), family.ground)
    else:
        c.rect((0, GROUND, WIDTH, HEIGHT), family.ground)


def backdrop_sky(c: Canvas, family: Family, variant: str) -> None:
    top, bottom = family.tint, mix(family.tint, WHITE, 0.6)
    for y in range(0, HEIGHT, 4):
        c.rect((0, y, WIDTH, y + 4), mix(top, bottom, y / HEIGHT))
    for x, y, s in ((180, 120, 1.1), (560, 210, 0.8), (960, 90, 1.0)):
        cloud(c, Item("cloud", x, s, ground=y, look={"color": mix(WHITE, top, 0.2)}))
    c.ellipse((-120, HEIGHT - 70, 520, HEIGHT + 140), family.shade)
    c.ellipse((380, HEIGHT - 50, 1320, HEIGHT + 160), mix(family.shade, family.tint, 0.3))


def backdrop_plain(c: Canvas, family: Family, variant: str) -> None:
    for x in range(0, WIDTH + 1, 40):
        for y in range(0, HEIGHT + 1, 40):
            c.circle(x, y, 1.6, family.shade)


BACKDROPS = {"street": backdrop_street, "room": backdrop_room, "outdoor": backdrop_outdoor, "sky": backdrop_sky,
             "plain": backdrop_plain}


# --- Overlays -----------------------------------------------------------------------------------

LABEL_PAD_X, LABEL_HEIGHT = 16, round(LABEL_FONT_SIZE * 1.35)


def label_box(box: Box, label: str) -> Box:
    """Where the label chip for an object box goes: above the box, or inside it when the top is too close."""
    x0, y0, _, _ = box
    w = text_width(label) + 2 * LABEL_PAD_X
    top = y0 - LABEL_HEIGHT if y0 - LABEL_HEIGHT >= 8 else y0
    left = min(max(x0, 8), WIDTH - 8 - w)
    return (left, top, left + w, top + LABEL_HEIGHT)


def draw_detection(c: Canvas, box: Box, label: str, accent: str, stroke: float = 4) -> Box:
    c.rect(box, "", radius=6, outline=accent, width=stroke)
    chip = label_box(box, label)
    c.rect(chip, accent, radius=6)
    c.text((chip[0] + LABEL_PAD_X, chip[3] - round(LABEL_FONT_SIZE * 0.36)), label, WHITE)
    return chip


def pad(box: Box, amount: float = 8) -> Box:
    return (box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount)


def draw_skeleton(c: Canvas, item: Item, accent: str) -> None:
    j = joints(item)
    for a, b in SKELETON:
        c.line([j[a], j[b]], accent, 5, round_caps=False)
    for name, (x, y) in j.items():
        if name not in ("head", "neck"):
            c.circle(x, y, 8, WHITE, outline=accent, width=4)
    c.circle(*j["head"], 8, WHITE, outline=accent, width=4)
