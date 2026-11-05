"""Deterministic geometric PushT illustration, not a measured rollout.

Uses the catalog's neutral/blue/teal/orange palette. The ten action markers
illustrate the output type and deliberately contain no scores or success claim.
"""

from pathlib import Path

from PIL import Image, ImageDraw


def draw(output: Path):
    image = Image.new("RGB", (1200, 480), "#F3F4F5")
    canvas = ImageDraw.Draw(image)
    black, blue, teal, orange = "#010F0E", "#5998DD", "#C0E8DB", "#D26728"
    canvas.rectangle((55, 45, 1145, 435), fill="#FFFFFF", outline=black, width=4)

    # Current T, pusher, and goal T share exact geometry and scale.
    def tee(x, y, color, outline=None):
        points = [
            (x, y),
            (x + 180, y),
            (x + 180, y + 60),
            (x + 120, y + 60),
            (x + 120, y + 200),
            (x + 60, y + 200),
            (x + 60, y + 60),
            (x, y + 60),
        ]
        canvas.polygon(points, fill=color, outline=outline)
        return points

    tee(275, 130, blue, black)
    tee(875, 130, teal, black)
    canvas.ellipse((170, 255, 230, 315), fill=black)
    waypoints = [(218 + i * 62, 285 - min(i, 4) * 11) for i in range(10)]
    canvas.line(waypoints, fill=orange, width=5)
    for x, y in waypoints:
        canvas.ellipse((x - 7, y - 7, x + 7, y + 7), fill=orange, outline=black, width=1)
    x, y = waypoints[-1]
    canvas.polygon([(x + 38, y), (x + 15, y - 14), (x + 15, y + 14)], fill=orange)
    # Direct visual matching shows the desired state without relying on color.
    canvas.line((875, 365, 1055, 365), fill=black, width=3)
    canvas.line((875, 355, 875, 375), fill=black, width=3)
    canvas.line((1055, 355, 1055, 375), fill=black, width=3)
    image.save(output, format="PNG", optimize=True)


if __name__ == "__main__":
    draw(Path(__file__).parent / "model/hero.png")
