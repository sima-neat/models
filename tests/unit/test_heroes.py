from __future__ import annotations

import io
import sys
from itertools import combinations
from pathlib import Path

import pytest
import yaml
from model_registry.documentation import MAX_HERO_BYTES, _validate_png

pytest.importorskip("PIL")
from PIL import Image, ImageChops  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools" / "heroes"))

import draw  # noqa: E402
import render  # noqa: E402
from specs import ARCHITECTURE_FAMILIES, SPECS, TASK_TEMPLATES, HeroSpec  # noqa: E402

MODEL_DIRS = sorted(path.parent for path in (REPOSITORY / "models").glob("*/model.yaml"))


def _model(model_dir: Path) -> dict:
    return yaml.safe_load((model_dir / "model.yaml").read_text())


def test_every_model_has_a_hero_spec_and_every_spec_a_model() -> None:
    assert sorted(path.name for path in MODEL_DIRS) == sorted(SPECS)


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=lambda path: path.name)
def test_hero_spec_follows_the_model_family_and_task(model_dir: Path) -> None:
    model, spec = _model(model_dir), SPECS[model_dir.name]
    assert spec.family == ARCHITECTURE_FAMILIES[model["architecture"]["id"]]
    assert spec.template == TASK_TEMPLATES[model["tasks"][0]]


def test_every_hero_has_its_own_scene() -> None:
    scenes = [spec.scene for spec in SPECS.values()]
    assert len(scenes) == len(set(scenes))
    contents = [tuple((item.kind, item.label) for item in spec.items) for spec in SPECS.values()]
    assert len(contents) == len(set(contents))


def test_labels_use_the_minimum_font_size() -> None:
    assert draw.LABEL_FONT_SIZE >= 48


def _check_layout(spec: HeroSpec) -> Image.Image:
    image, chips = render.render(spec)
    assert image.size == (draw.WIDTH, draw.HEIGHT)
    for chip in chips:
        assert 0 <= chip[0] < chip[2] <= draw.WIDTH and 0 <= chip[1] < chip[3] <= draw.HEIGHT, chip
    for a, b in combinations(chips, 2):
        assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    return image


@pytest.mark.parametrize("model_id", sorted(SPECS))
def test_committed_hero_matches_its_spec(model_id: str) -> None:
    image = _check_layout(SPECS[model_id])
    payload = render.encode(image)
    hero = REPOSITORY / "models" / model_id / "hero.png"
    assert len(payload) <= MAX_HERO_BYTES
    _validate_png(payload, hero)
    committed = Image.open(io.BytesIO(hero.read_bytes())).convert("RGB")
    # Allow for font rasterization differences between Pillow builds, not for a changed scene.
    changed = ImageChops.difference(committed, image).convert("L").point(lambda value: 255 if value > 24 else 0)
    assert changed.histogram()[255] / (draw.WIDTH * draw.HEIGHT) < 0.005, f"rerun tools/heroes/render.py {model_id}"


TEMPLATE_SAMPLES = {
    "keypoints": HeroSpec("yolo26", "keypoints", "sample pose", [
        draw.Item("person", 420, 1.0, "person", pose="wave"), draw.Item("person", 760, 1.0, "person", pose="walk")]),
    "depth": HeroSpec("yolo26", "depth", "sample depth", [
        draw.Item("tree", 200, 0.8, look={"depth": "0.8"}), draw.Item("car", 760, 1.2, look={"depth": "0.1"})]),
    "segmentation": HeroSpec("yolo26", "segmentation", "sample segmentation", [
        draw.Item("person", 300), draw.Item("dog", 560, mask=True)], prompt="dog"),
}


@pytest.mark.parametrize("template", sorted(set(TASK_TEMPLATES.values())))
def test_every_task_template_renders(template: str) -> None:
    spec = next((spec for spec in SPECS.values() if spec.template == template), TEMPLATE_SAMPLES.get(template))
    _validate_png(render.encode(_check_layout(spec)), Path(f"{template}.png"))
