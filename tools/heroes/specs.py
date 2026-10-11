"""Hero image specifications: one family per architecture family and one distinct scene per model.

Add a model by adding a HeroSpec under its model ID. Reuse its family; give it a scene no other
model uses. Add a family only for a new architecture family. See AGENTS.md, "Model hero images".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from draw import CLAY, CREAM, DENIM, GROUND, SAGE, SAND, SLATE, SLATE_DARK, Family, Item

FAMILIES: dict[str, Family] = {
    "resnet": Family("resnet", accent="#6D28D9", tint="#F5F2FD", shade="#DCD3F2", ground="#E7E1F7"),
    "rf_detr": Family("rf_detr", accent="#0F766E", tint="#EEF6F4", shade="#D5E7E2", ground="#E4D7C6"),
    "ssd_mobilenet": Family("ssd_mobilenet", accent="#C2410C", tint="#FBF4EC", shade="#DCE5D1",
                            ground="#D3DEC4"),
    "yolo26": Family("yolo26", accent="#2563EB", tint="#EEF3FC", shade="#DCE4F3", ground="#D3DBE7"),
}

# model.yaml architecture.id -> hero family. Variants of one design share a family.
ARCHITECTURE_FAMILIES: dict[str, str] = {
    "resnet_50": "resnet",
    "rf_detr": "rf_detr",
    "ssd_mobilenet_v1": "ssd_mobilenet",
    "ssd_mobilenet_v2": "ssd_mobilenet",
    "ssdlite_mobilenet_v3_large": "ssd_mobilenet",
    "yolo_26": "yolo26",
}

# model.yaml task -> template that shows that task's output on its input.
TASK_TEMPLATES: dict[str, str] = {
    "image_classification": "classification",
    "object_detection": "detection",
    "pose_estimation": "keypoints",
    "depth_estimation": "depth",
    "instance_segmentation": "segmentation",
    "semantic_segmentation": "segmentation",
}

FAMILY_BACKDROPS = {"resnet": "plain", "rf_detr": "room", "ssd_mobilenet": "outdoor", "yolo26": "street"}


@dataclass(frozen=True)
class HeroSpec:
    family: str
    template: str
    scene: str  # unique across all heroes
    items: list[Item]
    backdrop: str | None = None  # defaults to the family backdrop
    variant: str = ""
    scores: list[tuple[str, float]] = field(default_factory=list)  # classification only
    prompt: str = ""  # segmentation only
    inset: int | None = None  # index of an item to show magnified, for very small objects


ON_TABLE = GROUND - 120
ON_COUNTER = GROUND - 122

SPECS: dict[str, HeroSpec] = {
    "resnet_50": HeroSpec(
        "resnet", "classification", "cat portrait",
        [Item("cat", 300, 1.15, ground=400)],
        scores=[("tabby cat", 0.74), ("tiger cat", 0.17), ("Egyptian cat", 0.05)],
    ),
    "rf_detr_nano": HeroSpec(
        "rf_detr", "detection", "home desk",
        [Item("window", 230, 1.0, ground=300), Item("table", 560, look={"width": "440"}),
         Item("laptop", 500, 1.0, "laptop", ground=ON_TABLE), Item("cup", 680, 1.0, "cup", ground=ON_TABLE),
         Item("potted_plant", 960, 1.0, "potted plant")],
    ),
    "rf_detr_small": HeroSpec(
        "rf_detr", "detection", "kitchen counter",
        [Item("refrigerator", 190, 1.0, "refrigerator"), Item("counter", 720, look={"width": "720"}),
         Item("bottle", 560, 1.0, "bottle", ground=ON_COUNTER), Item("bowl", 860, 1.0, "bowl", ground=ON_COUNTER)],
        variant="kitchen",
    ),
    "rf_detr_medium": HeroSpec(
        "rf_detr", "detection", "living room",
        [Item("couch", 330, 1.0, "couch"), Item("clock", 640, 0.9, "clock", ground=176),
         Item("tv", 940, 1.0, "tv", ground=GROUND)],
    ),
    "rf_detr_large": HeroSpec(
        "rf_detr", "detection", "dining room",
        [Item("window", 1040, 1.0, ground=280), Item("chair", 370, 1.0, "chair"),
         Item("table", 600, 1.0, "dining table", look={"width": "320"}),
         Item("chair", 830, 1.0, "chair", look={"facing": "left"})],
    ),
    "ssd_mobilenet_v1": HeroSpec(
        "ssd_mobilenet", "detection", "pasture",
        [Item("tree", 1080, 1.2, ground=GROUND - 6), Item("horse", 320, 1.0, "horse"),
         Item("sheep", 660, 1.0, "sheep"), Item("sheep", 880, 0.85, "sheep")],
    ),
    "ssd_mobilenet_v2": HeroSpec(
        "ssd_mobilenet", "detection", "beach",
        [Item("umbrella", 280, 1.0, "umbrella"),
         Item("person", 700, 0.9, "person", pose="wave",
              look={"shirt": DENIM, "pants": SAND, "tether": "960,168"}),
         Item("kite", 960, 1.0, "kite", ground=150)],
        variant="shore",
    ),
    "ssd_mobilenet_v3": HeroSpec(
        "ssd_mobilenet", "detection", "harbor",
        [Item("boat", 360, 1.0, "boat", ground=GROUND - 12), Item("bird", 760, 1.0, "bird", ground=150),
         Item("person", 1000, 1.0, "person", look={"shirt": SAGE, "pants": SLATE_DARK})],
        variant="shore",
    ),
    "yolo26_tiny_drone": HeroSpec(
        "yolo26", "detection", "drones in open sky",
        [Item("drone", 260, 0.5, "drone", ground=200), Item("drone", 560, 0.45, "drone", ground=330)],
        backdrop="sky", inset=0,
    ),
    "yolo26n": HeroSpec(
        "yolo26", "detection", "crosswalk",
        [Item("person", 420, 1.0, "person", pose="walk", look={"shirt": CLAY}),
         Item("person", 630, 0.92, "person", pose="walk", look={"shirt": SAGE, "pants": DENIM}),
         Item("traffic_light", 900, 1.0, "traffic light")],
        variant="crosswalk",
    ),
    "yolo26s": HeroSpec(
        "yolo26", "detection", "bus stop",
        [Item("bus", 330, 1.0, "bus"), Item("person", 720, 1.0, "person", look={"shirt": DENIM}),
         Item("bench", 960, 1.0, "bench")],
    ),
    "yolo26m": HeroSpec(
        "yolo26", "detection", "dog walk",
        [Item("person", 340, 1.0, "person", pose="reach", look={"shirt": SAGE, "tether": "578,312"}),
         Item("dog", 530, 0.95, "dog"), Item("bicycle", 900, 1.15, "bicycle")],
    ),
    "yolo26l": HeroSpec(
        "yolo26", "detection", "parking lot",
        [Item("car", 280, 1.0, "car"), Item("car", 650, 1.0, "car", look={"body": SAGE}),
         Item("person", 960, 1.0, "person", look={"shirt": SAND, "pants": SLATE})],
    ),
    "yolo26x": HeroSpec(
        "yolo26", "detection", "loading zone",
        [Item("truck", 360, 1.0, "truck"),
         Item("person", 760, 1.0, "person", pose="walk", look={"shirt": CREAM, "pants": DENIM}),
         Item("person", 960, 0.95, "person", look={"shirt": CLAY})],
    ),
}
