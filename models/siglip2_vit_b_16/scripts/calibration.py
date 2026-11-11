"""SigLIP 2 ViT-B/16 image preprocessing and ImageNet sample selection, used for numerical fidelity and accuracy."""

from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# ImageNet class names from Big Vision commit 0127fb6b337ee2a27bf4e54dea79cff176527356:
# datasets/imagenet/class_names.py. Surgery also uses these names for fidelity prompts.
# fmt: off
CLASS_NAMES = (
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt",
    "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
    "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
    "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama",
    "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon",
    "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake",
    "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake",
    "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick",
    "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail",
    "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal",
    "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser",
    "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish",
    "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton",
    "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab",
    "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork",
    "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird",
    "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin",
    "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross",
    "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese",
    "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback",
    "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound",
    "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound",
    "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound",
    "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
    "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier",
    "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier",
    "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer",
    "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier",
    "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever",
    "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever",
    "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter",
    "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog",
    "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog",
    "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog",
    "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah",
    "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle",
    "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect",
    "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly",
    "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly",
    "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber",
    "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot",
    "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog",
    "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep",
    "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel",
    "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo",
    "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon",
    "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey",
    "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey",
    "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant",
    "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion",
    "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack",
    "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail",
    "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball",
    "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon",
    "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass",
    "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse",
    "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap",
    "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom",
    "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron",
    "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit",
    "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player",
    "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence",
    "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet",
    "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs",
    "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock",
    "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet",
    "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth",
    "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum",
    "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive",
    "entertainment center", "envelope", "espresso machine", "face powder", "feather boa",
    "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car",
    "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump",
    "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano",
    "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray",
    "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief",
    "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater",
    "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass",
    "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw",
    "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer",
    "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine",
    "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass",
    "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit",
    "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup",
    "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t",
    "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque",
    "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van",
    "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk",
    "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt",
    "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock",
    "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars",
    "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer",
    "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt",
    "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle",
    "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
    "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker",
    "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale",
    "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt",
    "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket",
    "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask",
    "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow",
    "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl",
    "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web",
    "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge",
    "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram",
    "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses",
    "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing",
    "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear",
    "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine",
    "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
    "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran",
    "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron",
    "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle",
    "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade",
    "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence",
    "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign",
    "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle",
    "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog",
    "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom",
    "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit",
    "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough",
    "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain",
    "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley",
    "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy",
    "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus",
    "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom",
    "bolete", "corn cob", "toilet paper",
)
# fmt: on

# The checkpoint's declared preprocessing (its preprocessor_config.json): a bilinear resize straight to 224x224,
# then mean and std 0.5, with channel constants in HWC order.
SIZE = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
# imagenet-2012/cal holds cal2012/<image> and imagenet-2012/val holds val2012/<image>, each with
# annotations/labels_<split>.json, whose category IDs are the ImageNet class indices.
IMAGE_SUFFIXES = (".jpeg",)
FIDELITY_SPLIT = "cal2012"
VALIDATION_SPLIT = "val2012"
CLASSES = 1000


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize and normalize one image into a float32 HWC tensor."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) * RESCALE - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    """Image paths relative to the dataset root, from an archive or an extracted directory."""
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    else:
        with tarfile.open(dataset, "r:gz") as archive:
            names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if name.lower().endswith(IMAGE_SUFFIXES))


def split_names(dataset: Path, split: str) -> list[str]:
    """The split's images; an archive holds them under its top-level directory."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == split]
    if not names:
        raise ValueError(f"{dataset} has no images under {split}/")
    return names


def validation_labels(dataset: Path) -> dict[str, int]:
    """Each validation image's class index, from the extracted dataset's labels file."""
    labels = json.loads((dataset / "annotations" / f"labels_{VALIDATION_SPLIT}.json").read_text(encoding="utf-8"))
    if len(labels["categories"]) != CLASSES:
        raise ValueError(f"expected {CLASSES} validation classes, found {len(labels['categories'])}")
    names = {image["id"]: f"{VALIDATION_SPLIT}/{image['file_name']}" for image in labels["images"]}
    return {names[annotation["image_id"]]: annotation["category_id"] for annotation in labels["annotations"]}


def read_images(dataset: Path, names: Iterable[str]) -> dict[str, bytes]:
    """Read the requested images; an archive is scanned once rather than seeked per image."""
    wanted = set(names)
    if dataset.is_dir():
        return {name: (dataset / name).read_bytes() for name in wanted}
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in wanted and member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    found[name] = stream.read()
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested images, such as {sorted(missing)[0]}")
    return found


def select_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Fidelity images selected as model.yaml declares: a seeded shuffle, then a slice."""
    names = split_names(dataset, FIDELITY_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"the selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]


def load_images(dataset: Path, names: list[str]) -> np.ndarray:
    """The named images, preprocessed and stacked in NHWC order."""
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
