"""Export CLIP ViT-B/16's image and text encoders, with a zero-shot ImageNet head, to one ONNX and check it.

The image encoder returns the image embedding and its zero-shot logits for the 1,000 ImageNet classes. The head holds
each class's text embedding, averaged over OpenAI's 80 prompt templates and scaled by CLIP's logit scale, so the
board scores ImageNet without a tokenizer. The text encoder takes token embeddings, because looking tokens up is a
data-dependent gather the MLA cannot run; the CPU looks them up and takes the row at the end-of-text token. Its
causal mask uses -1e4 instead of float32's minimum: softmax gives exactly 0 there either way, and -1e4 stays finite in
BF16. The export also returns fixed token embeddings for 64 prompts, which calibrate and check the text encoder.

With --text-probe <converted ONNX>, the script converts nothing. It takes the model.onnx a conversion wrote and writes
text_probe.npz, accuracy.py's check of the text encoder, into --output-dir: TEXT_PROBE_PROMPTS, their token IDs padded
to 77 with the end-of-text token, only the rows of the checkpoint's token embedding table those IDs use, and that
ONNX's FP32 end-of-text row for each prompt. The conversion tests check that the converted ONNX reproduces those rows,
so a conversion that disagrees with the file is not published. To change the file, convert, run --text-probe on that
model.onnx, then copy text_probe.npz to scripts/ and its SHA-256 to TEXT_PROBE_SHA256 in accuracy.py.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 224
TEXT_LENGTH = 77
EMBED_DIM = 512
CLASS_COUNT = 1000
CALIBRATION_PROMPTS = 64
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "token_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "image_embeds": [1, EMBED_DIM],
    "logits": [1, CLASS_COUNT],
    "text_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "calibration_token_embeds": [CALIBRATION_PROMPTS, TEXT_LENGTH, EMBED_DIM],
    "calibration_ends": [CALIBRATION_PROMPTS],
}
INPUTS = ("pixel_values", "token_embeds")
OUTPUTS = ("image_embeds", "logits", "text_embeds", "calibration_token_embeds", "calibration_ends")
SOURCE_FILES = ["pytorch_model.bin", "config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "special_tokens_map.json"]
SOURCE = ("file_set", "pytorch_model.bin", SOURCE_FILES)

# OpenAI's ImageNet class names and prompt templates for zero-shot evaluation, from
# https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# The names follow the ImageNet class index order (sorted WordNet IDs).
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
TEMPLATES = (
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.", "a rendering of a {}.",
    "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.", "a tattoo of a {}.",
    "the embroidered {}.", "a photo of a hard to see {}.", "a bright photo of a {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.", "a drawing of a {}.",
    "a photo of my {}.", "the plastic {}.", "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.", "a painting of a {}.",
    "a pixelated photo of the {}.", "a sculpture of the {}.", "a bright photo of the {}.",
    "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.", "a photo of the {}.",
    "a good photo of the {}.", "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.",
    "a doodle of a {}.", "a close-up photo of the {}.", "a photo of a {}.", "the origami {}.",
    "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.", "a origami {}.",
    "a low resolution photo of a {}.", "the toy {}.", "a rendition of the {}.",
    "a photo of the clean {}.", "a photo of a large {}.", "a rendition of a {}.",
    "a photo of a nice {}.", "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.", "a pixelated photo of a {}.",
    "itap of the {}.", "a jpeg corrupted photo of the {}.", "a good photo of a {}.", "a plushie {}.",
    "a photo of the nice {}.", "a photo of the small {}.", "a photo of the weird {}.", "the cartoon {}.",
    "art of the {}.", "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.", "itap of a {}.",
    "graffiti of the {}.", "a toy {}.", "itap of my {}.", "a photo of a cool {}.",
    "a photo of a small {}.", "a tattoo of the {}.",
)
# accuracy.py's check of the text encoder: OpenAI's "a photo of a {}." template on every 125th class. The text
# encoder calibrates on none of these prompts.
TEXT_PROBE_PROMPTS = tuple(f"a photo of a {CLASS_NAMES[index]}." for index in range(0, CLASS_COUNT, 125))


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    model_type = json.loads((directory / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type != "clip":
        raise ValueError(f"config.json names model_type {model_type!r}, expected 'clip'")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The CLIP export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [value.name for value in model.graph.output] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        dtype = onnx.TensorProto.INT64 if value.name == "calibration_ends" else onnx.TensorProto.FLOAT
        if tensor.elem_type != dtype or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded CLIP export")
    onnx.save(model, path)


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(source.parent)
    processor = CLIPProcessor.from_pretrained(source.parent, local_files_only=True)
    clip, loading = CLIPModel.from_pretrained(source.parent, local_files_only=True, attn_implementation="eager",
                                              output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not match the model: {loading}")
    clip = clip.float().eval()
    tokenizer = processor.tokenizer

    def tokens(prompts, padding="max_length"):
        return tokenizer(list(prompts), padding=padding, max_length=TEXT_LENGTH, truncation=True,
                         return_tensors="pt")["input_ids"]

    def text_features(input_ids):
        features = clip.get_text_features(input_ids=input_ids)
        return getattr(features, "pooler_output", features)

    with torch.inference_mode():
        # Each class's embedding is the normalized mean of its normalized prompt embeddings, as OpenAI evaluates it.
        # The text encoder is causal, so padding to the longest prompt gives the same features as padding to 77.
        class_embeds = []
        for name in CLASS_NAMES:
            features = text_features(tokens((template.format(name) for template in TEMPLATES), padding=True))
            mean = torch.nn.functional.normalize(features, dim=-1).mean(0)
            class_embeds.append(torch.nn.functional.normalize(mean, dim=0))
        class_weights = torch.stack(class_embeds).T * clip.logit_scale.exp()
        calibration_ids = tokens(
            TEMPLATES[index % len(TEMPLATES)].format(CLASS_NAMES[index * CLASS_COUNT // CALIBRATION_PROMPTS])
            for index in range(CALIBRATION_PROMPTS)
        )
        calibration_embeds = clip.text_model.embeddings.token_embedding(calibration_ids)
        # The end-of-text token has the highest id in CLIP's vocabulary.
        calibration_ends = calibration_ids.argmax(-1)

    class Clip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = clip
            self.register_buffer("class_weights", class_weights)
            self.register_buffer("calibration_token_embeds", calibration_embeds)
            self.register_buffer("calibration_ends", calibration_ends)
            self.register_buffer("mask", torch.full((TEXT_LENGTH, TEXT_LENGTH), -1e4).triu(1)[None, None])

        def forward(self, pixel_values, token_embeds):
            pooled = self.clip.vision_model(pixel_values=pixel_values).pooler_output
            image_embeds = self.clip.visual_projection(pooled)
            logits = (image_embeds / image_embeds.norm(dim=-1, keepdim=True)) @ self.class_weights
            text = self.clip.text_model
            positions = text.embeddings.position_embedding.weight[:TEXT_LENGTH][None]
            hidden = text.encoder(inputs_embeds=token_embeds + positions, attention_mask=self.mask).last_hidden_state
            text_embeds = self.clip.text_projection(text.final_layer_norm(hidden))
            return image_embeds, logits, text_embeds, self.calibration_token_embeds, self.calibration_ends

    model = Clip().eval()
    rng = np.random.RandomState(0)
    image = Image.fromarray(rng.randint(0, 256, (300, 400, 3), dtype=np.uint8))
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        torch.onnx.export(model, (pixel_values, calibration_embeds[:1]), str(output), input_names=list(INPUTS),
                          output_names=list(OUTPUTS), opset_version=17, dynamo=False)
    fold_constants(output)
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected_image = clip.get_image_features(pixel_values=pixel_values)
        expected_image = getattr(expected_image, "pooler_output", expected_image).numpy()
        expected_logits = (expected_image / np.linalg.norm(expected_image, axis=-1, keepdims=True)) @ class_weights.numpy()
        expected_text = text_features(calibration_ids[:4]).numpy()
    for index in range(4):
        image_embeds, logits, text_embeds, embeds, ends = session.run(
            None, {"pixel_values": pixel_values.numpy(), "token_embeds": calibration_embeds[index : index + 1].numpy()}
        )
        np.testing.assert_allclose(text_embeds[0, int(ends[index])], expected_text[index], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(image_embeds, expected_image, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(embeds, calibration_embeds.numpy())
    np.testing.assert_array_equal(ends, calibration_ends.numpy())
    print(f"PyTorch/ONNX check passed: {output}")
    return output


def write_text_probe(source, converted, output_dir):
    import onnxruntime as ort
    import torch
    from transformers import CLIPProcessor

    validate_config(source.parent)
    validate_export(converted)
    tokenizer = CLIPProcessor.from_pretrained(source.parent, local_files_only=True).tokenizer
    token_ids = tokenizer(list(TEXT_PROBE_PROMPTS), padding="max_length", max_length=TEXT_LENGTH, truncation=True,
                          return_tensors="np")["input_ids"]
    table = torch.load(source, map_location="cpu", weights_only=True)["text_model.embeddings.token_embedding.weight"]
    table = table.numpy()
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros(INTERFACE["pixel_values"], dtype=np.float32)
    text_embeds = []
    for ids in token_ids:
        (rows,) = session.run(["text_embeds"], {"pixel_values": pixel_values, "token_embeds": table[ids][None]})
        # The end-of-text token has the highest id in CLIP's vocabulary.
        text_embeds.append(rows[0, ids.argmax()])
    row_ids = np.unique(token_ids)
    arrays = {
        "prompts": np.asarray(TEXT_PROBE_PROMPTS),
        "token_ids": token_ids.astype(np.int32),
        "row_ids": row_ids.astype(np.int32),
        "rows": table[row_ids],
        "text_embeds": np.stack(text_embeds),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "text_probe.npz"
    # zipfile dates each member 1980-01-01, so the same arrays give the same bytes. np.savez does too, but it adds
    # ZIP64 fields, so its bytes differ from the file TEXT_PROBE_SHA256 pins.
    with zipfile.ZipFile(output, "w") as archive:
        for name, array in arrays.items():
            with archive.open(zipfile.ZipInfo(f"{name}.npy"), "w") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    print(f"Text probe written: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-probe", type=Path, metavar="ONNX",
                        help="write text_probe.npz for a converted model.onnx into --output-dir instead of converting")
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    if args.text_probe:
        write_text_probe(path, args.text_probe.resolve(), args.output_dir.resolve())
    else:
        convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
