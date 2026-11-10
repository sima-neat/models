"""Adapt CLIP ViT-B/16 in PyTorch, export ONNX and split its image and text encoders.

The image encoder keeps its raw embedding and adds fixed ImageNet class scores using OpenAI's 80 prompt templates.
The text encoder starts after token lookup and returns each token's projected row; the CPU selects end-of-text.
Its static causal mask uses -1e4, which gives zero masked softmax probabilities while remaining finite in BF16.
A separate prompts.npz holds 64 fixed prompts for numerical fidelity; neither encoder needs calibration.

Run with --weights weights.pth --output-dir graphs. To regenerate scripts/text_probe.npz, pass --text-probe
<graphs/model.onnx> and a fresh --output-dir, then update the checksum in accuracy.py and performance.py.
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx

IMAGE_SIZE = 224
TEXT_LENGTH = 77
EMBED_DIM = 512
CLASS_COUNT = 1000
FIDELITY_PROMPTS = 64
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "token_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "image_embeds": [1, EMBED_DIM],
    "logits": [1, CLASS_COUNT],
    "text_embeds": [1, TEXT_LENGTH, EMBED_DIM],
}
INPUTS = ("pixel_values", "token_embeds")
OUTPUTS = ("image_embeds", "logits", "text_embeds")
# fmt: off
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
# encoder checks these independently of the 64 reference prompts.
TEXT_PROBE_PROMPTS = tuple(f"a photo of a {CLASS_NAMES[index]}." for index in range(0, CLASS_COUNT, 125))
# fmt: on


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The CLIP export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [
        value.name for value in model.graph.output
    ] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        dtype = onnx.TensorProto.FLOAT
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


def build_image_encoder(clip, class_weights):
    """Use 4D channel-first transformer blocks and fused Q/K/V projections for the image encoder."""
    import torch
    from torch import nn

    def projection(linear):
        conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
        conv.weight.data.copy_(linear.weight.detach()[:, :, None, None])
        if linear.bias is not None:
            conv.bias.data.copy_(linear.bias.detach())
        return conv

    class ChannelNorm(nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.register_buffer("weight", norm.weight.detach()[None, :, None, None])
            self.register_buffer("bias", norm.bias.detach()[None, :, None, None])
            self.eps = norm.eps

        def forward(self, x):
            centered = x - x.mean(dim=1, keepdim=True)
            return (
                centered / torch.sqrt((centered * centered).mean(dim=1, keepdim=True) + self.eps) * self.weight
                + self.bias
            )

    class Attention(nn.Module):
        def __init__(self, attention):
            super().__init__()
            channels, self.heads = attention.embed_dim, attention.num_heads
            self.head_size = channels // self.heads
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.qkv.weight.data.copy_(
                torch.cat(
                    [
                        attention.q_proj.weight * self.head_size**-0.5,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ]
                )[:, :, None, None]
            )
            self.qkv.bias.data.copy_(
                torch.cat(
                    [
                        attention.q_proj.bias * self.head_size**-0.5,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ]
                )
            )
            self.output = projection(attention.out_proj)

        def split_heads(self, x):
            # Channel h*D+d belongs to head h; split/concat preserves that exact mapping.
            return torch.cat(torch.split(x, self.head_size, dim=1), dim=2)

        def forward(self, x):
            query, key, value = [self.split_heads(v) for v in self.qkv(x).chunk(3, dim=1)]
            scores = torch.einsum("nchw,nchq->nqhw", query, key)
            probs = torch.softmax(scores, dim=1)
            context = torch.einsum("nchw,nqhc->nqhw", probs, value)
            merged = torch.cat(torch.split(context, 1, dim=2), dim=1)
            return self.output(merged)

    class Block(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.norm1, self.norm2 = ChannelNorm(block.layer_norm1), ChannelNorm(block.layer_norm2)
            self.attention = Attention(block.self_attn)
            self.fc1, self.fc2 = projection(block.mlp.fc1), projection(block.mlp.fc2)

        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            hidden = self.fc1(self.norm2(x))
            # The pinned CLIP checkpoint uses QuickGELU in both encoders.
            hidden = hidden * torch.sigmoid(1.702 * hidden)
            return x + self.fc2(hidden)

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            vision = clip.vision_model
            assert clip.config.vision_config.hidden_act == "quick_gelu"
            self.patches = vision.embeddings.patch_embedding
            self.register_buffer("class_embedding", vision.embeddings.class_embedding.detach()[None, :, None, None])
            self.register_buffer("positions", vision.embeddings.position_embedding.weight.detach().T[None, :, None, :])
            self.pre_norm = ChannelNorm(vision.pre_layrnorm)
            self.post_norm = ChannelNorm(vision.post_layernorm)
            self.blocks = nn.ModuleList(Block(block) for block in vision.encoder.layers)
            self.projection = projection(clip.visual_projection)
            self.register_buffer("class_weights", class_weights)
            self.register_buffer("norm_scale", torch.full((1, EMBED_DIM, 1, 1), EMBED_DIM**-0.5))

        def forward(self, pixels):
            patches = self.patches(pixels).reshape(1, 768, 1, 196)
            image = self.pre_norm(torch.cat([self.class_embedding, patches], dim=3) + self.positions)
            for block in self.blocks:
                image = block(image)
            image = self.projection(self.post_norm(image[:, :, :, :1]))
            embedding = image.reshape(1, EMBED_DIM)
            # Native BF16 RMSNorm computes L2 normalization after scaling by 1/sqrt(512).
            # A standalone reciprocal square root would use an INT8 lookup table and require calibration.
            image = image / torch.sqrt((image * image).mean(dim=1, keepdim=True) + 1e-12)
            logits = (image * self.norm_scale).reshape(1, EMBED_DIM) @ self.class_weights
            return embedding, logits

    return ImageEncoder().float().eval()


def load_clip(source):
    """Restore the upstream checkpoint and its processor from the conversion bundle."""
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    assets = checkpoint["assets"]
    with tempfile.TemporaryDirectory(prefix="clip-processor-") as directory:
        for name, content in assets.items():
            (Path(directory) / name).write_text(content, encoding="utf-8")
        processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)
    config = CLIPConfig.from_dict(json.loads(assets["config.json"]))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    clip = CLIPModel(config).float().eval()
    state = checkpoint["state_dict"]
    # Older checkpoints persist these deterministic buffers; current Transformers does not.
    for name, length in (("text_model", TEXT_LENGTH), ("vision_model", 197)):
        positions = state.pop(f"{name}.embeddings.position_ids")
        if not torch.equal(positions, torch.arange(length)[None]):
            raise ValueError(f"unexpected {name} position IDs")
    clip.load_state_dict(state, strict=True)
    return clip, processor


def tokenize(tokenizer, prompts, padding="max_length"):
    return tokenizer(list(prompts), padding=padding, max_length=TEXT_LENGTH, truncation=True, return_tensors="pt")[
        "input_ids"
    ]


def text_features(clip, input_ids):
    features = clip.get_text_features(input_ids=input_ids)
    return getattr(features, "pooler_output", features)


def build_class_weights(clip, tokenizer):
    """Build OpenAI's normalized 80-prompt ensemble for the fixed ImageNet head."""
    import torch

    with torch.inference_mode():
        # Each class's embedding is the normalized mean of its normalized prompt embeddings, as OpenAI evaluates it.
        # The text encoder is causal, so padding to the longest prompt gives the same features as padding to 77.
        class_embeds = []
        for name in CLASS_NAMES:
            features = text_features(
                clip, tokenize(tokenizer, (template.format(name) for template in TEMPLATES), padding=True)
            )
            mean = torch.nn.functional.normalize(features, dim=-1).mean(0)
            class_embeds.append(torch.nn.functional.normalize(mean, dim=0))
        return torch.stack(class_embeds).T * clip.logit_scale.exp()


def prepare_prompts(clip, tokenizer):
    """Prepare the fixed token IDs and embeddings used for export and fidelity checks."""
    import torch

    with torch.inference_mode():
        prompt_ids = tokenize(
            tokenizer,
            (
                TEMPLATES[index % len(TEMPLATES)].format(CLASS_NAMES[index * CLASS_COUNT // FIDELITY_PROMPTS])
                for index in range(FIDELITY_PROMPTS)
            ),
        )
        prompt_embeds = clip.text_model.embeddings.token_embedding(prompt_ids)
    return prompt_ids, prompt_embeds


def build_model(clip, class_weights):
    """Compose the optimized image encoder and the static text transformer."""
    import torch

    class Clip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text = clip.text_model
            self.text_projection = clip.text_projection
            self.image_encoder = build_image_encoder(clip, class_weights)
            self.register_buffer("mask", torch.full((TEXT_LENGTH, TEXT_LENGTH), -1e4).triu(1)[None, None])

        def forward(self, pixel_values, token_embeds):
            image_embeds, logits = self.image_encoder(pixel_values)
            text = self.text
            positions = text.embeddings.position_embedding.weight[:TEXT_LENGTH][None]
            hidden = text.encoder(inputs_embeds=token_embeds + positions, attention_mask=self.mask).last_hidden_state
            text_embeds = self.text_projection(text.final_layer_norm(hidden))
            return image_embeds, logits, text_embeds

    return Clip().eval()


def export_onnx(model, pixel_values, prompt_embeds, output):
    """Export the fixed interface and fold constant ONNX expressions."""
    import torch

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, prompt_embeds[:1]),
            str(output),
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)


def compare_checkpoint(output, clip, class_weights, pixel_values, prompt_ids, prompt_embeds):
    """Compare the exported image, class-score and end-of-text outputs with upstream CLIP."""
    import onnxruntime as ort
    import torch

    # The end-of-text token has the highest id in CLIP's vocabulary.
    prompt_ends = prompt_ids.argmax(-1)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected_image = clip.get_image_features(pixel_values=pixel_values)
        expected_image = getattr(expected_image, "pooler_output", expected_image).numpy()
        expected_logits = (
            expected_image / np.linalg.norm(expected_image, axis=-1, keepdims=True)
        ) @ class_weights.numpy()
        expected_text = text_features(clip, prompt_ids[:4]).numpy()
    for index in range(4):
        image_embeds, logits, text_embeds = session.run(
            None, {"pixel_values": pixel_values.numpy(), "token_embeds": prompt_embeds[index : index + 1].numpy()}
        )
        np.testing.assert_allclose(text_embeds[0, int(prompt_ends[index])], expected_text[index], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(image_embeds, expected_image, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-3, atol=1e-2)


def export_model(source, output_dir):
    """Restore CLIP, prepare its static inputs, export the graph and verify checkpoint parity."""
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    clip, processor = load_clip(source)
    class_weights = build_class_weights(clip, processor.tokenizer)
    prompt_ids, prompt_embeds = prepare_prompts(clip, processor.tokenizer)
    model = build_model(clip, class_weights)
    rng = np.random.RandomState(0)
    image = Image.fromarray(rng.randint(0, 256, (300, 400, 3), dtype=np.uint8))
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    export_onnx(model, pixel_values, prompt_embeds, output)
    compare_checkpoint(output, clip, class_weights, pixel_values, prompt_ids, prompt_embeds)
    np.savez(output_dir / "prompts.npz", prompts=prompt_embeds.numpy(), ends=prompt_ids.argmax(-1).numpy())
    print(f"PyTorch/ONNX check passed: {output}")
    return output


def write_text_probe(source, converted, output_dir):
    import onnxruntime as ort
    import torch
    from transformers import CLIPProcessor

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_export(converted)
    with tempfile.TemporaryDirectory(prefix="clip-tokenizer-") as directory:
        for name, content in checkpoint["assets"].items():
            (Path(directory) / name).write_text(content, encoding="utf-8")
        tokenizer = CLIPProcessor.from_pretrained(directory, local_files_only=True).tokenizer
    token_ids = tokenizer(
        list(TEXT_PROBE_PROMPTS), padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="np"
    )["input_ids"]
    table = checkpoint["state_dict"]["text_model.embeddings.token_embedding.weight"].numpy()
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


COMPONENT_GRAPHS = {
    "image_encoder": {
        "inputs": {"pixel_values": tuple(INTERFACE["pixel_values"])},
        "outputs": {name: tuple(INTERFACE[name]) for name in ("image_embeds", "logits")},
    },
    "text_encoder": {
        "inputs": {"token_embeds": tuple(INTERFACE["token_embeds"])},
        "outputs": {"text_embeds": tuple(INTERFACE["text_embeds"])},
    },
}


def graph_paths(directory):
    return {component: directory / f"{component}.onnx" for component in COMPONENT_GRAPHS}


def prepare_model(weights, directory):
    """Adapt CLIP in PyTorch, export it, and split the independent encoders for compilation."""
    source = export_model(weights, directory)
    extractor = onnx.utils.Extractor(onnx.load(source))
    paths = graph_paths(directory)
    for component, path in paths.items():
        graph = COMPONENT_GRAPHS[component]
        model = extractor.extract_model(list(graph["inputs"]), list(graph["outputs"]))
        onnx.checker.check_model(model)
        onnx.save(model, path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-probe", type=Path, metavar="ONNX", help="regenerate the text probe from this export")
    args = parser.parse_args()
    if args.text_probe:
        write_text_probe(args.weights, args.text_probe, args.output_dir)
    else:
        prepare_model(args.weights, args.output_dir)


if __name__ == "__main__":
    main()
