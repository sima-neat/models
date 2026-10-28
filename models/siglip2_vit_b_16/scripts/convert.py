"""Export SigLIP 2 base patch16-224's image and text encoders, with a zero-shot ImageNet head, to one ONNX and check it.

The image encoder returns the image embedding and its zero-shot logits for the 1,000 ImageNet classes. The head holds
each class's text embedding, averaged over the 7 prompt templates OpenAI selected for ImageNet and scaled by the
model's logit scale, plus its logit bias, so the board scores ImageNet without a tokenizer. The text encoder takes
token embeddings, because looking tokens up in the 256,000-token vocabulary is a data-dependent gather the MLA cannot
run; the CPU looks them up. It returns the pooled text embedding, which SigLIP takes from the last of its 64 tokens.
The export also returns fixed token embeddings for 64 prompts, which calibrate and check the text encoder.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 224
TEXT_LENGTH = 64
EMBED_DIM = 768
CLASS_COUNT = 1000
CALIBRATION_PROMPTS = 64
# Prompts per batch while the class embeddings are computed: 40 classes of 7 templates.
BATCH = 280
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "token_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "image_embeds": [1, EMBED_DIM],
    "logits": [1, CLASS_COUNT],
    "text_embeds": [1, EMBED_DIM],
    "calibration_token_embeds": [CALIBRATION_PROMPTS, TEXT_LENGTH, EMBED_DIM],
}
INPUTS = ("pixel_values", "token_embeds")
OUTPUTS = ("image_embeds", "logits", "text_embeds", "calibration_token_embeds")
SOURCE_FILES = ["model.safetensors", "config.json", "preprocessor_config.json", "special_tokens_map.json",
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
SOURCE = ("file_set", "model.safetensors", SOURCE_FILES)

# OpenAI's ImageNet class names and the 7 prompt templates its forward selection chose over the 80, from
# https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# The names follow the ImageNet class index order (sorted WordNet IDs). SigLIP 2 was trained on lowercase text.
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
    "itap of a {}.", "a bad photo of the {}.", "a origami {}.", "a photo of the large {}.", "a {} in a video game.",
    "art of the {}.", "a photo of the small {}.",
)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    model_type = json.loads((directory / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type != "siglip":
        raise ValueError(f"config.json names model_type {model_type!r}, expected 'siglip'")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The SigLIP 2 export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [value.name for value in model.graph.output] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]:
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
        raise RuntimeError("onnxsim could not validate the folded SigLIP 2 export")
    onnx.save(model, path)


def prompts(class_names, templates):
    """Every template for every class, lowercased as SigLIP 2 was trained, class by class."""
    return [template.format(name).lower() for name in class_names for template in templates]


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipModel

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(source.parent)
    processor = AutoProcessor.from_pretrained(source.parent, local_files_only=True)
    siglip, loading = SiglipModel.from_pretrained(source.parent, local_files_only=True, attn_implementation="eager",
                                                  output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not match the model: {loading}")
    siglip = siglip.float().eval()
    tokenizer = processor.tokenizer

    def tokens(texts):
        # SigLIP 2 reads 64 tokens padded to the end, as Transformers' processor gives them.
        return tokenizer(list(texts), padding="max_length", max_length=TEXT_LENGTH, truncation=True,
                         return_tensors="pt")["input_ids"]

    def text_features(input_ids):
        features = siglip.get_text_features(input_ids=input_ids)
        return getattr(features, "pooler_output", features)

    with torch.inference_mode():
        # Each class's embedding is the normalized mean of its normalized prompt embeddings, as OpenAI evaluates it.
        texts = prompts(CLASS_NAMES, TEMPLATES)
        features = torch.cat([text_features(tokens(texts[start : start + BATCH])) for start in range(0, len(texts), BATCH)])
        features = torch.nn.functional.normalize(features, dim=-1).reshape(CLASS_COUNT, len(TEMPLATES), EMBED_DIM)
        class_embeds = torch.nn.functional.normalize(features.mean(1), dim=-1)
        class_weights = class_embeds.T * siglip.logit_scale.exp()
        calibration_ids = tokens(
            TEMPLATES[index % len(TEMPLATES)].format(CLASS_NAMES[index * CLASS_COUNT // CALIBRATION_PROMPTS]).lower()
            for index in range(CALIBRATION_PROMPTS)
        )
        calibration_embeds = siglip.text_model.embeddings.token_embedding(calibration_ids)

    class SigLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.siglip = siglip
            self.register_buffer("class_weights", class_weights)
            self.register_buffer("logit_bias", siglip.logit_bias.detach().clone())
            self.register_buffer("calibration_token_embeds", calibration_embeds)

        def forward(self, pixel_values, token_embeds):
            image_embeds = self.siglip.vision_model(pixel_values=pixel_values).pooler_output
            logits = (image_embeds / image_embeds.norm(dim=-1, keepdim=True)) @ self.class_weights + self.logit_bias
            text = self.siglip.text_model
            hidden = text.encoder(inputs_embeds=text.embeddings(inputs_embeds=token_embeds)).last_hidden_state
            text_embeds = text.head(text.final_layer_norm(hidden)[:, -1, :])
            return image_embeds, logits, text_embeds, self.calibration_token_embeds

    model = SigLIP().eval()
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
        expected_image = siglip.get_image_features(pixel_values=pixel_values)
        expected_image = getattr(expected_image, "pooler_output", expected_image).numpy()
        normalized = expected_image / np.linalg.norm(expected_image, axis=-1, keepdims=True)
        expected_logits = normalized @ class_weights.numpy() + siglip.logit_bias.detach().numpy()
        expected_text = text_features(calibration_ids[:4]).numpy()
    for index in range(4):
        image_embeds, logits, text_embeds, embeds = session.run(
            None, {"pixel_values": pixel_values.numpy(), "token_embeds": calibration_embeds[index : index + 1].numpy()}
        )
        np.testing.assert_allclose(text_embeds[0], expected_text[index], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(image_embeds, expected_image, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(embeds, calibration_embeds.numpy())
    print(f"PyTorch/ONNX check passed: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
