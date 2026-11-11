"""Measure the BF16 text encoder and mixed detector independently with Neat.

One text sample is one prepared 77-token prompt. One detector sample is one
640x640 normalized RGB image plus 80 cached prompt vectors and a mask, with
input quantization and YOLOv8 BoxDecode/NMS. CPU JPEG loading/letterboxing and prompt
updates are outside these independent model timings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tarfile

import model_report
import numpy as np

COMPONENTS = ("text_encoder", "detector")


def parse_mpks(values):
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("use --mpk text_encoder=<path> --mpk detector=<path>")
    result = dict(pairs)
    if len(pairs) != len(result) or set(result) != set(COMPONENTS):
        raise ValueError("expected exactly text_encoder and detector")
    return {name: Path(path) for name, path in result.items()}


def options(decode=False):
    import pyneat

    value = pyneat.ModelOptions()
    value.preprocess.kind = pyneat.InputKind.Tensor
    if decode:
        value.decode_type = pyneat.BoxDecodeType.YoloV8
        value.decode_type_option = pyneat.BoxDecodeTypeOption.GroupedByRoleProbability
        value.num_classes = 80
        value.score_threshold = .25
        value.nms_iou_threshold = .7
        value.top_k = 10
        value.boxdecode_original_width = value.boxdecode_original_height = 640
        value.boxdecode_resize_mode = pyneat.ResizeMode.Letterbox
    return value


class Component:
    def __init__(self, archive, decode=False):
        import pyneat

        with tarfile.open(archive, "r:gz") as source:
            (name,) = [n for n in source.getnames() if n.endswith("_mpk.json")]
            self.names = [item["name"] for item in json.load(source.extractfile(name))["input_nodes"]]
        self.model = pyneat.Model(str(archive), options(decode))
        self.specs = self.model.input_specs()
        self.runner = None

    def tensors(self, inputs):
        import pyneat

        if set(inputs) != set(self.names):
            raise ValueError(f"expected inputs {self.names}, got {list(inputs)}")
        result = []
        for name, spec in zip(self.names, self.specs, strict=True):
            value = np.asarray(inputs[name], np.float32)
            shape = tuple(spec.shape)
            if value.shape != shape:
                if value.shape[0] != 1 or value.shape[1:] != shape:
                    raise ValueError(f"{name}: expected {shape}, got {value.shape}")
                value = value[0]
            result.append(pyneat.Tensor.from_numpy(np.ascontiguousarray(value), copy=True,
                          layout=pyneat.TensorLayout.HWC if value.ndim in (3, 4) else pyneat.TensorLayout.Unknown))
        return result

    def run(self, tensors):
        if self.runner is None:
            self.runner = self.model.build(tensors)
        output = self.runner.run(tensors, timeout_ms=120000)
        if not output:
            raise RuntimeError("model produced no output")
        return [value.to_numpy(copy=True) for value in output]

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None


def prompt_assets(archive, resources=None):
    from tokenizers import Tokenizer

    # Compile staging nests each component; the installed package puts MPKs at its root.
    archive = Path(archive)
    root = Path(resources) if resources is not None else (
        archive.parent.parent if archive.parent.name == "text_encoder" else archive.parent)
    assets = root / "prompt_assets"
    table = np.load(assets / "token_embedding_table.npy", allow_pickle=False)
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    if table.shape != (49408, 512) or table.dtype != np.float32:
        raise ValueError("unexpected CLIP lookup table")
    tokenizer.no_truncation()
    tokenizer.enable_padding(length=77, direction="right", pad_id=49407, pad_token="<|endoftext|>")
    return tokenizer, table


def prompt_input(prompt, tokenizer, table):
    (encoded,) = tokenizer.encode_batch([prompt])
    if len(encoded.ids) != 77:
        raise ValueError("prompt exceeds CLIP's 77-token context")
    ids = np.asarray(encoded.ids, np.int64)
    end = int(ids.argmax())
    if ids[end] != 49407:
        raise ValueError("unexpected CLIP end-of-text token")
    selector = np.zeros((1, 1, 96, 1), np.float32)
    selector[0, 0, end, 0] = 1
    return {"token_embeddings": table[ids][None, None], "eos_selector": selector}


def encode_classes(archive, prompts, resources=None):
    if not 1 <= len(prompts) <= 80 or any(not prompt.strip() for prompt in prompts):
        raise ValueError("provide 1..80 nonempty prompts")
    tokenizer, table = prompt_assets(archive, resources)
    encoder = Component(archive)
    embeddings, mask = np.zeros((1, 80, 512), np.float32), np.zeros((1, 80), np.float32)
    try:
        for i, prompt in enumerate(prompts):
            (result,) = encoder.run(encoder.tensors(prompt_input(prompt, tokenizer, table)))
            vector = np.asarray(result, np.float32).reshape(512)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or abs(norm - 1) > .02:
                raise ValueError(f"invalid CLIP result for {prompt!r}")
            embeddings[0, i], mask[0, i] = vector / norm, 1
    finally:
        encoder.close()
    return embeddings, mask


def benchmark(mpks, dataset, output, resources=None):
    import pyneat

    from calibration import preprocess_image

    annotations = json.loads((dataset / "annotations/instances_val2017.json").read_text())
    categories = sorted(annotations["categories"], key=lambda item: item["id"])
    if len(categories) != 80:
        raise ValueError("expected COCO's 80 classes")
    record = random.Random(0).choice(sorted(annotations["images"], key=lambda item: item["id"]))
    image = preprocess_image((dataset / "val2017" / record["file_name"]).read_bytes())
    embeddings, mask = encode_classes(mpks["text_encoder"], [item["name"] for item in categories], resources)
    tokenizer, table = prompt_assets(mpks["text_encoder"], resources)
    reports = {}
    for name in COMPONENTS:
        component = Component(mpks[name], decode=name == "detector")
        try:
            inputs = (prompt_input("person", tokenizer, table) if name == "text_encoder" else
                      {"images": image[None], "text_embeddings": embeddings, "prompt_mask": mask})
            settings = pyneat.BenchmarkOptions()
            settings.inputs = component.tensors(inputs)
            settings.num_samples = 1000
            settings.min_duration_ms = 60000
            settings.include_mla_layer_stats = True
            settings.include_mla_stage_stats = True
            settings.original_width = settings.original_height = 640
            settings.resize_mode = pyneat.ResizeMode.Letterbox
            workload = ("One prepared prompt, 77-token context; BF16 transformer, EOS pooling, projection and L2 normalization; "
                        "FP32-to-BF16 input cast included; CPU tokenization and lookup excluded" if name == "text_encoder" else
                        f"COCO val2017 {record['file_name']}, prepared 640x640 RGB /255, 80 cached BF16 CLIP vectors, all slots active; "
                        "input quantization, detector, YOLOv8 GroupedByRoleProbability BoxDecode, top 10 above .25, NMS .7; "
                        "CPU image preparation and prompt encoding excluded")
            reports[name] = model_report.from_performance(component.model.benchmark(settings), {name: component.model},
                                                          workload=workload)
        finally:
            component.close()
        model_report.from_component_performance(reports).save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, help="package root containing prompt_assets; defaults to the MPK location")
    args = parser.parse_args()
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        parser.error("expected coco-2017-val")
    benchmark(parse_mpks(args.mpk), Path(directory), args.output, args.resources)


if __name__ == "__main__":
    main()
