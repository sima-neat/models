#!/usr/bin/env python3
"""Compile EfficientSAM ViT-T's image encoder and prompt decoder from the author's ONNX files, and measure fidelity.

The encoder is folded at a fixed 1x3x1024x1024 input, so its own resize to 1024x1024 is a no-op and is removed. The
decoder becomes a static graph: the prompt encoding (sin and cos of the points, plus an embedding per point type)
runs in the graph on one points tensor of normalized coordinates and one-hot point types; the prompt's shape logic
folds away; and the final resize to the image size moves to the CPU. Both graphs are checked against the original
files before they compile, and the encoder compiles without AFE's simulator check of its MLA code, which does not fit
in the compile runner's memory at 4096 tokens. Numerical fidelity runs the quantized encoder into the quantized decoder
on coco-2017 cal images with the box of each image's largest annotated object, against the original float32 graphs.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import MAX_POINTS, POINT_COLUMNS, POINT_TYPES, load_samples, prompt_inputs, select_samples
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from onnx import numpy_helper, shape_inference
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "efficientsam_vit_t"
COMPONENT_IDS = ("image_encoder", "prompt_decoder")
SOURCE_FILES = {"image_encoder": "efficientsam_ti_encoder.onnx", "prompt_decoder": "efficientsam_ti_decoder.onnx"}
COMPONENT_GRAPHS = {
    "image_encoder": {
        "inputs": {"batched_images": (1, 3, 1024, 1024)},
        "outputs": {"image_embeddings": (1, 256, 64, 64)},
    },
    "prompt_decoder": {
        "inputs": {"image_embeddings": (1, 256, 64, 64), "points": (1, MAX_POINTS, POINT_COLUMNS)},
        "outputs": {"iou_predictions": (1, 1, 3), "low_res_masks": (1, 3, 256, 256)},
    },
}
TOKENS = "/mask_decoder/Concat_8_output_0"
OUTPUT_TOKENS = 5
# A box on a 1024x1024 image: the decoder's prompt-shape logic folds at this prompt, and its values do not matter.
FOLDING_PROMPT = {
    "batched_point_coords": np.asarray([[[[180.0, 140.0], [620.0, 760.0]]]], np.float32),
    "batched_point_labels": np.asarray([[[2.0, 3.0]]], np.float32),
    "orig_im_size": np.asarray([1024, 1024], np.int64),
}
# Prompts the decoder check runs on both graphs: (points, point types, (height, width)).
CHECK_PROMPTS = [
    ([(180.0, 140.0), (620.0, 760.0)], ["box_top_left", "box_bottom_right"], (1024, 1024)),
    ([(40.0, 30.0), (600.0, 420.0)], ["box_top_left", "box_bottom_right"], (480, 640)),
    ([(320.0, 240.0)], ["positive"], (480, 640)),
    ([(100.0, 700.0), (500.0, 200.0)], ["positive", "negative"], (900, 600)),
    ([(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0), (90.0, 99.0), (5.0, 9.0)],
     ["positive", "negative", "positive", "positive", "negative", "positive"], (100, 120)),
]
# Files the compilation phases pass to each other in the temporary directory.
REFERENCE_OUTPUTS = "reference_outputs.npz"
CALIBRATION_EMBEDDINGS = "calibration_embeddings.npz"
QUANTIZED_EMBEDDINGS = "quantized_embeddings.npz"
FIDELITY_REPORT = "fidelity.json"


def _float32_bytes(shape: tuple[int, ...]) -> int:
    return math.prod(shape) * 4


def _component_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    """Return each component's build parameters in the build profile."""
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}


def _precision_label(component_parameters: dict[str, dict[str, Any]]) -> str:
    precisions = {parameters["precision"].upper() for parameters in component_parameters.values()}
    if len(precisions) != 1:
        raise ValueError("both components of a build profile use one precision")
    return precisions.pop()


def _component_name(profile_id: str, component_id: str) -> str:
    """Name a component's quantized model and its build directory."""
    return f"{MODEL_ID}_{profile_id}_{component_id}"


def _session(model: onnx.ModelProto | Path) -> ort.InferenceSession:
    payload = model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model)
    return ort.InferenceSession(payload, providers=["CPUExecutionProvider"])


# Graph preparation.


def _known_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Every tensor whose shape shape inference fully knows."""
    inferred = shape_inference.infer_shapes(model).graph
    shapes = {}
    for value in [*inferred.value_info, *inferred.input, *inferred.output]:
        dims = value.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = [dim.dim_value for dim in dims]
    for item in inferred.initializer:
        shapes[item.name] = list(item.dims)
    return shapes


def _fold(model: onnx.ModelProto, shapes: dict[str, list[int]] | None = None) -> onnx.ModelProto:
    """Fold the graph at fixed shapes. Shape nodes of tensors with a known shape become constants first, so the
    result does not depend on how much the installed onnxsim folds on its own."""
    from onnxsim import simplify  # only the Model Compiler's environment has onnxsim

    folded, valid = simplify(
        model, overwrite_input_shapes=shapes, dynamic_input_shape=False, perform_optimization=False
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded graph")
    for _ in range(8):
        known = _known_shapes(folded)
        replaced = 0
        for node in list(folded.graph.node):
            if node.op_type == "Shape" and node.input[0] in known:
                attributes = {
                    attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
                }
                value = known[node.input[0]][attributes.get("start", 0):attributes.get("end", None)]
                constant = onnx.helper.make_node("Constant", [], list(node.output),
                                                 value=numpy_helper.from_array(np.asarray(value, np.int64)))
                folded.graph.node.insert(list(folded.graph.node).index(node), constant)
                folded.graph.node.remove(node)
                replaced += 1
        if not replaced:
            break
        folded, valid = simplify(folded, dynamic_input_shape=False, perform_optimization=False)
        if not valid:
            raise RuntimeError("onnxsim could not validate the folded graph")
    return _drop_unused_constants(folded)


def _drop_unused_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop the initializers that no node reads; folding keeps the constants of the nodes it folded away."""

    def read(graph: onnx.GraphProto) -> set[str]:
        names = {value.name for value in graph.output}
        for node in graph.node:
            names.update(node.input)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    names |= read(attribute.g)
        return names

    used = read(model.graph)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def _remove_leftovers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Inline If nodes with a constant condition, drop Tiles that repeat by ones and Resizes to their input's own shape,
    and drop nodes no output depends on; folding leaves these for the MLA to reject."""
    graph = model.graph
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    inferred = shape_inference.infer_shapes(model).graph
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*inferred.value_info, *inferred.input]
    }
    renames: dict[str, str] = {}
    for node in list(graph.node):
        if node.op_type == "If" and node.input[0] in constants:
            taken = "then_branch" if bool(constants[node.input[0]].reshape(-1)[0]) else "else_branch"
            branch = next(attribute.g for attribute in node.attribute if attribute.name == taken)
            if len(branch.node) != 1 or branch.node[0].op_type != "Identity":
                raise RuntimeError(f"{node.name}: the {taken} is not a single Identity")
            source = branch.node[0].input[0]
            inner = {item.name: item for item in branch.initializer}
            if source in inner:
                value = numpy_helper.to_array(inner[source])
                graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
                constants[node.output[0]] = value
            else:
                renames[node.output[0]] = source
            graph.node.remove(node)
        elif node.op_type == "Tile" and node.input[1] in constants and np.all(constants[node.input[1]] == 1):
            renames[node.output[0]] = node.input[0]
            graph.node.remove(node)
        elif node.op_type == "Resize" and len(node.input) == 4 and node.input[3] in constants:
            if constants[node.input[3]].tolist() == shapes.get(node.input[0]):
                renames[node.output[0]] = node.input[0]
                graph.node.remove(node)
    for node in graph.node:
        for index, name in enumerate(node.input):
            while name in renames:
                name = renames[name]
            node.input[index] = name
    needed = {output.name for output in graph.output}
    for node in reversed(list(graph.node)):
        if any(name in needed for name in node.output):
            needed.update(name for name in node.input if name)
    kept = [node for node in graph.node if any(name in needed for name in node.output)]
    del graph.node[:]
    graph.node.extend(kept)
    return model


def _group_norms_as_pools(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite the upscaling's normalization so the MLA can run it.

    EfficientSAM normalizes all of a feature map's values together (a group norm with one group), exported as
    Reshape to one row, InstanceNormalization, and Reshape back. AFE turns that into one layer norm over a million
    values, which the MLA cannot take. Every channel has the same size, so per-channel GlobalAveragePools averaged over
    the channels give the same mean and variance.
    """
    graph = model.graph
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    for norm in [node for node in graph.node if node.op_type == "InstanceNormalization"]:
        flatten = producers[norm.input[0]]
        (restore,) = [node for node in graph.node if norm.output[0] in node.input]
        scale, bias = constants.get(norm.input[1]), constants.get(norm.input[2])
        if flatten.op_type != "Reshape" or restore.op_type != "Reshape" or scale is None or bias is None or \
                not np.all(scale == 1) or not np.all(bias == 0):
            raise RuntimeError(f"{norm.name} is not the one-group normalization this recipe rewrites")
        epsilon = next(onnx.helper.get_attribute_value(item) for item in norm.attribute if item.name == "epsilon")
        source, target, prefix = flatten.input[0], restore.output[0], norm.name.replace("/", "_")
        graph.initializer.append(numpy_helper.from_array(np.asarray(epsilon, np.float32), f"{prefix}_epsilon"))
        nodes = [
            onnx.helper.make_node("GlobalAveragePool", [source], [f"{prefix}_channel_means"]),
            onnx.helper.make_node("ReduceMean", [f"{prefix}_channel_means"], [f"{prefix}_mean"], axes=[1], keepdims=1),
            onnx.helper.make_node("Sub", [source, f"{prefix}_mean"], [f"{prefix}_centred"]),
            onnx.helper.make_node("Mul", [f"{prefix}_centred", f"{prefix}_centred"], [f"{prefix}_squares"]),
            onnx.helper.make_node("GlobalAveragePool", [f"{prefix}_squares"], [f"{prefix}_channel_variances"]),
            onnx.helper.make_node(
                "ReduceMean", [f"{prefix}_channel_variances"], [f"{prefix}_variance"], axes=[1], keepdims=1
            ),
            onnx.helper.make_node("Add", [f"{prefix}_variance", f"{prefix}_epsilon"], [f"{prefix}_shifted"]),
            onnx.helper.make_node("Sqrt", [f"{prefix}_shifted"], [f"{prefix}_deviation"]),
            onnx.helper.make_node("Div", [f"{prefix}_centred", f"{prefix}_deviation"], [target]),
        ]
        position = list(graph.node).index(flatten)
        for node in (flatten, norm, restore):
            graph.node.remove(node)
        for offset, node in enumerate(nodes):
            graph.node.insert(position + offset, node)
    return model


def _rename(model: onnx.ModelProto, mapping: dict[str, str]) -> onnx.ModelProto:
    graph = model.graph
    for value in [*graph.input, *graph.output, *graph.value_info]:
        value.name = mapping.get(value.name, value.name)
    for node in graph.node:
        node.input[:] = [mapping.get(name, name) for name in node.input]
        node.output[:] = [mapping.get(name, name) for name in node.output]
    return model


def encoder_graph(source: Path) -> onnx.ModelProto:
    """The encoder at a fixed 1x3x1024x1024 input, checked against the original file."""
    shapes = {name: list(shape) for name, shape in COMPONENT_GRAPHS["image_encoder"]["inputs"].items()}
    model = _fold(_remove_leftovers(_fold(onnx.load(source), shapes)))
    image = np.random.default_rng(0).uniform(0, 1, shapes["batched_images"]).astype(np.float32)
    expected = _session(source).run(None, {"batched_images": image})[0]
    actual = _session(model).run(None, {"batched_images": image})[0]
    if actual.shape != expected.shape or np.abs(actual - expected).max() > 1e-3:
        raise RuntimeError(f"the folded encoder differs from the original: {np.abs(actual - expected).max()}")
    return model


def positional_encoding(coords: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    """EfficientSAM's point encoding of normalized coordinates: sin and cos of 2*pi*((2c - 1) @ G)."""
    projected = 2 * np.pi * ((2 * coords - 1) @ gaussian)
    return np.concatenate([np.sin(projected), np.cos(projected)], -1)


def _original_decoder(source: Path, extra_outputs: tuple[str, ...] = ()) -> tuple[ort.InferenceSession, str]:
    model = onnx.load(source)
    low_res = next(node for node in model.graph.node if node.name == "/Resize").input[0]
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in (low_res, *extra_outputs))
    return _session(model), low_res


def _original_decode(session: ort.InferenceSession, embeddings: np.ndarray, points: list[tuple[float, float]],
                     types: list[str], size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The original decoder's outputs for a prompt; it takes labels -1 to 3 for padding, negative, positive and the
    two box corners, and pads the prompt itself."""
    feeds = {
        "image_embeddings": embeddings,
        "batched_point_coords": np.asarray([[points]], np.float32),
        "batched_point_labels": np.asarray([[[POINT_TYPES.index(kind) - 1.0 for kind in types]]], np.float32),
        "orig_im_size": np.asarray(size, np.int64),
    }
    return dict(zip([output.name for output in session.get_outputs()], session.run(None, feeds)))


def prompt_constants(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The point-encoding matrix, the decoder's output tokens, and each point type's embedding.

    The export folds the type embeddings into constants, so they are read back from the tokens it builds: for each
    point type, a token row minus the point encoding of its point.
    """
    gaussian = next(numpy_helper.to_array(item) for item in onnx.load(source).graph.initializer
                    if item.name.endswith("positional_encoding_gaussian_matrix"))
    session, _ = _original_decoder(source, (TOKENS,))
    embeddings = np.zeros(COMPONENT_GRAPHS["prompt_decoder"]["inputs"]["image_embeddings"], np.float32)
    point, size = (300.0, 200.0), (1024, 1024)
    table: dict[int, np.ndarray] = {}
    output_tokens = None
    for index, kind in enumerate(POINT_TYPES[1:], start=1):
        tokens = _original_decode(session, embeddings, [point], [kind], size)[TOKENS][0]
        encoding = positional_encoding(prompt_inputs([point], [kind], size)["points"][0, :, :2], gaussian)
        table[index] = tokens[OUTPUT_TOKENS] - encoding[0]
        table[0] = tokens[OUTPUT_TOKENS + 1] - encoding[1]  # the first padding row
        output_tokens = tokens[:OUTPUT_TOKENS]
    return gaussian, output_tokens, np.stack([table[index] for index in range(len(POINT_TYPES))])


def decoder_graph(source: Path) -> onnx.ModelProto:
    """The prompt decoder as a static graph on (image_embeddings, points), checked against the original file on
    several prompts."""
    gaussian, output_tokens, table = prompt_constants(source)
    model = onnx.load(source)
    graph = model.graph
    low_res = next(node for node in graph.node if node.name == "/Resize").input[0]
    # The tokens become an input and the prompt inputs constants, so only the mask decoder remains after folding.
    graph.node.remove(next(node for node in graph.node if TOKENS in node.output))
    tokens_shape = [1, OUTPUT_TOKENS + MAX_POINTS, output_tokens.shape[1]]
    graph.input.append(onnx.helper.make_tensor_value_info(TOKENS, onnx.TensorProto.FLOAT, tokens_shape))
    kept = [value for value in graph.input if value.name not in FOLDING_PROMPT]
    del graph.input[:]
    graph.input.extend(kept)
    graph.initializer.extend(numpy_helper.from_array(value, name) for name, value in FOLDING_PROMPT.items())
    outputs = [value for value in graph.output if value.name == "iou_predictions"]
    masks = onnx.helper.make_tensor_value_info(low_res, onnx.TensorProto.FLOAT,
                                               list(COMPONENT_GRAPHS["prompt_decoder"]["outputs"]["low_res_masks"]))
    del graph.output[:]
    graph.output.extend([*outputs, masks])
    shapes = {
        "image_embeddings": list(COMPONENT_GRAPHS["prompt_decoder"]["inputs"]["image_embeddings"]),
        TOKENS: tokens_shape,
    }
    model = _fold(_remove_leftovers(_fold(_remove_leftovers(_fold(model, shapes)))))
    model = _rename(_group_norms_as_pools(model), {TOKENS: "tokens", low_res: "low_res_masks"})

    # The prompt encoding, in front of the mask decoder. Constant MatMuls read each points row's coordinates and its
    # type's embedding; the other columns meet zero rows.
    graph = model.graph
    prompt_nodes = [
        onnx.helper.make_node("MatMul", ["points", "prompt_coordinate_columns"], ["prompt_coords"]),
        onnx.helper.make_node("Mul", ["prompt_coords", "prompt_two"], ["prompt_doubled"]),
        onnx.helper.make_node("Sub", ["prompt_doubled", "prompt_one"], ["prompt_centred"]),
        onnx.helper.make_node("MatMul", ["prompt_centred", "prompt_gaussian"], ["prompt_projected"]),
        onnx.helper.make_node("Mul", ["prompt_projected", "prompt_two_pi"], ["prompt_angles"]),
        onnx.helper.make_node("Sin", ["prompt_angles"], ["prompt_sin"]),
        onnx.helper.make_node("Cos", ["prompt_angles"], ["prompt_cos"]),
        onnx.helper.make_node("Concat", ["prompt_sin", "prompt_cos"], ["prompt_encoding"], axis=-1),
        onnx.helper.make_node("MatMul", ["points", "prompt_type_embeddings"], ["prompt_type_rows"]),
        onnx.helper.make_node("Add", ["prompt_encoding", "prompt_type_rows"], ["prompt_rows"]),
        onnx.helper.make_node("Concat", ["prompt_output_tokens", "prompt_rows"], ["tokens"], axis=1),
    ]
    coordinate_columns = np.zeros((POINT_COLUMNS, 2), np.float32)
    coordinate_columns[0, 0] = coordinate_columns[1, 1] = 1.0
    type_embeddings = np.zeros((POINT_COLUMNS, table.shape[1]), np.float32)
    type_embeddings[2 : 2 + len(POINT_TYPES)] = table
    graph.initializer.extend([
        numpy_helper.from_array(coordinate_columns, "prompt_coordinate_columns"),
        numpy_helper.from_array(np.asarray(2.0, np.float32), "prompt_two"),
        numpy_helper.from_array(np.asarray(1.0, np.float32), "prompt_one"),
        numpy_helper.from_array(np.asarray(2 * np.pi, np.float32), "prompt_two_pi"),
        numpy_helper.from_array(gaussian.astype(np.float32), "prompt_gaussian"),
        numpy_helper.from_array(type_embeddings, "prompt_type_embeddings"),
        numpy_helper.from_array(output_tokens[None].astype(np.float32), "prompt_output_tokens"),
    ])
    inputs = [value for value in graph.input if value.name != "tokens"]
    inputs.append(onnx.helper.make_tensor_value_info("points", onnx.TensorProto.FLOAT,
                                                     list(COMPONENT_GRAPHS["prompt_decoder"]["inputs"]["points"])))
    del graph.input[:]
    graph.input.extend(inputs)
    nodes = [*prompt_nodes, *graph.node]
    del graph.node[:]
    graph.node.extend(nodes)
    model = _fold(model)
    onnx.checker.check_model(model)
    _check_decoder(source, model)
    return model


def _check_decoder(source: Path, model: onnx.ModelProto) -> None:
    """The static decoder gives the original's masks and scores on every check prompt."""
    original, low_res = _original_decoder(source)
    static = _session(model)
    names = [output.name for output in static.get_outputs()]
    embeddings = np.random.default_rng(1).normal(0, 1, COMPONENT_GRAPHS["prompt_decoder"]["inputs"]["image_embeddings"])
    embeddings = embeddings.astype(np.float32)
    for points, types, size in CHECK_PROMPTS:
        expected = _original_decode(original, embeddings, points, types, size)
        actual = dict(
            zip(names, static.run(None, {"image_embeddings": embeddings, **prompt_inputs(points, types, size)}))
        )
        for name, reference in (("low_res_masks", expected[low_res]), ("iou_predictions", expected["iou_predictions"])):
            # Relative to the output's range: the rewritten normalization sums in a different order than the original.
            difference = float(np.abs(actual[name] - reference).max()) / max(1.0, float(np.abs(reference).max()))
            if actual[name].shape != reference.shape or difference > 1e-3:
                raise RuntimeError(
                    f"the static decoder's {name} differs from the original for {types}: {difference:.2e}"
                )
            logging.info("decoder check %s %s: max difference %.2e of the output's range", types, name, difference)


def graph_paths(directory: Path) -> dict[str, Path]:
    return {component_id: directory / f"{component_id}.onnx" for component_id in COMPONENT_IDS}


def _prepare_phase(source_dir: Path, temporary_dir: Path, calibration_dataset: Path,
                   calibration: list[tuple[str, list[float]]], fidelity_dataset: Path,
                   fidelity: list[tuple[str, list[float]]]) -> None:
    """Build and check both graphs, and save the float32 references the later phases need."""
    paths = graph_paths(temporary_dir / "graphs")
    paths["image_encoder"].parent.mkdir(parents=True)
    sources = {component_id: source_dir / name for component_id, name in SOURCE_FILES.items()}
    onnx.save(encoder_graph(sources["image_encoder"]), paths["image_encoder"])
    onnx.save(decoder_graph(sources["prompt_decoder"]), paths["prompt_decoder"])
    encoder = _session(sources["image_encoder"])
    decoder = _session(paths["prompt_decoder"])

    def embed(image: np.ndarray) -> np.ndarray:
        return encoder.run(None, {"batched_images": image[None]})[0]

    calibration_samples = load_samples(calibration_dataset, calibration)
    np.savez(
        temporary_dir / CALIBRATION_EMBEDDINGS,
        embeddings=np.concatenate([embed(image) for image, _ in calibration_samples]),
    )
    references: dict[str, list[np.ndarray]] = {"image_embeddings": [], "iou_predictions": [], "low_res_masks": []}
    names = [output.name for output in decoder.get_outputs()]
    for image, prompt in load_samples(fidelity_dataset, fidelity):
        embeddings = embed(image)
        outputs = dict(zip(names, decoder.run(None, {"image_embeddings": embeddings, **prompt})))
        references["image_embeddings"].append(embeddings)
        for name in ("iou_predictions", "low_res_masks"):
            references[name].append(outputs[name])
    np.savez(temporary_dir / REFERENCE_OUTPUTS, **{name: np.concatenate(values) for name, values in references.items()})


# Quantization and compilation.


def _quantization_config(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        # AFE quantizes these graphs only with calibrated ranges, even at BF16.
        .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        # with_activation_quantization turns per-token quantization on for BF16; these builds do not use it.
        .with_per_token_quant(False)
    )


def _afe_layout(array: np.ndarray) -> np.ndarray:
    """AFE takes 4-D tensors in NHWC order and other tensors as they are."""
    return np.ascontiguousarray(array.transpose(0, 2, 3, 1) if array.ndim == 4 else array)


def _quantize_component(graph: Path, name: str, component_id: str, parameters: dict[str, Any],
                        calibration: dict[str, np.ndarray]) -> Any:
    shapes = COMPONENT_GRAPHS[component_id]["inputs"]
    importer = onnx_source(str(graph), shape_dict=shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    data = {InputName(input_name): _afe_layout(calibration[input_name]) for input_name in shapes}
    return loaded.quantize(
        calibration_data=convert_data_generator_to_iterable(DataGenerator(data)),
        quantization_config=_quantization_config(parameters),
        model_name=name,
        log_level=logging.INFO,
    )


def _read_mpk_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        return names, json.load(source)


def audit_mpk(archive: Path, component_id: str) -> dict[str, Any]:
    """Require one MLA plugin, no CPU plugins and the component's inputs."""
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = manifest.get("plugins", [])
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, "
            f"host={len(host_plugins)}, shared_objects={shared_objects}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    expected = {name: _float32_bytes(shape) for name, shape in COMPONENT_GRAPHS[component_id]["inputs"].items()}
    if public_inputs != expected:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {public_inputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins)}


def _skip_simulator_check() -> None:
    """Compile without AFE's check that replays the generated MLA code in the MLA simulator.

    At 1024x1024 the encoder attends over 4096 tokens, and the simulator runs out of memory on the 32 GiB compile
    runner. The check only verifies the generated code and does not change it. performance.py checks the compiled
    encoder on a DevKit instead, through the mask the compiled models segment.
    """
    from afe.backends.mla.afe_to_n2a_compiler.n2a_compiler_operations import MLACompilerConfig
    from afe.core import compile_networks

    if getattr(compile_networks, "MLACompilerConfig", None) is not MLACompilerConfig:
        raise RuntimeError("this Model Compiler configures the MLA compile differently; revisit the simulator check")
    compile_networks.MLACompilerConfig = functools.partial(MLACompilerConfig, verify_mlc_files=False)


def _find_mpk(build_dir: Path, component_id: str) -> Path:
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def _compile_component(quantized: Any, build_dir: Path, component_id: str, precision: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=False)
    quantized.compile(
        output_path=str(build_dir),
        batch_size=1,
        compress=True,
        log_level=logging.INFO,
        tessellate_parameters=None,
        l2_caching_mode=L2CachingMode.NONE,
        preserve=True,
        deployable=True,
        retained_temporary_directory_name=None,
        use_power_limits=False,
        max_mla_power=None,
        # The option applies to every layer norm, and INT8 ones reject it.
        layer_norm_use_fp32_intermediates=precision == "w16a16",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archive = _find_mpk(build_dir, component_id)
    audit_mpk(archive, component_id)
    return archive


@dataclass
class _OutputError:
    """Running sums of one output's error against the original float32 graphs, over every fidelity sample."""

    name: str
    count: int = 0
    dot: float = 0.0
    expected_squared: float = 0.0
    actual_squared: float = 0.0
    error_squared: float = 0.0
    error_absolute: float = 0.0
    max_absolute_error: float = 0.0

    def add(self, expected: np.ndarray, actual: np.ndarray) -> None:
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{self.name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{self.name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        self.count += expected.size
        self.dot += float(np.dot(expected.ravel(), actual.ravel()))
        self.expected_squared += float(np.sum(np.square(expected)))
        self.actual_squared += float(np.sum(np.square(actual)))
        self.error_squared += float(np.sum(np.square(error)))
        self.error_absolute += float(np.sum(absolute))
        self.max_absolute_error = max(self.max_absolute_error, float(np.max(absolute)))

    def summary(self) -> dict[str, Any]:
        expected_rms = max(np.sqrt(self.expected_squared / self.count), np.finfo(np.float64).eps)
        norm = np.sqrt(self.expected_squared * self.actual_squared)
        cosine = self.dot / norm if norm else float(self.expected_squared == self.actual_squared)
        return {
            "name": self.name,
            "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
            "normalized_rmse": float(np.sqrt(self.error_squared / self.count) / expected_rms),
            "mean_absolute_error": float(self.error_absolute / self.count),
            "max_absolute_error": self.max_absolute_error,
        }


def _execute(model: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    outputs = model.execute(
        inputs={InputName(name): _afe_layout(value) for name, value in inputs.items()}, use_jax=True
    )
    return [np.asarray(output) for output in outputs]


def _by_shape(outputs: list[np.ndarray], component_id: str, name: str) -> np.ndarray:
    """A quantized component's output, matched by its shape, which differs between the outputs; 4-D outputs come back
    NHWC."""
    shape = COMPONENT_GRAPHS[component_id]["outputs"][name]
    for output in outputs:
        if output.shape == shape:
            return output
        if output.ndim == 4 and output.transpose(0, 3, 1, 2).shape == shape:
            return output.transpose(0, 3, 1, 2)
    raise RuntimeError(
        f"the quantized {component_id} returned no {name} of shape {shape}: {[o.shape for o in outputs]}"
    )


def _encoder_phase(graph: Path, temporary_dir: Path, profile_id: str, parameters: dict[str, Any],
                   calibration_dataset: Path, calibration: list[tuple[str, list[float]]], fidelity_dataset: Path,
                   fidelity: list[tuple[str, list[float]]]) -> None:
    """Compile the encoder, and save its embeddings of the fidelity images for the decoder phase."""
    images = np.stack([image for image, _ in load_samples(calibration_dataset, calibration)])
    name = _component_name(profile_id, "image_encoder")
    encoder = _quantize_component(graph, name, "image_encoder", parameters, {"batched_images": images})
    _skip_simulator_check()
    _compile_component(encoder, temporary_dir / name, "image_encoder", parameters["precision"])
    embeddings = [_by_shape(_execute(encoder, {"batched_images": image[None]}), "image_encoder", "image_embeddings")
                  for image, _ in load_samples(fidelity_dataset, fidelity)]
    np.savez(temporary_dir / QUANTIZED_EMBEDDINGS, embeddings=np.concatenate(embeddings))


def _decoder_phase(graph: Path, temporary_dir: Path, profile_id: str, parameters: dict[str, Any],
                   calibration_dataset: Path, calibration: list[tuple[str, list[float]]], fidelity_dataset: Path,
                   fidelity: list[tuple[str, list[float]]]) -> None:
    """Compile the decoder on the float32 encoder's embeddings, then run it on the quantized encoder's embeddings of
    the fidelity images and compare every output with the float32 chain."""
    with np.load(temporary_dir / CALIBRATION_EMBEDDINGS) as saved:
        calibration_embeddings = saved["embeddings"]
    prompts = [prompt for _, prompt in load_samples(calibration_dataset, calibration)]
    data = {
        "image_embeddings": calibration_embeddings,
        "points": np.concatenate([prompt["points"] for prompt in prompts]),
    }
    name = _component_name(profile_id, "prompt_decoder")
    decoder = _quantize_component(graph, name, "prompt_decoder", parameters, data)
    _compile_component(decoder, temporary_dir / name, "prompt_decoder", parameters["precision"])
    with np.load(temporary_dir / QUANTIZED_EMBEDDINGS) as saved:
        quantized_embeddings = saved["embeddings"]
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = {key: saved[key] for key in saved.files}
    errors = {name: _OutputError(name) for name in ("image_embeddings", "iou_predictions", "low_res_masks")}
    for index, (_, prompt) in enumerate(load_samples(fidelity_dataset, fidelity)):
        errors["image_embeddings"].add(
            reference["image_embeddings"][index : index + 1], quantized_embeddings[index : index + 1]
        )
        outputs = _execute(decoder, {"image_embeddings": quantized_embeddings[index : index + 1], **prompt})
        for output in ("iou_predictions", "low_res_masks"):
            errors[output].add(reference[output][index : index + 1], _by_shape(outputs, "prompt_decoder", output))
    (temporary_dir / "decoder_fidelity.json").write_text(json.dumps([error.summary() for error in errors.values()]))


def _execute_phase(target: Callable[..., None], arguments: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    try:
        target(**arguments)
    except SystemExit as error:
        # AFE reports a failed quantization or compilation by raising SystemExit(), which exits with code 0.
        raise RuntimeError(f"{target.__name__} stopped in AFE; the AFE log above names the cause") from error


def _run_phase(name: str, target: Callable[..., None], **arguments: Any) -> None:
    """Run one phase in its own process, so the next starts with the memory the previous one used."""
    logging.info("Starting %s", name)
    process = multiprocessing.get_context("spawn").Process(name=name, target=_execute_phase, args=(target, arguments))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.exitcode}")
    logging.info("Completed %s", name)


# Outputs.


def _write_fidelity(temporary_dir: Path, profile_id: str, config: dict[str, Any], precision: str) -> None:
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "Original float32 ONNX encoder and decoder (ONNX Runtime)",
        "candidate": f"AFE {precision} image encoder feeding the prompt decoder (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": config["selection"],
        "samples": config["selection"]["samples"],
        "metrics": json.loads((temporary_dir / "decoder_fidelity.json").read_text()),
    }
    (temporary_dir / FIDELITY_REPORT).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_component_archives(temporary_dir: Path, profile_id: str, output_dir: Path) -> dict[str, Path]:
    """Copy each component's MPK to <output_dir>/<component>/."""
    archives = {}
    for component_id in COMPONENT_IDS:
        built = _find_mpk(temporary_dir / _component_name(profile_id, component_id), component_id)
        component_dir = output_dir / component_id
        component_dir.mkdir()
        archives[component_id] = component_dir / built.name
        shutil.copyfile(built, archives[component_id])
    return archives


def _write_compile_outputs(output_dir: Path, profile_id: str, archives: dict[str, Path],
                           graph_hashes: dict[str, str]) -> None:
    """Write compile-outputs.json, which records each MPK and derived graph."""
    manifest = {
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "compile_invocation_id": profile_id,
        "outputs": [
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archives[component_id].name}",
                "mpk_sha256": sha256_file(archives[component_id])[0],
            }
            for component_id in COMPONENT_IDS
        ],
        "derived_graphs": [
            {"component_id": component_id, "sha256": graph_hashes[component_id]} for component_id in COMPONENT_IDS
        ],
    }
    (output_dir / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")


def compile_model(source: Path, profile_id: str, calibration_dataset: Path, fidelity_dataset: Path,
                  output_dir: Path) -> dict[str, Path]:
    metadata = load_metadata(MODEL_DIR)
    component_parameters = _component_parameters(metadata, profile_id)
    precision = _precision_label(component_parameters)
    fidelity_config = metadata["numerical_fidelity"]
    selection = fidelity_config["selection"]
    seed, offset, samples = (int(selection[key]) for key in ("seed", "offset", "samples"))
    count = int(component_parameters["image_encoder"]["calibration_samples"])
    if offset < count:
        raise ValueError("the fidelity selection must begin after the calibration images")
    calibration = select_samples(calibration_dataset, seed, 0, count)
    fidelity = select_samples(fidelity_dataset, seed, offset, samples)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        graphs = graph_paths(temporary_dir / "graphs")
        datasets = {"calibration_dataset": calibration_dataset, "calibration": calibration,
                    "fidelity_dataset": fidelity_dataset, "fidelity": fidelity}
        _run_phase("EfficientSAM graph preparation", _prepare_phase, source_dir=source.parent,
                   temporary_dir=temporary_dir, **datasets)
        graph_hashes = {component_id: sha256_file(path)[0] for component_id, path in graphs.items()}
        _run_phase("EfficientSAM image encoder compilation", _encoder_phase, graph=graphs["image_encoder"],
                   temporary_dir=temporary_dir, profile_id=profile_id,
                   parameters=component_parameters["image_encoder"], **datasets)
        _run_phase("EfficientSAM prompt decoder compilation and fidelity", _decoder_phase,
                   graph=graphs["prompt_decoder"], temporary_dir=temporary_dir, profile_id=profile_id,
                   parameters=component_parameters["prompt_decoder"], **datasets)
        _write_fidelity(temporary_dir, profile_id, fidelity_config, precision)
        output_dir.mkdir(parents=True)
        archives = _copy_component_archives(temporary_dir, profile_id, output_dir)
        shutil.copyfile(temporary_dir / FIDELITY_REPORT, output_dir / FIDELITY_REPORT)
        _write_compile_outputs(output_dir, profile_id, archives, graph_hashes)
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the encoder ONNX; the decoder ONNX sits beside it")
    parser.add_argument("--dataset", type=Path, required=True, help="calibration images")
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve()
    if source.name != SOURCE_FILES["image_encoder"] or not source.is_file():
        parser.error(f"--model must be the pinned {SOURCE_FILES['image_encoder']}")
    if not (source.parent / SOURCE_FILES["prompt_decoder"]).is_file():
        parser.error(f"{SOURCE_FILES['prompt_decoder']} must sit beside the encoder")
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = compile_model(
        source, args.build_profile, dataset, fidelity_dataset, args.output_dir.expanduser().resolve()
    )
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
