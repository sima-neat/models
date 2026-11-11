"""Export fixed-input text and detector graphs and apply compiler precision hints.

Source-equivalent rewrites: fused evaluation BN; bounded, masked maximum of
sigmoids in the neck; shared text normalization; raw DFL heads and sigmoid
probabilities for Neat's GroupedByRoleProbability YOLOv8 decoder.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from precision import annotate, regions

SLOTS = 80
COMPONENTS = ("text_encoder", "detector")
DETECTOR_INPUTS = {"images": (1, 3, 640, 640), "text_embeddings": (1, SLOTS, 512), "prompt_mask": (1, SLOTS)}
TEXT_INPUTS = {"token_embeddings": (1, 512, 1, 77), "eos_selector": (1, 1, 1, 96)}
CALIBRATION_PROMPT_ROWS = 32
BF16_MAX = np.float32(3.3895313892515355e38)


def finite_clip_bounds(model):
    """BF16 compilation requires finite clamp bounds; normalized input norms cannot reach this maximum."""
    for node in model.graph.node:
        if node.op_type == "Clip" and (len(node.input) == 2 or not node.input[2]):
            name = node.name + "/finite_max"
            model.graph.initializer.append(numpy_helper.from_array(np.array(BF16_MAX), name))
            if len(node.input) == 2:
                node.input.append(name)
            else:
                node.input[2] = name
    return model


def export(module, inputs, outputs, output):
    import torch
    from onnxsim import simplify

    example = tuple(torch.from_numpy(value) for value in inputs.values())
    with torch.inference_mode():
        torch.onnx.export(module, example, str(output), input_names=list(inputs), output_names=outputs,
                          opset_version=13, dynamo=False, do_constant_folding=True)
    graph, valid = simplify(onnx.load(output))
    if not valid:
        raise RuntimeError("ONNX simplification failed")
    graph = finite_clip_bounds(graph)
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = module(*example)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    for i, (actual, reference) in enumerate(zip(session.run(None, inputs), expected, strict=True)):
        np.testing.assert_allclose(actual, reference.numpy(), atol=0.01 if i < 3 and len(outputs) == 6 else 0.0002,
                                   rtol=0.001, err_msg=outputs[i])
    return graph


def prepare_model(source: Path, directory: Path):
    import torch
    from clip import TextEncoder, token_table
    from detector import Detector, OUTPUTS

    directory.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    state, assets = bundle["state_dict"], bundle["assets"]
    rng = np.random.default_rng(20261110)
    embeddings = rng.normal(size=(1, SLOTS, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    detector = Detector(state)
    inputs = {"images": rng.random(DETECTOR_INPUTS["images"], dtype=np.float32),
              "text_embeddings": embeddings, "prompt_mask": np.ones((1, SLOTS), np.float32)}
    reference_path = directory / "detector_fp32.onnx"
    graph = export(detector, inputs, OUTPUTS, reference_path)
    selected = regions(graph)["projection_attention"]
    graph, _ = annotate(graph, selected)
    detector_path = directory / "detector.onnx"
    onnx.save(graph, detector_path)

    table = token_table(state)
    ids = np.full((1, 77), 49407, np.int64)
    ids[0, :3] = [49406, 2533, 49407]  # "person", padded to the fixed context
    selector = np.zeros(TEXT_INPUTS["eos_selector"], np.float32)
    selector[0, 0, 0, 2] = 1
    text_inputs = {"token_embeddings": table[ids].transpose(0, 2, 1)[:, :, None, :], "eos_selector": selector}
    encoder = TextEncoder(state)
    text_path = directory / "text_encoder.onnx"
    export(encoder, text_inputs, ["text_embeddings"], text_path)
    # Simulation-only graph: independent prompts occupy rows, keeping batch one.
    # The separately compiled text_encoder.onnx and its application contract stay unchanged.
    calibration_inputs = {name: np.repeat(value, CALIBRATION_PROMPT_ROWS, axis=2)
                          for name, value in text_inputs.items()}
    export(encoder, calibration_inputs, ["text_embeddings"], directory / "text_calibration.onnx")
    np.save(directory / "token_embedding_table.npy", table, allow_pickle=False)
    for name, value in assets.items():
        destination = directory / Path(name).name
        destination.write_text(value, encoding="utf-8")
    return {"text_encoder": text_path, "detector": detector_path}, encoder, table
