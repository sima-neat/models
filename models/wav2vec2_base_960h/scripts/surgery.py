"""Split the complete export at the CPU frontend/MLA encoder boundary."""

from pathlib import Path

BOUNDARY = "/wrapped/wav2vec2/encoder/layer_norm/LayerNormalization_output_0"


def split(source: Path, output: Path) -> tuple[Path, Path]:
    import onnx
    from onnx import TensorProto, helper
    from onnx.utils import Extractor

    graph = onnx.load(source)
    if [item.name for item in graph.graph.input] != ["input_values"] or [item.name for item in graph.graph.output] != [
        "logits"
    ]:
        raise ValueError("Expected the complete Wav2Vec2 CTC export")
    if BOUNDARY not in {name for node in graph.graph.node for name in node.output}:
        raise ValueError("Expected encoder boundary is missing")
    graph.graph.value_info.append(helper.make_tensor_value_info(BOUNDARY, TensorProto.FLOAT, [1, 249, 768]))
    graph = onnx.shape_inference.infer_shapes(graph)
    extractor = Extractor(graph)
    output.mkdir(parents=True, exist_ok=True)
    paths = output / "frontend.onnx", output / "encoder.onnx"
    for path, inputs, outputs in ((paths[0], ["input_values"], [BOUNDARY]), (paths[1], [BOUNDARY], ["logits"])):
        part = extractor.extract_model(inputs, outputs)
        onnx.checker.check_model(part)
        onnx.save(part, path)
    return paths
