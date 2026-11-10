"""Source precision hints for the qualified projection/attention BF16 islands."""
from collections import defaultdict
import copy
import onnx
from onnx import helper

LAYOUT = {"Reshape", "Transpose", "Slice", "Split", "Concat", "Tile", "Identity", "Expand"}

def index(model):
    producers = {value: node for node in model.graph.node for value in node.output}
    users = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            users[value].append(node)
    for output in model.graph.output:
        users[output.name].append(None)
    shapes = {value.name: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
              for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    return producers, users, shapes


def ancestors(model, outputs, stops=()):
    producers, _, _ = index(model)
    pending, found = list(outputs), set()
    while pending:
        value = pending.pop()
        if value in stops or value not in producers:
            continue
        node = producers[value]
        if node.name in found:
            continue
        found.add(node.name)
        pending.extend(node.input)
    return found


def regions(model):
    names = {node.name: node for node in model.graph.node}
    projections = {f"/cls_preds.{i}/cls_preds.{i}.2/Conv" for i in range(3)}
    if not projections <= names.keys():
        raise ValueError("expected the three YOLO-World class projection convolutions")
    embeddings = {value for name in projections for value in names[name].output}
    matching = ancestors(model, [value.name for value in model.graph.output][3:], embeddings)
    attention = {node.name for node in model.graph.node
                 if "/attn_block/" in node.name and "/project_conv/" not in node.name}
    output_names = {output.name for output in model.graph.output}
    outputs = {node.name for node in model.graph.node if output_names.intersection(node.output)}
    return {"projection_attention": projections | matching | attention | outputs}


def annotate(model, selected, bf16_text=True):
    model = copy.deepcopy(model)
    _, users, _ = index(model)

    def readers(value):
        result = []
        for user in users[value]:
            if user is not None and user.op_type in LAYOUT and user.name not in selected:
                for output in user.output:
                    result.extend(readers(output))
            else:
                result.append(user)
        return result

    marks = dict.fromkeys(selected, "bfloat16")
    for node in model.graph.node:
        if node.name in selected:
            for output in node.output:
                for reader in readers(output):
                    if reader is not None and reader.name not in selected:
                        marks[reader.name] = "int8"
    if bf16_text:
        for reader in readers("text_embeddings"):
            if reader is not None and reader.name not in selected:
                marks[reader.name] = "int8"
    rewritten = []
    if bf16_text:
        for node in model.graph.node:
            for i, value in enumerate(node.input):
                if value == "text_embeddings":
                    node.input[i] = "text_embeddings/bf16"
        rewritten.append(helper.make_node("AnnotatePrecision", ["text_embeddings"], ["text_embeddings/bf16"],
            name="text_embeddings/AnnotatePrecision", domain="ai.sima", precision="bfloat16"))
    for node in model.graph.node:
        rewritten.append(node)
        if node.name in marks:
            for i, value in enumerate(list(node.output)):
                node.output[i] = value + "/unannotated"
                rewritten.append(helper.make_node("AnnotatePrecision", [node.output[i]], [value],
                    name=node.name + f"/AnnotatePrecision_{i}", domain="ai.sima", precision=marks[node.name]))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.opset_import.append(helper.make_opsetid("ai.sima", 1))
    onnx.checker.check_model(model)
    return model, marks

