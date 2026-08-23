#!/usr/bin/env python3
"""Bake the selected EdgeTAM P1 QAT values into deploy-source Q/DQ ONNX.

The prepared P1 graphs duplicate several shared PyTorch modules and split the
decoder attention projections into physical heads.  Mapping by exported names
alone is therefore brittle.  This tool reconstructs the selected model,
matches every learned ONNX callsite to its base weight numerically, applies
the selected per-output W8 codes, and inserts the selected asymmetric A8 grids
at the exact affine input/output boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qparams(quantizer) -> tuple[float, int]:
    scale, zero_point = quantizer.qparams()
    return float(scale.detach()), int(zero_point.detach().round())


def fake_weight(module) -> np.ndarray:
    weight = module.base.weight.detach().cpu().numpy().astype(np.float32)
    scales = module.wq.log_scale.detach().exp().clamp_min(1e-8).cpu().numpy()
    shape = [1] * weight.ndim
    shape[module.wq.axis] = scales.size
    view = scales.reshape(shape)
    codes = np.clip(np.rint(weight / view), -127, 127).astype(np.int8)
    return codes.astype(np.float32) * view


def variants(weight: np.ndarray, target_shape: tuple[int, ...]):
    emitted = set()

    def add(tag: str, value: np.ndarray, selector: tuple):
        if value.shape == target_shape and (tag, selector) not in emitted:
            emitted.add((tag, selector))
            yield tag, value, selector

    yield from add("direct", weight, ("direct",))
    if weight.ndim == 2:
        yield from add("transpose", weight.T, ("transpose",))
        yield from add("direct_4d", weight[:, :, None, None], ("direct_4d",))
    if weight.shape[0] % target_shape[0] == 0 and weight.shape[1:] == target_shape[1:]:
        parts = weight.shape[0] // target_shape[0]
        for index in range(parts):
            value = weight[index * target_shape[0] : (index + 1) * target_shape[0]]
            yield from add(f"chunk_{index}_of_{parts}", value, ("chunk", index, parts))
    if (
        weight.ndim == 2
        and len(target_shape) == 4
        and target_shape[2:] == (1, 1)
        and weight.shape[0] % target_shape[0] == 0
        and weight.shape[1] == target_shape[1]
    ):
        parts = weight.shape[0] // target_shape[0]
        for index in range(parts):
            value = weight[index * target_shape[0] : (index + 1) * target_shape[0], :, None, None]
            yield from add(f"chunk_4d_{index}_of_{parts}", value, ("chunk_4d", index, parts))


def transform_weight(weight: np.ndarray, selector: tuple) -> np.ndarray:
    kind = selector[0]
    if kind == "direct":
        return weight
    if kind == "transpose":
        return weight.T
    if kind == "direct_4d":
        return weight[:, :, None, None]
    if kind in ("chunk", "chunk_4d"):
        _, index, parts = selector
        size = weight.shape[0] // parts
        value = weight[index * size : (index + 1) * size]
        return value if kind == "chunk" else value[:, :, None, None]
    raise ValueError(f"unknown weight transform: {selector}")


def append_qdq(
    model: onnx.ModelProto,
    tag: str,
    source: str,
    output: str,
    scale: float,
    zero_point: int,
) -> list[onnx.NodeProto]:
    scale_name = tag + "_scale"
    zero_name = tag + "_zero_point"
    quantized_name = tag + "_quantized"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(scale, np.float32), scale_name),
            numpy_helper.from_array(np.asarray(zero_point, np.int8), zero_name),
        ]
    )
    return [
        helper.make_node(
            "QuantizeLinear",
            [source, scale_name, zero_name],
            [quantized_name],
            name=tag + "_QuantizeLinear",
        ),
        helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name, zero_name],
            [output],
            name=tag + "_DequantizeLinear",
        ),
    ]


def build_one(source: Path, output: Path, modules: dict, qat_affine) -> dict:
    model = onnx.load(source, load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    consumers = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for value in node.input:
            consumers[value].append(index)

    mapped: dict[int, tuple[str, tuple, float]] = {}
    special_qkv: dict[int, tuple[str, list[str]]] = {}
    special_fuser: dict[int, str] = {}
    learned_types = {"Conv", "ConvTranspose", "MatMul", "Gemm"}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in learned_types or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        if node.input[1] in ("att__v_1972", "att__v_1979"):
            layer = 0 if node.input[1].endswith("1972") else 1
            names = [f"memory_attention.layers.{layer}.self_attn.{kind}_proj" for kind in "qkv"]
            special_qkv[index] = (f"memory_attention.layers.{layer}.self_attn", names)
            continue
        if node.name in ("node_MatMul_163", "node_MatMul_165"):
            layer = 0 if node.name.endswith("163") else 1
            special_fuser[index] = f"memory_encoder.fuser.layers.{layer}.pwconv2"
            continue
        source_weight = numpy_helper.to_array(initializers[node.input[1]]).astype(np.float32)
        hits = []
        for name, module in modules.items():
            if not isinstance(module, qat_affine):
                continue
            base = module.base.weight.detach().cpu().numpy().astype(np.float32)
            for tag, value, selector in variants(base, source_weight.shape):
                delta = float(np.max(np.abs(source_weight - value))) if source_weight.size else 0.0
                if delta < 1e-5:
                    hits.append((delta, name, selector, tag))
        unique = sorted({item[1] for item in hits})
        if len(unique) != 1:
            raise RuntimeError(f"learned callsite {node.name!r} has {len(unique)} numerical module matches: {unique}")
        best = min(item for item in hits if item[1] == unique[0])
        mapped[index] = (best[1], best[2], best[0])

    expected = sum(
        1
        for node in model.graph.node
        if node.op_type in learned_types and len(node.input) > 1 and node.input[1] in initializers
    )
    if len(mapped) + len(special_qkv) + len(special_fuser) != expected:
        raise RuntimeError("not every learned callsite was classified")

    # MatMul/Gemm exports usually hold the module bias in the immediately
    # following Add.  QAT output quantization belongs after that bias.
    output_owner: dict[int, tuple[int, str]] = {}
    for index, (name, _selector, _delta) in mapped.items():
        node = model.graph.node[index]
        owner_index = index
        owner_output = node.output[0]
        if node.op_type == "MatMul" and len(consumers[node.output[0]]) == 1:
            candidate_index = consumers[node.output[0]][0]
            candidate = model.graph.node[candidate_index]
            if candidate.op_type == "Add" and any(value in initializers for value in candidate.input):
                owner_index = candidate_index
                owner_output = candidate.output[0]
        output_owner[index] = (owner_index, owner_output)

    before: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    after: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    rows = []
    for index, (name, selector, source_delta) in mapped.items():
        node = model.graph.node[index]
        module = modules[name]
        replacement = transform_weight(fake_weight(module), selector).astype(np.float32)
        old = numpy_helper.to_array(initializers[node.input[1]])
        if replacement.shape != old.shape:
            raise RuntimeError(f"replacement shape mismatch for {node.name}: {replacement.shape} != {old.shape}")
        initializers[node.input[1]].CopyFrom(numpy_helper.from_array(replacement, node.input[1]))

        input_scale, input_zero = qparams(module.inq)
        original_input = node.input[0]
        input_dq = f"edgetam_qat_input_{index}_dequantized"
        before[index].extend(
            append_qdq(model, f"edgetam_qat_input_{index}", original_input, input_dq, input_scale, input_zero)
        )
        node.input[0] = input_dq

        owner_index, owner_output = output_owner[index]
        raw_output = f"edgetam_qat_output_{index}_raw"
        owner = model.graph.node[owner_index]
        owner.output[list(owner.output).index(owner_output)] = raw_output
        output_scale, output_zero = qparams(module.outq)
        after[owner_index].extend(
            append_qdq(
                model,
                f"edgetam_qat_output_{index}",
                raw_output,
                owner_output,
                output_scale,
                output_zero,
            )
        )
        rows.append(
            {
                "node": node.name,
                "module": name,
                "weight_transform": list(selector),
                "source_weight_max_abs_delta": source_delta,
                "input_scale": input_scale,
                "input_zero_point": input_zero,
                "output_scale": output_scale,
                "output_zero_point": output_zero,
            }
        )

    # The exported memory self-attention combines q/k/v weights into one
    # MatMul, but applies three biases after Split.  Preserve the selected W8
    # codes and all three output grids.  The three learned input grids differ
    # only slightly; the fused physical operator must use one grid, so q_proj
    # owns it and the exact deviation is retained in the receipt.
    qkv_rows = []
    for index, (prefix, names) in special_qkv.items():
        node = model.graph.node[index]
        pieces = [fake_weight(modules[name]).T for name in names]
        replacement = np.concatenate(pieces, axis=1).astype(np.float32)
        initializers[node.input[1]].CopyFrom(numpy_helper.from_array(replacement, node.input[1]))
        input_values = [qparams(modules[name].inq) for name in names]
        input_scale, input_zero = input_values[0]
        original_input = node.input[0]
        input_dq = f"edgetam_qat_fused_qkv_input_{index}_dequantized"
        before[index].extend(
            append_qdq(
                model,
                f"edgetam_qat_fused_qkv_input_{index}",
                original_input,
                input_dq,
                input_scale,
                input_zero,
            )
        )
        node.input[0] = input_dq
        split_index = consumers[node.output[0]][0]
        split = model.graph.node[split_index]
        if split.op_type != "Split" or len(split.output) != 3:
            raise RuntimeError(f"unexpected fused qkv consumer for {prefix}")
        branches = []
        for branch, (name, split_output) in enumerate(zip(names, split.output)):
            add_index = consumers[split_output][0]
            add = model.graph.node[add_index]
            if add.op_type != "Add":
                raise RuntimeError(f"qkv branch {name} is not followed by bias Add")
            original_output = add.output[0]
            raw_output = f"edgetam_qat_fused_qkv_output_{index}_{branch}_raw"
            add.output[0] = raw_output
            scale, zero = qparams(modules[name].outq)
            after[add_index].extend(
                append_qdq(
                    model,
                    f"edgetam_qat_fused_qkv_output_{index}_{branch}",
                    raw_output,
                    original_output,
                    scale,
                    zero,
                )
            )
            branches.append({"module": name, "scale": scale, "zero_point": zero})
        qkv_rows.append(
            {
                "node": node.name,
                "prefix": prefix,
                "shared_input": {"scale": input_scale, "zero_point": input_zero},
                "source_input_qparams": input_values,
                "maximum_input_scale_delta": max(abs(value[0] - input_scale) for value in input_values),
                "branches": branches,
            }
        )

    # The prepared memory encoder folds LayerScale into pwconv2.  Undo that
    # fold, quantize at the original module output, then explicitly reapply the
    # recovered per-channel LayerScale vector.
    fuser_rows = []
    for index, name in special_fuser.items():
        node = model.graph.node[index]
        module = modules[name]
        source_weight = numpy_helper.to_array(initializers[node.input[1]]).astype(np.float32)
        base_transposed = module.base.weight.detach().cpu().numpy().astype(np.float32).T
        ratio = (source_weight * base_transposed).sum(axis=0) / np.maximum(
            (base_transposed * base_transposed).sum(axis=0), 1e-30
        )
        fit_error = float(np.max(np.abs(source_weight - base_transposed * ratio)))
        if fit_error >= 2e-6:
            raise RuntimeError(f"cannot recover folded LayerScale for {name}: {fit_error}")
        replacement = fake_weight(module).T.astype(np.float32)
        initializers[node.input[1]].CopyFrom(numpy_helper.from_array(replacement, node.input[1]))
        input_scale, input_zero = qparams(module.inq)
        original_input = node.input[0]
        input_dq = f"edgetam_qat_fuser_input_{index}_dequantized"
        before[index].extend(
            append_qdq(model, f"edgetam_qat_fuser_input_{index}", original_input, input_dq, input_scale, input_zero)
        )
        node.input[0] = input_dq
        add_index = consumers[node.output[0]][0]
        add = model.graph.node[add_index]
        if add.op_type != "Add":
            raise RuntimeError(f"folded fuser {name} is not followed by bias Add")
        bias_name = next(value for value in add.input if value in initializers)
        bias = module.base.bias.detach().cpu().numpy().astype(np.float32)
        initializers[bias_name].CopyFrom(numpy_helper.from_array(bias, bias_name))
        original_output = add.output[0]
        prequant = f"edgetam_qat_fuser_output_{index}_prequant"
        add.output[0] = prequant
        quantized = f"edgetam_qat_fuser_output_{index}_dequantized"
        output_scale, output_zero = qparams(module.outq)
        after[add_index].extend(
            append_qdq(
                model,
                f"edgetam_qat_fuser_output_{index}",
                prequant,
                quantized,
                output_scale,
                output_zero,
            )
        )
        ratio_name = f"edgetam_qat_fuser_layerscale_{index}"
        model.graph.initializer.extend([numpy_helper.from_array(ratio.astype(np.float32), ratio_name)])
        after[add_index].append(
            helper.make_node(
                "Mul",
                [quantized, ratio_name],
                [original_output],
                name=f"edgetam_qat_fuser_layerscale_{index}_Mul",
            )
        )
        fuser_rows.append(
            {
                "node": node.name,
                "module": name,
                "layerscale_fit_max_abs_error": fit_error,
                "input_scale": input_scale,
                "input_zero_point": input_zero,
                "output_scale": output_scale,
                "output_zero_point": output_zero,
            }
        )

    ordered = []
    for index, node in enumerate(model.graph.node):
        ordered.extend(before.get(index, ()))
        ordered.append(node)
        ordered.extend(after.get(index, ()))
    model.graph.ClearField("node")
    model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    onnx.save(model, temporary)
    os.replace(temporary, output)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_size_bytes": output.stat().st_size,
        "learned_callsite_count": expected,
        "regular_callsite_count": len(rows),
        "fused_qkv_callsite_count": len(qkv_rows),
        "folded_fuser_callsite_count": len(fuser_rows),
        "regular": rows,
        "fused_qkv": qkv_rows,
        "folded_fuser": fuser_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--recurrent-source", type=Path, required=True)
    parser.add_argument("--memory-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [
        str(args.bundle / "tools"),
        str(args.bundle / "runtime/site-packages"),
        str(args.bundle / "vendor"),
    ]
    import model_qat
    from target_real_affine_qat_v2 import QAT_AFFINE

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _fold = model_qat.build_float_training_model("cpu")
    coverage = model_qat.convert_target_real(model)
    model_qat.materialize_qparams_from_state(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    modules = dict(model.named_modules())
    entries = []
    for name, source in (
        ("image_prompt_p3", args.prompt_source),
        ("recurrent_s1_p1", args.recurrent_source),
        ("memory_encoder_p1", args.memory_source),
    ):
        entries.append(build_one(source, args.output_dir / f"edgetam_{name}_qat_qdq.onnx", modules, QAT_AFFINE))
    receipt = {
        "schema_version": 1,
        "status": "PASS_WITH_FUSED_QKV_SHARED_INPUT_GRID",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "coverage": coverage,
        "entries": entries,
        "evidence_level": "ONNX QDQ preview; AFE, compiler, simulator, and board evidence pending",
    }
    receipt_path = args.output_dir / "QDQ_EXPORT_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "entries": [
                    {
                        "output": item["output"],
                        "sha256": item["output_sha256"],
                        "learned_callsite_count": item["learned_callsite_count"],
                    }
                    for item in entries
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
