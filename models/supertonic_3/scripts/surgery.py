"""Prepare the two fixed-profile Supertonic 3 ONNX components for Modalix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from runtime_inputs import MAX_SEQUENCE_LENGTH, ROPE_TABLE_KEY, build_time_table, pack_rope_input
from vector_transforms import (
    externalize,
    extract,
    lift,
    metrics,
    optimize,
    ort_session,
    source_constants,
    staticize,
    style_nchw,
)
from vocoder_surgery import (
    LATENT_LENGTH,
    VOCODER_SOURCE_SHA256,
)
from vocoder_surgery import (
    optimize as optimize_vocoder,
)
from vocoder_surgery import (
    staticize_and_simplify as staticize_vocoder,
)
from vocoder_surgery import (
    validate_equivalence as validate_vocoder_equivalence,
)

COMPONENTS = ("vector_estimator", "vocoder")
SURGERY_SEED = 1101
VECTOR_GRAPH_SHA256 = "79f6c51e73f4c8c20450fcede73fccbc183f4b22e5e20b78d64bc3aa585def5a"
VOCODER_GRAPH_SHA256 = "b6d233bb3cd6062f613fcc2e8b9192b1447408db06995b215f1da0d66f1a9684"
# Natural (latent, text) lengths and steps at which the derived graph must reproduce Supertone's own.
SOURCE_CHECK_LENGTHS = ((1, 1), (24, 18), (160, 128), (192, 192))
SOURCE_CHECK_STEPS = (0, 3, 7)
# The derived graph measured at most 1.7e-5 max abs and 1.3e-6 relative L2 from the source graph.
SOURCE_TOLERANCE = {"max_abs": 1e-4, "relative_l2": 1e-5, "min_cosine": 0.9999999}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_reference(path: Path) -> None:
    """Create deterministic inputs used only for surgery equivalence checks."""

    rng = np.random.default_rng(SURGERY_SEED)
    latent_mask = np.zeros((1, 1, LATENT_LENGTH), dtype=np.float32)
    latent_mask[..., :160] = 1.0
    text_mask = np.zeros((1, 1, MAX_SEQUENCE_LENGTH), dtype=np.float32)
    text_mask[..., :128] = 1.0
    np.savez(
        path,
        noisy_latent_padded=rng.standard_normal((1, 144, LATENT_LENGTH), dtype=np.float32),
        text_emb_padded=rng.standard_normal((1, 256, MAX_SEQUENCE_LENGTH), dtype=np.float32),
        style_ttl=rng.standard_normal((1, 50, 256), dtype=np.float32),
        latent_mask_padded=latent_mask,
        text_mask_padded=text_mask,
    )


def _padded(value: np.ndarray) -> np.ndarray:
    """A [1, C, length] array zero-padded to [1, C, 1, MAX_SEQUENCE_LENGTH]."""
    padded = np.zeros((1, value.shape[1], 1, MAX_SEQUENCE_LENGTH), dtype=np.float32)
    padded[..., : value.shape[2]] = value[:, :, None, :]
    return padded


def validate_against_source(source: Path, derived: Path, runtime_data: Path, steps: int) -> dict[str, Any]:
    """Compare a full guided step of Supertone's graph at its natural length with two derived-graph calls.

    The source graph takes unpadded inputs and returns the guided update. The derived graph runs at the
    compiled length, once with the text and voice and once with the unconditional runtime data, and the
    host applies x + (4 c - 3 u) / steps. This covers every surgery stage, including the guidance split and
    the mask-aware edge fill at padded positions.
    """

    reference = ort_session(source)
    candidate = ort_session(derived)
    constants = source_constants(source)
    with np.load(runtime_data) as archive:
        rope = archive[ROPE_TABLE_KEY]
    times = build_time_table(steps)
    rng = np.random.default_rng(SURGERY_SEED)
    style = rng.standard_normal((1, 50, 256), dtype=np.float32)
    comparisons = []
    for latent_length, text_length in SOURCE_CHECK_LENGTHS:
        latent = rng.standard_normal((1, 144, latent_length), dtype=np.float32)
        text = rng.standard_normal((1, 256, text_length), dtype=np.float32)
        latent_mask = _padded(np.ones((1, 1, latent_length), np.float32))
        text_mask = _padded(np.ones((1, 1, text_length), np.float32)).transpose(0, 3, 2, 1)
        unconditional_text = np.broadcast_to(constants["unconditional_text"], (1, 256, MAX_SEQUENCE_LENGTH))
        branches = (
            (_padded(text), style, constants["conditional_style_key"]),
            (
                np.ascontiguousarray(unconditional_text[:, :, None, :], dtype=np.float32),
                constants["unconditional_style_value"],
                constants["unconditional_style_key"],
            ),
        )
        for step in SOURCE_CHECK_STEPS:
            expected = reference.run(None, {
                "noisy_latent": latent,
                "text_emb": text,
                "style_ttl": style,
                "latent_mask": np.ones((1, 1, latent_length), np.float32),
                "text_mask": np.ones((1, 1, text_length), np.float32),
                "current_step": np.asarray([step], np.float32),
                "total_step": np.asarray([steps], np.float32),
            })[0]
            conditional, unconditional = (
                candidate.run(None, {
                    "noisy_latent": _padded(latent),
                    "text_emb": text_emb,
                    "style_ttl": style_nchw(style_value),
                    "style_key": style_nchw(style_key),
                    "latent_mask": latent_mask,
                    "text_mask": text_mask,
                    "time_sinusoidal": times[step],
                    "rope_tables": pack_rope_input(rope, latent_mask, text_mask),
                })[0][:, :, 0, :latent_length]
                for text_emb, style_value, style_key in branches
            )
            actual = latent + (4.0 * conditional - 3.0 * unconditional) / steps
            comparisons.append({
                "latent_length": latent_length, "text_length": text_length, "step": step, **metrics(expected, actual)
            })
    worst = {
        "max_abs": max(item["max_abs"] for item in comparisons),
        "relative_l2": max(item["relative_l2"] for item in comparisons),
        "min_cosine": min(item["cosine"] for item in comparisons),
    }
    if (
        worst["max_abs"] > SOURCE_TOLERANCE["max_abs"]
        or worst["relative_l2"] > SOURCE_TOLERANCE["relative_l2"]
        or worst["min_cosine"] < SOURCE_TOLERANCE["min_cosine"]
    ):
        raise RuntimeError(f"derived vector estimator departs from the source: {worst}, allowed {SOURCE_TOLERANCE}")
    return {"comparisons": comparisons, "worst": worst, "tolerance": SOURCE_TOLERANCE}


def _prepare_vector(source: Path, output_dir: Path, steps: int) -> dict[str, Any]:
    intermediate = output_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    reference = intermediate / "deterministic_reference.npz"
    _vector_reference(reference)

    static = intermediate / "vector_estimator.static.onnx"
    rank3 = intermediate / "vector_field.rank3.onnx"
    all4d = intermediate / "vector_field.all4d.onnx"
    optimized = intermediate / "vector_field.all4d.opt.onnx"
    final = output_dir / "supertonic_vector_estimator.onnx"
    runtime_data = output_dir / "supertonic_runtime_data.npz"

    stages: dict[str, Any] = {}
    stages["staticize"] = staticize(source, static)
    stages["extract"] = extract(static, rank3)
    stages["lift"] = lift(rank3, all4d)
    stages["optimize"] = optimize(all4d, optimized)
    stages["externalize"] = externalize(
        optimized,
        final,
        runtime_data,
        source,
        [reference],
        steps,
    )
    stages["source_equivalence"] = validate_against_source(source, final, runtime_data, steps)
    graph_hash = sha256(final)
    runtime_data_hash = sha256(runtime_data)
    if graph_hash != VECTOR_GRAPH_SHA256:
        raise RuntimeError(f"vector graph hash {graph_hash} != {VECTOR_GRAPH_SHA256}")
    report = {
        "component": "vector_estimator",
        "source_sha256": sha256(source),
        "graph_sha256": graph_hash,
        "runtime_data_sha256": runtime_data_hash,
        "stages": stages,
    }
    return {
        "model": final,
        "runtime_data": runtime_data,
        "report": report,
    }


def _prepare_vocoder(source: Path, output_dir: Path) -> dict[str, Any]:
    actual_hash = sha256(source)
    if actual_hash != VOCODER_SOURCE_SHA256:
        raise ValueError(f"unexpected vocoder source hash: {actual_hash}")

    final = output_dir / "supertonic_vocoder.onnx"
    model = staticize_vocoder(source)
    report = optimize_vocoder(model, final)
    graph_hash = sha256(final)
    if graph_hash != VOCODER_GRAPH_SHA256:
        raise RuntimeError(f"vocoder graph hash {graph_hash} != {VOCODER_GRAPH_SHA256}")

    reference = output_dir / "deterministic_vocoder_reference.npz"
    latent = np.random.default_rng(SURGERY_SEED).standard_normal((1, 144, 160), dtype=np.float32)
    np.savez(reference, final_latent_natural=latent)
    report["validation"] = validate_vocoder_equivalence(source, final, [reference])
    report["source"] = {"path": str(source), "sha256": actual_hash}
    report["graph_sha256"] = graph_hash
    return {"model": final, "report": report}


def prepare_model(
    source: str | Path,
    output_dir: str | Path,
    component: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Apply the declared component surgery.

    Returns the prepared ONNX as ``model`` and a report; the vector estimator also returns ``runtime_data``, the
    tables the application reads at run time.
    """

    source_path = Path(source).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if component not in COMPONENTS:
        raise ValueError(f"unsupported Supertonic component: {component}")
    if parameters.get("component") != component:
        raise ValueError(f"surgery component parameter must be {component!r}")
    latent_length = int(parameters.get("latent_length", 0))
    if latent_length != LATENT_LENGTH:
        raise ValueError(f"unsupported latent length: {latent_length}")
    destination.mkdir(parents=True, exist_ok=False)
    if component == "vector_estimator":
        text_length = int(parameters.get("text_length", 0))
        steps = int(parameters.get("steps", 0))
        if text_length != MAX_SEQUENCE_LENGTH:
            raise ValueError(f"unsupported text length: {text_length}")
        if steps < 1:
            raise ValueError("surgery steps must be positive")
        return _prepare_vector(source_path, destination, steps)
    return _prepare_vocoder(source_path, destination)
