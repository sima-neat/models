"""Prepare the two fixed-profile Supertonic 3 ONNX components for Modalix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from runtime_inputs import MAX_SEQUENCE_LENGTH
from vector_transforms import externalize, extract, lift, optimize, staticize
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
        "generated": {runtime_data.name: runtime_data},
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
    return {"model": final, "generated": {}, "report": report}


def prepare_model(
    source: str | Path,
    output_dir: str | Path,
    component: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Apply the declared component surgery.

    Returns the prepared ONNX as ``model``, a report, and ``generated``: the files the MPK needs at run time, by
    the name model.yaml declares under the component's ``generated_files``.
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
