"""Preserve the public output ABI when reloading a qualified ModelSDK model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def restore_public_output_labels(model: Any, output_names: Sequence[str]) -> dict[str, Any]:
    """Restore labels omitted by ModelSDK 2.1.3 serialization, then verify them.

    ``Model.save`` records the labelled output tuple in the serialized graph, but
    ``Model.load`` in ModelSDK 2.1.3 does not restore ``AwesomeNet.output_labels``.
    Compilation then exposes internal ``dequantize_*`` names in the MPK.  There is
    no public setter in that SDK release, so keep the compatibility shim isolated
    here and fail closed if a future SDK presents different state.
    """
    expected = list(output_names)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError(f"public output labels must be non-empty and unique: {expected}")

    net = getattr(model, "_net", None)
    if net is None:
        raise RuntimeError("ModelSDK model has no AwesomeNet graph")
    current = getattr(net, "output_labels", None)
    if current is not None and list(current) != expected:
        raise RuntimeError(f"ModelSDK public output labels changed: expected {expected}, found {current}")

    restored = current is None
    if restored:
        if not hasattr(net, "_output_labels"):
            raise RuntimeError("ModelSDK graph has no output-label storage")
        net._output_labels = expected

    verified = getattr(net, "output_labels", None)
    if verified is None or list(verified) != expected:
        raise RuntimeError(f"failed to restore ModelSDK public output labels: {verified}")
    return {"names": expected, "restored_after_load": restored}
