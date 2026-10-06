#!/usr/bin/env python3
"""Select and validate the qualified TorchVision SSDLite precision source."""

from __future__ import annotations

import sys
from pathlib import Path

QAT_DIR = Path(__file__).resolve().parent / "qat"
sys.path.insert(0, str(QAT_DIR))
try:
    from contract import CONVOLUTION_COUNT, INPUT_NAME, INPUT_SHAPE, OUTPUTS
    from source_bundle import prepare_source
finally:
    sys.path.remove(str(QAT_DIR))

__all__ = ["CONVOLUTION_COUNT", "INPUT_NAME", "INPUT_SHAPE", "OUTPUTS"]


def prepare_variant_model(source: Path, destination: Path, precision: str) -> Path:
    """Extract one immutable ONNX member without changing its numerical graph."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"SSDLite source bundle does not exist: {source}")
    if source == destination:
        raise ValueError("SSDLite source bundle and prepared ONNX destination must differ")
    return prepare_source(source, destination, precision)


def prepare_model(source: Path, destination: Path) -> Path:
    """Prepare the FP32 source consumed by both BF16 variants."""

    return prepare_variant_model(source, destination, "bf16")


def prepare_int8_model(source: Path, destination: Path) -> Path:
    """Prepare the learned QDQ source consumed by both INT8 variants."""

    return prepare_variant_model(source, destination, "int8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--precision", choices=("int8", "bf16"), default="bf16")
    args = parser.parse_args()
    prepare_variant_model(args.source, args.destination, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
