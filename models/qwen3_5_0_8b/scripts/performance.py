"""Measure text buckets and an encoder-sized image using the installed MoLE and LLiMa runtime."""

import argparse
from pathlib import Path

from model_registry.llima_benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_benchmark(args.deployment, args.output)
