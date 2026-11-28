"""Measure a compiled LLM/VLM with MoLE, on the datasets this model's model.yaml selects."""

from pathlib import Path

from model_report.generation_benchmark import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
