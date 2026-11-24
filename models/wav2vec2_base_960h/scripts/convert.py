"""Export the complete pinned Wav2Vec2 CTC checkpoint to ONNX."""

import argparse
from pathlib import Path

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]


def convert(source: Path, output_dir: Path) -> Path:
    import torch
    from transformers import Wav2Vec2ForCTC

    torch.set_num_threads(4)
    model = Wav2Vec2ForCTC.from_pretrained(source.parent, local_files_only=True).eval()

    class Outputs(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, audio):
            return self.wrapped(audio).logits

    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.onnx.export(
        Outputs(model),
        (torch.zeros((1, 80000), dtype=torch.float32),),
        str(output),
        input_names=["input_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR)["conversion"]
    if args.conversion not in conversions:
        parser.error(f"Unknown conversion {args.conversion!r}")
    source, *_ = Fetcher().fetch_source(conversions[args.conversion]["source"])
    convert(source, args.output_dir.resolve())


if __name__ == "__main__":
    main()
