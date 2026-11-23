"""Score the FP32 source model on FLEURS English test with accuracy.py's pipeline and scorer, on a host CPU.

Run it from the repository root, on a host with the surgery and accuracy requirements installed:

    PYTHONPATH=tools python models/nemotron_3_5_asr_streaming_0_6b/scripts/fp32_baseline_accuracy.py \\
        --checkpoint <pinned source directory> --graphs <export directory> \\
        --dataset fleurs-en-us-test=<extracted dataset> --output fp32-baseline.json

surgery.py exports the FP32 encoder, predictor and joiner ONNX graphs from the pinned checkpoint (skipped when
--graphs already holds them). benchmark_runtime.Transcriber then runs unchanged except for where the tensors go:
the encoder runs as FP32 ONNX on ONNX Runtime instead of the BF16 MLA archive, and the predictor and joiner run
unquantized instead of INT8. Windowing, the log-mel frontend, greedy decoding, the recording order and the
model_report.speech scorer are accuracy.py's own, so the difference is the BF16 encoder and INT8 CPU models.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import accuracy
import numpy as np
import onnx
import onnxruntime as ort
from benchmark_runtime import Transcriber, read_audio
from model_report.speech import char_error_rate, word_error_rate
from tokenizers import Tokenizer


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class FP32Transcriber(Transcriber):
    """benchmark_runtime.Transcriber with the FP32 ONNX encoder, predictor and joiner on the host CPU."""

    def __init__(self, graphs: Path, threads: int):
        graph = onnx.load(graphs / "model.onnx", load_external_data=False)
        self.config = json.loads(next(item.value for item in graph.metadata_props if item.key == "sima.asr"))
        assets = self.config.pop("assets")
        self.processor = assets["processor_config.json"]
        self.tokenizer = Tokenizer.from_str(json.dumps(assets["tokenizer.json"]))
        frontend = {
            key: np.frombuffer(base64.b64decode(value["data"]), dtype="<f4").reshape(value["shape"])
            for key, value in self.config.pop("frontend").items()
        }
        self.filters, self.window = frontend["mel_filters"], frontend["window"]
        self.encoder = session(graphs / "model.onnx", threads)
        self.predictor, self.joiner = (session(graphs / f"{name}.onnx", 1) for name in ("predictor", "joiner"))
        self.ports = {value.name: list(value.shape) for value in self.encoder.get_inputs()}
        self.output_names = [value.name for value in self.encoder.get_outputs()]
        # Fresh streaming state for models whose encoder carries caches between calls; Parakeet has none.
        self.zero_states = {name: np.zeros(shape, np.float32) for name, shape in self.ports.items()}
        self.run = None

    def close(self):
        pass

    def tensor(self, value):
        return np.ascontiguousarray(value, dtype=np.float32)

    def encode(self, inputs):
        feeds = {}
        for name, shape in self.ports.items():
            value = np.asarray(inputs[name], dtype=np.float32)
            if value.size != int(np.prod(shape)):
                raise ValueError(f"{name}: {value.shape} does not fit the FP32 encoder input {shape}")
            # The runtime lays tensors out for the accelerator; only unit axes move, so element order is kept.
            moved = [dim for dim in value.shape if dim != 1] != [dim for dim in shape if dim != 1]
            if moved:
                raise ValueError(f"{name}: {value.shape} and {shape} order their non-unit axes differently")
            feeds[name] = value.reshape(shape)
        outputs = dict(zip(self.output_names, self.encoder.run(None, feeds), strict=True))
        encoded = outputs["encoded"]
        encoded = encoded.reshape(1, -1, encoded.shape[-1])
        if not np.isfinite(encoded).all():
            raise RuntimeError("Encoder returned nonfinite features")
        return encoded, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="pinned source directory with model.safetensors")
    parser.add_argument("--graphs", type=Path, required=True, help="FP32 export directory, created when missing")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=extracted-dataset")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for metrics and transcripts")
    parser.add_argument("--threads", type=int, default=8, help="encoder ONNX Runtime threads")
    parser.add_argument("--limit", type=int, help="score only the first recordings, for a quick local check")
    args = parser.parse_args()
    name, separator, path = args.dataset.partition("=")
    if name != accuracy.DATASET or not separator:
        parser.error(f"--dataset must be {accuracy.DATASET}=directory")
    graphs = args.graphs.resolve()
    if not (graphs / "model.onnx").is_file():
        if args.checkpoint is None:
            parser.error("--checkpoint is required to export the FP32 graphs")
        from surgery import export_checkpoint

        export_checkpoint(args.checkpoint.resolve(strict=True), graphs)
    directory = Path(path)
    rows = accuracy.recordings(directory)
    selected = rows[: args.limit] if args.limit else rows
    references, transcripts = [], []
    started = time.monotonic()
    with FP32Transcriber(graphs, args.threads) as transcriber:
        for row in selected:
            audio = read_audio(directory / "audio/test" / row[1])
            if len(audio) != int(row[5]):
                raise ValueError(f"FLEURS sample count differs from the TSV: {row[1]}")
            references.append(row[2])
            transcripts.append(accuracy.transcribe(transcriber, audio))
    result = {
        "dataset": accuracy.DATASET,
        "split": "test",
        "evaluated_samples": len(selected),
        "available_samples": len(rows),
        "language": accuracy.LANGUAGE,
        "precision": "fp32",
        "inference_seconds": round(time.monotonic() - started, 1),
        "metrics": {
            "wer": word_error_rate(references, transcripts),
            "cer": char_error_rate(references, transcripts),
        },
        "transcripts": [
            dict(file=row[1], reference=reference, transcript=text)
            for row, reference, text in zip(selected, references, transcripts, strict=True)
        ],
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({name: round(value, 4) for name, value in result["metrics"].items()}))


if __name__ == "__main__":
    main()
