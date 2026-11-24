"""Validate runtime assets and the graph cut without requiring device hardware."""

import json
from types import SimpleNamespace

import numpy as np
from model_registry.runtime_files import read_runtime_files


def test_compile_publishes_frontend_and_vocabulary(compiled, load_script):
    files = {entry["path"] for entry in read_runtime_files(compiled)}
    assert files == {"model/frontend.onnx", "model/vocab.json"}
    vocab = json.loads((compiled / "model/vocab.json").read_text())
    assert len(vocab) == 32 and vocab["<pad>"] == 0
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)


def test_window_stitch_deduplicates_repeated_token_at_boundary(load_script):
    runtime = load_script("benchmark_runtime")
    outputs = []
    for positions in ({224: 5, 225: 6}, {24: 6, 25: 5, 27: 5}):
        ids = np.zeros(249, dtype=np.int64)
        for index, value in positions.items():
            ids[index] = value
        logits = np.full((249, 32), -1.0, dtype=np.float32)
        logits[np.arange(249), ids] = 1.0
        outputs.append(logits)

    class Runner:
        def run(self, tensors, timeout_ms):
            return [SimpleNamespace(to_numpy=lambda copy: outputs.pop(0))]

    transcriber = runtime.Transcriber.__new__(runtime.Transcriber)
    transcriber.vocab = {"<pad>": 0}
    transcriber.symbols = {0: "<pad>", 5: "A", 6: "B"}
    transcriber.frontend = SimpleNamespace(run=lambda *args: [np.zeros((1, 249, 768), np.float32)])
    transcriber.pn = SimpleNamespace(
        Tensor=SimpleNamespace(from_numpy=lambda value, **kwargs: value),
        TensorLayout=SimpleNamespace(HWC="HWC"),
    )
    transcriber.run = Runner()
    assert transcriber.transcribe(np.zeros(96000, np.float32)) == "AA"
    assert not outputs
