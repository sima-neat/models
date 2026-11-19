"""Shared end-to-end Wav2Vec2 inference for registry benchmarks."""

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 80000


def recordings(directory):
    rows = [line.split("\t") for line in (directory / "test.tsv").read_text(encoding="utf-8").splitlines()]
    if not rows or any(len(row) != 7 or Path(row[1]).name != row[1] or not row[1].endswith(".wav") for row in rows):
        raise ValueError("Expected the original FLEURS test.tsv")
    rows.sort(key=lambda row: row[1])
    if len({row[1] for row in rows}) != len(rows):
        raise ValueError("FLEURS filenames must be unique")
    return rows


def read_audio(path):
    from scipy.io import wavfile

    rate, audio = wavfile.read(path)
    if rate != SAMPLE_RATE or audio.ndim != 1 or audio.dtype not in (np.int16, np.float32):
        raise ValueError(f"Expected mono 16 kHz PCM16 or float32 WAV: {path}")
    result = audio.astype(np.float32) / (32768 if audio.dtype == np.int16 else 1)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return result


def runtime_directory(mpk, resources=None):
    roots = ([Path(resources)] if resources else []) + [Path(mpk).parent]
    for root in roots:
        for candidate in (root, root / "model"):
            if (candidate / "frontend.onnx").is_file() and (candidate / "vocab.json").is_file():
                return candidate
    raise FileNotFoundError("Could not find both frontend.onnx and vocab.json in the package")


class Transcriber:
    def __init__(self, mpk, directory, threads=8):
        import onnxruntime as ort
        import pyneat

        self.pn = pyneat
        self.vocab = json.loads((directory / "vocab.json").read_text())
        self.symbols = {value: key for key, value in self.vocab.items()}
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.frontend = ort.InferenceSession(
            str(directory / "frontend.onnx"), options, providers=["CPUExecutionProvider"]
        )
        opts = pyneat.ModelOptions()
        opts.name_suffix = "_wav2vec2_bench"
        opts.preprocess.kind = pyneat.InputKind.Tensor
        opts.preprocess.enable = pyneat.AutoFlag.Off
        opts.preprocess.input_max_width = 768
        self.model = pyneat.Model(str(mpk), opts)
        self.run = None

    def transcribe(self, audio):
        if not len(audio):
            return ""
        tokens = []
        previous = None
        for start in range(0, len(audio), 64000):
            chunk = audio[start : start + WINDOW]
            actual_samples = len(chunk)
            chunk = np.pad(chunk, (0, WINDOW - actual_samples))
            chunk = ((chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7))[None, :]
            boundary = np.ascontiguousarray(self.frontend.run(None, {"input_values": chunk})[0])
            tensor = self.pn.Tensor.from_numpy(boundary, copy=False, layout=self.pn.TensorLayout.HWC)
            if self.run is None:
                self.run = self.model.build([tensor])
            outputs = self.run.run([tensor], timeout_ms=120000)
            logits = np.asarray(outputs[0].to_numpy(copy=True)).reshape(249, 32)
            # Wav2Vec2 convolutional output length: floor((n - 400) / 320) + 1.
            frames = max(0, min(249, (actual_samples - 400) // 320 + 1))
            # Discard half a second of context at internal edges. Adjacent
            # windows retain contiguous 20 ms output frames without duplicates.
            first = 25 if start else 0
            last = min(frames, 225) if start + WINDOW < len(audio) else frames
            for value in logits[first:last].argmax(axis=-1):
                value = int(value)
                if value != self.vocab["<pad>"] and value != previous:
                    tokens.append(self.symbols[value])
                previous = value
            if start + WINDOW >= len(audio):
                break
        return "".join(tokens).replace("|", " ").strip()

    def close(self):
        if self.run is not None:
            self.run.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
