"""Parakeet's Python runtime: log-mel frontend, BF16 windowed encoder calls and INT8 CPU TDT decoding.

performance.py and accuracy.py both run the model through this module. It is not a compile input and not
shipped in the model download; it is part of both benchmark fingerprints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def log_mel(audio, filters, window, *, center=True, normalize=False):
    from scipy.fft import rfft

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 2:
        raise ValueError("Audio window must contain at least two samples")
    emphasized = np.concatenate((audio[:1], audio[1:] - np.float32(0.97) * audio[:-1]))
    if center:
        emphasized = np.pad(emphasized, (256, 256))
    if len(emphasized) < 512:
        raise ValueError("Audio window is shorter than the Fourier transform")
    frames = np.lib.stride_tricks.sliding_window_view(emphasized, 512)[::160]
    padded_window = np.pad(window, (56, 56))
    spectrum = rfft(frames * padded_window, axis=-1)
    magnitudes = np.sqrt(spectrum.real**2 + spectrum.imag**2) ** 2
    features = np.log(magnitudes @ filters.T + np.float32(2**-24))
    valid = len(audio) // 160 if center else len(features)
    features = features[:valid]
    if normalize:
        if valid < 2:
            raise ValueError("Normalized audio needs at least two feature frames")
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, ddof=1, keepdims=True)
        features = (features - mean) / (std + np.float32(1e-5))
    return features.astype(np.float32)[None]


def bf16_bits(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class Decoder:
    """Greedy TDT decoding with prediction caching and duration carry between chunks."""

    def __init__(self, predictor, joiner, config):
        self.predictor, self.joiner, self.config = predictor, joiner, config
        self.tokens = []
        self.hidden = np.zeros((2, 1, 640), np.float32)
        self.cell = self.hidden.copy()
        self.skip = 0
        self.predict(config["blank_token_id"])

    def predict(self, token):
        self.prediction, self.hidden, self.cell = self.predictor.run(
            None, dict(token=np.array([[token]], np.int64), hidden=self.hidden, cell=self.cell)
        )

    def step(self, encoded):
        frame = self.skip
        while frame < encoded.shape[1]:
            advance = 1
            for _ in range(self.config["max_symbols_per_step"]):
                values = self.joiner.run(
                    None, dict(encoder_frame=encoded[:, frame : frame + 1], predictor_output=self.prediction)
                )
                token = int(values[0].argmax())
                duration = int(values[1].argmax())
                if token != self.config["blank_token_id"]:
                    self.tokens.append(token)
                    self.predict(token)
                if duration or token == self.config["blank_token_id"]:
                    advance = max(1, duration)
                    break
            frame += advance
        self.skip = max(0, frame - encoded.shape[1])


class Transcriber:
    """Own one encoder Run. Call transcribe() for each independent utterance."""

    def __init__(self, mpk, package):
        import onnxruntime as ort
        import pyneat as pn
        from tokenizers import Tokenizer

        self.pn = pn
        package = Path(package)
        self.config = json.loads((package / "runtime_config.json").read_text())
        with np.load(package / "frontend.npz") as data:
            self.filters, self.window = data["mel_filters"], data["window"]
        self.tokenizer = Tokenizer.from_file(str(package / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.predictor, self.joiner = [
            ort.InferenceSession(str(package / (name + "_cpu_int8.onnx")), options, providers=["CPUExecutionProvider"])
            for name in ("predictor", "joiner")
        ]
        options = pn.ModelOptions()
        terminal = pn.InferenceTerminalPolicy()
        terminal.mla_only = True
        options.inference_terminal = terminal
        self.model = pn.Model(Path(mpk), options)
        self.graph = pn.Graph("speech_encoder")
        ingress = pn.InputOptions()
        ingress.payload_type = pn.PayloadType.Tensor
        ingress.format = pn.Format.BF16
        self.graph.add(pn.nodes.input("features", ingress))
        self.graph.add(self.model.inference())
        self.graph.add(pn.nodes.output("encoded"))
        self.run = None

    def close(self):
        if self.run is not None:
            self.run.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def tensor(self, value):
        tensor = self.pn.Tensor.from_numpy(bf16_bits(value), copy=True, memory=self.pn.TensorMemory.EV74)
        tensor.dtype = self.pn.TensorDType.BFloat16
        return tensor

    def encode(self, inputs):
        ordered = [inputs[port["name"]] for port in self.config["inputs"]]
        if self.run is None:
            options = self.pn.RunOptions()
            options.output_memory = self.pn.OutputMemory.ZeroCopy
            options.startup_preflight = False
            self.run = self.graph.build(ordered, options)
        tensors = self.run.run(ordered, 60000)
        if len(tensors) != len(self.config["outputs"]):
            raise RuntimeError(f"Encoder returned {len(tensors)} tensors; expected {len(self.config['outputs'])}")
        outputs = dict(zip((port["name"] for port in self.config["outputs"]), tensors, strict=True))
        holder = self.pn.Sample()
        holder.tensor = outputs["encoded"]
        words = holder.tensor
        words.dtype = self.pn.TensorDType.UInt16
        encoded = (words.to_numpy().astype(np.uint32) << 16).view(np.float32).copy()
        if not np.isfinite(encoded).all():
            raise RuntimeError("Encoder returned nonfinite features")
        return encoded, outputs

    def transcribe(self, audio, language="auto"):
        """Yield cumulative text after each 2 s central window with 2 s lookahead."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono PCM samples")
        decoder = Decoder(self.predictor, self.joiner, self.config)
        if language != "auto":
            raise ValueError("Parakeet has no language prompt")
        for target in range(0, len(audio), 32000):
            start, end = max(0, target - 160000), min(len(audio), target + 64000)
            raw = audio[start:end]
            if len(raw) < 320:
                raw = np.pad(raw, (0, 320 - len(raw)))
            features = log_mel(raw, self.filters, self.window, normalize=True)
            valid = min(1400, features.shape[1])
            fixed = np.pad(features[:, :valid], ((0, 0), (0, 1400 - valid), (0, 0)))[..., None]
            masks = np.zeros((1, 1232, 1, 1), np.float32)
            for stride, offset, size in zip((2, 4, 8), (0, 704, 1056), (700, 350, 175), strict=True):
                masks[:, offset : offset + min((valid + stride - 1) // stride, size)] = 1
            bias = np.zeros((1, 1, 176, 1), np.float32)
            bias[:, :, (valid + 7) // 8 :] = -10000
            encoded, _ = self.encode(
                dict(
                    features=self.tensor(fixed), padding_masks=self.tensor(masks), attention_bias=self.tensor(bias)
                )
            )
            first = (target - start) // 1280
            count = min(25, (len(audio) - target + 1279) // 1280)
            decoder.step(encoded[:, first : first + count])
            yield self.tokenizer.decode(decoder.tokens, skip_special_tokens=True)


def read_audio(path):
    from scipy.io import wavfile

    rate, audio = wavfile.read(path)
    if rate != SAMPLE_RATE or audio.ndim != 1 or audio.dtype not in (np.dtype('int16'), np.dtype('float32')):
        raise ValueError("Expected a 16 kHz mono PCM16 or float32 WAV")
    audio = audio.astype(np.float32) / 32768 if audio.dtype == np.int16 else audio
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite audio")
    return audio


def runtime_directory(mpk, resources=None):
    """The folder holding runtime_config.json and the CPU models.

    The accuracy runner passes it as --resources. The performance runner passes the model's pinned resources there
    instead, and the compiled runtime files sit beside the encoder archive.
    """
    roots = [Path(resources)] if resources else []
    roots.append(Path(mpk).parent)
    for root in roots:
        for candidate in (root, root / "model"):
            if (candidate / "runtime_config.json").is_file():
                return candidate
    raise FileNotFoundError(f"runtime_config.json is in none of {', '.join(str(root) for root in roots)}")
