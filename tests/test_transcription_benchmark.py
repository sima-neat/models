import json
import struct
import threading
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from model_report import transcription_benchmark as benchmark


def float_wav(path: Path, samples: list[float]) -> None:
    """A 16 kHz mono IEEE float32 WAV, as FLEURS stores it, with the optional fact chunk."""
    data = array("f", samples).tobytes()
    fmt = struct.pack("<HHIIHHH", 3, 1, 16000, 64000, 4, 32, 0)
    fact = struct.pack("<I", len(samples))
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<I", 4) + fact
    body += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_float_and_pcm16_wavs_read_back_as_the_same_samples(tmp_path):
    float_wav(tmp_path / "float.wav", [0.0, 0.5, -0.5, 0.25])
    (tmp_path / "pcm.wav").write_bytes(benchmark.wav_bytes(array("f", [0.0, 0.5, -0.5, 0.25])))
    assert list(benchmark.read_wav(tmp_path / "float.wav")) == [0.0, 0.5, -0.5, 0.25]
    assert list(benchmark.read_wav(tmp_path / "pcm.wav")) == pytest.approx([0.0, 0.5, -0.5, 0.25], abs=1e-4)


def test_other_wav_layouts_are_rejected(tmp_path):
    (tmp_path / "text.wav").write_bytes(b"not audio")
    with pytest.raises(ValueError, match="not a WAV"):
        benchmark.read_wav(tmp_path / "text.wav")


def fleurs(directory: Path, lengths: dict[str, int]) -> None:
    (directory / "audio/test").mkdir(parents=True)
    rows = []
    for index, (name, length) in enumerate(lengths.items()):
        float_wav(directory / "audio/test" / name, [index / 10] * length)
        rows.append(f"{index}\t{name}\ttext {index}\ttext {index}\tx\t{length}\tMALE")
    (directory / "test.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_the_first_30_seconds_split_back_into_one_upload_per_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "SECONDS", 1)
    fleurs(tmp_path, {"b.wav": 9000, "a.wav": 10000, "c.wav": 4000})
    audio, metadata = benchmark.load_dataset(tmp_path)
    # Filename order, and the last recording cut where the 16,000 samples end.
    assert [(item["id"], item["samples_used"]) for item in metadata["recordings"]] == [
        ("en_us/test/a", 10000), ("en_us/test/b", 6000),
    ]
    lengths = []
    for index, upload in enumerate(benchmark.pieces(audio, metadata)):
        (tmp_path / f"upload{index}.wav").write_bytes(upload)
        lengths.append(len(benchmark.read_wav(tmp_path / f"upload{index}.wav")))
    assert lengths == [10000, 6000]


def test_too_little_speech_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "SECONDS", 1)
    fleurs(tmp_path, {"a.wav": 8000})
    with pytest.raises(ValueError, match="less than 30 seconds"):
        benchmark.load_dataset(tmp_path)


class Server(BaseHTTPRequestHandler):
    """Answers like Neat's GenAI server: one JSON object, or server-sent events when the form says stream=true."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        assert b'name="model"\r\n\r\nasr' in body and b'name="file"; filename="speech.wav"' in body
        if b'name="stream"\r\n\r\ntrue' in body:
            events = [{"text": " "}, {"text": "Hello ", "ttft": 0.07}, {"text": "world."}, {"text": "", "tps": 120.0}]
            payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            payload, content_type = json.dumps({"text": " Hello world."}), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *_):
        pass


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_pass_joins_every_recording_and_streaming_reports_the_first_visible_text(server_url):
    upload = benchmark.wav_bytes(array("f", [0.0] * 160))
    elapsed, text = benchmark.transcribe_all(server_url, [upload, upload])
    assert text == "Hello world. Hello world." and elapsed > 0
    streamed = benchmark.first_text(server_url, upload)
    # The blank first chunk is not visible text; the server's own TTFT and TPS are kept as evidence.
    assert streamed["seconds"] > 0 and (streamed["server_ttft_s"], streamed["server_tps"]) == (0.07, 120.0)
