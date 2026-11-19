"""Check that conversion retains the complete upstream CTC forward pass."""

import numpy as np
from model_registry.fetch import Fetcher
from model_registry.model import load_model


def test_complete_export_matches_checkpoint(converted, model_dir):
    import onnx
    import onnxruntime as ort
    import torch
    from transformers import Wav2Vec2ForCTC

    onnx.checker.check_model(str(converted))
    source, *_ = Fetcher().fetch_source(load_model(model_dir)["conversion"]["default"]["source"])
    torch.set_num_threads(4)
    model = Wav2Vec2ForCTC.from_pretrained(source.parent, local_files_only=True).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(converted), options, providers=["CPUExecutionProvider"])
    for seed in (10, 11):
        audio = np.random.default_rng(seed).normal(size=(1, 80000)).astype(np.float32)
        with torch.no_grad():
            reference = model(torch.from_numpy(audio)).logits.numpy()
        actual = session.run(None, {"input_values": audio})[0]
        np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=1e-3)
        assert np.array_equal(actual.argmax(-1), reference.argmax(-1))


def test_graph_split_preserves_ctc_logits(converted, tmp_path, load_script):
    import onnxruntime as ort

    frontend, encoder = load_script("surgery").split(converted, tmp_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sessions = [
        ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
        for p in (converted, frontend, encoder)
    ]
    audio = np.random.default_rng(12).normal(size=(1, 80000)).astype(np.float32)
    original = sessions[0].run(None, {"input_values": audio})[0]
    boundary = sessions[1].run(None, {"input_values": audio})[0]
    reconstructed = sessions[2].run(None, {sessions[2].get_inputs()[0].name: boundary})[0]
    np.testing.assert_allclose(reconstructed, original, atol=3e-3, rtol=1e-3)
    assert np.array_equal(original.argmax(-1), reconstructed.argmax(-1))
