from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_arguments_name_exactly_both_encoders(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")

    mpks = script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz", "text_encoder=/data/text_mpk.tar.gz"])
    assert list(mpks) == ["image_encoder", "text_encoder"]
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and text_encoder=<path>"):
        script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz"])


def test_the_runner_can_pass_the_package_resources(tmp_path, monkeypatch, load_script) -> None:
    script = load_script("performance")
    benchmarked = []
    monkeypatch.setattr(script, "benchmark_encoders", lambda *arguments: benchmarked.append(arguments))
    monkeypatch.setattr(
        "sys.argv",
        [
            "performance.py",
            "--mpk=image_encoder=image.tar.gz",
            "--mpk=text_encoder=text.tar.gz",
            f"--dataset=imagenet-2012-val={tmp_path}",
            f"--output={tmp_path / 'out'}",
            f"--resources={tmp_path / 'resources'}",
        ],
    )

    script.main()

    mpks = {"image_encoder": Path("image.tar.gz"), "text_encoder": Path("text.tar.gz")}
    assert benchmarked == [(mpks, "imagenet-2012-val", tmp_path, tmp_path / "out")]


def test_the_cpu_looks_the_probe_prompt_up_for_the_text_encoder(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("performance")
    prompt = script.load_prompt()

    embeds = script.token_embeddings(prompt)
    assert embeds.shape == (1, 64, 768)
    for position, token in enumerate(prompt.token_ids):
        np.testing.assert_array_equal(embeds[0, position], prompt.rows[token])


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["image_encoder", "text_encoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_image_encoder_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    edges = performance["components"]["image_encoder"]["execution_plan"]["edges"]
    assert any(tensor.get("shape") == [1, 720, 1280, 3] for edge in edges for tensor in edge.get("tensors", []))


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


def test_each_encoder_has_an_independent_core_benchmark(tmp_path, monkeypatch, load_script):
    script = load_script("performance")
    image, text = object(), object()
    monkeypatch.setattr(script, "sample_image", lambda dataset: Path("sample.JPEG"))
    monkeypatch.setattr(script, "frame_tensor", lambda path: image)
    monkeypatch.setattr(script, "load_prompt", lambda: "prompt")
    monkeypatch.setattr(script, "look_up", lambda prompt: text)
    monkeypatch.setattr(script.pyneat, "BenchmarkOptions", SimpleNamespace)
    monkeypatch.setattr(
        script.pyneat,
        "ModelOptions",
        lambda: SimpleNamespace(
            preprocess=SimpleNamespace(
                resize=SimpleNamespace(), normalize=SimpleNamespace(), color_convert=SimpleNamespace()
            )
        ),
    )
    measured, combined, saved = [], [], []

    class Model:
        def __init__(self, mpk, options):
            self.mpk, self.options = mpk, options

        def benchmark(self, options):
            measured.append((self.mpk, self.options, options))
            return SimpleNamespace(fps=100.0 * len(measured), latency_ms=10.0 / len(measured))

    monkeypatch.setattr(script.pyneat, "Model", Model)
    monkeypatch.setattr(
        script.model_report, "from_performance", lambda benchmark, models, **kw: (benchmark, models, kw)
    )
    monkeypatch.setattr(
        script.model_report,
        "from_component_performance",
        lambda reports: combined.append(reports) or SimpleNamespace(save=saved.append),
    )
    script.benchmark_encoders(
        {"image_encoder": Path("image.tar.gz"), "text_encoder": Path("text.tar.gz")},
        "imagenet-2012-val",
        tmp_path,
        tmp_path / "out",
    )
    assert [item[0] for item in measured] == ["image.tar.gz", "text.tar.gz"]
    assert [item[2].inputs for item in measured] == [[image], [text]]
    assert measured[0][1].preprocess.kind == script.pyneat.InputKind.Image
    assert measured[1][1].preprocess.kind == script.pyneat.InputKind.Tensor
    assert all(options.include_mla_layer_stats and options.include_mla_stage_stats for _, _, options in measured)
    assert list(combined[0]) == ["image_encoder", "text_encoder"]
    assert combined[0]["image_encoder"][0].fps != combined[0]["text_encoder"][0].fps
    assert saved == [tmp_path / "out"]
