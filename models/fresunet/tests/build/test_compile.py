from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[2]
BAND_NAMES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")
# Each band's grid step relative to the 10 m bands.
STEPS = {"B02": 1, "B03": 1, "B04": 1, "B08": 1, "B05": 2, "B06": 2, "B07": 2, "B8A": 2, "B11": 2, "B12": 2,
         "B01": 6, "B09": 6, "B10": 6}


def write_city(root: Path, name: str, height: int, width: int, changed: tuple[slice, slice]) -> None:
    """One synthetic OSCD city: both dates' 13 bands as uint16 TIFFs, each band a constant, and its change map."""
    for date, offset in (("imgs_1", 0), ("imgs_2", 100)):
        folder = root / name / date
        folder.mkdir(parents=True)
        for index, band in enumerate(BAND_NAMES):
            step = STEPS[band]
            size = (-(-height // step), -(-width // step))
            value = 1000 + 10 * index + offset
            assert cv2.imwrite(str(folder / f"S2A_TILE_{band}.tif"), np.full(size, value, np.uint16))
    (root / name / "cm").mkdir()
    label = np.zeros((height, width), np.uint8)
    label[changed] = 255
    assert cv2.imwrite(str(root / name / "cm" / "cm.png"), label)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "OSCD"
    write_city(root, "alpha", 120, 96, (slice(0, 60), slice(0, 96)))
    write_city(root, "beta", 600, 660, (slice(300, 600), slice(0, 660)))
    write_city(root, "gamma", 60, 72, (slice(0, 0), slice(0, 0)))
    (root / "train.txt").write_text("alpha,beta\n")
    (root / "test.txt").write_text("gamma\n")
    return tmp_path


def test_cities_follow_the_split_files_order(load_script: Callable[[str], ModuleType], dataset: Path) -> None:
    oscd = load_script("oscd")
    root = oscd.dataset_root(dataset)

    assert oscd.cities(root, "train") == ["alpha", "beta"]
    assert oscd.cities(root, "test") == ["gamma"]


def test_city_input_stacks_both_dates_normalized_at_544(
    load_script: Callable[[str], ModuleType], dataset: Path
) -> None:
    oscd = load_script("oscd")
    root = oscd.dataset_root(dataset)

    for name in ("alpha", "beta"):
        sample = oscd.city_input(root, name)

        assert (sample.shape, sample.dtype) == ((544, 544, 26), np.float32)
        # Every band is constant, so away from the edges, where the cubic upsampling of the coarse bands fades, each
        # date's channels keep their band order and normalization.
        values = sample[272, 272]
        assert np.allclose(values[:13], values[13:], atol=1e-5)
        bands = [BAND_NAMES.index(band) for band, _ in oscd.BANDS]
        expected = 1000 + 10 * np.asarray(bands, np.float64)
        expected = (expected - expected.mean()) / expected.std()
        np.testing.assert_allclose(values[:13], expected, atol=1e-4)


def test_change_map_resizes_the_label_to_544(load_script: Callable[[str], ModuleType], dataset: Path) -> None:
    oscd = load_script("oscd")
    root = oscd.dataset_root(dataset)

    alpha = oscd.change_map(root, "alpha")
    beta = oscd.change_map(root, "beta")

    assert (alpha.shape, alpha.dtype) == ((544, 544), np.bool_)
    assert alpha[:272].all() and not alpha[272:].any()
    assert not beta[:272].any() and beta[272:].all()
    assert not oscd.change_map(root, "gamma").any()


def test_prediction_takes_the_argmax_in_either_layout(load_script: Callable[[str], ModuleType]) -> None:
    oscd = load_script("oscd")
    logits = np.zeros((1, 2, 544, 544), np.float32)
    logits[0, 1, :10] = 1.0

    channels_first = oscd.prediction(logits)
    channels_last = oscd.prediction(logits[0].transpose(1, 2, 0)[None])

    assert channels_first[:10].all() and not channels_first[10:].any()
    assert (channels_first == channels_last).all()
    with pytest.raises(ValueError, match="unexpected FresUNet output shape"):
        oscd.prediction(np.zeros((2, 100, 100)))


def test_change_score_sums_pixels_over_cities(load_script: Callable[[str], ModuleType]) -> None:
    oscd = load_script("oscd")
    score = oscd.ChangeScore()
    score.add(np.array([True, True, False, False]), np.array([True, False, True, False]))
    score.add(np.array([True, False]), np.array([True, False]))

    assert (score.tp, score.fp, score.fn, score.tn) == (2, 1, 1, 2)
    metrics = score.metrics()
    assert metrics["change_precision"] == pytest.approx(2 / 3)
    assert metrics["change_recall"] == pytest.approx(2 / 3)
    assert metrics["change_f1"] == pytest.approx(2 / 3)


def test_calibration_reads_the_first_train_cities_from_the_archive(
    load_script: Callable[[str], ModuleType], dataset: Path, tmp_path: Path
) -> None:
    compile_script = load_script("compile")
    archive = tmp_path / "datasets__OSCD.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(dataset / "OSCD", arcname="OSCD")

    samples = compile_script.calibration_inputs(archive, 1)

    assert (samples.shape, samples.dtype) == ((1, 544, 544, 26), np.float32)
    with pytest.raises(ValueError, match="requires 3 train cities"):
        compile_script.calibration_inputs(dataset, 3)


def test_each_declared_profile_compiles_with_its_precision(load_script: Callable[[str], ModuleType]) -> None:
    compile_script = load_script("compile")

    assert compile_script._parameters("w16a16")["precision"] == "w16a16"
    assert compile_script._parameters("w8a8")["precision"] == "w8a8"
    for profile in ("w16a16", "w8a8"):
        parameters = compile_script._parameters(profile)
        assert (parameters["calibration_samples"], parameters["calibration_bins"]) == (14, 1024)
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_script.quantization("w4a8", 1024)


def _mpk(path: Path, processors: list[str]) -> Path:
    manifest = json.dumps({"plugins": [{"processor": processor} for processor in processors]}).encode()
    with tarfile.open(path, "w:gz") as tar:
        member = tarfile.TarInfo("fresunet_mpk.json")
        member.size = len(manifest)
        tar.addfile(member, io.BytesIO(manifest))
    return path


def test_mpk_audit_requires_one_mla_segment_and_no_host_code(
    load_script: Callable[[str], ModuleType], tmp_path: Path
) -> None:
    compile_script = load_script("compile")

    compile_script.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74", "EV74"]))
    for name, processors in (("two", ["MLA", "EV74", "MLA"]), ("host", ["MLA", "A65"]), ("none", ["EV74"])):
        with pytest.raises(RuntimeError, match="invalid FresUNet MPK"):
            compile_script.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors))


def test_compiled_build_has_one_mla_segment(compiled: Path, load_script: Callable[[str], ModuleType]) -> None:
    compile_script = load_script("compile")
    (archive,) = sorted(compiled.glob("model/*_mpk.tar.gz"))

    compile_script.audit_mpk(archive)
