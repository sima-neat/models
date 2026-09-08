from __future__ import annotations

import hashlib
import io
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .accuracy_config import select_accuracy_config
from .errors import ModelRegistryError
from .model import load_and_validate_model

LOG = logging.getLogger("model_registry.accuracy")


class AccuracyError(ModelRegistryError):
    """Raised when declarative accuracy evaluation cannot complete safely."""


@dataclass(frozen=True)
class SampleRecord:
    member_name: str
    label: int | None


@dataclass(frozen=True)
class WorkerTask:
    reference_model: Path
    candidate_model: Path
    dataset: Path
    records: tuple[SampleRecord, ...]
    preprocessing: tuple[dict[str, Any], ...]
    postprocessing: tuple[dict[str, Any], ...]
    source_layout: str
    reference_layout: str
    candidate_layout: str
    input_name: str
    classes: int
    top_k: tuple[int, ...]
    use_jax: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(path: Path, expected_sha256: str, expected_size: int, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AccuracyError(f"{label} does not exist: {resolved}")
    actual_size = resolved.stat().st_size
    if actual_size != expected_size:
        raise AccuracyError(f"{label} size mismatch: expected {expected_size}, found {actual_size}")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise AccuracyError(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}")
    return resolved


def resolve_dataset(
    dataset_config: dict[str, Any],
    dataset_path: Path | None,
    cache_dir: Path | None = None,
) -> Path:
    if dataset_path is not None:
        return _validate_file(
            dataset_path,
            str(dataset_config["sha256"]),
            int(dataset_config["size_bytes"]),
            "dataset",
        )
    command_prefix: list[str]
    executable = shutil.which("dataset-registry")
    if executable is not None:
        command_prefix = [executable]
    else:
        installed_script = Path(sys.executable).resolve().with_name("dataset-registry")
        if installed_script.is_file():
            command_prefix = [str(installed_script)]
        else:
            command_prefix = [sys.executable, "-m", "dataset_registry"]
    command = [
        *command_prefix,
        "resolve",
        str(dataset_config["id"]),
        "--variant",
        str(dataset_config["variant"]),
    ]
    if cache_dir is not None:
        command.extend(("--cache-dir", str(cache_dir.expanduser().resolve())))
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        result = json.loads(completed.stdout)
        resolved = Path(result["path"])
    except FileNotFoundError as exc:
        raise AccuracyError("dataset-registry is required to resolve the configured dataset") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, TypeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise AccuracyError(f"dataset-registry resolution failed: {detail.strip()}") from exc
    return _validate_file(
        resolved,
        str(dataset_config["sha256"]),
        int(dataset_config["size_bytes"]),
        "dataset",
    )


def _safe_parts(name: str) -> tuple[str, ...]:
    normalized = name.removeprefix("./")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise AccuracyError(f"unsafe dataset archive member: {name}")
    return path.parts


def index_split(dataset: Path, dataset_config: dict[str, Any], split_name: str) -> list[SampleRecord]:
    split = dataset_config["splits"][split_name]
    prefix = tuple(PurePosixPath(str(split["prefix"])).parts)
    suffixes = tuple(str(item).lower() for item in dataset_config["loader"]["image_suffixes"])
    members: list[tuple[str, tuple[str, ...]]] = []
    with tarfile.open(dataset, "r:gz") as archive:
        for member in archive.getmembers():
            parts = _safe_parts(member.name)
            if member.issym() or member.islnk():
                if parts[: len(prefix)] == prefix:
                    raise AccuracyError(f"dataset archive links are not allowed: {member.name}")
                continue
            if member.isfile() and parts[: len(prefix)] == prefix and parts and parts[-1].lower().endswith(suffixes):
                members.append((member.name, parts))
    members.sort(key=lambda item: item[0])
    expected_samples = int(split["samples"])
    if len(members) != expected_samples:
        raise AccuracyError(f"dataset split {split_name!r} expected {expected_samples} images, found {len(members)}")

    strategy = split.get("label_strategy", "none")
    if strategy == "none":
        return [SampleRecord(name, None) for name, _ in members]
    if strategy != "lexicographic_parent_directory":
        raise AccuracyError(f"unsupported label strategy: {strategy}")
    class_names = sorted({parts[-2] for _, parts in members})
    expected_classes = int(split.get("classes", 0))
    if len(class_names) != expected_classes:
        raise AccuracyError(
            f"dataset split {split_name!r} expected {expected_classes} classes, found {len(class_names)}"
        )
    class_index = {name: index for index, name in enumerate(class_names)}
    records = [SampleRecord(name, class_index[parts[-2]]) for name, parts in members]
    counts = [0] * expected_classes
    for record in records:
        assert record.label is not None
        counts[record.label] += 1
    if min(counts) != max(counts):
        raise AccuracyError(
            f"dataset split {split_name!r} is not class balanced: class-count range {min(counts)}..{max(counts)}"
        )
    return records


def select_sample_indices(available: int, sample_count: int) -> list[int]:
    if sample_count < 1 or sample_count > available:
        raise AccuracyError(f"sample count must be in 1..{available}, found {sample_count}")
    if sample_count == 1:
        return [0]
    return [index * (available - 1) // (sample_count - 1) for index in range(sample_count)]


def _as_float_array(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def preprocess_image(data: bytes, pipeline: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> Any:
    import numpy as np
    from PIL import Image

    value: Any = data
    for stage in pipeline:
        operation = stage["op"]
        parameters = stage["parameters"]
        if operation == "decode_image":
            with Image.open(io.BytesIO(value)) as source:
                value = source.convert(str(parameters.get("color_space", "RGB"))).copy()
        elif operation == "resize":
            if not isinstance(value, Image.Image):
                raise AccuracyError("resize must run before conversion to an array")
            shortest_edge = int(parameters["shortest_edge"])
            width, height = value.size
            if width <= height:
                resized = (shortest_edge, int(shortest_edge * height / width))
            else:
                resized = (int(shortest_edge * width / height), shortest_edge)
            resampling = {
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "nearest": Image.Resampling.NEAREST,
            }
            try:
                method = resampling[str(parameters["resample"]).lower()]
            except KeyError as exc:
                raise AccuracyError(f"unsupported resize resample mode: {parameters['resample']}") from exc
            value = value.resize(resized, resample=method)
        elif operation == "center_crop":
            if not isinstance(value, Image.Image):
                raise AccuracyError("center_crop must run before conversion to an array")
            width = int(parameters["width"])
            height = int(parameters["height"])
            left = (value.width - width) // 2
            top = (value.height - height) // 2
            value = value.crop((left, top, left + width, top + height))
        elif operation == "rescale":
            value = _as_float_array(value) * np.float32(parameters["factor"])
        elif operation == "normalize":
            array = _as_float_array(value)
            mean = np.asarray(parameters["mean"], dtype=np.float32)
            std = np.asarray(parameters["std"], dtype=np.float32)
            value = (array - mean) / std
        elif operation == "transpose":
            array = _as_float_array(value)
            source_layout = str(parameters["from"])
            target_layout = str(parameters["to"])
            if sorted(source_layout) != sorted(target_layout) or array.ndim != len(source_layout):
                raise AccuracyError(f"invalid transpose {source_layout}->{target_layout} for shape {array.shape}")
            value = np.transpose(array, tuple(source_layout.index(axis) for axis in target_layout))
        else:
            raise AccuracyError(f"unsupported preprocessing operation: {operation}")
    array = _as_float_array(value)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise AccuracyError(f"expected one finite rank-3 image tensor, found shape {array.shape}")
    return array


def convert_sample_layout(sample: Any, source_layout: str, target_layout: str) -> Any:
    import numpy as np

    source = source_layout.removeprefix("N")
    target = target_layout.removeprefix("N")
    if sorted(source) != sorted(target) or sample.ndim != len(source):
        raise AccuracyError(f"cannot convert shape {sample.shape} from {source_layout} to {target_layout}")
    converted = sample if source == target else np.transpose(sample, tuple(source.index(axis) for axis in target))
    return converted[None]


def _read_member(archive: tarfile.TarFile, record: SampleRecord) -> bytes:
    try:
        member = archive.getmember(record.member_name)
    except KeyError as exc:
        raise AccuracyError(f"dataset member disappeared: {record.member_name}") from exc
    stream = archive.extractfile(member)
    if stream is None:
        raise AccuracyError(f"cannot read dataset member: {record.member_name}")
    return stream.read()


def load_calibration_samples(
    model_dir: str | Path,
    dataset_path: Path,
    sample_count: int,
    *,
    test_id: str | None = None,
    output_layout: str = "NHWC",
) -> Any:
    import numpy as np

    model = load_and_validate_model(model_dir)
    config_path, config = select_accuracy_config(model_dir, model, test_id)
    relative_config = config_path.relative_to(Path(model_dir).expanduser().resolve()).as_posix()
    test_case = next(case for case in model["tests"]["correctness"] if case["config"] == relative_config)
    component = next(item for item in model["components"] if item["id"] == test_case["scope"]["component"])
    entrypoint = component["interface"]["entrypoints"][0]
    dataset = resolve_dataset(config["dataset"], dataset_path)
    records = index_split(dataset, config["dataset"], "calibration")
    if sample_count != len(records):
        raise AccuracyError(f"calibration requires exactly {len(records)} samples, found request for {sample_count}")
    model_input = entrypoint["inputs"]
    if len(model_input) != 1:
        raise AccuracyError("the image-classification evaluator currently supports exactly one model input")
    source_layout = str(model_input[0]["layout"])
    samples = []
    with tarfile.open(dataset, "r:gz") as archive:
        for record in records:
            sample = preprocess_image(_read_member(archive, record), component["preprocessing"]["pipeline"])
            samples.append(convert_sample_layout(sample, source_layout, output_layout)[0])
    return np.stack(samples).astype(np.float32, copy=False)


def _classification_scores(
    logits: Any,
    pipeline: tuple[dict[str, Any], ...],
    maximum_top_k: int,
) -> Any:
    import numpy as np

    values = np.asarray(logits)
    for stage in pipeline:
        operation = stage["op"]
        parameters = stage["parameters"]
        if operation == "softmax":
            axis = int(parameters["axis"])
            shifted = values - np.max(values, axis=axis, keepdims=True)
            exponent = np.exp(shifted)
            values = exponent / np.sum(exponent, axis=axis, keepdims=True)
        elif operation == "top_k":
            if int(parameters["k"]) < maximum_top_k:
                raise AccuracyError(
                    f"postprocessing top_k={parameters['k']} is smaller than requested metric k={maximum_top_k}"
                )
        else:
            raise AccuracyError(f"unsupported classification postprocessing operation: {operation}")
    return values


def _prediction(
    logits: Any,
    classes: int,
    top_k: tuple[int, ...],
    postprocessing: tuple[dict[str, Any], ...],
) -> tuple[int, dict[int, set[int]]]:
    import numpy as np

    values = np.asarray(_classification_scores(logits, postprocessing, max(top_k))).reshape(-1)
    if values.size != classes or not np.isfinite(values).all():
        raise AccuracyError(f"expected {classes} finite logits, found shape {np.asarray(logits).shape}")
    top1 = int(np.argmax(values))
    selections = {k: set(int(index) for index in np.argpartition(values, -k)[-k:]) for k in top_k}
    return top1, selections


def _run_worker(task: WorkerTask) -> dict[str, int]:
    logging.basicConfig(level=logging.WARNING, force=True)
    try:
        import onnxruntime as ort
        from afe.apis.defines import InputName
        from afe.apis.model import Model as AfeModel
    except ImportError as exc:
        raise AccuracyError(
            "accuracy execution requires ONNX Runtime and AFE inside the activated Model Compiler environment"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        str(task.reference_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate = AfeModel.load(
        model_name=task.candidate_model.name,
        network_directory=str(task.candidate_model.parent),
    )
    counts = {f"reference_top{k}": 0 for k in task.top_k}
    counts.update({f"candidate_top{k}": 0 for k in task.top_k})
    counts["top1_agreement"] = 0
    with tarfile.open(task.dataset, "r:gz") as archive:
        for record in task.records:
            if record.label is None:
                raise AccuracyError(f"validation sample has no label: {record.member_name}")
            sample = preprocess_image(_read_member(archive, record), task.preprocessing)
            reference_input = convert_sample_layout(sample, task.source_layout, task.reference_layout)
            candidate_input = convert_sample_layout(sample, task.source_layout, task.candidate_layout)
            reference_logits = reference.run(None, {task.input_name: reference_input})[0]
            candidate_outputs = candidate.execute(
                inputs={InputName(task.input_name): candidate_input},
                use_jax=task.use_jax,
            )
            if len(candidate_outputs) != 1:
                raise AccuracyError(
                    f"expected one candidate output for {record.member_name}, found {len(candidate_outputs)}"
                )
            reference_top1, reference_topk = _prediction(
                reference_logits, task.classes, task.top_k, task.postprocessing
            )
            candidate_top1, candidate_topk = _prediction(
                candidate_outputs[0], task.classes, task.top_k, task.postprocessing
            )
            for k in task.top_k:
                counts[f"reference_top{k}"] += int(record.label in reference_topk[k])
                counts[f"candidate_top{k}"] += int(record.label in candidate_topk[k])
            counts["top1_agreement"] += int(reference_top1 == candidate_top1)
    return counts


def _metrics(counts: dict[str, int], sample_count: int, top_k: tuple[int, ...]) -> dict[str, Any]:
    reference = {f"top{k}_accuracy": counts[f"reference_top{k}"] / sample_count for k in top_k}
    candidate = {f"top{k}_accuracy": counts[f"candidate_top{k}"] / sample_count for k in top_k}
    comparison = {f"top{k}_delta": candidate[f"top{k}_accuracy"] - reference[f"top{k}_accuracy"] for k in top_k}
    comparison["top1_agreement"] = counts["top1_agreement"] / sample_count
    return {"reference": reference, "candidate": candidate, "comparison": comparison}


def evaluate_accuracy(
    model_dir: str | Path,
    build_profile_id: str,
    component_id: str,
    candidate_model: Path,
    *,
    reference_model: Path | None = None,
    dataset_path: Path | None = None,
    dataset_cache_dir: Path | None = None,
    model_cache_dir: Path | None = None,
    test_id: str | None = None,
    profile_name: str = "regular",
    sample_count: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    from .fetch import Fetcher
    from .source import SourceSpec

    started = time.perf_counter()
    directory = Path(model_dir).expanduser().resolve()
    model = load_and_validate_model(directory)
    config_path, config = select_accuracy_config(
        directory,
        model,
        test_id,
        build_profile_id=build_profile_id,
        component_id=component_id,
    )
    profiles = [item for item in model["build_profiles"] if item["id"] == build_profile_id]
    if len(profiles) != 1:
        raise AccuracyError(f"{directory}: expected exactly one build profile named {build_profile_id!r}")
    build_profile = profiles[0]
    bindings = [item for item in build_profile["components"] if item["id"] == component_id]
    if len(bindings) != 1:
        raise AccuracyError(
            f"{directory}: build profile {build_profile_id!r} does not contain component {component_id!r}"
        )
    binding = bindings[0]
    if build_profile["target"] != "modalix":
        raise AccuracyError(f"build profile {build_profile_id!r} does not target Modalix")

    relative_config = config_path.relative_to(directory).as_posix()
    test_case = next(case for case in model["tests"]["correctness"] if case["config"] == relative_config)
    if build_profile_id not in test_case["build_profiles"]:
        raise AccuracyError(f"accuracy test {config_path} does not apply to build profile {build_profile_id!r}")
    if test_case["scope"] != {"type": "component", "component": component_id}:
        raise AccuracyError(f"accuracy test {config_path} does not target component {component_id!r}")
    component = next(item for item in model["components"] if item["id"] == component_id)
    entrypoint = component["interface"]["entrypoints"][0]

    calibration_id = binding.get("calibration")
    calibration_entry = next(
        (item for item in model["calibrations"] if item["id"] == calibration_id), None
    )
    calibration = calibration_entry.get("dataset") if calibration_entry is not None else None
    dataset_config = config["dataset"]
    if calibration is None or any(calibration[key] != dataset_config[key] for key in calibration):
        raise AccuracyError("accuracy dataset identity does not match model calibration dataset metadata")
    dataset = resolve_dataset(dataset_config, dataset_path, dataset_cache_dir)

    source = component["source"]
    if source["type"] == "file":
        source_spec = SourceSpec.from_mapping(source)
        source_sha256 = reference_sha256 = source_spec.sha256
        if reference_model is None:
            reference, _ = Fetcher(model_cache_dir).fetch(source_spec)
        else:
            reference = _validate_file(reference_model, source_spec.sha256, source_spec.size_bytes, "reference model")
    elif source["type"] == "onnx_bundle":
        reference, _, source_sha256, _ = Fetcher(model_cache_dir).fetch_source(source)
        reference_sha256 = sha256_file(reference)
        if reference_model is not None:
            _validate_file(reference_model, reference_sha256, reference.stat().st_size, "reference model")
        # Execute the verified bundle so external tensor data stays checksum-verified,
        # even when the caller supplies an identical graph from another directory.
    else:
        raise AccuracyError("the image-classification evaluator requires a file or ONNX bundle component source")
    candidate = candidate_model.expanduser().resolve()
    if not candidate.is_file():
        raise AccuracyError(f"candidate model does not exist: {candidate}")

    evaluation_profiles = config["profiles"]
    if profile_name not in evaluation_profiles:
        raise AccuracyError(
            f"unknown accuracy profile {profile_name!r}; choose from {', '.join(sorted(evaluation_profiles))}"
        )
    profile = evaluation_profiles[profile_name]
    validation = index_split(dataset, dataset_config, "validation")
    configured_samples = len(validation) if profile["samples"] == "all" else int(profile["samples"])
    requested_samples = sample_count if sample_count is not None else configured_samples
    selected_indices = select_sample_indices(len(validation), requested_samples)
    selected = [validation[index] for index in selected_indices]

    configured_workers = profile["workers"]
    if workers is None:
        if configured_workers == "auto":
            worker_count = max(1, min(8, (os.cpu_count() or 2) // 2))
        else:
            worker_count = int(configured_workers)
    else:
        worker_count = workers
    if worker_count < 1:
        raise AccuracyError("workers must be at least 1")
    worker_count = min(worker_count, requested_samples)

    model_inputs = entrypoint["inputs"]
    if len(model_inputs) != 1:
        raise AccuracyError("the image-classification evaluator currently supports exactly one model input")
    input_name = str(binding["parameters"]["input_name"])
    if input_name != model_inputs[0]["name"]:
        raise AccuracyError(f"build-profile input {input_name!r} does not match component interface")
    executors = config["executors"]
    if executors["reference"]["type"] != "onnxruntime" or executors["candidate"]["type"] != "afe_sima":
        raise AccuracyError("the current engine requires onnxruntime reference and afe_sima candidate executors")
    if executors["reference"]["artifact"] != "source" or executors["candidate"]["artifact"] != "compiled_component":
        raise AccuracyError("executor artifact roles must be source reference and compiled_component candidate")
    if executors["reference"]["input_layout"] != "NCHW" or executors["candidate"]["input_layout"] != "NHWC":
        raise AccuracyError("the current engine requires NCHW reference and NHWC AFE candidate inputs")

    top_k = tuple(sorted({int(metric["k"]) for metric in config["metrics"] if metric["type"] == "top_k_accuracy"}))
    if not top_k or 1 not in top_k:
        raise AccuracyError("classification accuracy metrics must include top_k_accuracy with k: 1")
    shard_size = (len(selected) + worker_count - 1) // worker_count
    shards = [selected[start : start + shard_size] for start in range(0, len(selected), shard_size)]
    tasks = [
        WorkerTask(
            reference_model=reference,
            candidate_model=candidate,
            dataset=dataset,
            records=tuple(shard),
            preprocessing=tuple(component["preprocessing"]["pipeline"]),
            postprocessing=tuple(component["postprocessing"]["pipeline"]),
            source_layout=str(model_inputs[0]["layout"]),
            reference_layout=str(executors["reference"]["input_layout"]),
            candidate_layout=str(executors["candidate"]["input_layout"]),
            input_name=input_name,
            classes=int(dataset_config["splits"]["validation"]["classes"]),
            top_k=top_k,
            use_jax=bool(executors["candidate"].get("use_jax", False)),
        )
        for shard in shards
    ]
    counts = {f"reference_top{k}": 0 for k in top_k}
    counts.update({f"candidate_top{k}": 0 for k in top_k})
    counts["top1_agreement"] = 0
    try:
        if len(tasks) == 1:
            partials = [_run_worker(tasks[0])]
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
                futures = [executor.submit(_run_worker, task) for task in tasks]
                partials = []
                for completed, future in enumerate(as_completed(futures), start=1):
                    partials.append(future.result())
                    LOG.info("Accuracy worker %d/%d complete", completed, len(futures))
    except AccuracyError:
        raise
    except Exception as exc:
        raise AccuracyError(f"accuracy worker failed: {type(exc).__name__}: {exc}") from exc
    for partial in partials:
        for name, value in partial.items():
            counts[name] += int(value)

    selection_sha256 = hashlib.sha256(",".join(str(index) for index in selected_indices).encode("ascii")).hexdigest()
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "report_scope": str(profile["report_scope"]),
        "model": {
            "id": model["id"],
            "component_id": component_id,
            "source_sha256": source_sha256,
            "build_profile_id": build_profile_id,
        },
        "artifacts": {
            "reference": {"type": "onnx", "sha256": reference_sha256, "size_bytes": reference.stat().st_size},
            "candidate": {
                "type": "sima",
                "sha256": sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            },
        },
        "dataset": {
            "id": dataset_config["id"],
            "variant": dataset_config["variant"],
            "sha256": dataset_config["sha256"],
            "split": "validation",
            "available_samples": len(validation),
            "evaluated_samples": requested_samples,
            "sampling": {
                "method": profile["sampling"],
                "seed": None,
                "selection_sha256": selection_sha256,
            },
        },
        "evaluation": {
            "config": str(config_path.relative_to(directory)),
            "profile": profile_name,
            "status": config["status"],
        },
        "execution": {
            "workers": len(tasks),
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round(requested_samples / elapsed, 3),
            "reference": executors["reference"]["type"],
            "candidate": executors["candidate"]["type"],
        },
        "counts": counts,
        "metrics": _metrics(counts, requested_samples, top_k),
    }
