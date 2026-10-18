"""Measure warm selected-model inference against the one-second monitor cadence."""

import argparse
import hashlib
import importlib.metadata
import math
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from hearhere_audio.predict import AudioPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = PROJECT_ROOT / "examples" / "1-31482-A-42.wav"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "inference_benchmark.local.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(audio_path: Path, output_path: Path, iterations: int) -> dict:
    if iterations < 5:
        raise ValueError("Use at least five warm iterations")
    startup_started = time.perf_counter()
    predictor = AudioPredictor()
    model_startup_ms = (time.perf_counter() - startup_started) * 1000
    warmup_started = time.perf_counter()
    predictor.predict_file(audio_path)
    first_warmup_ms = (time.perf_counter() - warmup_started) * 1000
    latencies_ms = []
    for index in range(iterations):
        started = time.perf_counter()
        predictor.predict_file(audio_path)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        print(f"Measured warm inference {index + 1}/{iterations}")

    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    model_path = Path(predictor.model_path)
    try:
        report_audio_path = audio_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        report_audio_path = str(audio_path.resolve())
    report = {
        "artifact_status": "local_warm_inference_benchmark",
        "model_version": predictor.model_version,
        "model_sha256": sha256_file(model_path),
        "audio_path": report_audio_path,
        "audio_sha256": sha256_file(audio_path),
        "iterations": iterations,
        "request_cadence_ms": 1000,
        "model_startup_ms": model_startup_ms,
        "first_inference_warmup_ms": first_warmup_ms,
        "latency_ms": {
            "minimum": min(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": ordered[p95_index],
            "maximum": max(latencies_ms),
        },
        "within_one_second_cadence": max(latencies_ms) < 1000,
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": importlib.metadata.version("torch"),
            "torchlibrosa": importlib.metadata.version("torchlibrosa"),
        },
        "notes": "Warm local predictor timing includes WAV header validation, preprocessing, embedding extraction and classifier scoring; it excludes HTTP transport and model startup. This local result does not verify LinuxONE latency.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    report = run_benchmark(args.audio.resolve(), args.output.resolve(), args.iterations)
    print(json.dumps(report["latency_ms"], indent=2))
    print(f"Saved benchmark: {args.output.resolve()}")


if __name__ == "__main__":
    main()
