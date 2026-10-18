"""JSON command-line prediction for a single WAV recording."""

import argparse
import json
import sys

from .features import AudioProcessingError
from .predict import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, ModelLoadError, PredictionError, predict_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify one WAV of at most five seconds using HearHere.")
    parser.add_argument("audio_path", help="WAV file to classify")
    parser.add_argument("--model", default=None, help="Override the model file declared in model metadata")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Path to matching model metadata JSON")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output")
    args = parser.parse_args(argv)
    try:
        result = predict_file(args.audio_path, args.model, args.metadata)
    except (AudioProcessingError, ModelLoadError, PredictionError) as exc:
        code = 2 if isinstance(exc, AudioProcessingError) else 3 if isinstance(exc, ModelLoadError) else 4
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}), file=sys.stderr)
        return code
    print(json.dumps(result, indent=2 if args.pretty else None, allow_nan=False))
    return 0
