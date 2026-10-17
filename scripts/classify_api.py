"""Upload a WAV to the existing backend and preserve its full JSON response."""

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearhere_audio.api_client import ClassifyAPIError, ClassifyClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload one WAV to HearHere's existing POST /classify.")
    parser.add_argument("audio_path")
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8000 (without /classify)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        client = ClassifyClient(args.base_url, timeout=args.timeout)
        start = time.perf_counter()
        result = client.classify_file(args.audio_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except (ClassifyAPIError, ValueError) as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2 if args.pretty else None, allow_nan=False))
    print(f"Request round-trip: {elapsed_ms:.1f} ms (includes upload, server work and response; excludes capture)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
