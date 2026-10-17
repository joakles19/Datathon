"""Consume B's /classify API without loading a model or applying alert rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from .config import CLASSES


class ClassifyAPIError(RuntimeError):
    """The API request failed or the returned response violates the contract."""


class ClassifyHTTPError(ClassifyAPIError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


def _score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and math.isfinite(value)
    )


def validate_classify_response(payload: Any) -> dict[str, Any]:
    """Validate structure and score mapping while preserving server policy output."""
    if not isinstance(payload, dict):
        raise ClassifyAPIError("/classify must return a JSON object")
    fields = {"label", "predicted_label", "confidence", "action", "priority", "message", "event_id", "scores", "model_version"}
    if not fields.issubset(payload):
        raise ClassifyAPIError("Missing /classify fields: " + ", ".join(sorted(fields - payload.keys())))
    scores = payload["scores"]
    if not isinstance(scores, dict) or set(scores) != set(CLASSES) or not all(_score(v) for v in scores.values()):
        raise ClassifyAPIError("scores must contain eight finite HearHere class scores between 0 and 1")
    # Normalization is an estimator-output consistency check, not calibration.
    if not math.isclose(sum(scores.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ClassifyAPIError("Class scores do not sum to one within tolerance")
    for field in ("label", "predicted_label"):
        if not isinstance(payload[field], str) or payload[field] not in scores:
            raise ClassifyAPIError(f"{field} must name one of the eight classes")
    if not math.isclose(scores[payload["predicted_label"]], max(scores.values()), rel_tol=0, abs_tol=1e-12):
        raise ClassifyAPIError("predicted_label does not have a highest class score")
    if not _score(payload["confidence"]) or not math.isclose(
        payload["confidence"], scores[payload["label"]], rel_tol=0, abs_tol=1e-12,
    ):
        raise ClassifyAPIError("confidence must equal the selected label's score")
    if not isinstance(payload["action"], str) or payload["action"] not in {"alert", "ignore", "suppressed"}:
        raise ClassifyAPIError("Unrecognized server action")
    for field in ("priority", "message"):
        if payload[field] is not None and not isinstance(payload[field], str):
            raise ClassifyAPIError(f"{field} must be text or null")
    event_id = payload["event_id"]
    if event_id is not None and (isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0):
        raise ClassifyAPIError("event_id must be a positive integer or null")
    if not isinstance(payload["model_version"], str) or not payload["model_version"].strip():
        raise ClassifyAPIError("model_version must be a non-empty string")
    return payload


class ClassifyClient:
    """POST one WAV under multipart field 'file'; never retry automatically.

    The server owns thresholds, alert selection and events. A timeout can occur
    after an event was created, so a failed request is not retried by this client.
    """

    def __init__(self, base_url: str, *, timeout: float = 60.0, max_upload_bytes: int = 10 * 1024 * 1024):
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.query or url.fragment or url.username or url.password:
            raise ValueError("base_url must be an HTTP(S) server URL without credentials, query or fragment")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if isinstance(max_upload_bytes, bool) or not isinstance(max_upload_bytes, int) or max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be a positive integer")
        self.endpoint = base_url.rstrip("/") + "/classify"
        self.timeout = float(timeout)
        self.max_upload_bytes = max_upload_bytes

    def classify_file(self, audio_path: str | Path) -> dict[str, Any]:
        path = Path(audio_path)
        try:
            with path.open("rb") as audio:
                data = audio.read(self.max_upload_bytes + 1)
        except OSError as exc:
            raise ClassifyAPIError(f"Could not read upload file: {path}") from exc
        if not data:
            raise ClassifyAPIError("Audio upload is empty")
        if len(data) > self.max_upload_bytes:
            raise ClassifyAPIError("Audio upload exceeds the configured byte limit")
        if len(data) < 12 or data[:4] not in {b"RIFF", b"RIFX", b"RF64"} or data[8:12] != b"WAVE":
            raise ClassifyAPIError("Expected a WAV container; filename extension alone is insufficient")
        boundary = "hearhere-" + uuid.uuid4().hex
        # A fixed multipart filename avoids header escaping of a caller's path.
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="upload.wav"\r\n'
            'Content-Type: audio/wav\r\n\r\n'
        ).encode("ascii") + data + f"\r\n--{boundary}--\r\n".encode("ascii")
        request = Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        })
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_bytes = response.read(64 * 1024 + 1)
        except HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            finally:
                exc.close()
            try:
                error_body = json.loads(detail)
                if isinstance(error_body, dict):
                    detail = error_body.get("detail", detail)
            except ValueError:
                pass
            if not isinstance(detail, str):
                detail = json.dumps(detail)
            raise ClassifyHTTPError(exc.code, detail) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ClassifyAPIError("Could not complete /classify request; no automatic retry was made") from exc
        if len(response_bytes) > 64 * 1024:
            raise ClassifyAPIError("/classify response exceeds the 64 KiB limit")
        try:
            payload = json.loads(response_bytes)
        except (ValueError, UnicodeError) as exc:
            raise ClassifyAPIError("/classify returned invalid JSON") from exc
        return validate_classify_response(payload)
