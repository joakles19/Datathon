from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from hearhere_audio.api_client import ClassifyAPIError, ClassifyClient, ClassifyHTTPError, validate_classify_response


SAMPLE = Path(__file__).resolve().parents[1] / "examples/1-31482-A-42.wav"


def response_payload():
    # Deliberately different prediction and alert candidate: no re-selection here.
    return {
        "label": "siren", "predicted_label": "rain", "confidence": 0.31,
        "action": "alert", "priority": "high", "message": "Siren detected", "event_id": 17,
        "scores": {"door_wood_knock": 0.08, "engine": 0.04, "glass_breaking": 0.06,
                   "keyboard_typing": 0.05, "rain": 0.40, "siren": 0.31, "vacuum_cleaner": 0.03, "wind": 0.03},
        "model_version": "hearhere-rf-v1",
    }


class APIClientTests(unittest.TestCase):
    def test_preserves_different_predicted_and_selected_labels(self):
        payload = response_payload()
        self.assertIs(validate_classify_response(payload), payload)
        self.assertEqual(payload["predicted_label"], "rain")
        self.assertEqual(payload["label"], "siren")
        self.assertEqual(payload["confidence"], 0.31)

    def test_preserves_server_suppression(self):
        payload = response_payload()
        payload.update(action="suppressed", message="repeat within cooldown")
        self.assertEqual(validate_classify_response(payload)["action"], "suppressed")

    def test_accepts_ignore_with_null_event_fields(self):
        payload = response_payload()
        payload.update(label="rain", confidence=0.4, action="ignore", priority=None, message=None, event_id=None)
        self.assertEqual(validate_classify_response(payload)["action"], "ignore")

    def test_rejects_missing_fields_invalid_scores_and_wrong_mapping(self):
        changes = (
            lambda p: p.pop("predicted_label"),
            lambda p: p["scores"].update(siren=float("nan")),
            lambda p: p.update(confidence=0.4),
            lambda p: p.update(predicted_label="siren"),
            lambda p: p.update(event_id=True),
            lambda p: p.update(action="invented_action"),
        )
        for change in changes:
            payload = deepcopy(response_payload())
            change(payload)
            with self.subTest(change=change), self.assertRaises(ClassifyAPIError):
                validate_classify_response(payload)

    def test_multipart_upload_uses_file_field_and_preserves_payload(self):
        payload = response_payload()
        with patch("hearhere_audio.api_client.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as send:
            actual = ClassifyClient("http://127.0.0.1:8000").classify_file(SAMPLE)
        self.assertEqual(actual, payload)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/classify")
        self.assertEqual(request.get_method(), "POST")
        self.assertIn(b'name="file"; filename="upload.wav"', request.data)
        self.assertIn(SAMPLE.read_bytes(), request.data)
        self.assertEqual(send.call_args.kwargs["timeout"], 60.0)

    def test_http_error_preserves_status_and_detail_without_retry(self):
        error = HTTPError("http://127.0.0.1/classify", 422, "invalid audio", {}, io.BytesIO(b'{"detail":"Audio is longer than 5 seconds"}'))
        with patch("hearhere_audio.api_client.urlopen", side_effect=error) as send:
            with self.assertRaises(ClassifyHTTPError) as caught:
                ClassifyClient("http://127.0.0.1").classify_file(SAMPLE)
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn("longer than 5 seconds", caught.exception.detail)
        self.assertEqual(send.call_count, 1)

    def test_connection_failure_is_reported_without_retry(self):
        with patch("hearhere_audio.api_client.urlopen", side_effect=URLError("refused")) as send:
            with self.assertRaises(ClassifyAPIError):
                ClassifyClient("http://127.0.0.1").classify_file(SAMPLE)
        self.assertEqual(send.call_count, 1)

    def test_nonjson_response_is_reported(self):
        with patch("hearhere_audio.api_client.urlopen", return_value=io.BytesIO(b"<html>not JSON</html>")):
            with self.assertRaisesRegex(ClassifyAPIError, "invalid JSON"):
                ClassifyClient("http://127.0.0.1").classify_file(SAMPLE)

    def test_response_size_is_bounded(self):
        with patch("hearhere_audio.api_client.urlopen", return_value=io.BytesIO(b"x" * (64 * 1024 + 1))):
            with self.assertRaisesRegex(ClassifyAPIError, "64 KiB"):
                ClassifyClient("http://127.0.0.1").classify_file(SAMPLE)

    def test_rejects_oversized_upload_before_network(self):
        with patch("hearhere_audio.api_client.urlopen") as send:
            with self.assertRaisesRegex(ClassifyAPIError, "byte limit"):
                ClassifyClient("http://127.0.0.1", max_upload_bytes=16).classify_file(SAMPLE)
            send.assert_not_called()

    def test_bad_url_and_timeout_are_rejected(self):
        for url in ("file:///tmp", "not-a-url", "https://example.com/#fragment", "http://user:password@example.com"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ClassifyClient(url)
        for timeout in (0, float("nan"), -1):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                ClassifyClient("http://127.0.0.1", timeout=timeout)


if __name__ == "__main__":
    unittest.main()
