"""Contract and integration tests using A1's real model and supplied WAV.

The reference WAV belongs to fold 1; these tests measure integration,
not held-out classification accuracy. No model is trained here.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import soundfile as sf

from hearhere_audio.backend_adapter import predict_label_score
from hearhere_audio.cli import main
from hearhere_audio.features import AudioProcessingError, extract_features_from_file
from hearhere_audio.predict import AudioPredictor, ModelLoadError, PredictionError, predict_file


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/hearhere_rf.joblib"
METADATA = ROOT / "models/model_metadata.json"
SAMPLE = ROOT / "examples/1-31482-A-42.wav"
REFERENCE = ROOT / "tests/sample_reference.json"


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        cls.predictor = AudioPredictor(MODEL, METADATA)
        cls.result = cls.predictor.predict_file(SAMPLE)
        cls.temp_root = ROOT / ".test_tmp"
        cls.temp_root.mkdir(exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.path = Path(self.temp.name)

    def tearDown(self):
        # Verify the absolute cleanup target is within this project's test area.
        if not self.path.resolve().is_relative_to(self.temp_root.resolve()):
            raise RuntimeError("Unexpected test cleanup path")
        self.temp.cleanup()

    def wav(self, name="input.wav", data=None, sr=16000, subtype="PCM_16", format="WAV"):
        if data is None:
            data = np.zeros(16000, dtype=np.float32)
        path = self.path / name
        sf.write(path, data, sr, subtype=subtype, format=format)
        return path

    def metadata(self, update):
        data = json.loads(METADATA.read_text())
        update(data)
        path = self.path / "metadata.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_real_sample_output_and_class_score_mapping(self):
        self.assertEqual(set(self.result), {"predicted_class", "model_score", "scores", "model_version", "processing_ms"})
        self.assertEqual(hashlib.sha256(SAMPLE.read_bytes()).hexdigest(), self.reference["wav_sha256"])
        self.assertEqual(self.result["predicted_class"], self.reference["predicted_class"])
        self.assertEqual(list(self.result["scores"]), list(self.predictor.model.classes_))
        self.assertEqual(self.result["model_version"], self.reference["model_version"])
        for label, expected in self.reference["scores"].items():
            self.assertAlmostEqual(self.result["scores"][label], expected, places=12)
        self.assertEqual(self.result["model_score"], self.result["scores"][self.result["predicted_class"]])
        self.assertGreaterEqual(self.result["processing_ms"], 0)
        json.dumps(self.result, allow_nan=False)

    def test_sample_features_match_review_reference(self):
        actual = extract_features_from_file(SAMPLE)
        np.testing.assert_allclose(actual, self.reference["features"], rtol=1e-5, atol=1e-4)

    def test_convenience_function_matches_persistent_predictor(self):
        result = predict_file(SAMPLE, MODEL, METADATA)
        self.assertEqual(result["predicted_class"], self.result["predicted_class"])
        self.assertEqual(result["scores"], self.result["scores"])

    def test_predictor_loads_model_once_for_multiple_calls(self):
        with patch("hearhere_audio.predict.joblib.load", wraps=joblib.load) as load:
            predictor = AudioPredictor(MODEL, METADATA)
            first = predictor.predict_file(SAMPLE)
            second = predictor.predict_file(SAMPLE)
            self.assertEqual(load.call_count, 1)
        self.assertEqual(first["scores"], second["scores"])

    def test_save_reload_preserves_real_model_output(self):
        model_path = self.path / "roundtrip.joblib"
        joblib.dump(self.predictor.model, model_path)
        meta_path = self.metadata(lambda m: m.update(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()))
        loaded = AudioPredictor(model_path, meta_path)
        result = loaded.predict_file(SAMPLE)
        self.assertEqual(result["predicted_class"], self.result["predicted_class"])
        np.testing.assert_allclose(list(result["scores"].values()), list(self.result["scores"].values()), rtol=0, atol=1e-12)

    def test_tuple_adapter_matches_full_result(self):
        label, score = predict_label_score(SAMPLE, self.predictor)
        self.assertIsInstance(label, str)
        self.assertIsInstance(score, float)
        self.assertEqual((label, score), (self.result["predicted_class"], self.result["model_score"]))

    def test_short_stereo_wav_is_accepted(self):
        path = self.wav(data=np.zeros((8000, 2), dtype=np.float32))
        result = self.predictor.predict_file(path)
        self.assertEqual(len(result["scores"]), 8)

    def test_missing_empty_corrupt_and_zero_frame_inputs(self):
        empty = self.path / "empty.wav"
        empty.write_bytes(b"")
        corrupt = self.path / "corrupt.wav"
        corrupt.write_bytes(b"RIFF_corrupt_audio")
        no_frames = self.wav("zero_frames.wav", np.empty(0))
        for path in (self.path / "missing.wav", empty, corrupt, no_frames, self.path):
            with self.subTest(path=path.name), self.assertRaises(AudioProcessingError):
                self.predictor.predict_file(path)

    def test_nonfinite_wav_samples_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                path = self.wav(data=np.array([0.0, value], dtype=np.float32), subtype="FLOAT")
                with self.assertRaisesRegex(AudioProcessingError, "non-finite"):
                    self.predictor.predict_file(path)

    def test_long_wav_is_rejected_before_feature_extraction(self):
        for sr in (16000, 44100):
            path = self.wav(data=np.zeros(sr * 5 + 1), sr=sr)
            with self.subTest(sr=sr), patch("hearhere_audio.predict.extract_features_from_file") as extract:
                with self.assertRaisesRegex(AudioProcessingError, "longer than 5 seconds"):
                    self.predictor.predict_file(path)
                extract.assert_not_called()

    def test_nonwav_container_and_disguised_extension_are_rejected(self):
        flac = self.wav("input.flac", format="FLAC")
        disguised = self.path / "disguised.wav"
        disguised.write_bytes(flac.read_bytes())
        for path in (flac, disguised):
            with self.subTest(path=path.name), self.assertRaisesRegex(AudioProcessingError, "Expected WAV"):
                self.predictor.predict_file(path)

    def test_file_byte_limit_is_enforced(self):
        predictor = AudioPredictor(MODEL, METADATA, max_audio_bytes=1024)
        with self.assertRaisesRegex(AudioProcessingError, "byte limit"):
            predictor.predict_file(SAMPLE)

    def test_checksum_mismatch_is_rejected_before_deserialization(self):
        meta_path = self.metadata(lambda m: m.update(model_sha256="0" * 64))
        with patch("hearhere_audio.predict.joblib.load") as load:
            with self.assertRaisesRegex(ModelLoadError, "checksum"):
                AudioPredictor(MODEL, meta_path)
            load.assert_not_called()

    def test_reordered_metadata_classes_are_rejected(self):
        meta_path = self.metadata(lambda m: m["class_order"].reverse())
        with self.assertRaisesRegex(ModelLoadError, "class order"):
            AudioPredictor(MODEL, meta_path)

    def test_wrong_feature_order_is_rejected(self):
        meta_path = self.metadata(lambda m: m["feature_extractor"].update(feature_order="wrong"))
        with self.assertRaisesRegex(ModelLoadError, "feature_order"):
            AudioPredictor(MODEL, meta_path)

    def test_missing_and_invalid_metadata_and_model_are_clear_errors(self):
        invalid_json = self.path / "broken.json"
        invalid_json.write_text("not JSON")
        for path in (self.path / "missing.json", invalid_json):
            with self.subTest(path=path.name), self.assertRaises(ModelLoadError):
                AudioPredictor(MODEL, path)
        with self.assertRaisesRegex(ModelLoadError, "Could not load model"):
            AudioPredictor(self.path / "missing.joblib", METADATA)

    def test_invalid_model_scores_are_rejected(self):
        with patch.object(self.predictor.model, "predict_proba", return_value=np.full((1, 8), np.nan)):
            with self.assertRaisesRegex(PredictionError, "invalid class scores"):
                self.predictor.predict_file(SAMPLE)

    def test_cli_success_is_json(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([str(SAMPLE), "--model", str(MODEL), "--metadata", str(METADATA)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["predicted_class"], self.result["predicted_class"])
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_audio_error_is_stderr_json_and_exit_2(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([str(self.path / "missing.wav")])
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["type"], "AudioProcessingError")

    def test_cli_model_error_is_stderr_json_and_exit_3(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([str(SAMPLE), "--model", str(self.path / "missing.joblib")])
        self.assertEqual(exit_code, 3)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["type"], "ModelLoadError")


if __name__ == "__main__":
    unittest.main()
