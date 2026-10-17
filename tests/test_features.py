import numpy as np
import pytest

from hearhere_audio.config import FEATURE_ORDER, SAMPLE_RATE, WINDOW_SAMPLES
from hearhere_audio.features import (
    AudioProcessingError,
    extract_features,
    extract_features_from_bytes,
    extract_features_from_file,
    prepare_audio,
)


def test_short_audio_is_zero_padded_to_fixed_window():
    samples = np.ones(SAMPLE_RATE, dtype=np.float32)

    prepared = prepare_audio(samples, SAMPLE_RATE)

    assert prepared.shape == (WINDOW_SAMPLES,)
    np.testing.assert_array_equal(prepared[:SAMPLE_RATE], samples)
    np.testing.assert_array_equal(prepared[SAMPLE_RATE:], 0)


def test_stereo_audio_is_mixed_down_to_mono():
    stereo = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    stereo[:, 0] = 1.0
    stereo[:, 1] = -0.5

    prepared = prepare_audio(stereo, SAMPLE_RATE)

    np.testing.assert_allclose(prepared[:SAMPLE_RATE], 0.25)


def test_audio_longer_than_five_seconds_is_rejected():
    samples = np.zeros(WINDOW_SAMPLES + 1, dtype=np.float32)

    with pytest.raises(AudioProcessingError, match="longer than 5 seconds"):
        prepare_audio(samples, SAMPLE_RATE)


@pytest.mark.parametrize(
    "samples, sample_rate, message",
    [
        (np.array([], dtype=np.float32), SAMPLE_RATE, "empty"),
        (np.array([np.nan], dtype=np.float32), SAMPLE_RATE, "non-finite"),
        (np.zeros(SAMPLE_RATE), 0, "positive integer"),
        (np.zeros((2, SAMPLE_RATE)), SAMPLE_RATE, "one or two channels"),
    ],
)
def test_invalid_waveforms_are_rejected(samples, sample_rate, message):
    with pytest.raises(AudioProcessingError, match=message):
        prepare_audio(samples, sample_rate)


def test_feature_vector_has_fixed_order_shape_and_finite_values():
    samples = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)

    features = extract_features(samples.astype(np.float32), SAMPLE_RATE)

    assert len(FEATURE_ORDER) == 40
    assert features.shape == (40,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_file_entrypoint_reads_wav_and_resamples(tmp_path):
    import soundfile as sf

    sample_rate = 44_100
    samples = np.zeros((sample_rate, 2), dtype=np.float32)
    samples[1000:1200, 0] = 0.8
    samples[2000:2200, 1] = -0.6
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, samples, sample_rate)

    features = extract_features_from_file(audio_path)

    assert features.shape == (40,)
    assert np.isfinite(features).all()


def test_bytes_entrypoint_matches_file_features(tmp_path):
    import soundfile as sf

    samples = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, samples.astype(np.float32), SAMPLE_RATE)

    from_file = extract_features_from_file(audio_path)
    from_bytes = extract_features_from_bytes(audio_path.read_bytes())

    np.testing.assert_array_equal(from_bytes, from_file)


def test_file_entrypoint_reports_decode_errors(tmp_path):
    with pytest.raises(AudioProcessingError, match="Could not decode"):
        extract_features_from_file(tmp_path / "missing.wav")
