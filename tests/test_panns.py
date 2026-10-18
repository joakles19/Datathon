import hashlib
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from hearhere_audio import panns


class FakeEmbeddingModel:
    def __new__(cls):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.25))

            def forward(self, inputs):
                embedding = self.value.expand(inputs.shape[0], panns.EMBEDDING_DIM)
                return {"embedding": embedding}

        return Model()


def make_extractor(tmp_path, monkeypatch):
    import torch

    model = FakeEmbeddingModel()
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    monkeypatch.setattr(panns, "CHECKPOINT_SIZE", checkpoint_path.stat().st_size)
    monkeypatch.setattr(
        panns,
        "CHECKPOINT_MD5",
        hashlib.md5(checkpoint_path.read_bytes(), usedforsecurity=False).hexdigest(),
    )
    return panns.PannsEmbeddingExtractor(checkpoint_path, model=FakeEmbeddingModel())


def test_checkpoint_rejects_unexpected_size(tmp_path, monkeypatch):
    path = tmp_path / "bad.pth"
    path.write_bytes(b"bad")
    monkeypatch.setattr(panns, "CHECKPOINT_SIZE", 4)

    with pytest.raises(panns.PannsError, match="unexpected size"):
        panns._validate_checkpoint(path)


def test_checkpoint_rejects_wrong_published_checksum(tmp_path, monkeypatch):
    path = tmp_path / "bad.pth"
    path.write_bytes(b"fake weights")
    monkeypatch.setattr(panns, "CHECKPOINT_SIZE", path.stat().st_size)
    monkeypatch.setattr(panns, "CHECKPOINT_MD5", "0" * 32)

    with pytest.raises(panns.PannsError, match="MD5"):
        panns._validate_checkpoint(path)


def test_embedding_has_fixed_shape_finite_values_and_is_repeatable(tmp_path, monkeypatch):
    extractor = make_extractor(tmp_path, monkeypatch)
    waveform = np.zeros(panns.WINDOW_SAMPLES, dtype=np.float32)

    first = extractor.extract_embedding(waveform)
    second = extractor.extract_embedding(waveform)

    assert first.shape == (panns.EMBEDDING_DIM,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_wav_file_and_bytes_share_preprocessing_and_embedding(tmp_path, monkeypatch):
    extractor = make_extractor(tmp_path, monkeypatch)
    sample_rate = 44_100
    samples = np.zeros((sample_rate, 2), dtype=np.float32)
    samples[1000:1300, 0] = 0.7
    samples[2000:2300, 1] = -0.3
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, samples, sample_rate)
    audio = BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV")

    from_file = panns.extract_embedding_from_file(audio_path, extractor)
    from_bytes = panns.extract_embedding_from_bytes(audio.getvalue(), extractor)

    np.testing.assert_array_equal(from_file, from_bytes)
    assert from_file.shape == (panns.EMBEDDING_DIM,)
    assert np.isfinite(from_file).all()


def test_embedding_rejects_invalid_audio(tmp_path, monkeypatch):
    extractor = make_extractor(tmp_path, monkeypatch)

    with pytest.raises(panns.PannsError, match="empty"):
        panns.extract_embedding(extractor, np.array([], dtype=np.float32), 16_000)

    with pytest.raises(panns.PannsError, match="non-finite"):
        panns.extract_embedding(extractor, np.array([np.nan], dtype=np.float32), 16_000)
