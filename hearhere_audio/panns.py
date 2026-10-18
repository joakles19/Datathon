"""CPU PANNs CNN14 embeddings for HearHere.

Architecture follows Qiuqiang Kong's MIT-licensed AudioSet tagging CNN:
https://github.com/qiuqiangkong/audioset_tagging_cnn
The 16 kHz AudioSet checkpoint is distributed separately under CC BY 4.0:
https://doi.org/10.5281/zenodo.3987831
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.request import urlopen

import librosa
import numpy as np

from .features import AudioProcessingError, prepare_audio


CHECKPOINT_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Cnn14_16k_mAP%3D0.438.pth?download=1"
)
CHECKPOINT_SIZE = 358_668_570
CHECKPOINT_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
EMBEDDING_DIM = 2048
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 5 * SAMPLE_RATE
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "hearhere" / "panns" / "Cnn14_16k_mAP=0.438.pth"


class PannsError(RuntimeError):
    """Checkpoint or embedding extraction failed."""


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise PannsError(f"PANNs checkpoint does not exist: {path}")
    if path.stat().st_size != CHECKPOINT_SIZE:
        raise PannsError(
            f"PANNs checkpoint has an unexpected size: {path.stat().st_size} bytes"
        )
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_MD5:
        raise PannsError("PANNs checkpoint MD5 does not match the Zenodo release")


def ensure_checkpoint(path: str | Path = DEFAULT_CACHE_PATH) -> tuple[Path, str]:
    """Download the official weights explicitly and return path and SHA-256.

    This function is never called implicitly during module import or server
    startup. A partial or unverified file is never left at the cache path.
    """
    destination = Path(path).expanduser()
    if destination.is_file():
        _validate_checkpoint(destination)
        return destination, checkpoint_sha256(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            received = 0
            with urlopen(CHECKPOINT_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > CHECKPOINT_SIZE:
                        raise PannsError("PANNs checkpoint download exceeds expected size")
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        _validate_checkpoint(temporary_path)
        digest = checkpoint_sha256(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination, digest
    except PannsError:
        raise
    except Exception as exc:
        raise PannsError(f"Could not download PANNs checkpoint: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _init_layer(layer: Any) -> None:
    import torch.nn as nn

    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def _init_batch_norm(layer: Any) -> None:
    layer.bias.data.fill_(0.0)
    layer.weight.data.fill_(1.0)


def _build_cnn14_16k():
    """Construct the upstream Cnn14_16k topology with checkpoint keys intact."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            _init_layer(self.conv1)
            _init_layer(self.conv2)
            _init_batch_norm(self.bn1)
            _init_batch_norm(self.bn2)

        def forward(self, inputs, pool_size=(2, 2)):
            output = functional.relu_(self.bn1(self.conv1(inputs)))
            output = functional.relu_(self.bn2(self.conv2(output)))
            return functional.avg_pool2d(output, kernel_size=pool_size)

    class Cnn14_16k(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectrogram_extractor = Spectrogram(
                n_fft=512,
                hop_length=160,
                win_length=512,
                window="hann",
                center=True,
                pad_mode="reflect",
                freeze_parameters=True,
            )
            self.logmel_extractor = LogmelFilterBank(
                sr=16000,
                n_fft=512,
                n_mels=64,
                fmin=50,
                fmax=8000,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True,
            )
            self.bn0 = nn.BatchNorm2d(64)
            channels = (1, 64, 128, 256, 512, 1024, 2048)
            for block in range(1, 7):
                setattr(self, f"conv_block{block}", ConvBlock(channels[block - 1], channels[block]))
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            self.fc_audioset = nn.Linear(2048, 527, bias=True)
            _init_batch_norm(self.bn0)
            _init_layer(self.fc1)
            _init_layer(self.fc_audioset)

        def forward(self, inputs):
            output = self.spectrogram_extractor(inputs)
            output = self.logmel_extractor(output)
            output = output.transpose(1, 3)
            output = self.bn0(output)
            output = output.transpose(1, 3)
            for block in range(1, 7):
                pool = (1, 1) if block == 6 else (2, 2)
                output = getattr(self, f"conv_block{block}")(output, pool_size=pool)
            output = torch.mean(output, dim=3)
            output = torch.max(output, dim=2).values + torch.mean(output, dim=2)
            output = functional.relu_(self.fc1(output))
            embedding = functional.dropout(output, p=0.5, training=self.training)
            clipwise_output = torch.sigmoid(self.fc_audioset(output))
            return {"clipwise_output": clipwise_output, "embedding": embedding}

    return Cnn14_16k()


class PannsEmbeddingExtractor:
    """Load frozen CPU CNN14 once and extract one 2048-D vector per window."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CACHE_PATH, *, model=None) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        _validate_checkpoint(self.checkpoint_path)
        try:
            import torch

            self._torch = torch
            self.model = model if model is not None else _build_cnn14_16k()
            # This upstream release stores NumPy metadata with its weights.
            # The fixed published MD5 check above gates this legacy pickle load.
            try:
                checkpoint = torch.load(
                    self.checkpoint_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                # Torch 1.10 is the approved s390x wheel and predates this kwarg.
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to("cpu")
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        except PannsError:
            raise
        except Exception as exc:
            raise PannsError(f"Could not load PANNs CNN14 checkpoint: {exc}") from exc

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.shape != (WINDOW_SAMPLES,):
            raise PannsError("PANNs input must be one mono five-second 16 kHz window")
        if not np.isfinite(samples).all():
            raise PannsError("PANNs input contains non-finite samples")
        with self._torch.inference_mode():
            result = self.model(self._torch.from_numpy(np.ascontiguousarray(samples))[None, :])
        embedding = result["embedding"].detach().cpu().numpy()[0].astype(np.float32, copy=False)
        if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
            raise PannsError("PANNs CNN14 produced an invalid embedding")
        return np.ascontiguousarray(embedding)


def extract_embedding(
    extractor: PannsEmbeddingExtractor, samples: np.ndarray, source_sample_rate: int
) -> np.ndarray:
    """Prepare arbitrary decoded audio with the shared contract and embed it."""
    try:
        waveform = prepare_audio(samples, source_sample_rate)
    except AudioProcessingError as exc:
        raise PannsError(str(exc)) from exc
    return extractor.extract_embedding(waveform)


def extract_embedding_from_file(
    audio_path: str | Path, extractor: PannsEmbeddingExtractor
) -> np.ndarray:
    """Read a WAV and use the same preprocessing path as online inference."""
    try:
        samples, source_sample_rate = librosa.load(
            str(audio_path), sr=None, mono=True, dtype=np.float32
        )
    except Exception as exc:
        raise PannsError(f"Could not decode audio file: {audio_path}") from exc
    return extract_embedding(extractor, samples, source_sample_rate)


def extract_embedding_from_bytes(
    audio_bytes: bytes, extractor: PannsEmbeddingExtractor
) -> np.ndarray:
    """Decode uploaded audio and use the shared preprocessing path."""
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise PannsError("Audio bytes must be a non-empty bytes value")
    try:
        samples, source_sample_rate = librosa.load(
            BytesIO(audio_bytes), sr=None, mono=True, dtype=np.float32
        )
    except Exception as exc:
        raise PannsError("Could not decode uploaded audio bytes") from exc
    return extract_embedding(extractor, samples, source_sample_rate)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the verified HearHere PANNs checkpoint")
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.download_checkpoint:
        parser.error("pass --download-checkpoint to fetch and verify the 359 MB weights")
    path, sha256 = ensure_checkpoint()
    print(f"Verified PANNs checkpoint: {path}")
    print(f"SHA-256: {sha256}")
    print("Source: Zenodo record 3987831; checkpoint license CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
