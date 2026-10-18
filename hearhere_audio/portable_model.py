"""NumPy-only inference for the frozen PANNs logistic-regression head.

LinuxONE's approved s390x runtime has no supported scikit-learn wheel. The
portable artifact stores the fitted StandardScaler and LogisticRegression
arrays, so its prediction math remains identical without importing sklearn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PortableModelError(RuntimeError):
    """Raised when a portable classifier artifact is invalid."""


class PortableLogisticRegression:
    """A validated StandardScaler + multinomial logistic regression head."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            with np.load(self.path, allow_pickle=False) as artifact:
                self.classes_ = np.asarray(artifact["classes"]).astype(str)
                self.mean_ = np.asarray(artifact["mean"], dtype=np.float64)
                self.scale_ = np.asarray(artifact["scale"], dtype=np.float64)
                self.coef_ = np.asarray(artifact["coef"], dtype=np.float64)
                self.intercept_ = np.asarray(artifact["intercept"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            raise PortableModelError(f"Could not read portable model: {self.path}") from exc
        self.n_features_in_ = int(self.mean_.size)
        if (
            self.classes_.ndim != 1
            or self.classes_.size == 0
            or self.mean_.shape != (self.n_features_in_,)
            or self.scale_.shape != (self.n_features_in_,)
            or self.coef_.shape != (self.classes_.size, self.n_features_in_)
            or self.intercept_.shape != (self.classes_.size,)
            or not np.isfinite(self.mean_).all()
            or not np.isfinite(self.scale_).all()
            or not np.isfinite(self.coef_).all()
            or not np.isfinite(self.intercept_).all()
            or np.any(self.scale_ <= 0)
        ):
            raise PortableModelError("Portable model arrays have inconsistent shapes or values")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise PortableModelError(
                f"Portable model expects {self.n_features_in_} features, got {values.shape}"
            )
        logits = ((values - self.mean_) / self.scale_) @ self.coef_.T + self.intercept_
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def load_portable_model(path: str | Path) -> PortableLogisticRegression:
    return PortableLogisticRegression(path)
