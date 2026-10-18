"""Export the fitted PANNs logistic head for runtimes without scikit-learn."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def export(source: Path, destination: Path) -> None:
    model = joblib.load(source)
    scaler = model.named_steps["standard_scaler"]
    logistic = model.named_steps["logistic_regression"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        classes=np.asarray(logistic.classes_, dtype="U64"),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coef=np.asarray(logistic.coef_, dtype=np.float64),
        intercept=np.asarray(logistic.intercept_, dtype=np.float64),
    )
    print(f"Exported portable model: {destination}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "models" / "hearhere_panns_lr.joblib")
    parser.add_argument("--destination", type=Path, default=root / "models" / "hearhere_panns_lr_portable.npz")
    args = parser.parse_args()
    export(args.source.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
