"""Synthetic data for GPSI examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_gpsi_data(
    n_subjects: int = 240,
    n_features: int = 8,
    n_subgroups: int = 2,
    noise: float = 0.12,
    missing_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate unordered cross-sectional subjects with latent progression stages."""
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, n_subgroups, size=n_subjects)
    stage = rng.uniform(0.0, 1.0, size=n_subjects)
    X = np.zeros((n_subjects, n_features), dtype=np.float64)

    for j in range(n_features):
        phase = 0.4 * j
        for c in range(n_subgroups):
            mask = clusters == c
            s = stage[mask]
            if c % 2 == 0:
                signal = np.sin(np.pi * (s + phase / n_features)) + 0.35 * s
            else:
                signal = np.cos(np.pi * (s + phase / n_features)) - 0.25 * s
            X[mask, j] = signal + rng.normal(0.0, noise, size=mask.sum())

    missing = rng.uniform(size=X.shape) < missing_rate
    X[missing] = np.nan

    frame = pd.DataFrame(X, columns=[f"feature_{j + 1}" for j in range(n_features)])
    frame.insert(0, "true_stage", stage)
    frame.insert(0, "true_cluster", clusters)
    frame.insert(0, "subject_id", [f"S{i + 1:04d}" for i in range(n_subjects)])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    data = simulate_gpsi_data()
    data.to_csv("data/simulated/simulated_gpsi_data.csv", index=False)


if __name__ == "__main__":
    main()
