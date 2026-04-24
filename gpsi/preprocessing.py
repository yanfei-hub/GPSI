"""Data processing utilities corresponding to Figure 1A."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    missing_threshold: float = 0.4,
    variance_threshold: float = 0.0,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Select features, impute missing values, and Z-score normalize.

    Parameters
    ----------
    frame:
        Input table with one row per subject.
    feature_cols:
        Numeric feature columns used by GPSI.
    missing_threshold:
        Drop features with missing fraction above this threshold.
    variance_threshold:
        Drop near-constant features after imputation.
    """
    X_raw = frame[feature_cols].copy()
    missing_fraction = X_raw.isna().mean()
    kept_missing = missing_fraction[missing_fraction <= missing_threshold].index.tolist()
    if not kept_missing:
        raise RuntimeError("No features remained after missingness filtering.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw[kept_missing])

    selector = VarianceThreshold(threshold=variance_threshold)
    X_selected = selector.fit_transform(X_imputed)
    selected_cols = [col for col, keep in zip(kept_missing, selector.get_support()) if keep]
    if not selected_cols:
        raise RuntimeError("No features remained after variance filtering.")

    X_scaled = StandardScaler().fit_transform(X_selected).astype(np.float32)
    meta = {
        "n_features_raw": float(len(feature_cols)),
        "n_features_after_missing_filter": float(len(kept_missing)),
        "n_features_after_variance_filter": float(len(selected_cols)),
        "missing_threshold": float(missing_threshold),
        "variance_threshold": float(variance_threshold),
        "n_imputed_values": float(np.isnan(X_raw[kept_missing].to_numpy(dtype=np.float64)).sum()),
    }
    return X_scaled, selected_cols, meta

