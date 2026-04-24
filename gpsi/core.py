"""Core GPSI implementation.

This module is data-source agnostic. It expects a numeric matrix `X` with one
row per subject and performs:

1. k-means subgroup estimation with elbow-based subgroup selection.
2. Per-subgroup Gaussian Process for Stage Inference.
3. Metropolis-Hastings proposals for latent stages with MAP hyperparameter
   updates, following the Figure 1C workflow in the GPSI paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class KSelectionResult:
    best_k: int
    score_df: pd.DataFrame
    labels: np.ndarray


@dataclass
class StageResult:
    pseudo_stage: np.ndarray
    stage_bin: np.ndarray
    metadata: dict[str, dict[str, float]]


def select_kmeans_clusters(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    seed: int = 42,
    metric_sample_size: int = 8000,
    n_init: int = 25,
) -> KSelectionResult:
    """Select number of subgroups using an elbow score.

    Silhouette, Davies-Bouldin, and Calinski-Harabasz scores are reported as
    diagnostics, but `best_k` is chosen by the elbow strength.
    """
    rng = np.random.default_rng(seed)
    metric_idx = np.arange(X.shape[0])
    if X.shape[0] > metric_sample_size:
        metric_idx = rng.choice(X.shape[0], size=metric_sample_size, replace=False)

    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    max_k = min(k_max, X.shape[0] - 1)
    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        labels = km.fit_predict(X)
        labels_by_k[k] = labels
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X[metric_idx], labels[metric_idx])),
                "davies_bouldin": float(davies_bouldin_score(X[metric_idx], labels[metric_idx])),
                "calinski_harabasz": float(calinski_harabasz_score(X[metric_idx], labels[metric_idx])),
                "metric_n": int(len(metric_idx)),
            }
        )

    score_df = pd.DataFrame(rows)
    inertia = score_df["inertia"].to_numpy()
    if len(inertia) >= 3:
        x = score_df["k"].to_numpy(dtype=float)
        line = np.interp(x, [x[0], x[-1]], [inertia[0], inertia[-1]])
        elbow = (line - inertia) / max(float(line.max() - inertia.min()), 1e-8)
    else:
        elbow = np.zeros(len(score_df))
    score_df["elbow_strength"] = elbow
    score_df["selection_score"] = score_df["elbow_strength"]

    if float(score_df["elbow_strength"].max()) <= 0:
        best_k = int(score_df.iloc[0]["k"])
    else:
        best_k = int(score_df.loc[score_df["elbow_strength"].idxmax(), "k"])
    return KSelectionResult(best_k=best_k, score_df=score_df, labels=labels_by_k[best_k].astype(np.int16))


def _prepare_gp_observations(X: np.ndarray, seed: int, n_pca_features: int) -> np.ndarray:
    n_comp = min(n_pca_features, X.shape[1], max(1, X.shape[0] - 1))
    Y = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    return StandardScaler().fit_transform(Y).astype(np.float64)


def _initial_stage_logits(Y: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(Y[:, 0])).astype(np.float64)
    s = (order + 0.5) / max(float(len(order)), 1.0)
    return np.log(s / (1.0 - s))


def _rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _clip_stage(stage: np.ndarray) -> np.ndarray:
    return np.clip(stage.astype(np.float32), 1e-6, 1.0)


def _log_gp_marginal(Y: np.ndarray, s: np.ndarray, lambda_j2: np.ndarray, sigma_0j2: np.ndarray) -> float:
    n, p = Y.shape
    dist2 = (s[:, None] - s[None, :]) ** 2
    eye = np.eye(n)
    constant = n * math.log(2.0 * math.pi)
    total = 0.0
    for j in range(p):
        K = np.exp(-lambda_j2[j] * dist2) + (sigma_0j2[j] + 1e-6) * eye
        try:
            chol = np.linalg.cholesky(K)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, Y[:, j]))
        except np.linalg.LinAlgError:
            return -math.inf
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        total += -0.5 * (Y[:, j] @ alpha + logdet + constant)
    return float(total)


class GPSIMH:
    """Metropolis-Hastings GPSI estimator with hyperparameter updates."""

    def __init__(
        self,
        n_pca_features: int = 12,
        iterations: int = 1200,
        burn_in: int = 400,
        alpha: float = 2.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        sigma_s2: float = 0.0064,
        hyper_steps: int = 3,
        hyper_lr: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.n_pca_features = n_pca_features
        self.iterations = iterations
        self.burn_in = burn_in
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_s2 = sigma_s2
        self.hyper_steps = hyper_steps
        self.hyper_lr = hyper_lr
        self.seed = seed

    def _log_posterior_s(
        self,
        Y: np.ndarray,
        logits_s: np.ndarray,
        lambda_j2: np.ndarray,
        sigma_0j2: np.ndarray,
    ) -> float:
        s = 1.0 / (1.0 + np.exp(-logits_s))
        logp = _log_gp_marginal(Y, s, lambda_j2, sigma_0j2)
        if not np.isfinite(logp):
            return -math.inf
        # Uniform prior over s in [0, 1], with logit proposal Jacobian.
        return float(logp + np.sum(np.log(s) + np.log1p(-s)))

    def _hyperparameter_objective(
        self,
        Y: torch.Tensor,
        s: torch.Tensor,
        log_lambda_j2: torch.Tensor,
        log_sigma_0j2: torch.Tensor,
    ) -> torch.Tensor:
        n, p = Y.shape
        dist2 = (s[:, None] - s[None, :]).pow(2)
        eye = torch.eye(n, dtype=Y.dtype)
        lambda_j2 = torch.exp(log_lambda_j2)
        sigma_0j2 = torch.exp(log_sigma_0j2)
        nll = torch.tensor(0.0, dtype=Y.dtype)
        constant = n * math.log(2.0 * math.pi)
        for j in range(p):
            K = torch.exp(-lambda_j2[j] * dist2) + (sigma_0j2[j] + 1e-6) * eye
            chol = torch.linalg.cholesky(K)
            alpha_sol = torch.cholesky_solve(Y[:, [j]], chol).squeeze(1)
            logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
            nll = nll + 0.5 * (Y[:, j] @ alpha_sol + logdet + constant)

        nll = nll - torch.sum(torch.log(torch.tensor(self.gamma, dtype=Y.dtype)) - self.gamma * lambda_j2 + log_lambda_j2)
        nll = nll - torch.sum(
            self.alpha * math.log(self.beta)
            - math.lgamma(self.alpha)
            - (self.alpha + 1.0) * log_sigma_0j2
            - self.beta / sigma_0j2
            + log_sigma_0j2
        )
        return nll

    def _update_hyperparameters(
        self,
        Y_np: np.ndarray,
        logits_s: np.ndarray,
        log_lambda_j2: np.ndarray,
        log_sigma_0j2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        dtype = torch.float64
        Y = torch.tensor(Y_np, dtype=dtype)
        s = torch.tensor(1.0 / (1.0 + np.exp(-logits_s)), dtype=dtype)
        lambda_param = torch.tensor(log_lambda_j2, dtype=dtype, requires_grad=True)
        sigma_param = torch.tensor(log_sigma_0j2, dtype=dtype, requires_grad=True)
        opt = torch.optim.Adam([lambda_param, sigma_param], lr=self.hyper_lr)
        loss_value = math.nan
        for _ in range(self.hyper_steps):
            opt.zero_grad()
            loss = self._hyperparameter_objective(Y, s, lambda_param, sigma_param)
            loss.backward()
            opt.step()
            with torch.no_grad():
                lambda_param.clamp_(math.log(1e-3), math.log(1e3))
                sigma_param.clamp_(math.log(1e-5), math.log(10.0))
            loss_value = float(loss.detach().cpu())
        return lambda_param.detach().cpu().numpy(), sigma_param.detach().cpu().numpy(), loss_value

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        Y = _prepare_gp_observations(X, seed=self.seed, n_pca_features=self.n_pca_features)
        p = Y.shape[1]
        logits_s = _initial_stage_logits(Y)
        log_lambda_j2 = np.zeros(p, dtype=np.float64)
        log_sigma_0j2 = np.full(p, -1.2, dtype=np.float64)
        current = self._log_posterior_s(Y, logits_s, np.exp(log_lambda_j2), np.exp(log_sigma_0j2))

        samples = []
        accepted = 0
        last_hyper_loss = math.nan
        proposal_sd = math.sqrt(self.sigma_s2)
        for it in range(self.iterations):
            prop_s = logits_s + rng.normal(0.0, proposal_sd, size=logits_s.shape)
            proposed = self._log_posterior_s(Y, prop_s, np.exp(log_lambda_j2), np.exp(log_sigma_0j2))
            if np.log(rng.uniform()) < proposed - current:
                logits_s = prop_s
                current = proposed
                accepted += 1

            log_lambda_j2, log_sigma_0j2, last_hyper_loss = self._update_hyperparameters(
                Y, logits_s, log_lambda_j2, log_sigma_0j2
            )
            current = self._log_posterior_s(Y, logits_s, np.exp(log_lambda_j2), np.exp(log_sigma_0j2))
            if it >= self.burn_in:
                samples.append(1.0 / (1.0 + np.exp(-logits_s)))

        if not samples:
            samples.append(1.0 / (1.0 + np.exp(-logits_s)))
        stage = _clip_stage(np.mean(np.vstack(samples), axis=0))
        return stage, {
            "method": "mh_hyperupdate",
            "n_fit_patients": float(X.shape[0]),
            "n_pca_features": float(Y.shape[1]),
            "iterations": float(self.iterations),
            "burn_in": float(self.burn_in),
            "acceptance_rate": float(accepted / max(self.iterations, 1)),
            "final_log_posterior": float(current),
            "final_expected_complete_loglik_neg": float(last_hyper_loss),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "sigma_s2": float(self.sigma_s2),
            "lambda_j2_mean": float(np.exp(log_lambda_j2).mean()),
            "sigma_0j2_mean": float(np.exp(log_sigma_0j2).mean()),
        }


def fit_gpsi_stage_mh(
    X: np.ndarray,
    seed: int = 42,
    n_pca_features: int = 12,
    iterations: int = 1200,
    burn_in: int = 400,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convenience wrapper around :class:`GPSIMH`."""
    return GPSIMH(n_pca_features=n_pca_features, iterations=iterations, burn_in=burn_in, seed=seed).fit(X)


def assign_gpsi_stages_by_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    stage_bins: int = 15,
    max_fit_subjects: int = 160,
    n_pca_features: int = 12,
    mh_iterations: int = 800,
    mh_burn_in: int = 250,
) -> StageResult:
    """Fit GPSI stages inside each subgroup and return one stage per row."""
    rng = np.random.default_rng(seed)
    pseudo_stage = np.full(X.shape[0], np.nan, dtype=np.float32)
    stage_bin = np.full(X.shape[0], -1, dtype=np.int16)
    metadata: dict[str, dict[str, float]] = {}

    for cluster in sorted(np.unique(labels)):
        idx = np.where(labels == cluster)[0]
        if len(idx) < 8:
            pseudo_stage[idx] = _rank01(np.arange(len(idx)))
            metadata[str(int(cluster))] = {"n_subjects": float(len(idx)), "skipped_small_cluster": 1.0}
            stage_bin[idx] = np.clip(np.ceil(pseudo_stage[idx] * stage_bins), 1, stage_bins).astype(np.int16)
            continue

        fit_idx_local = np.arange(len(idx))
        if len(idx) > max_fit_subjects:
            fit_idx_local = rng.choice(len(idx), size=max_fit_subjects, replace=False)
        fit_idx = idx[fit_idx_local]
        fit_stage, meta = fit_gpsi_stage_mh(
            X[fit_idx],
            seed=seed + int(cluster),
            n_pca_features=n_pca_features,
            iterations=mh_iterations,
            burn_in=mh_burn_in,
        )

        if len(fit_idx) == len(idx):
            pseudo_stage[idx] = fit_stage
        else:
            nn = NearestNeighbors(n_neighbors=min(15, len(fit_idx)), metric="euclidean")
            nn.fit(X[fit_idx])
            distances, nn_idx = nn.kneighbors(X[idx])
            weights = 1.0 / (distances + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)
            pseudo_stage[idx] = (weights * fit_stage[nn_idx]).sum(axis=1).astype(np.float32)

        stage_bin[idx] = np.clip(np.ceil(pseudo_stage[idx] * stage_bins), 1, stage_bins).astype(np.int16)
        meta["n_subjects"] = float(len(idx))
        metadata[str(int(cluster))] = meta

    return StageResult(pseudo_stage=pseudo_stage, stage_bin=stage_bin, metadata=metadata)
