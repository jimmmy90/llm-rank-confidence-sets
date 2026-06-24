#!/usr/bin/env python3
"""Rank confidence sets (Mogstad–Romano–Shaikh–Wilhelm) for an LLM benchmark ranking.

Setup. n questions, p models. X is the n×p per-item score matrix,
X[i, j] ∈ [0, 1]  (1 = model j correct on question i; fractional if averaged
over epochs). Define

    θ̂ = X̄                                   (column means; accuracy vector)
    Σ̂ = Cov(X̄) = 1/(n(n-1)) Σ_i (X_i-X̄)(X_i-X̄)ᵀ   (covariance of the MEAN vector)

A parametric bootstrap draws Z ~ N(0, Σ̂). The rank confidence set for model j is

    { |N⁻_j| + 1, …, p − |N⁺_j| }

where N⁻_j / N⁺_j are the competitors significantly better / worse than j.

Coverage note: this quantifies *item-sampling* uncertainty only (the benchmark
questions as a sample from a question population). It does NOT capture
generation stochasticity or prompt sensitivity. State that when reporting.

Reference: Mogstad, Romano, Shaikh & Wilhelm, "Inference for Ranks with
Applications to Mobility across Neighbourhoods and Academic Achievement across
Countries", Review of Economic Studies 91(1), 2024.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 (+2): point estimate and the FULL covariance of the accuracy vector
# ─────────────────────────────────────────────────────────────────────────────
def estimate_theta_and_cov(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """θ̂ = column means of X;  Σ̂ = covariance of the MEAN vector X̄.

    Parameters
    ----------
    X : (n, p) array — per-item score matrix.

    Returns
    -------
    theta : (p,)   — θ̂_j = (1/n) Σ_i X[i, j]
    Sigma : (p, p) — Σ̂ = 1/(n(n-1)) Σ_i (X_i − X̄)(X_i − X̄)ᵀ
                     i.e. the sample covariance of the ROWS of X, divided by n.
                     The 1/n is folded in: Sigma IS Cov(X̄), used directly
                     downstream — do NOT divide by n again.
    """
    n = X.shape[0]
    theta = X.mean(axis=0)
    centered = X - theta
    Sigma = centered.T @ centered / (n * (n - 1))  # == Cov(X̄); the 1/n is folded in
    return theta, Sigma


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: standard error of every pairwise difference (the shared-item gain)
# ─────────────────────────────────────────────────────────────────────────────
def pairwise_diff_se(Sigma: np.ndarray) -> np.ndarray:
    """se_{jk} = sqrt( Σ_jj + Σ_kk − 2·Σ_jk )  for every pair (j, k).

    Parameters
    ----------
    Sigma : (p, p) — covariance of X̄ from `estimate_theta_and_cov`.

    Returns
    -------
    SE : (p, p) — SE[j, k] = standard error of (θ̂_j − θ̂_k); diagonal = 0.

    The −2·Σ_jk term is the whole point: because all models answer the SAME
    questions, θ̂_j and θ̂_k are positively correlated, so the difference
    variance is SMALLER than the independence value Σ_jj + Σ_kk. Clip tiny
    negative values (floating-point noise) up to 0 before the sqrt.
    """
    d = np.diag(Sigma)
    var = d[:, None] + d[None, :] - 2.0 * Sigma  # Σ_jj + Σ_kk − 2Σ_jk
    return np.sqrt(np.clip(var, 0.0, None))


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: parametric-bootstrap critical value(s)
# ─────────────────────────────────────────────────────────────────────────────
def critical_values(
    Sigma: np.ndarray,
    se_jk: np.ndarray,
    alpha: float,
    R: int,
    simultaneous: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap critical value(s) for the standardized pairwise statistic.

    Draw Z⁽¹…R⁾ ~ N(0, Σ). For each draw form  T_{jk} = |Z_j − Z_k| / se_{jk}
    (the k = j entries are ignored). Then:

      marginal      : for each j,  c_j = (1−alpha) quantile over draws of
                      max_{k≠j} T_{jk}                      → array shape (p,)
      simultaneous  : a single c = (1−alpha) quantile over draws of
                      max_j max_{k≠j} T_{jk}, broadcast to  → array shape (p,)

    Returns
    -------
    c : (p,) — critical value to use for each target j.
    """
    p = Sigma.shape[0]

    # ---- provided: draw the Gaussian bootstrap sample (svd handles singular Σ)
    Z = rng.multivariate_normal(
        np.zeros(p), Sigma, size=R, method="svd", check_valid="ignore"
    )  # (R, p)

    # ---- provided: standardized |Z_j − Z_k| / se_jk, with degenerate pairs out
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.abs(Z[:, :, None] - Z[:, None, :]) / se_jk[None, :, :]  # (R, p, p)
    # Ignore k = j (se = 0 on the diagonal) AND any degenerate pair with se = 0
    # (identical / zero-variance columns): such pairs carry no sampling
    # uncertainty and are resolved deterministically in `rank_set` (half-width
    # se·c = 0), so they must never win — or pollute — the bootstrap max here.
    T[:, se_jk <= 0.0] = -np.inf

    if simultaneous:
        allmax = T.max(axis=(1, 2))  # (R,) max over BOTH j and k per draw
        return np.full(p, np.quantile(allmax, 1 - alpha))
    maxk = T.max(axis=2)  # (R, p) max over competitors k for each target j
    return np.quantile(maxk, 1 - alpha, axis=0)  # (p,)


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4 (+5): turn critical values into a rank confidence set per model
# ─────────────────────────────────────────────────────────────────────────────
def rank_set(
    theta: np.ndarray, se_jk: np.ndarray, c: np.ndarray
) -> list[tuple[int, int]]:
    """Rank confidence set for every model (1 = best).

    For target j and competitor k (k ≠ j) the difference CI is
        (θ_j − θ_k) ± se_jk[j, k] · c[j].
    Count
        N⁻_j = #{ k ≠ j : CI entirely below 0 }   (k significantly BETTER than j)
        N⁺_j = #{ k ≠ j : CI entirely above 0 }   (k significantly WORSE  than j)
    The rank set of j is the integer interval
        [ N⁻_j + 1 ,  p − N⁺_j ].

    Parameters
    ----------
    theta : (p,)   accuracy vector
    se_jk : (p, p) pairwise difference SEs
    c     : (p,)   critical values (one per target j)

    Returns
    -------
    list of (best_rank, worst_rank), one tuple per model, 1-indexed.

    Notes:
      • exclude k = j from both counts;
      • "CI entirely below 0"  ⇔  (θ_j − θ_k) + se_jk[j, k]·c[j] < 0;
      • "CI entirely above 0"  ⇔  (θ_j − θ_k) − se_jk[j, k]·c[j] > 0.
    """
    p = len(theta)
    out: list[tuple[int, int]] = []
    for j in range(p):
        n_minus = n_plus = 0
        for k in range(p):
            if k == j:
                continue
            diff = theta[j] - theta[k]
            half = se_jk[j, k] * c[j]
            if diff + half < 0:      # CI entirely below 0 → k significantly better
                n_minus += 1
            elif diff - half > 0:    # CI entirely above 0 → k significantly worse
                n_plus += 1
        out.append((n_minus + 1, p - n_plus))
    return out


def wilson_ci(p_hat: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion — the per-model accuracy column.

    This is a SEPARATE object from the rank set (a marginal CI on one model's
    accuracy), shown only for readability. Ranks are decided by `rank_set`.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class RankResult:
    models: list[str]
    n: int
    theta: np.ndarray  # (p,) accuracy
    acc_ci: list[tuple[float, float]]  # Wilson CI per model
    naive: list[tuple[int, int]]  # naive baseline rank set (see naive_rank_sets)
    marginal: list[tuple[int, int]]  # MRSW marginal rank set per model
    simultaneous: list[tuple[int, int]]  # MRSW simultaneous rank set
    se_jk: np.ndarray  # (p, p) MRSW pairwise diff SE (paired, with -2*Sigma_jk)
    se_indep: np.ndarray  # (p, p) naive diff SE (independence: sqrt(Sigma_jj+Sigma_kk))
    c_marginal: np.ndarray  # (p,)
    c_simultaneous: np.ndarray  # (p,)
    z_naive: float  # fixed normal quantile used by the naive baseline
    notes: list[str]


def naive_rank_sets(
    theta: np.ndarray, Sigma: np.ndarray, alpha: float
) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    """The 'traditional' baseline we contrast MRSW against.

    Two deliberate simplifications vs MRSW, so the comparison isolates exactly
    what MRSW buys:
      1. INDEPENDENCE — difference SE = sqrt(Sigma_jj + Sigma_kk), dropping the
         shared-item covariance term -2*Sigma_jk (i.e. pretends models were
         evaluated on disjoint question samples).
      2. NO multiple-comparison correction — uses the fixed marginal normal
         quantile z_{1-alpha/2} (≈1.96) for every pairwise test, instead of the
         bootstrap max-statistic critical value that controls error across all
         p-1 comparisons simultaneously.
    This is essentially the common "do the two models' marginal CIs overlap?"
    rule. Reuses `rank_set`, so it is apples-to-apples with the MRSW sets.
    Returns (rank_sets, se_indep, z).
    """
    d = np.diag(Sigma)
    se_indep = np.sqrt(np.clip(d[:, None] + d[None, :], 0.0, None))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = len(theta)
    return rank_set(theta, se_indep, np.full(p, z)), se_indep, z


def _finite(c: np.ndarray) -> np.ndarray:
    """Clamp non-finite critical values to 0 (a target whose every pair is
    degenerate has an empty bootstrap max → -inf quantile); 0 means 'resolve
    only by the deterministic half-width = 0 rule in rank_set'."""
    return np.where(np.isfinite(c), c, 0.0)


def rank_confidence_sets(
    X: np.ndarray,
    models: list[str],
    alpha: float = 0.05,
    R: int = 2000,
    seed: int = 0,
) -> RankResult:
    """Orchestrator: chain steps 1→4 and assemble both marginal and
    simultaneous rank confidence sets."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    notes: list[str] = []
    if p < 2:
        raise ValueError("rank confidence sets need at least p = 2 models")

    theta, Sigma = estimate_theta_and_cov(X)
    se_jk = pairwise_diff_se(Sigma)

    off = ~np.eye(p, dtype=bool)
    if np.any((se_jk <= 0) & off):
        notes.append(
            "Some model pairs have zero score variance (identical or constant "
            "columns); those pairs were resolved deterministically by the sign "
            "of the accuracy gap, not by the bootstrap."
        )

    # Same seed for both so the two share one bootstrap sample → comparable.
    c_marg = _finite(critical_values(Sigma, se_jk, alpha, R, False, np.random.default_rng(seed)))
    c_sim = _finite(critical_values(Sigma, se_jk, alpha, R, True, np.random.default_rng(seed)))

    marginal = rank_set(theta, se_jk, c_marg)
    simultaneous = rank_set(theta, se_jk, c_sim)
    naive, se_indep, z_naive = naive_rank_sets(theta, Sigma, alpha)
    acc_ci = [wilson_ci(float(t), n, alpha) for t in theta]

    return RankResult(
        models=list(models),
        n=n,
        theta=theta,
        acc_ci=acc_ci,
        naive=naive,
        marginal=marginal,
        simultaneous=simultaneous,
        se_jk=se_jk,
        se_indep=se_indep,
        c_marginal=c_marg,
        c_simultaneous=c_sim,
        z_naive=z_naive,
        notes=notes,
    )


def tau_best(simultaneous: list[tuple[int, int]], tau: int) -> list[int]:
    """τ-Best confidence set: indices of models that CANNOT be excluded from the
    top τ, derived from the SIMULTANEOUS rank sets (MRSW, simpler method).

    A model qualifies iff its simultaneous best-possible rank (the lower bound)
    is ≤ τ. Equivalent to the one-sided construction (worst rank forced to p,
    keep models whose interval [best, p] contains τ): because the bootstrap maxes
    over all ordered pairs, the one-sided critical value equals the two-sided
    simultaneous one, so the simultaneous lower bound IS the one-sided bound.

    Returns model indices (0-based) sorted best-first.
    """
    return [j for j, (lo, _hi) in enumerate(simultaneous) if lo <= tau]


def tau_worst(simultaneous: list[tuple[int, int]], tau: int, p: int | None = None) -> list[int]:
    """τ-Worst confidence set: models that cannot be excluded from the bottom τ.

    Equivalent to applying τ-best to the negated scores; concretely, a model
    qualifies iff its simultaneous worst-possible rank (upper bound) is ≥ p−τ+1.
    """
    p = p if p is not None else len(simultaneous)
    thr = p - tau + 1
    return [j for j, (_lo, hi) in enumerate(simultaneous) if hi >= thr]
