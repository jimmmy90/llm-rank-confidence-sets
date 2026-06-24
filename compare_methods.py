#!/usr/bin/env python3
"""Compare three rank-inference methods on the real eval logs, isolating each
ingredient, and write outputs/paper/method_comparison.md.

  (1) Naive            : independent marginal CIs + fixed z_{1-a/2}, no multiple-
                         comparison correction  ("just look at se / overlapping CIs")
  (2) MRSW independent : bootstrap multiple-comparison critical value, but ASSUMES
                         accuracy rates independent (diagonal covariance; se_jk =
                         sqrt(Sig_jj+Sig_kk))           <-- the "independence" experiment
  (3) MRSW full        : (2) + the shared-item covariance term -2*Sig_jk (our method)

Decomposition:  naive --[+MC correction]--> MRSW indep --[+shared-item cov]--> MRSW full
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import mrsw
from analyze import build_score_matrix, collect_runs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ALPHA, R, SEED = 0.05, 2000, 0


_RELABEL = {"deepseek-chat": "deepseek-v3"}  # OpenRouter slug serves DeepSeek V3
def short(m):
    name = m.split("/")[-1]
    return _RELABEL.get(name, name)
def fmt(p): return f"{p[0]}" if p[0] == p[1] else f"{p[0]}-{p[1]}"
def width(p): return p[1] - p[0] + 1


def mrsw_sets(theta, Sigma):
    """Return (marginal, simultaneous) rank sets for a given covariance matrix."""
    se = mrsw.pairwise_diff_se(Sigma)
    cm = mrsw._finite(mrsw.critical_values(Sigma, se, ALPHA, R, False, np.random.default_rng(SEED)))
    cs = mrsw._finite(mrsw.critical_values(Sigma, se, ALPHA, R, True, np.random.default_rng(SEED)))
    return mrsw.rank_set(theta, se, cm), mrsw.rank_set(theta, se, cs)


def summary(sets, p):
    sing = sum(1 for s in sets if s[0] == s[1])
    return sing, sum(width(s) for s in sets) / p


def main():
    runs = collect_runs()
    X, models = build_score_matrix([r["model"] for r in runs])
    n, p = X.shape
    theta, Sigma = mrsw.estimate_theta_and_cov(X)
    Sigma_diag = np.diag(np.diag(Sigma))  # independence assumption

    # average pairwise correlation of per-item correctness (justifies the cov term)
    C = np.corrcoef(X, rowvar=False)
    off = C[~np.eye(p, dtype=bool)]
    avg_corr = off.mean()

    naive, _, z = mrsw.naive_rank_sets(theta, Sigma, ALPHA)
    indep_m, indep_s = mrsw_sets(theta, Sigma_diag)
    full_m, full_s = mrsw_sets(theta, Sigma)

    order = list(range(p))  # already accuracy-desc from collect_runs
    L = [
        "# Method comparison — rank sets on MMLU-Pro\n",
        f"n={n} questions, p={p} models, alpha={ALPHA}, R={R}. "
        f"Mean pairwise correlation of per-item correctness across models: "
        f"**{avg_corr:.3f}** (positive ⇒ the shared-item term −2Σ_jk matters).\n",
        "| # | Model | Acc % | Naive | Indep marginal | Indep simult. | Full marginal | Full simult. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in order:
        L.append(
            f"| {i+1} | {short(models[i])} | {theta[i]*100:.1f} | {fmt(naive[i])} "
            f"| {fmt(indep_m[i])} | {fmt(indep_s[i])} "
            f"| {fmt(full_m[i])} | {fmt(full_s[i])} |"
        )
    # discrimination summary
    rows = [
        ("Naive (indep CI, no MC)", *summary(naive, p)),
        ("MRSW independent — marginal", *summary(indep_m, p)),
        ("MRSW independent — simultaneous", *summary(indep_s, p)),
        ("MRSW full — marginal", *summary(full_m, p)),
        ("MRSW full — simultaneous", *summary(full_s, p)),
    ]
    L += [
        "\n## Resolving power (singletons = models pinned to one exact rank)\n",
        "| Method | Singletons | Avg width |",
        "|---|---|---|",
    ]
    for name, s, w in rows:
        L.append(f"| {name} | {s}/{p} | {w:.2f} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "method_comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\navg_corr={avg_corr:.4f}  z_naive={z:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
