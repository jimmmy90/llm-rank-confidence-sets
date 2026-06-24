#!/usr/bin/env python3
"""Property-based self-checks for the MRSW rank confidence sets in `mrsw.py`.

Run:  python test_mrsw.py
Each check prints PASS/FAIL with a short explanation of what it verifies.
"""
from __future__ import annotations

import numpy as np

import mrsw


def _bernoulli(rng, p, n):
    return (rng.random(n) < p).astype(float)


def check_theta_and_cov():
    rng = np.random.default_rng(0)
    X = rng.random((150, 3))
    theta, Sigma = mrsw.estimate_theta_and_cov(X)
    n = X.shape[0]
    assert np.allclose(theta, X.mean(0)), "theta must equal column means"
    expected = np.cov(X, rowvar=False, bias=False) / n  # Cov(X̄) = S/n
    assert np.allclose(Sigma, expected), "Sigma must be Cov(X̄) = sample-cov / n"
    assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric"
    print("PASS  estimate_theta_and_cov: θ̂ = means, Σ̂ = Cov(X̄) = S/n, symmetric")


def check_pairwise_diff_se():
    # Two positively-correlated columns: full-cov SE must beat the independence SE.
    rng = np.random.default_rng(1)
    base = _bernoulli(rng, 0.7, 400)
    flip = rng.random(400) < 0.1
    a = base.copy()
    b = np.where(flip, 1 - base, base)  # correlated with a, not identical
    X = np.column_stack([a, b])
    _, Sigma = mrsw.estimate_theta_and_cov(X)
    SE = mrsw.pairwise_diff_se(Sigma)
    assert np.allclose(np.diag(SE), 0.0), "diagonal SE must be 0"
    assert np.allclose(SE, SE.T), "SE must be symmetric"
    indep = np.sqrt(Sigma[0, 0] + Sigma[1, 1])
    assert SE[0, 1] < indep, (
        f"shared-item SE ({SE[0,1]:.5f}) must be < independence SE ({indep:.5f}) "
        "— the −2Σ_jk term is missing or wrong"
    )
    print(f"PASS  pairwise_diff_se: shared-item SE {SE[0,1]:.5f} < independence {indep:.5f}")


def check_dominance():
    # Model 0 correct on every item → must be ranked exactly 1st.
    rng = np.random.default_rng(2)
    n = 300
    X = np.column_stack([
        np.ones(n),
        _bernoulli(rng, 0.70, n),
        _bernoulli(rng, 0.45, n),
    ])
    res = mrsw.rank_confidence_sets(X, ["dom", "mid", "low"], R=3000, seed=7)
    assert res.marginal[0] == (1, 1), f"dominant model must be rank {{1}}, got {res.marginal[0]}"
    print(f"PASS  dominance: all-correct model → rank set {res.marginal[0]}")


def check_ties():
    # Models 0 and 1 identical, both clearly above model 2.
    rng = np.random.default_rng(3)
    n = 300
    col = _bernoulli(rng, 0.78, n)
    X = np.column_stack([col, col.copy(), _bernoulli(rng, 0.40, n)])
    res = mrsw.rank_confidence_sets(X, ["a", "b", "c"], R=3000, seed=7)
    assert res.marginal[0] == res.marginal[1], (
        f"identical models must share a rank set, got {res.marginal[0]} vs {res.marginal[1]}"
    )
    lo, hi = res.marginal[0]
    assert lo <= 1 <= hi and lo <= 2 <= hi, (
        f"tied top pair should admit ranks 1 and 2, got {res.marginal[0]}"
    )
    assert res.marginal[2][0] >= 2, f"clearly-worst model should not be rank 1, got {res.marginal[2]}"
    print(f"PASS  ties: identical pair share set {res.marginal[0]}, worst = {res.marginal[2]}")


def check_simultaneous_wider():
    # Simultaneous sets must contain the marginal sets (larger critical value).
    rng = np.random.default_rng(4)
    n = 250
    X = np.column_stack([
        _bernoulli(rng, 0.80, n),
        _bernoulli(rng, 0.76, n),
        _bernoulli(rng, 0.72, n),
        _bernoulli(rng, 0.55, n),
    ])
    res = mrsw.rank_confidence_sets(X, ["m1", "m2", "m3", "m4"], R=4000, seed=11)
    assert np.all(res.c_simultaneous >= res.c_marginal - 1e-9), (
        "simultaneous critical value must be ≥ marginal"
    )
    for j, ((bm, wm), (bs, ws)) in enumerate(zip(res.marginal, res.simultaneous)):
        assert bs <= bm and ws >= wm, (
            f"model {j}: simultaneous {(bs,ws)} must contain marginal {(bm,wm)}"
        )
    print("PASS  simultaneous ⊇ marginal for every model")


def check_reproducible():
    rng = np.random.default_rng(5)
    X = (rng.random((200, 4)) < 0.7).astype(float)
    r1 = mrsw.rank_confidence_sets(X, list("abcd"), R=2000, seed=42)
    r2 = mrsw.rank_confidence_sets(X, list("abcd"), R=2000, seed=42)
    assert r1.marginal == r2.marginal and r1.simultaneous == r2.simultaneous
    print("PASS  reproducible: same seed → identical rank sets")


def main() -> int:
    checks = [
        check_theta_and_cov,
        check_pairwise_diff_se,
        check_dominance,
        check_ties,
        check_simultaneous_wider,
        check_reproducible,
    ]
    failed = 0
    for c in checks:
        try:
            c()
        except AssertionError as e:
            print(f"FAIL  {c.__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed} check(s) failed.")
        return 1
    print("All checks passed ✔  — rank confidence sets are correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
