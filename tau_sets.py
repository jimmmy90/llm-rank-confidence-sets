#!/usr/bin/env python3
"""τ-Best / τ-Worst confidence sets on the real eval logs (default τ=3 → top-3 /
bottom-3). Built from the MRSW SIMULTANEOUS rank sets. Writes outputs/paper/tau_sets.md.

    python tau_sets.py [--tau 3] [--alpha 0.05] [--R 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mrsw
from analyze import build_score_matrix, collect_runs

OUT = Path(__file__).resolve().parent / "outputs"


_RELABEL = {"deepseek-chat": "deepseek-v3"}  # OpenRouter slug serves DeepSeek V3
def short(m):
    name = m.split("/")[-1]
    return _RELABEL.get(name, name)
def fmt(p): return f"{p[0]}" if p[0] == p[1] else f"{p[0]}–{p[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = collect_runs()
    X, models = build_score_matrix([r["model"] for r in runs])
    res = mrsw.rank_confidence_sets(X, models, alpha=a.alpha, R=a.R, seed=a.seed)
    p = len(models)
    conf = int(round((1 - a.alpha) * 100))

    best = mrsw.tau_best(res.simultaneous, a.tau)
    worst = mrsw.tau_worst(res.simultaneous, a.tau, p)

    L = [
        f"# τ-Best / τ-Worst confidence sets (τ={a.tau}) — MMLU-Pro\n",
        f"{conf}% **simultaneous** confidence sets for membership in the top-{a.tau} and "
        f"bottom-{a.tau} (n={res.n} questions, p={p} models). A model is included iff it "
        f"cannot be statistically excluded from that tier under joint coverage.\n",
        f"## Top-{a.tau} (τ-Best): {len(best)} candidates for {a.tau} slots\n",
        f"Models whose simultaneous best-possible rank ≤ {a.tau}:\n",
        "| Model | Acc % | Simult. rank set | Best rank |",
        "|---|---|---|---|",
    ]
    for j in best:
        L.append(f"| {short(models[j])} | {res.theta[j]*100:.1f} | {fmt(res.simultaneous[j])} | {res.simultaneous[j][0]} |")
    L += [
        f"\n## Bottom-{a.tau} (τ-Worst): {len(worst)} candidates for {a.tau} slots\n",
        f"Models whose simultaneous worst-possible rank ≥ {p - a.tau + 1}:\n",
        "| Model | Acc % | Simult. rank set | Worst rank |",
        "|---|---|---|---|",
    ]
    for j in worst:
        L.append(f"| {short(models[j])} | {res.theta[j]*100:.1f} | {fmt(res.simultaneous[j])} | {res.simultaneous[j][1]} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tau_sets.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
