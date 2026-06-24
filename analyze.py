#!/usr/bin/env python3
"""Turn the eval_set logs into a ranking with MRSW rank confidence sets.

Outputs:
    outputs/ranking.csv / ranking.md   -> measured ranking + accuracy CI + rank sets
    outputs/rank_confidence.md         -> MRSW rank confidence sets + pairwise diffs

Usage:
    python analyze.py                         # full analysis (rank CS on, defaults)
    python analyze.py --alpha 0.05 --R 2000   # confidence level / bootstrap draws
    python analyze.py --seed 0                # bootstrap RNG seed (reproducible)
    python analyze.py --no-rankcs             # accuracy ranking only, skip inference
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inspect_ai.log import list_eval_logs, read_eval_log

import mrsw

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "logs"
OUT = ROOT / "outputs"

# Robustly turn a Score.value ("C"/"I"/1.0/...) into a float in [0,1].
try:  # inspect_ai ships the canonical mapper
    from inspect_ai.scorer import value_to_float as _vtf

    _TO_FLOAT = _vtf()
except Exception:  # pragma: no cover - defensive fallback

    def _TO_FLOAT(v):
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}.get(v.upper(), float("nan"))
        return float("nan")


def _accuracy(log) -> float | None:
    """Pull the accuracy metric (0-1) out of an EvalLog's results."""
    results = getattr(log, "results", None)
    if not results or not getattr(results, "scores", None):
        return None
    for score in results.scores:
        metrics = getattr(score, "metrics", {}) or {}
        if "accuracy" in metrics:
            return float(metrics["accuracy"].value)
    return None


def collect_runs() -> list[dict]:
    rows = []
    for info in list_eval_logs(str(LOG_DIR)):
        log = read_eval_log(info, header_only=True)
        if getattr(log, "status", None) != "success":
            continue
        acc = _accuracy(log)
        if acc is None:
            continue
        rows.append(
            {
                "model": log.eval.model,
                "accuracy_pct": round(acc * 100, 1),
                "epochs": getattr(getattr(log.eval, "config", None), "epochs", None),
            }
        )
    # de-dup (resumed runs): keep best accuracy per model
    best: dict[str, dict] = {}
    for r in rows:
        if r["model"] not in best or r["accuracy_pct"] > best[r["model"]]["accuracy_pct"]:
            best[r["model"]] = r
    return sorted(best.values(), key=lambda r: r["accuracy_pct"], reverse=True)


# ── per-item correctness extraction (for the MRSW n×p matrix) ────────────────
def _chosen_log_per_model() -> dict[str, object]:
    """model id -> the successful EvalLogInfo with the most samples."""
    chosen: dict[str, tuple[object, int]] = {}
    for info in list_eval_logs(str(LOG_DIR)):
        log = read_eval_log(info, header_only=True)
        if getattr(log, "status", None) != "success":
            continue
        model = log.eval.model
        nsamp = getattr(log.eval.dataset, "samples", 0) or 0
        if model not in chosen or nsamp > chosen[model][1]:
            chosen[model] = (info, nsamp)
    return {m: info for m, (info, _) in chosen.items()}


def _per_question_scores(log) -> dict[object, float]:
    """question id -> mean score over epochs, for one model's full log."""
    acc: dict[object, list[float]] = {}
    for s in getattr(log, "samples", None) or []:
        scores = getattr(s, "scores", None)
        if not scores:
            continue
        val = None
        for sc in scores.values():  # one scorer for gpqa; take the first usable
            v = _TO_FLOAT(getattr(sc, "value", None))
            if v == v:  # not NaN
                val = v
                break
        if val is None:
            continue
        acc.setdefault(s.id, []).append(val)
    return {qid: float(np.mean(vs)) for qid, vs in acc.items()}


def build_score_matrix(model_order: list[str]) -> tuple[np.ndarray | None, list[str]]:
    """Build the n×p per-question score matrix aligned across models.

    Returns (X, models) where columns follow `model_order` (point-rank order).
    X is None if fewer than 2 models or no shared questions.
    """
    info_by_model = _chosen_log_per_model()
    per_model: dict[str, dict] = {}
    for m in model_order:
        info = info_by_model.get(m)
        if info is None:
            continue
        per_model[m] = _per_question_scores(read_eval_log(info, header_only=False))

    models = [m for m in model_order if per_model.get(m)]
    if len(models) < 2:
        return None, models
    common = sorted(
        set.intersection(*(set(per_model[m]) for m in models)), key=str
    )
    if not common:
        return None, models
    X = np.array([[per_model[m][q] for m in models] for q in common], dtype=float)
    return X, models


# ── formatting helpers ───────────────────────────────────────────────────────
def _fmt_set(pair: tuple[int, int]) -> str:
    lo, hi = pair
    return f"{{{lo}}}" if lo == hi else f"{{{lo}–{hi}}}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def write_ranking(runs: list[dict], rank: dict | None) -> None:
    import csv

    OUT.mkdir(parents=True, exist_ok=True)

    if rank:
        fields = [
            "point_rank", "model", "accuracy_pct", "acc_ci_lo_pct", "acc_ci_hi_pct",
            "rank_naive_lo", "rank_naive_hi",
            "rank_marginal_lo", "rank_marginal_hi", "rank_sim_lo", "rank_sim_hi", "epochs",
        ]
    else:
        fields = ["rank", "model", "accuracy_pct", "epochs"]

    with (OUT / "ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(runs, 1):
            if rank and r["model"] in rank:
                d = rank[r["model"]]
                w.writerow({
                    "point_rank": i, "model": r["model"], "accuracy_pct": r["accuracy_pct"],
                    "acc_ci_lo_pct": round(d["acc_ci"][0] * 100, 1),
                    "acc_ci_hi_pct": round(d["acc_ci"][1] * 100, 1),
                    "rank_naive_lo": d["naive"][0], "rank_naive_hi": d["naive"][1],
                    "rank_marginal_lo": d["marginal"][0], "rank_marginal_hi": d["marginal"][1],
                    "rank_sim_lo": d["simultaneous"][0], "rank_sim_hi": d["simultaneous"][1],
                    "epochs": r["epochs"],
                })
            elif rank:
                w.writerow({"point_rank": i, "model": r["model"], "accuracy_pct": r["accuracy_pct"], "epochs": r["epochs"]})
            else:
                w.writerow({"rank": i, "model": r["model"], "accuracy_pct": r["accuracy_pct"], "epochs": r["epochs"]})

    if rank:
        lines = [
            "# MMLU-Pro — measured ranking with rank confidence sets\n",
            "`Point rank` sorts by accuracy. The rank-set columns give each model's",
            "plausible ranks under three methods: **naive** (independent marginal CIs,",
            "no multiple-comparison correction — the common practice), **MRSW marginal**,",
            "and **MRSW simultaneous**. See rank_confidence.md for the method + contrast.\n",
            "| Point rank | Model | Accuracy % | 95% Acc CI | Rank (naive) | Rank (MRSW marg.) | Rank (MRSW sim.) | Epochs |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(runs, 1):
            d = rank.get(r["model"])
            if d:
                lines.append(
                    f"| {i} | `{r['model']}` | {r['accuracy_pct']} | {_fmt_ci(d['acc_ci'])} "
                    f"| {_fmt_set(d['naive'])} | {_fmt_set(d['marginal'])} | {_fmt_set(d['simultaneous'])} | {r['epochs']} |"
                )
            else:
                lines.append(f"| {i} | `{r['model']}` | {r['accuracy_pct']} | — | — | — | — | {r['epochs']} |")
    else:
        lines = ["# Measured ranking (accuracy)\n", "| Rank | Model | Accuracy % | Epochs |", "|---|---|---|---|"]
        for i, r in enumerate(runs, 1):
            lines.append(f"| {i} | `{r['model']}` | {r['accuracy_pct']} | {r['epochs']} |")
    (OUT / "ranking.md").write_text("\n".join(lines) + "\n")


def _width(pair: tuple[int, int]) -> int:
    return pair[1] - pair[0] + 1


def write_rank_confidence(res: "mrsw.RankResult", alpha: float, R: int) -> None:
    conf = int(round((1 - alpha) * 100))
    lines = [
        "# MMLU-Pro — rank confidence sets (Mogstad–Romano–Shaikh–Wilhelm)\n",
        f"Method: parametric-bootstrap marginal & simultaneous confidence sets for "
        f"ranks (Mogstad, Romano, Shaikh & Wilhelm, REStud 2024), {conf}% level, "
        f"R={R} draws, n={res.n} shared questions, p={len(res.models)} models.\n",
        "Each model's accuracy is θ̂ = mean per-question score; the full covariance "
        "of the accuracy vector is estimated from the **shared item bank**, so "
        "pairwise difference SEs use the paired term −2·Σ_jk (tighter than assuming "
        "independence). A model's rank set is the interval of ranks not excluded by "
        "the bootstrap. **Coverage is for item-sampling uncertainty only** — it does "
        "not capture generation stochasticity or prompt sensitivity.\n",
        "## Rank confidence sets — three methods side by side\n",
        "**Naive** = independent marginal CIs + fixed z (no multiple-comparison "
        "correction); the common practice. **MRSW marginal / simultaneous** add the "
        "shared-item covariance and proper error control across comparisons.\n",
        "| Point rank | Model | Accuracy % | 95% Acc CI | Rank (naive) | Rank (MRSW marg.) | Rank (MRSW sim.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(res.models, 1):
        lines.append(
            f"| {i} | `{m}` | {res.theta[i-1]*100:.1f} | {_fmt_ci(res.acc_ci[i-1])} "
            f"| {_fmt_set(res.naive[i-1])} | {_fmt_set(res.marginal[i-1])} | {_fmt_set(res.simultaneous[i-1])} |"
        )

    # Discrimination summary: the headline contrast. Tighter = more resolving power.
    p = len(res.models)
    def _stats(sets):
        singletons = sum(1 for s in sets if s[0] == s[1])
        avg_w = sum(_width(s) for s in sets) / p
        return singletons, avg_w
    n_sing, n_w = _stats(res.naive)
    m_sing, m_w = _stats(res.marginal)
    s_sing, s_w = _stats(res.simultaneous)
    lines += [
        "\n## Discrimination summary (the headline)\n",
        "Singletons = models pinned to one exact rank; Avg width = mean rank-set size.\n",
        "| Method | Models pinned to a single rank | Avg rank-set width |",
        "|---|---|---|",
        f"| Naive (independent marginal CIs, no MC correction) | {n_sing}/{p} | {n_w:.2f} |",
        f"| MRSW marginal | {m_sing}/{p} | {m_w:.2f} |",
        f"| MRSW simultaneous | {s_sing}/{p} | {s_w:.2f} |",
        "\n**Reading this correctly (not 'tighter = better'):** the naive method makes "
        "all p−1 pairwise calls at level α with *no* multiple-comparison correction and "
        "assumes independence, so it is **anti-conservative** — it tends to pin more "
        "models to single ranks, but those pins do **not** carry valid 95% coverage of "
        "the true rank (its apparent precision is unwarranted). MRSW controls error "
        "across all comparisons (valid coverage); its **simultaneous** column is the "
        "honest 'which orderings are truly resolved'. The paired −2·Σ_jk term then "
        "*recovers* power that the independence assumption would otherwise throw away, "
        "partly offsetting the (correct) widening from multiple-comparison control. The "
        "scientific point is validity, not raw narrowness.",
    ]

    # Pairwise differences (marginal): which orderings are statistically resolved.
    lines += [
        "\n## Pairwise accuracy differences (marginal, {0}%)\n".format(conf),
        "Δ = (row model) − (col model), in points. ✓ = difference CI excludes 0 "
        "(ordering statistically resolved); · = not resolved (statistical tie).\n",
        "| Model A vs B | Δ (pts) | 95% diff CI (pts) | Resolved? |",
        "|---|---|---|---|",
    ]
    p = len(res.models)
    theta = res.theta
    for j in range(p):
        for k in range(j + 1, p):
            diff = (theta[j] - theta[k]) * 100
            half = res.se_jk[j, k] * res.c_marginal[j] * 100
            lo, hi = diff - half, diff + half
            resolved = "✓" if (lo > 0 or hi < 0) else "·"
            lines.append(
                f"| `{res.models[j]}` vs `{res.models[k]}` | {diff:+.1f} "
                f"| [{lo:+.1f}, {hi:+.1f}] | {resolved} |"
            )

    for note in res.notes:
        lines.append(f"\n> ⚠️ {note}")
    (OUT / "rank_confidence.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=0.05, help="1-alpha = confidence level (default 0.05)")
    ap.add_argument("--R", type=int, default=2000, help="bootstrap draws (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (default 0)")
    ap.add_argument("--no-rankcs", action="store_true", help="skip MRSW rank confidence sets")
    args = ap.parse_args()

    runs = collect_runs()
    if not runs:
        print(f"No successful eval logs in {LOG_DIR}. Run: python run_ranking.py")
        return 1

    rank_by_model: dict | None = None
    if not args.no_rankcs:
        X, models = build_score_matrix([r["model"] for r in runs])
        if X is None:
            print(f"Rank CS skipped: need ≥2 models with shared questions (have {len(models)}).")
        else:
            res = mrsw.rank_confidence_sets(X, models, alpha=args.alpha, R=args.R, seed=args.seed)
            rank_by_model = {
                m: {
                    "acc_ci": res.acc_ci[i],
                    "naive": res.naive[i],
                    "marginal": res.marginal[i],
                    "simultaneous": res.simultaneous[i],
                }
                for i, m in enumerate(models)
            }
            write_rank_confidence(res, args.alpha, args.R)

    write_ranking(runs, rank_by_model)

    outs = "ranking.md, ranking.csv" + (", rank_confidence.md" if rank_by_model else "")
    print(f"Wrote {outs} to {OUT}")
    for i, r in enumerate(runs, 1):
        extra = ""
        if rank_by_model and r["model"] in rank_by_model:
            d = rank_by_model[r["model"]]
            extra = f"  rank set (marg) {_fmt_set(d['marginal'])}  (sim) {_fmt_set(d['simultaneous'])}"
        print(f"  {i}. {r['model']:40s} {r['accuracy_pct']}%{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
