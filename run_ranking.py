#!/usr/bin/env python3
"""Run a benchmark across every model we have an API key for, via Inspect's
resumable eval_set. Models come from models.yaml; a model is included only when
its `key_env` is present in the environment and its id is verified (not VERIFY/...).

Benchmarks:
    gpqa      -> GPQA-Diamond, 198 questions (near-saturated; frontier models)
    mmlu_pro  -> MMLU-Pro, STRATIFIED-sampled to --n questions across 14 subjects
                 (better discrimination for mid-tier models; default)

Usage:
    python run_ranking.py                              # mmlu_pro, n=500, epochs=1
    python run_ranking.py --benchmark gpqa --epochs 4  # official GPQA protocol
    python run_ranking.py --n 1000                     # tighter rank sets (more $)
    python run_ranking.py --limit 5                    # smoke test: 5 questions
    python run_ranking.py --models openrouter/qwen/qwen-2.5-72b-instruct
    python run_ranking.py --dry-run                    # print selection and exit
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "logs"
VERIFY_SENTINEL = "VERIFY"


def load_run_models() -> list[dict]:
    data = yaml.safe_load((ROOT / "models.yaml").read_text())
    return data.get("run_models", [])


def select_models(entries: list[dict], override: list[str] | None) -> list[dict]:
    """Keep models that are verified, have a real id, and whose key env is set.

    `override` (explicit ids on the CLI) bypasses the yaml filtering entirely.
    """
    if override:
        return [{"leaderboard_name": m, "inspect_model_id": m, "key_env": None} for m in override]

    selected, skipped = [], []
    for e in entries:
        mid = str(e.get("inspect_model_id", ""))
        key_env = e.get("key_env")
        has_key = bool(key_env and os.environ.get(key_env, "").strip())
        runnable = e.get("verified") and VERIFY_SENTINEL not in mid
        if runnable and has_key:
            selected.append(e)
        else:
            reason = "id not verified" if not runnable else f"no {key_env}"
            skipped.append((e.get("leaderboard_name"), mid, reason))

    if skipped:
        print("Skipped (not runnable yet):", file=sys.stderr)
        for name, mid, reason in skipped:
            print(f"  - {name:32s} {mid:40s} [{reason}]", file=sys.stderr)
    return selected


def stratified_sample(dataset, n: int, seed: int):
    """Proportional stratified sample of ~n MMLU-Pro questions across the 14
    subjects (sample.metadata['subject']), with a fixed seed for reproducibility.

    MMLU-Pro is ordered by subject, so a plain --limit would only hit the first
    few subjects. This keeps the slice representative; treating the benchmark as
    a sample of a question population is exactly what the rank CIs quantify.
    """
    from inspect_ai.dataset import MemoryDataset

    samples = list(dataset)
    total = len(samples)
    by_subject: dict[str, list] = {}
    for s in samples:
        by_subject.setdefault(s.metadata["subject"], []).append(s)

    rng = random.Random(seed)
    picked: list = []
    for subject, ss in sorted(by_subject.items()):
        k = max(1, round(n * len(ss) / total))
        ss = ss[:]  # don't mutate the original
        rng.shuffle(ss)
        picked.extend(ss[:k])
    rng.shuffle(picked)
    return MemoryDataset(picked, name=getattr(dataset, "name", "mmlu_pro"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=["gpqa", "mmlu_pro"], default="mmlu_pro",
                    help="which benchmark to run (default mmlu_pro)")
    ap.add_argument("--n", type=int, default=500,
                    help="mmlu_pro: stratified sample size across subjects (default 500)")
    ap.add_argument("--limit", type=int, default=None, help="further cap questions (smoke test)")
    ap.add_argument("--epochs", type=int, default=1, help="samples/question (default 1)")
    ap.add_argument("--models", nargs="*", default=None, help="explicit provider/model ids (bypass yaml)")
    ap.add_argument("--cot", action="store_true", default=True, help="gpqa: chain-of-thought (default on)")
    ap.add_argument("--seed", type=int, default=0, help="stratified-sample seed (default 0)")
    ap.add_argument("--dry-run", action="store_true", help="print selection and exit")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    selected = select_models(load_run_models(), args.models)
    if not selected:
        print(
            "\nNo runnable models. Set OPENROUTER_API_KEY in .env "
            "or pass --models <provider/model-id>.",
            file=sys.stderr,
        )
        return 1

    model_ids = [m["inspect_model_id"] for m in selected]
    depth = "full (198)" if args.benchmark == "gpqa" else f"stratified n≈{args.n}"
    print(f"\nRunning {args.benchmark} on:")
    for m in selected:
        print(f"  + {m['inspect_model_id']}")
    print(f"epochs={args.epochs}  questions={args.limit or depth}  seed={args.seed}  log_dir={LOG_DIR}")

    if args.dry_run:
        return 0

    # Import here so --dry-run / --help work without the heavy import + dataset fetch.
    from inspect_ai import eval_set

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    eval_kwargs: dict = {}

    if args.benchmark == "gpqa":
        from inspect_evals.gpqa import gpqa_diamond

        task = gpqa_diamond(cot=args.cot, epochs=args.epochs)  # epochs baked into the task
    else:
        from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro

        task = mmlu_pro(shuffle=False)
        task.dataset = stratified_sample(task.dataset, args.n, args.seed)
        eval_kwargs["epochs"] = args.epochs  # mmlu_pro takes epochs at the eval level
        print(f"  sampled {len(list(task.dataset))} questions across subjects (seed={args.seed})")

    success, logs = eval_set(
        tasks=[task],
        model=model_ids,
        log_dir=str(LOG_DIR),
        limit=args.limit,
        retry_attempts=3,
        **eval_kwargs,
    )
    print(f"\neval_set complete: success={success}, {len(logs)} log(s) in {LOG_DIR}")
    print("Next: python analyze.py")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
