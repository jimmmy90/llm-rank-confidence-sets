# MMLU-Pro — rank confidence sets (Mogstad–Romano–Shaikh–Wilhelm)

Method: parametric-bootstrap marginal & simultaneous confidence sets for ranks (Mogstad, Romano, Shaikh & Wilhelm, REStud 2024), 95% level, R=2000 draws, n=2000 shared questions, p=13 models.

Each model's accuracy is θ̂ = mean per-question score; the full covariance of the accuracy vector is estimated from the **shared item bank**, so pairwise difference SEs use the paired term −2·Σ_jk (tighter than assuming independence). A model's rank set is the interval of ranks not excluded by the bootstrap. **Coverage is for item-sampling uncertainty only** — it does not capture generation stochasticity or prompt sensitivity.

## Rank confidence sets — three methods side by side

**Naive** = independent marginal CIs + fixed z (no multiple-comparison correction); the common practice. **MRSW marginal / simultaneous** add the shared-item covariance and proper error control across comparisons.

| Point rank | Model | Accuracy % | 95% Acc CI | Rank (naive) | Rank (MRSW marg.) | Rank (MRSW sim.) |
|---|---|---|---|---|---|---|
| 1 | `openrouter/deepseek/deepseek-chat` | 78.8 | [77.0, 80.6] | {1} | {1} | {1–2} |
| 2 | `openrouter/qwen/qwen3-14b` | 75.8 | [73.9, 77.7] | {2–4} | {2–4} | {1–4} |
| 3 | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | 74.8 | [72.8, 76.6] | {2–4} | {2–4} | {2–4} |
| 4 | `openrouter/qwen/qwen3-8b` | 74.1 | [72.1, 76.0] | {2–4} | {2–4} | {2–5} |
| 5 | `openrouter/z-ai/glm-4.7-flash` | 70.9 | [68.9, 72.8] | {5–6} | {5–6} | {4–7} |
| 6 | `openrouter/meta-llama/llama-3.3-70b-instruct` | 70.2 | [68.2, 72.2] | {5–7} | {5–7} | {5–7} |
| 7 | `openrouter/mistralai/mistral-small-3.2-24b-instruct` | 67.8 | [65.7, 69.8] | {6–7} | {6–7} | {5–7} |
| 8 | `openrouter/microsoft/phi-4` | 63.2 | [61.1, 65.3] | {8} | {8} | {8} |
| 9 | `openrouter/google/gemma-3-12b-it` | 58.5 | [56.3, 60.6] | {9} | {9} | {9} |
| 10 | `openrouter/qwen/qwen-2.5-7b-instruct` | 52.9 | [50.8, 55.1] | {10} | {10} | {10} |
| 11 | `openrouter/google/gemma-3-4b-it` | 43.9 | [41.7, 46.1] | {11} | {11} | {11–12} |
| 12 | `openrouter/meta-llama/llama-3.1-8b-instruct` | 40.1 | [38.0, 42.3] | {12} | {12} | {11–12} |
| 13 | `openrouter/meta-llama/llama-3.2-3b-instruct` | 22.1 | [20.3, 24.0] | {13} | {13} | {13} |

## Discrimination summary (the headline)

Singletons = models pinned to one exact rank; Avg width = mean rank-set size.

| Method | Models pinned to a single rank | Avg rank-set width |
|---|---|---|
| Naive (independent marginal CIs, no MC correction) | 7/13 | 1.77 |
| MRSW marginal | 7/13 | 1.77 |
| MRSW simultaneous | 4/13 | 2.38 |

**Reading this correctly (not 'tighter = better'):** the naive method makes all p−1 pairwise calls at level α with *no* multiple-comparison correction and assumes independence, so it is **anti-conservative** — it tends to pin more models to single ranks, but those pins do **not** carry valid 95% coverage of the true rank (its apparent precision is unwarranted). MRSW controls error across all comparisons (valid coverage); its **simultaneous** column is the honest 'which orderings are truly resolved'. The paired −2·Σ_jk term then *recovers* power that the independence assumption would otherwise throw away, partly offsetting the (correct) widening from multiple-comparison control. The scientific point is validity, not raw narrowness.

## Pairwise accuracy differences (marginal, 95%)

Δ = (row model) − (col model), in points. ✓ = difference CI excludes 0 (ordering statistically resolved); · = not resolved (statistical tie).

| Model A vs B | Δ (pts) | 95% diff CI (pts) | Resolved? |
|---|---|---|---|
| `openrouter/deepseek/deepseek-chat` vs `openrouter/qwen/qwen3-14b` | +3.0 | [+0.5, +5.5] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | +4.1 | [+1.6, +6.6] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/qwen/qwen3-8b` | +4.7 | [+2.2, +7.3] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/z-ai/glm-4.7-flash` | +8.0 | [+5.3, +10.6] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/meta-llama/llama-3.3-70b-instruct` | +8.7 | [+5.9, +11.4] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +11.1 | [+8.3, +13.9] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/microsoft/phi-4` | +15.6 | [+12.6, +18.7] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/google/gemma-3-12b-it` | +20.4 | [+17.3, +23.4] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +25.9 | [+22.7, +29.1] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/google/gemma-3-4b-it` | +34.9 | [+31.5, +38.4] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +38.7 | [+35.3, +42.2] | ✓ |
| `openrouter/deepseek/deepseek-chat` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +56.8 | [+53.4, +60.1] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | +1.1 | [-1.3, +3.5] | · |
| `openrouter/qwen/qwen3-14b` vs `openrouter/qwen/qwen3-8b` | +1.7 | [-0.5, +4.0] | · |
| `openrouter/qwen/qwen3-14b` vs `openrouter/z-ai/glm-4.7-flash` | +4.9 | [+2.3, +7.6] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/meta-llama/llama-3.3-70b-instruct` | +5.6 | [+2.8, +8.5] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +8.1 | [+5.4, +10.8] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/microsoft/phi-4` | +12.6 | [+9.7, +15.6] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/google/gemma-3-12b-it` | +17.3 | [+14.3, +20.4] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +22.9 | [+19.7, +26.1] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/google/gemma-3-4b-it` | +31.9 | [+28.5, +35.4] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +35.7 | [+32.3, +39.2] | ✓ |
| `openrouter/qwen/qwen3-14b` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +53.8 | [+50.4, +57.1] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/qwen/qwen3-8b` | +0.7 | [-1.8, +3.1] | · |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/z-ai/glm-4.7-flash` | +3.9 | [+1.1, +6.6] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/meta-llama/llama-3.3-70b-instruct` | +4.6 | [+1.8, +7.3] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +7.0 | [+4.3, +9.7] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/microsoft/phi-4` | +11.6 | [+8.6, +14.5] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/google/gemma-3-12b-it` | +16.3 | [+13.3, +19.2] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +21.8 | [+18.7, +24.9] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/google/gemma-3-4b-it` | +30.9 | [+27.5, +34.2] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +34.7 | [+31.2, +38.1] | ✓ |
| `openrouter/qwen/qwen3-30b-a3b-instruct-2507` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +52.7 | [+49.2, +56.1] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/z-ai/glm-4.7-flash` | +3.2 | [+0.5, +5.9] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/meta-llama/llama-3.3-70b-instruct` | +3.9 | [+1.0, +6.8] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +6.3 | [+3.6, +9.1] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/microsoft/phi-4` | +10.9 | [+7.9, +13.9] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/google/gemma-3-12b-it` | +15.6 | [+12.6, +18.6] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +21.2 | [+18.1, +24.2] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/google/gemma-3-4b-it` | +30.2 | [+26.9, +33.5] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +34.0 | [+30.5, +37.5] | ✓ |
| `openrouter/qwen/qwen3-8b` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +52.0 | [+48.6, +55.4] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/meta-llama/llama-3.3-70b-instruct` | +0.7 | [-2.3, +3.7] | · |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +3.1 | [+0.2, +6.1] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/microsoft/phi-4` | +7.7 | [+4.6, +10.8] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/google/gemma-3-12b-it` | +12.4 | [+9.3, +15.5] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +17.9 | [+14.6, +21.3] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/google/gemma-3-4b-it` | +27.0 | [+23.5, +30.5] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +30.8 | [+27.3, +34.3] | ✓ |
| `openrouter/z-ai/glm-4.7-flash` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +48.8 | [+45.3, +52.3] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/mistralai/mistral-small-3.2-24b-instruct` | +2.4 | [-0.3, +5.2] | · |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/microsoft/phi-4` | +7.0 | [+3.9, +10.1] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/google/gemma-3-12b-it` | +11.7 | [+8.6, +14.8] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +17.2 | [+14.1, +20.4] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/google/gemma-3-4b-it` | +26.3 | [+22.8, +29.8] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +30.1 | [+26.7, +33.5] | ✓ |
| `openrouter/meta-llama/llama-3.3-70b-instruct` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +48.1 | [+44.7, +51.5] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/microsoft/phi-4` | +4.5 | [+1.6, +7.5] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/google/gemma-3-12b-it` | +9.3 | [+6.3, +12.2] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +14.8 | [+11.7, +17.9] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/google/gemma-3-4b-it` | +23.8 | [+20.4, +27.3] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +27.6 | [+24.2, +31.1] | ✓ |
| `openrouter/mistralai/mistral-small-3.2-24b-instruct` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +45.6 | [+42.1, +49.2] | ✓ |
| `openrouter/microsoft/phi-4` vs `openrouter/google/gemma-3-12b-it` | +4.7 | [+1.6, +7.8] | ✓ |
| `openrouter/microsoft/phi-4` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +10.3 | [+7.1, +13.4] | ✓ |
| `openrouter/microsoft/phi-4` vs `openrouter/google/gemma-3-4b-it` | +19.3 | [+15.9, +22.7] | ✓ |
| `openrouter/microsoft/phi-4` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +23.1 | [+19.7, +26.5] | ✓ |
| `openrouter/microsoft/phi-4` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +41.1 | [+37.7, +44.5] | ✓ |
| `openrouter/google/gemma-3-12b-it` vs `openrouter/qwen/qwen-2.5-7b-instruct` | +5.5 | [+2.4, +8.7] | ✓ |
| `openrouter/google/gemma-3-12b-it` vs `openrouter/google/gemma-3-4b-it` | +14.6 | [+11.3, +17.9] | ✓ |
| `openrouter/google/gemma-3-12b-it` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +18.4 | [+14.9, +21.9] | ✓ |
| `openrouter/google/gemma-3-12b-it` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +36.4 | [+32.9, +39.9] | ✓ |
| `openrouter/qwen/qwen-2.5-7b-instruct` vs `openrouter/google/gemma-3-4b-it` | +9.0 | [+5.8, +12.3] | ✓ |
| `openrouter/qwen/qwen-2.5-7b-instruct` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +12.8 | [+9.5, +16.2] | ✓ |
| `openrouter/qwen/qwen-2.5-7b-instruct` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +30.9 | [+27.5, +34.2] | ✓ |
| `openrouter/google/gemma-3-4b-it` vs `openrouter/meta-llama/llama-3.1-8b-instruct` | +3.8 | [+0.4, +7.2] | ✓ |
| `openrouter/google/gemma-3-4b-it` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +21.8 | [+18.5, +25.1] | ✓ |
| `openrouter/meta-llama/llama-3.1-8b-instruct` vs `openrouter/meta-llama/llama-3.2-3b-instruct` | +18.0 | [+14.9, +21.1] | ✓ |
