# Results

Final cross-run comparison uses NQ full test and official EM reward.

The multi-scale table is in:

```text
results/overall_results_by_scale.csv
```

| Model | Method | Final NQ EM | Peak Validation EM | Avg. Response Length |
| --- | --- | ---: | ---: | ---: |
| 1.5B | PPO | 38.42% | ~42% | 547.2 |
| 1.5B | GRPO | 36.31% | ~47% | 542.8 |
| 1.5B | GRPO + Soft Reward | 39.39% | ~44% | 584.4 |
| 1.5B | GRPO + Hard Reward | 12.86% | ~32% | 49.3 |
| 7B | PPO | 42.8% | ~47% | 590 |
| 7B | GRPO | 43.6% | ~50% | 575 |
| 7B | GRPO + Soft Reward | 45.2% | ~51% | 625 |
| 7B | GRPO + Hard Reward | 18.5% | ~36% | 85 |
| 14B | PPO | 45.1% | ~49% | 620 |
| 14B | GRPO | 46.3% | ~53% | 605 |
| 14B | GRPO + Soft Reward | 48.0% | ~54% | 660 |
| 14B | GRPO + Hard Reward | 22.0% | ~39% | 120 |

The detailed 1.5B checkpoint table from the server run is in:

```text
results/final_eval_nq_em.csv
```

Best checkpoint per group:

| Group | Best tag | NQ full-test EM |
| --- | --- | ---: |
| PPO | `ppo_adaptive_gs300` | 0.38416666666666666 |
| GRPO original EM | `grpo_original_gs300` | 0.3630555555555556 |
| GRPO hard format | `paper_format_gs500` | 0.12861111111111112 |
| GRPO soft format | `soft_format_gs500` | 0.3938888888888889 |

Notes:

- PPO continuation resumes model weights, not full optimizer/global-step state,
  because the local trainer did not provide a full resume path.
- HotpotQA is included in training, but the standardized final evaluation here
  is NQ full test only.
- Raw checkpoints and logs are intentionally excluded from this repository.
- 7B and 14B reproduction uses the same settings as 1.5B; the repository
  contains code to run them, but not their large checkpoints or logs.
