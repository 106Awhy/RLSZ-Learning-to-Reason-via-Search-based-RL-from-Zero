# RLSZ-Learning-to-Reason-via-Search-based-RL-from-Zero

This repository contains the lightweight code needed to reproduce the RLSZ experiments across 1.5B, 7B, and 14B model scales.

## Experiments

Four training groups are covered:

1. `ppo`: PPO with the official Search-R1 exact-match reward.
2. `grpo_original`: GRPO with the official Search-R1 exact-match reward.
3. `grpo_paper_format`: GRPO with a hard format reward, `lambda_f=0.2`.
4. `grpo_soft_format`: GRPO with a soft continuous format reward, `lambda_f=0.2`.

Each method is defined for Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, and
Qwen2.5-14B-Instruct. All 7B and 14B experiments use the same settings as the
1.5B experiments: NQ + HotpotQA training data, local Wikipedia 2018 retrieval,
E5/FAISS, `topk=3`, and `max_turns=4`.

## Repository Layout

```text
configs/                 Experiment definitions
docs/                    Reproduction notes and data/model sources
patches/                 Minimal patch against the official Search-R1 repo
results/                 Small CSV result tables only
scripts/                 Setup, data prep, training, and evaluation entrypoints
src/                     Custom reward source file
```

## Quick Start

The full run is GPU-heavy and downloads large public artifacts. By default,
work files are created under `.work/`; override with `SEARCH_R1_ROOT`.

```bash
git clone <this-repo>
cd search-r1-qwen25-15b-repro

# Create envs, clone Search-R1, install dependencies, and apply the reward patch.
bash scripts/reproduce.sh setup

# Download all three models plus retrieval files, then build NQ + HotpotQA data.
bash scripts/reproduce.sh prepare-data all

# Launch retriever, then run one experiment.
bash scripts/reproduce.sh launch-retriever
bash scripts/reproduce.sh train 7b grpo_soft_format

# 14B may need environment overrides on smaller GPU nodes.
TENSOR_MODEL_PARALLEL_SIZE=2 TRAIN_BATCH_SIZE=12 PPO_MINI_BATCH_SIZE=30 \
  bash scripts/reproduce.sh train 14b grpo_soft_format

# Evaluate a checkpoint on NQ full test with official EM reward.
bash scripts/reproduce.sh eval 7b grpo_soft_format /path/to/actor/global_step_500
```

For a long end-to-end run:

```bash
bash scripts/reproduce.sh all
```

`all` is intentionally literal: it may take days and requires enough disk for
the retrieval index, datasets, model, checkpoints, and logs.

## Results Snapshot

The 1.5B server run used for this package is summarized in
`results/final_eval_nq_em.csv`. The multi-scale table supplied with the paper
draft is stored in `results/overall_results_by_scale.csv`.

Best final NQ EM by scale and method:

| Scale | PPO | GRPO | GRPO + Soft Reward | GRPO + Hard Reward |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 38.42% | 36.31% | 39.39% | 12.86% |
| 7B | 42.8% | 43.6% | 45.2% | 18.5% |
| 14B | 45.1% | 46.3% | 48.0% | 22.0% |

Best 1.5B NQ full-test official EM scores from raw logs:

- PPO: `0.384167` at `ppo_adaptive_gs300`
- GRPO original EM: `0.363056` at `grpo_original_gs300`
- GRPO hard format reward: `0.128611` at `paper_format_gs500`
- GRPO soft format reward: `0.393889` at `soft_format_gs500`

