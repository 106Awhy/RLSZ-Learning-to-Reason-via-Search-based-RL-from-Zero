# Data And Model Sources

This repository does not contain large artifacts. Reproduction downloads or
generates them under `SEARCH_R1_ROOT`, defaulting to `.work/`.

## Models

- 1.5B: `Qwen/Qwen2.5-1.5B-Instruct`
- 7B: `Qwen/Qwen2.5-7B-Instruct`
- 14B: `Qwen/Qwen2.5-14B-Instruct`
- Preferred download path in the original run: ModelScope
- Local targets:
  - `$SEARCH_R1_ROOT/models/Qwen2.5-1.5B-Instruct`
  - `$SEARCH_R1_ROOT/models/Qwen2.5-7B-Instruct`
  - `$SEARCH_R1_ROOT/models/Qwen2.5-14B-Instruct`

The reproduction script downloads all three when run as
`bash scripts/reproduce.sh prepare-data all`. The original server used
ModelScope because direct Hugging Face downloads were unstable on that network.

## Training Data

The training data is generated from public datasets:

- NQ via the official Search-R1 data processing script:
  `scripts/data_process/nq_search.py`
- HotpotQA via `RUC-NLPIR/FlashRAG_datasets`, config `hotpotqa`
- Merged train file:
  `$SEARCH_R1_ROOT/data/processed/nq_hotpotqa_train.parquet`

Original generated counts:

```text
NQ train:       79,168
HotpotQA train: 90,447
Merged train:  169,615
NQ test:         3,610
```

## Retrieval

The original run used the Search-R1 local retriever with:

- Wikipedia 2018 corpus: `wiki-18.jsonl`
- E5 FAISS index: `e5_Flat.index`

The reproducibility script downloads these from public Hugging Face dataset
mirrors. The retrieval index is large and is not tracked in git.

## Evaluation

Final comparison uses NQ full test and the official Search-R1 exact-match
reward for every checkpoint, including checkpoints trained with custom reward
formats.

The 7B and 14B runs use the same data, retrieval files, reward choices, and
evaluation path as the 1.5B runs.

For constrained GPU nodes, the script keeps the same default setting but allows
runtime overrides such as `TENSOR_MODEL_PARALLEL_SIZE`, `TRAIN_BATCH_SIZE`,
`PPO_MINI_BATCH_SIZE`, `PPO_MICRO_BATCH_SIZE`, and
`VLLM_GPU_MEMORY_UTILIZATION`.
