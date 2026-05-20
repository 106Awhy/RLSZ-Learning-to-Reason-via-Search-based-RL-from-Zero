# Scripts

- `reproduce.sh`: main entrypoint for setup, data/model preparation, training,
  and evaluation across `1.5b`, `7b`, and `14b`.
- `apply_reward_patch.sh`: installs `qa_custom.py` and patches Search-R1's
  reward selector.
- `process_hotpotqa_search.py`: converts HotpotQA from FlashRAG format to the
  Search-R1 parquet schema.
- `merge_train_data.py`: merges NQ and HotpotQA train parquets.

The long-running training commands expect a live retriever at
`http://127.0.0.1:8000/retrieve`.

Examples:

```bash
bash scripts/reproduce.sh prepare-data all
bash scripts/reproduce.sh train 14b grpo_soft_format
bash scripts/reproduce.sh eval 14b grpo_soft_format /path/to/actor/global_step_500
```
