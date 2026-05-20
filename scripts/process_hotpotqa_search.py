import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

def make_prefix(question: str) -> str:
    question = question.strip()
    if question and question[-1] != '?':
        question += '?'
    return f"""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

root = Path(os.environ.get('ROOT', '.work')).resolve()
out_dir = root / 'data/processed/hotpotqa'
out_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa', split='train')
print('loaded RUC-NLPIR/FlashRAG_datasets hotpotqa', len(ds))
rows = []
for idx, ex in enumerate(ds):
    target = ex.get('golden_answers') or ex.get('answer')
    rows.append({
        'data_source': 'hotpotqa',
        'prompt': [{'role': 'user', 'content': make_prefix(ex['question'])}],
        'ability': 'fact-reasoning',
        'reward_model': {'style': 'rule', 'ground_truth': {'target': target}},
        'extra_info': {'split': 'train', 'index': idx},
    })
df = pd.DataFrame(rows)
out = out_dir / 'hotpotqa_train_search.parquet'
df.to_parquet(out, index=False)
print('saved', out, len(df))
