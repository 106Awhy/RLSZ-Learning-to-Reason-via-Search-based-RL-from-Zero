import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ.get('ROOT', '.work')).resolve()
nq_path = root / 'data/processed/nq/train.parquet'
hotpot_path = root / 'data/processed/hotpotqa/hotpotqa_train_search.parquet'
out_path = root / 'data/processed/nq_hotpotqa_train.parquet'
assert nq_path.exists(), f'Missing NQ file: {nq_path}'
assert hotpot_path.exists(), f'Missing HotpotQA file: {hotpot_path}'
nq = pd.read_parquet(nq_path)
hp = pd.read_parquet(hotpot_path)
df = pd.concat([nq, hp], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
summary = {'nq_path': str(nq_path), 'hotpot_path': str(hotpot_path), 'out_path': str(out_path), 'nq_count': len(nq), 'hotpotqa_count': len(hp), 'total_count': len(df), 'columns': list(df.columns)}
(root / 'reports').mkdir(parents=True, exist_ok=True)
(root / 'reports/train_data_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
