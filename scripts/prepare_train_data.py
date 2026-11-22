"""Download Search-R1's NQ + HotpotQA training split and keep a small shuffled sample -> data/train.jsonl.

Usage: python scripts/prepare_train_data.py [--n 2000]
Bamboogle is never used for training (it is only in data/bamboogle.csv).
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=2000)
args = p.parse_args()

path = hf_hub_download("PeterJinGo/nq_hotpotqa_train", "train.parquet", repo_type="dataset")
table = pq.read_table(path, columns=["question", "golden_answers", "data_source"]).to_pandas()
sample = table.sample(n=args.n, random_state=0)  # fixed seed so the sample is reproducible

out = Path(__file__).parent.parent / "data" / "train.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for row in sample.itertuples():
        f.write(json.dumps({"question": row.question, "answers": list(row.golden_answers), "source": row.data_source}) + "\n")
print(f"wrote {len(sample)} questions to {out}")
