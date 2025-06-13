# scripts/merge_train_data.py
import os, json
import pandas as pd

os.makedirs('data/raw', exist_ok=True)
out_path = 'data/raw/train.jsonl'
with open(out_path, 'w', encoding='utf8') as out:
    # 1) Kaggle AI-vs-Human CSV
    df = pd.read_csv('data/AI_Human.csv')
    for _, row in df.iterrows():
        out.write(json.dumps({'text': row['text'],      'label': 0}) + '\n')
        out.write(json.dumps({'text': row['generated'], 'label': 1}) + '\n')


    # 2) RAID text files (assume all AI-written)
    for root, _, files in os.walk('data/raid-main'):
        for fn in files:
            path = os.path.join(root, fn)
            with open(path, encoding='utf8', errors='ignore') as f:
                text = f.read().strip()
            if text:
                out.write(json.dumps({'text': text, 'label': 1}) + '\n')

print(f"Wrote merged train set to {out_path}")
