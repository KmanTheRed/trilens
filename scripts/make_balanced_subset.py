#!/usr/bin/env python3
"""
Reservoir-sample exactly 15 000 rows per class from a huge JSONL file
and write them to a new JSONL.

Usage:
  python -m scripts.make_balanced_subset \
         --in  data/raw/train.jsonl \
         --out data/raw/train_15k.jsonl \
         --k   15000 --seed 42
"""

import argparse, json, random, pathlib, collections

TEXT_KEYS  = ("text", "content", "body", "human_text", "machine_text")
LABEL_KEYS = ("label", "is_ai")

def label_and_text(row):
    txt = next((row[k] for k in TEXT_KEYS if k in row), None)
    lbl = next((row[k] for k in LABEL_KEYS if k in row), None)
    if lbl is None:
        if "machine_text" in row: lbl, txt = 1, row["machine_text"]
        elif "human_text"   in row: lbl, txt = 0, row["human_text"]
    return lbl, txt

def main(a):
    random.seed(a.seed)
    buckets = {0: [], 1: []}            # store raw json strings
    seen    = collections.Counter()

    with open(a.in_path, encoding="utf8") as fh:
        for raw in fh:
            row = json.loads(raw)
            lbl, txt = label_and_text(row)
            if lbl not in (0,1) or not isinstance(txt, str) or not txt.strip():
                continue
            seen[lbl] += 1
            b = buckets[lbl]
            if len(b) < a.k:
                b.append(raw)
            else:
                j = random.randrange(seen[lbl])
                if j < a.k:
                    b[j] = raw

    print(f"kept {len(buckets[0])} human + {len(buckets[1])} AI rows")
    pathlib.Path(a.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_path, "w", encoding="utf8") as fout:
        fout.writelines(buckets[0] + buckets[1])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="in_path",  required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--k",   type=int, default=15000,
                   help="rows per class to keep")
    p.add_argument("--seed",type=int, default=0)
    main(p.parse_args())
