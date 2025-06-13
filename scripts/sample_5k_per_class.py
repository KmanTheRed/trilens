#!/usr/bin/env python3
"""
Create a balanced 5k/5k sample when the CSV has columns:
   text, source, ...
Label is derived:
   0 = human / reference
   1 = anything else (LLM)
Usage:
   python sample_5k_per_class.py \
           --csv  data/unseen/data.csv \
           --out  data/unseen/balanced_10k.jsonl
"""

import csv, sys
# increase CSV field size limit to handle large text rows
csv.field_size_limit(10**8)
import json, random, argparse, pathlib, collections, re

HUMAN_PAT  = re.compile(r"human|reference", re.I)   # regex for human sources

def label_from_source(src: str) -> int:
    return 0 if HUMAN_PAT.search(src) else 1

def main(a):
    random.seed(a.seed)
    buckets  = {0: [], 1: []}
    counter  = collections.Counter()

    with open(a.csv, newline='', encoding="utf8") as fh:
        rdr = csv.DictReader(fh)
        if "text" not in rdr.fieldnames or "source" not in rdr.fieldnames:
            raise SystemExit("CSV must contain 'text' and 'source' columns")
        for row in rdr:
            txt  = row["text"].strip()
            lbl  = label_from_source(row["source"])
            if not txt:
                continue
            counter[lbl] += 1
            B = buckets[lbl]
            if len(B) < a.k:
                B.append((txt,lbl))
            else:                               # reservoir
                j = random.randrange(counter[lbl])
                if j < a.k:
                    B[j] = (txt,lbl)

    dst = pathlib.Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf8") as out:
        for lbl in (0,1):
            for txt,l in buckets[lbl]:
                out.write(json.dumps({"text": txt, "label": l}) + "\n")

    print(f"✓ wrote {dst}  ({len(buckets[0])} human + {len(buckets[1])} AI)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--out",  default="data/unseen/balanced_10k.jsonl")
    ap.add_argument("-k",     type=int, default=5000, help="per-class sample")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
