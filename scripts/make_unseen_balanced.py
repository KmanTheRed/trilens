
#!/usr/bin/env python3
"""
Make a balanced 10 k (5 k human + 5 k AI) JSONL from a CSV whose columns are:
    text, source, ...
We label rows:
    0  if source contains "human" or "reference"
    1  otherwise (any model name).
"""
import csv, json, random, re, argparse, pathlib, collections, sys

csv.field_size_limit(10**8)                         # lift 128 kB limit
HUMAN_PAT = re.compile(r"(human|reference)", re.I)  # tweak if needed

def label(src): return 0 if HUMAN_PAT.search(src) else 1

def main(a):
    random.seed(a.seed)
    buckets, seen = {0: [], 1: []}, collections.Counter()

    with open(a.csv, newline='', encoding="utf8") as fh:
        rdr = csv.DictReader(fh)
        if "text" not in rdr.fieldnames or "source" not in rdr.fieldnames:
            sys.exit("CSV must have 'text' and 'source' columns")
        for row in rdr:
            txt = row["text"].strip()
            if not txt: continue
            lbl = label(row["source"])
            seen[lbl] += 1
            B, k = buckets[lbl], a.k
            if len(B) < k:
                B.append((txt,lbl))
            else:               # reservoir
                j = random.randrange(seen[lbl])
                if j < k: B[j] = (txt,lbl)

    dst = pathlib.Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf8") as f:
        for lbl in (0,1):
            for txt,l in buckets[lbl]:
                f.write(json.dumps({"text": txt, "label": l}) + "\n")
    print(f"✓ wrote {dst}  ({len(buckets[0])} human + {len(buckets[1])} AI)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="e.g. data/unseen/data.csv")
    ap.add_argument("--out", default="data/unseen/balanced_10k_unseen.jsonl")
    ap.add_argument("-k", type=int, default=5000, help="samples per class")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())

