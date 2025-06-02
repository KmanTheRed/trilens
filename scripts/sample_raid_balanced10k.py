#!/usr/bin/env python3
"""
Make a balanced 10 k RAID subset
--------------------------------
Reads:
    data/raid/sample_50k_human.jsonl
    data/raid/sample_50k_machine.jsonl
and writes:
    data/raid/balanced_10k_raid.jsonl   (5 000 human + 5 000 AI)

Each input row already has {"text": ..., "label": 0|1}.
If not, we infer label from the filename.
"""
import json, random, pathlib, collections, argparse, sys, gzip

def reservoir(src_path, label_val, k, bucket, counter):
    open_fn = gzip.open if src_path.suffix == ".gz" else open
    with open_fn(src_path, encoding="utf8") as fh:
        for line in fh:
            obj = json.loads(line)
            txt = obj.get("text", "").strip()
            if not txt:
                continue
            lbl = obj.get("label", label_val)
            counter[lbl] += 1
            if len(bucket[lbl]) < k:
                bucket[lbl].append({"text": txt, "label": lbl})
            else:
                j = random.randrange(counter[lbl])
                if j < k:
                    bucket[lbl][j] = {"text": txt, "label": lbl}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--per_class", type=int, default=5000,
                    help="samples per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    base = pathlib.Path("data/raid")
    human_file   = base / "sample_50k_human.jsonl"
    machine_file = base / "sample_50k_machine.jsonl"

    if not (human_file.exists() and machine_file.exists()):
        sys.exit("✖ Expected sample_50k_human.jsonl and sample_50k_machine.jsonl in data/raid/")

    buckets = {0: [], 1: []}
    counter = collections.Counter()

    # Fill reservoirs
    reservoir(human_file,   0, args.per_class, buckets, counter)
    reservoir(machine_file, 1, args.per_class, buckets, counter)

    dst = base / "balanced_10k_raid.jsonl"
    with dst.open("w", encoding="utf8") as out:
        for lbl in (0, 1):
            for row in buckets[lbl]:
                out.write(json.dumps(row) + "\n")

    print(f"✓ wrote {dst}  ({len(buckets[0])} human + {len(buckets[1])} AI)")

if __name__ == "__main__":
    main()

