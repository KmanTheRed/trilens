import pandas as pd, os, json, random, pathlib, argparse, re, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices={"train","test","extra"})
    ap.add_argument("--out",   default="data/raid/balanced_20k_raid.jsonl")
    ap.add_argument("-k", type=int, default=10_000, help="per-class sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # We only care about “test”, so look for test_full.csv (non-adv)
    csv_path = os.path.join("data/raid", f"{args.split}_full.csv")
    if not pathlib.Path(csv_path).exists():
        sys.exit(f"ERROR: cannot find {csv_path}. Did you wget test_none.csv → {csv_path}?")

    print(f"Loading RAID CSV from {csv_path} …")
    df = pd.read_csv(csv_path)

    # confirm we have both columns
    if "generation" not in df.columns or "model" not in df.columns:
        sys.exit(f"Unexpected columns in {csv_path}: {list(df.columns)}")

    human_re = re.compile(r"(human|reference)", re.I)
    df["label"] = df["model"].apply(lambda m: 0 if human_re.search(str(m)) else 1)

    human_df = df[df.label == 0].sample(n=args.k, random_state=args.seed)
    ai_df    = df[df.label == 1].sample(n=args.k, random_state=args.seed)
    balanced = pd.concat([human_df, ai_df]).sample(frac=1, random_state=args.seed)

    dst = pathlib.Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf8") as fout:
        for _, row in balanced.iterrows():
            fout.write(json.dumps({
                "text":  row["generation"],
                "label": int(row["label"])
            }) + "\n")

    print(f"✓ Wrote {dst} ({args.k} human + {args.k} AI)")

if __name__ == "__main__":
    main()
