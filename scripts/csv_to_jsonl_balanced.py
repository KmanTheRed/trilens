
import csv, json, pathlib, sys, glob, collections

src_glob = "data/raw/human_ai_5k/*.csv"
files = glob.glob(src_glob)
if not files:
    sys.exit(f"✖  No CSV found at {src_glob}")
src = files[0]
print("→ using", src)

# open once to discover header
with open(src, newline='', encoding="utf8") as fh:
    reader = csv.DictReader(fh)
    header = reader.fieldnames
print("CSV columns:", header)

# decide which column is the label (first integer-ish column)
label_col = None
sample_row = next(csv.DictReader(open(src, newline='', encoding="utf8")))
for col in header:
    try:
        int(sample_row[col])
        label_col = col
        break
    except (ValueError, TypeError):
        continue
if label_col is None:
    sys.exit("✖  Could not find an integer label column. Edit script and set label_col manually.")

text_col = next((c for c in header if c.lower().startswith("text")), header[0])

dst = pathlib.Path("data/raw/train_5k.jsonl")
dst.parent.mkdir(parents=True, exist_ok=True)

label_count = collections.Counter()
with open(src, newline='', encoding="utf8") as fh, dst.open("w", encoding="utf8") as out:
    reader = csv.DictReader(fh)
    for row in reader:
        try:
            lbl = int(row[label_col])
        except ValueError:
            continue
        txt = row[text_col]
        label_count[lbl] += 1
        out.write(json.dumps({"text": txt, "label": lbl}) + "\n")

print(f"✓ wrote {dst} with {label_count[0]} human + {label_count[1]} AI rows")

