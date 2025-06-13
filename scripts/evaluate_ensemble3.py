#!/usr/bin/env python3
"""
Evaluate the Tri-Lens ensemble on one or more JSONL files.

Usage
-----
  python scripts/evaluate_ensemble3.py \
      --ckpt models/ensemble.pkl \
      --gcn  models/syntax_gcn_balanced_10k.pt \
      --threshold 0.23 \
      data/raid/balanced_10k_raid.jsonl  [more.jsonl ...]

The script understands two row formats:
  • {"machine_text": "..."} / {"human_text": "..."}
  • {"text": "...", "label": 0|1}
"""
import sys, pathlib, argparse, glob, json, joblib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# --- lens imports -----------------------------------------------------------
from lenses.lens1_curvature import PerturbationCurvature
from lenses.lens2_compress   import CompressionAnomaly
from lenses.lens3_syntax_gcn import SyntaxGCN, doc_to_graph

# ----------------------------------------------------------------------------
_gcn = None
def gcn_score(txt: str, ckpt: str, device="cuda") -> float:
    global _gcn
    if _gcn is None:
        in_dim = doc_to_graph("dummy").x.size(1)
        _gcn   = SyntaxGCN(in_dim=in_dim).to(device)
        _gcn.load_state_dict(torch.load(ckpt, map_location=device))
        _gcn.eval()
    g = doc_to_graph(txt); g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(_gcn(g.to(device))).item()

# ----------------------------------------------------------------------------
def row_to_label_text(obj):
    """Return (label,text) or (None,None) if row unusable."""
    if "label" in obj and "text" in obj:                 # generic
        return int(obj["label"]), obj["text"]
    if "machine_text" in obj:
        return 1, obj["machine_text"]
    if "human_text" in obj:
        return 0, obj["human_text"]
    return None, None

# ----------------------------------------------------------------------------
def evaluate_file(fp, curv, comp, clf, gcn_ckpt, device, thresh):
    y, feats = [], []
    with open(fp, encoding="utf8") as fh:
        for line in tqdm(fh, desc=fp, unit="lines", leave=False):
            obj = json.loads(line)
            lbl, txt = row_to_label_text(obj)
            if txt is None:        # skip malformed
                continue
            y.append(lbl)
            feats.append([curv.score(txt),
                          comp.score(txt),
                          gcn_score(txt, gcn_ckpt, device)])
    probs = clf.predict_proba(feats)[:,1]
    preds = (probs >= thresh).astype(int)
    return accuracy_score(y, preds), f1_score(y, preds)

# ----------------------------------------------------------------------------
def main(a):
    curv, comp = PerturbationCurvature(device=a.device), CompressionAnomaly()
    clf        = joblib.load(a.ckpt)
    files      = a.files or sorted(glob.glob("data/raw/cs162-final-dev/*.jsonl"))

    print(f"\nEvaluating on {len(files)} file(s)  (threshold={a.threshold})\n" + "="*40)
    for fp in files:
        acc, f1 = evaluate_file(fp, curv, comp, clf, a.gcn, a.device, a.threshold)
        print(f"{fp:<60} Acc {acc:.4f}  F1 {f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ensemble.pkl")
    ap.add_argument("--gcn",  required=True, help="Syntax-GCN .pt file")
    ap.add_argument("--threshold", type=float, default=0.50, help="prob cut-off")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("files", nargs="*", help="JSONL(s) to score")
    main(ap.parse_args())
