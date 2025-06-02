import argparse, glob, json, torch, joblib
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
# --- lens imports -----------------------------------------------------------
from lenses.lens1_curvature import PerturbationCurvature
from lenses.lens2_compress   import CompressionAnomaly
from lenses.lens3_syntax_gcn import SyntaxGCN, doc_to_graph

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_gcn_model = None
def gcn_score(text: str, ckpt: str, device: str = "cuda") -> float:
    """Lazy-load Syntax-GCN and return probability for one text."""
    global _gcn_model
    if _gcn_model is None:
        in_dim = doc_to_graph("dummy").x.size(1)
        _gcn_model = SyntaxGCN(in_dim=in_dim).to(device)
        _gcn_model.load_state_dict(torch.load(ckpt, map_location=device))
        _gcn_model.eval()
    g = doc_to_graph(text)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(_gcn_model(g.to(device))).item()

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(a):
    THRESH = a.threshold  # allow override from command line
    curv = PerturbationCurvature(device=a.device)
    comp = CompressionAnomaly()
    clf  = joblib.load(a.ckpt)

    files = a.files or sorted(glob.glob("data/raw/cs162-final-dev/*.jsonl"))
    print(f"\nEvaluating ensemble on {len(files)} file(s)\n" + "="*40)

    for fp in files:
        y_true, feats = [], []
        with open(fp, encoding="utf8") as fh:
            for line in tqdm(fh, desc=fp, unit="lines", leave=False):
                obj   = json.loads(line)
                label = 1 if "machine_text" in obj else 0
                text  = obj.get("machine_text") or obj.get("human_text")
                x     = [curv.score(text),
                         comp.score(text),
                         gcn_score(text, a.gcn, a.device)]
                y_true.append(label)
                feats.append(x)

        # Get probabilities and apply threshold
        probs = clf.predict_proba(feats)[:, 1]  # probability of class 1 (AI)
        preds = (probs >= THRESH).astype(int)

        acc   = accuracy_score(y_true, preds)
        f1    = f1_score(y_true, preds)
        print(f"{fp:<55}  Acc {acc:.4f}  F1 {f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ensemble.pkl")
    ap.add_argument("--gcn",  required=True, help="Syntax-GCN .pt file")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--threshold", type=float, default=0.10, help="classification threshold (default=0.10)")
    ap.add_argument("files", nargs="*", help="JSONL files (optional)")
    main(ap.parse_args())

