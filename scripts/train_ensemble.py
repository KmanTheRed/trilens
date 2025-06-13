#!/usr/bin/env python3
"""
Train (or re-train) the logistic-regression ensemble on a JSONL training file.
It extracts 3 features per text:
    • curvature (Lens-1)
    • compression anomaly (Lens-2)
    • syntax-gcn probability (Lens-3)

Example:
    python -m scripts.train_ensemble \
        --train_jsonl data/train.jsonl \
        --gcn_ckpt   models/syntax_gcn_balanced_10k.pt \
        --out        models/ensemble.pkl
"""
import json
import joblib
import argparse
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

from lenses.lens1_curvature import PerturbationCurvature
from lenses.lens2_compress import CompressionAnomaly
from lenses.lens3_syntax_gcn import SyntaxGCN, doc_to_graph


def main(args):
    # device can be "cuda", "cpu", or "cuda:0", etc.
    dev = args.device

    # initialize your three feature extractors
    curv = PerturbationCurvature(device=dev)
    comp = CompressionAnomaly()

    # load the GCN
    in_dim = doc_to_graph("dummy").x.size(1)
    gcn = SyntaxGCN(in_dim=in_dim).to(dev)
    gcn.load_state_dict(torch.load(args.gcn_ckpt, map_location=dev))
    gcn.eval()

    def gcn_score(txt: str) -> float:
        g = doc_to_graph(txt)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            return torch.sigmoid(gcn(g.to(dev))).item()

    # 1) Pre-count lines for an accurate ETA
    with open(args.train_jsonl, "r", encoding="utf8") as f:
        total = sum(1 for _ in f)

    X, y = [], []
    # 2) Rigorous tqdm bar with ETA and postfix stats
    with open(args.train_jsonl, "r", encoding="utf8") as fh:
        pbar = tqdm(
            fh,
            desc="⟳ extracting features",
            total=total,
            unit="samples",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.3,
        )
        for row in pbar:
            obj = json.loads(row)
            y.append(obj["label"])
            txt = obj["text"]

            c1 = curv.score(txt)
            c2 = comp.score(txt)
            c3 = gcn_score(txt)
            X.append([c1, c2, c3])

            # 3) Display the latest feature values in the bar
            pbar.set_postfix(
                curv=f"{c1:.3f}",
                comp=f"{c2:.3f}",
                gcn=f"{c3:.3f}",
                refresh=False,
            )

    # fit & dump
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, class_weight="balanced"),
    )
    model.fit(X, y)
    joblib.dump(model, args.out)
    print("✓ saved ensemble →", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True, help="Path to JSONL training file")
    parser.add_argument("--gcn_ckpt",    required=True, help="Path to syntax-GCN checkpoint (.pt)")
    parser.add_argument("--out",         required=True, help="Where to write the ensemble (.pkl)")
    parser.add_argument("--device",      default="cuda", help="torch device string (cuda or cpu)")
    args = parser.parse_args()
    main(args)
