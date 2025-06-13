#!/usr/bin/env python3
import sys, os

# ensure project root is on PYTHONPATH so we can import lenses/
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse, json, torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from lenses.lens3_syntax_gcn import SyntaxGCN, doc_to_graph

def main():
    p = argparse.ArgumentParser(
        description="Evaluate Syntax GCN lens on a JSONL file"
    )
    p.add_argument(
        "--ckpt", "-c",
        required=True,
        help="Path to your .pt checkpoint"
    )
    p.add_argument(
        "--file", "-f",
        required=True,
        help="Single JSONL file to score"
    )
    p.add_argument(
        "--device", "-d",
        default="cuda",
        help="Torch device: cuda or cpu"
    )
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Probability cutoff for predicting machine_text (default=0.5)"
    )
    args = p.parse_args()

    # Load model
    in_dim = doc_to_graph("dummy").x.size(1)
    model = SyntaxGCN(in_dim=in_dim).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    # Count lines for progress bar
    with open(args.file, encoding="utf8") as fh:
        total_lines = sum(1 for _ in fh)

    y_true, y_pred = [], []
    with open(args.file, encoding="utf8") as fh, torch.no_grad():
        for line in tqdm(fh, total=total_lines, desc="Evaluating", unit="line"):
            obj  = json.loads(line)
            lbl  = 1 if "machine_text" in obj else 0
            text = obj.get("machine_text") or obj.get("human_text")

            # build graph & score
            g = doc_to_graph(text)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            prob = torch.sigmoid(model(g.to(args.device))).item()

            y_true.append(lbl)
            y_pred.append(1 if prob >= args.threshold else 0)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    print(f"{args.file}  |  Acc {acc:.4f}  F1 {f1:.4f}")

if __name__ == "__main__":
    main()
