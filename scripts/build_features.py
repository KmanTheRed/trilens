# File: scripts/build_features.py
#!/usr/bin/env python3
import sys, os, json, argparse
from tqdm import tqdm
import numpy as np
import torch

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lenses.lens1_curvature import PerturbationCurvature
from lenses.lens2_compress import CompressionAnomaly
from lenses.lens3_syntax_gcn import doc_to_graph, SyntaxGCN

def main(args):
    # Determine input JSONL path
    if args.input_jsonl:
        jsonl = args.input_jsonl
    elif args.data_root:
        jsonl = os.path.join(args.data_root, 'raw', 'train.jsonl')
    else:
        raise ValueError('Either --data_root or --input_jsonl must be provided')

    # Initialize lenses
    pc = PerturbationCurvature(k_swaps=args.k_swaps, device=args.device)
    ca = CompressionAnomaly()

    # Load trained GCN
    dummy = doc_to_graph("This is a test.")
    in_dim = dummy.x.size(1)
    gcn = SyntaxGCN(in_dim).to(args.device)
    gcn.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
    gcn.eval()

    # Count for progress bar
    total = sum(1 for _ in open(jsonl, encoding='utf8'))

    feats, labels = [], []
    with open(jsonl, encoding='utf8') as f:
        for line in tqdm(f, total=total, desc="Building features"):
            obj = json.loads(line)
            text = obj.get('text')
            label = obj.get('label', 0)
            if not isinstance(text, str):
                continue

            # Lens 1: perturbation curvature
            f1 = pc.score(text)
            # Lens 2: compression anomaly
            f2 = ca.score(text)

            # Lens 3: syntactic graph
            g = doc_to_graph(text)
            if g.x.size(0) == 0:
                # empty doc → assign neutral score
                f3 = 0.5
            else:
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
                g = g.to(args.device)
                with torch.no_grad():
                    f3 = gcn(g).item()

            feats.append([f1, f2, f3])
            labels.append(label)

    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    np.savez(args.output_path, X=X, y=y)
    print(f"Saved features to {args.output_path}: X.shape={X.shape}, y.shape={y.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Tri-Lens features with optional input_jsonl")
    parser.add_argument('--data_root',    type=str,
                        help="root of data folder containing raw/train.jsonl")
    parser.add_argument('--input_jsonl',  type=str,
                        help="path to jsonl override (e.g. raw/dev_standard.jsonl)")
    parser.add_argument('--ckpt_path',    type=str, required=True,
                        help="path to best_syntax_gcn.pt")
    parser.add_argument('--output_path',  type=str, required=True,
                        help="where to write .npz")
    parser.add_argument('--k_swaps',      type=int, default=5,
                        help="how many random synonym swaps")
    parser.add_argument('--device',       type=str, default='cpu',
                        help="cuda or cpu")
    args = parser.parse_args()
    main(args)
