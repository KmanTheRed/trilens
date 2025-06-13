#!/usr/bin/env python3
import sys
import os
import json
import argparse
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
    gcn = SyntaxGCN(in_dim=in_dim).to(args.device)
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

            # Lens 1: synonym‐swap curvature
            f1 = pc.score(text)
            # Lens 2: compression anomaly
            f2 = ca.score(text)
            # Lens 3: syntax‐GCN
            g = doc_to_graph(text)
            # **skip any empty graphs**
            if g.x.size(0) == 0:
                continue
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            g = g.to(args.device)
            with torch.no_grad():
                f3 = gcn(g).item()

            feats.append([f1, f2, f3])
            labels.append(label)

    # stack into arrays
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    np.savez(args.output_path, X=X, y=y)
    print(f"Saved features to {args.output_path}: X.shape={X.shape}, y.shape={y.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build Tri-Lens features with optional input_jsonl"
    )
    parser.add_argument(
        '--data_root', type=str,
        help='root folder containing raw/train.jsonl'
    )
    parser.add_argument(
        '--input_jsonl', type=str, default=None,
        help='path to a custom JSONL (overrides data_root/raw/train.jsonl)'
    )
    parser.add_argument(
        '--ckpt_path', type=str, required=True,
        help='path to best_syntax_gcn.pt'
    )
    parser.add_argument(
        '--output_path', type=str, default='features_train.npz',
        help='where to write features .npz'
    )
    parser.add_argument(
        '--k_swaps', type=int, default=5,
        help='number of synonym swaps for curvature lens'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='device to run on: cuda or cpu'
    )
    args = parser.parse_args()
    main(args)
