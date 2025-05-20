#!/usr/bin/env python3
"""
scripts/plot_lens_histograms.py

Usage:
  ./scripts/plot_lens_histograms.py <features.npz> --outdir <plot_dir>

Reads X,y from the .npz, then for each of the 3 lenses makes a
two-color histogram (human vs AI) and dumps it to PNGs in --outdir.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(features_path: str, outdir: str):
    data = np.load(features_path)
    X, y = data['X'], data['y']
    os.makedirs(outdir, exist_ok=True)

    lens_names = ["Lens 1 (curvature)", "Lens 2 (compression)", "Lens 3 (syntax)"]
    colors = ['C0','C1']
    labels = ['human','ai']

    for i, name in enumerate(lens_names):
        plt.figure(figsize=(6,4))
        # human
        plt.hist(X[y==0,i], bins=50, alpha=0.6,
                 label='human', color=colors[0], density=True)
        # ai
        plt.hist(X[y==1,i], bins=50, alpha=0.6,
                 label='ai',    color=colors[1], density=True)
        plt.title(name)
        plt.xlabel("score")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(outdir, f"hist_lens{i+1}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Wrote {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('features', help='.npz file with X and y arrays')
    p.add_argument('--outdir', required=True, help='where to save PNGs')
    args = p.parse_args()
    main(args.features, args.outdir)
