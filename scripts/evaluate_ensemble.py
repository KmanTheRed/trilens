#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

def main(features_path, model_path, threshold_path):
    # Load held‐out features
    data = np.load(features_path)
    X, y = data['X'], data['y']

    # Load model + threshold
    clf = joblib.load(model_path)
    thr_arr = np.load(threshold_path)
    thr = float(thr_arr.item() if thr_arr.shape else thr_arr)

    # Predict
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > thr).astype(int)

    # Metrics
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    precision, recall, _, _ = precision_recall_fscore_support(y, preds, average='binary')

    print(f"Evaluation on {features_path}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print("\nFull classification report:")
    print(classification_report(y, preds, target_names=['human','ai']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features',  type=str, required=True,
                        help='.npz with X,y from build_features.py')
    parser.add_argument('--model',     type=str, required=True,
                        help='the .pkl from train_ensemble.py')
    parser.add_argument('--threshold', type=str, required=True,
                        help='the .npy from train_ensemble.py')
    args = parser.parse_args()
    main(args.features, args.model, args.threshold)
