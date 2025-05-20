#!/usr/bin/env python3
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

def main(train_features_path: str,
         dev_features_path: str,
         output_model: str,
         threshold_path: str):
    # Load train features
    data_train = np.load(train_features_path)
    X_train, y_train = data_train['X'], data_train['y']

    # Load dev features for threshold calibration
    data_dev = np.load(dev_features_path)
    X_dev, y_dev = data_dev['X'], data_dev['y']

    # 1) Hyperparameter search over RF
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth':    [None, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    print(f"Best RF params: {grid.best_params_}")

    # 2) Calibrate threshold on dev
    probs_dev = best_clf.predict_proba(X_dev)[:, 1]
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.linspace(0, 1, 101):
        preds = (probs_dev > thr).astype(int)
        f1 = f1_score(y_dev, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # 3) Report dev‐set performance
    preds_dev = (probs_dev > best_thr).astype(int)
    acc_dev = accuracy_score(y_dev, preds_dev)
    prec, rec, f1_dev, _ = precision_recall_fscore_support(y_dev, preds_dev, average='binary')
    print(f"\nDev performance @ thr={best_thr:.2f}: Acc={acc_dev:.4f}, F1={f1_dev:.4f}, P={prec:.4f}, R={rec:.4f}")
    print("\nFull dev report:")
    print(classification_report(y_dev, preds_dev, target_names=['human','ai']))

    # 4) Save model + threshold
    joblib.dump(best_clf, output_model)
    np.save(threshold_path, np.array([best_thr]))
    print(f"\nSaved RF ensemble to {output_model}, threshold to {threshold_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RF ensemble on train + calibrate on dev"
    )
    parser.add_argument('--train_features', required=True,
                        help='.npz with train X,y')
    parser.add_argument('--dev_features',   required=True,
                        help='.npz with dev   X,y')
    parser.add_argument('--output_model',   required=True,
                        help='where to write RF .pkl')
    parser.add_argument('--threshold_path', required=True,
                        help='where to write threshold .npy')
    args = parser.parse_args()
    main(args.train_features, args.dev_features, args.output_model, args.threshold_path)
