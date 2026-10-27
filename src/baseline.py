"""Reference baseline evaluation harness on SGCC dataset."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute comprehensive classification metrics for imbalanced anomaly detection."""
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }


def train_and_evaluate_baseline(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "gradient_boosting",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train a baseline classifier with stratified split and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if model_type == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state, max_iter=100
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_prob)
    print("\n=== Baseline Evaluation Results ===")
    print(f"Model: {model_type}")
    print(
        f"Test Samples: {len(y_test)} (Fraud: {np.sum(y_test)}, Normal: {len(y_test) - np.sum(y_test)})"
    )
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    return metrics
