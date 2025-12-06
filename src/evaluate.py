"""Evaluation metrics and plotting for classification."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import MODELS_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)


def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """Save Precision-Recall curve plot."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, color="blue", lw=2, label=f"AUC = {auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_evaluation_report(report: Dict[str, Any], save_path: Path) -> None:
    """Save evaluation report to JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {save_path}")


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    plots_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate classifier: compute metrics, save plots and report."""
    plots_dir = plots_dir or PLOTS_DIR
    report_path = report_path or (MODELS_DIR / "evaluation_report.json")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2)),
    }

    logger.info("=" * 50)
    logger.info("CLASSIFICATION EVALUATION")
    logger.info(f"PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"F1: {metrics['f1_score']:.4f} | F2: {metrics['f2_score']:.4f}")
    logger.info("=" * 50)

    # Save plots
    save_roc_curve(y_true, y_proba, plots_dir / "roc_curve.png")
    save_precision_recall_curve(y_true, y_proba, plots_dir / "pr_curve.png")
    save_confusion_matrix(y_true, y_pred, plots_dir / "confusion_matrix.png")
    logger.info(f"Plots saved to {plots_dir}")

    # Save report
    report = {
        "metrics": metrics,
        "classification_report": classification_report(y_true, y_pred),
    }
    save_evaluation_report(report, report_path)

    return report