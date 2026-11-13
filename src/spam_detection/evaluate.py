"""
Cross-validation evaluation, metrics computation, and visualization module.

Implements:
1. Stratified 5-Fold Cross Validation with strict data leakage prevention.
2. Comprehensive evaluation metrics:
   - Accuracy
   - Macro-F1
   - Spam Precision
   - Spam Recall
   - Spam F1-Score
   - Ham False Positive Rate (Ham emails classified as Spam / Total actual Ham emails)
3. Out-of-fold predictions and probability tracking.
4. Exporting comparison tables to CSV and JSON.
5. High-resolution visualizations:
   - Confusion Matrix (outputs/figures/spam_confusion_matrix.png)
   - Threshold Tradeoff Curves (outputs/figures/spam_threshold_tradeoff.png)
   - ROC & Precision-Recall Curves (outputs/figures/spam_roc_pr_curve.png)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.base import clone

logger = logging.getLogger(__name__)


def evaluate_models_cv(
    models: Dict[str, Any],
    vectorizer_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Performs Stratified K-Fold Cross Validation across all models.
    Strictly fits the TF-IDF vectorizer only on X_train for each fold.
    Returns:
    - comparison_df: Summary DataFrame with mean and std for all metrics.
    - oof_predictions: Dictionary mapping model names to out-of-fold predictions.
    - oof_probabilities: Dictionary mapping model names to out-of-fold predicted probabilities (or None).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    records = []
    oof_predictions = {}
    oof_probabilities = {}

    for name, model in models.items():
        logger.info(f"Evaluating {name} with {n_splits}-fold CV...")
        
        fold_acc = []
        fold_macro_f1 = []
        fold_spam_prec = []
        fold_spam_rec = []
        fold_spam_f1 = []
        fold_ham_fpr = []
        
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros(len(y), dtype=float)
        has_proba = hasattr(model, "predict_proba") or hasattr(model, "decision_function")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            # Fit vectorizer strictly on training fold to prevent leakage
            vec = vectorizer_factory()
            X_train_vec = vec.fit_transform(X[train_idx])
            X_val_vec = vec.transform(X[val_idx])
            
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf = clone(model)
            clf.fit(X_train_vec, y_train)
            
            val_preds = clf.predict(X_val_vec)
            oof_preds[val_idx] = val_preds
            
            if hasattr(clf, "predict_proba"):
                val_probs = clf.predict_proba(X_val_vec)[:, 1]
                oof_probs[val_idx] = val_probs
            elif hasattr(clf, "decision_function"):
                # Normalize decision function to [0, 1] via logistic sigmoid for thresholding
                df_scores = clf.decision_function(X_val_vec)
                val_probs = 1.0 / (1.0 + np.exp(-df_scores))
                oof_probs[val_idx] = val_probs
            
            # Compute fold metrics
            acc = accuracy_score(y_val, val_preds)
            mf1 = f1_score(y_val, val_preds, average="macro")
            sp_p = precision_score(y_val, val_preds, pos_label=1, zero_division=0)
            sp_r = recall_score(y_val, val_preds, pos_label=1, zero_division=0)
            sp_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
            
            # Ham FPR = FP / (FP + TN) where true class is 0 (Ham)
            tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
            ham_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            fold_acc.append(acc)
            fold_macro_f1.append(mf1)
            fold_spam_prec.append(sp_p)
            fold_spam_rec.append(sp_r)
            fold_spam_f1.append(sp_f1)
            fold_ham_fpr.append(ham_fpr)

        oof_predictions[name] = oof_preds
        oof_probabilities[name] = oof_probs if has_proba else None

        records.append({
            "Model": name,
            "Accuracy_Mean": np.mean(fold_acc),
            "Accuracy_Std": np.std(fold_acc),
            "Macro_F1_Mean": np.mean(fold_macro_f1),
            "Macro_F1_Std": np.std(fold_macro_f1),
            "Spam_Precision_Mean": np.mean(fold_spam_prec),
            "Spam_Precision_Std": np.std(fold_spam_prec),
            "Spam_Recall_Mean": np.mean(fold_spam_rec),
            "Spam_Recall_Std": np.std(fold_spam_rec),
            "Spam_F1_Mean": np.mean(fold_spam_f1),
            "Spam_F1_Std": np.std(fold_spam_f1),
            "Ham_FPR_Mean": np.mean(fold_ham_fpr),
            "Ham_FPR_Std": np.std(fold_ham_fpr),
        })

    comparison_df = pd.DataFrame(records)
    # Sort primarily by Spam F1 descending
    comparison_df = comparison_df.sort_values(by="Spam_F1_Mean", ascending=False).reset_index(drop=True)
    return comparison_df, oof_predictions, oof_probabilities


def compute_threshold_tradeoffs(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> pd.DataFrame:
    """
    Computes performance metrics across various probability thresholds.
    """
    records = []
    for th in thresholds:
        preds = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        ham_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        records.append({
            "Threshold": th,
            "Accuracy": accuracy_score(y_true, preds),
            "Macro_F1": f1_score(y_true, preds, average="macro"),
            "Spam_Precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
            "Spam_Recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
            "Spam_F1": f1_score(y_true, preds, pos_label=1, zero_division=0),
            "Ham_FPR": ham_fpr,
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "True_Negatives": int(tn)
        })
    return pd.DataFrame(records)


def save_comparison_tables(
    df: pd.DataFrame,
    csv_path: str = "outputs/tables/spam_model_comparison.csv",
    json_path: str = "outputs/tables/spam_model_comparison.json"
):
    """Saves model comparison results in both CSV and JSON formats."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Model comparison saved to {csv_path} and {json_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str = "outputs/figures/spam_confusion_matrix.png"
):
    """
    Generates and saves a high-quality annotated confusion matrix plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    ham_fpr = fp / (fp + tn)
    spam_f1 = f1_score(y_true, y_pred, pos_label=1)

    plt.figure(figsize=(7, 6))
    labels = ["Ham (0)", "Spam (1)"]
    
    # Calculate percentages relative to class totals
    annot_matrix = np.empty_like(cm, dtype=object)
    annot_matrix[0, 0] = f"{tn:,}\n(TN: {tn/(tn+fp):.1%})"
    annot_matrix[0, 1] = f"{fp:,}\n(FP: {fp/(tn+fp):.1%})"
    annot_matrix[1, 0] = f"{fn:,}\n(FN: {fn/(fn+tp):.1%})"
    annot_matrix[1, 1] = f"{tp:,}\n(TP: {tp/(fn+tp):.1%})"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(
        f"Out-of-Fold Confusion Matrix\n{model_name}\nSpam F1: {spam_f1:.4f} | Ham FPR: {ham_fpr:.2%}",
        fontsize=13,
        pad=15,
        fontweight="bold"
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")


def plot_threshold_tradeoff(
    tradeoff_df: pd.DataFrame,
    model_name: str,
    output_path: str = "outputs/figures/spam_threshold_tradeoff.png"
):
    """
    Plots the operational tradeoff between Spam Precision, Recall, F1, and Ham FPR
    across decision thresholds. Optimized for print and column-width readability.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(6.5, 4.4))
    plt.plot(tradeoff_df["Threshold"], tradeoff_df["Spam_Precision"], marker="o", label="Spam Precision", color="#2ca02c", linewidth=2.8, markersize=6)
    plt.plot(tradeoff_df["Threshold"], tradeoff_df["Spam_Recall"], marker="s", label="Spam Recall", color="#ff7f0e", linewidth=2.8, markersize=6)
    plt.plot(tradeoff_df["Threshold"], tradeoff_df["Spam_F1"], marker="^", label="Spam F1-Score", color="#1f77b4", linewidth=3.2, markersize=7)
    plt.plot(tradeoff_df["Threshold"], tradeoff_df["Ham_FPR"], marker="x", label="Ham FPR (Cost)", color="#d62728", linewidth=2.8, linestyle="--", markersize=7)

    plt.axvline(x=0.5, color="#4b5563", linestyle=":", linewidth=2.0, alpha=0.85, label=r"Default ($\tau=0.50$)")
    plt.axvline(x=0.7, color="#7c3aed", linestyle="-.", linewidth=2.0, alpha=0.85, label=r"Constrained ($\tau=0.70$)")
    
    plt.title(f"Operational Decision Threshold Tradeoffs\n{model_name}", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Probability Decision Threshold ($\tau$)", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.ylim(-0.02, 1.03)
    plt.xticks(np.arange(0.1, 1.0, 0.1), fontsize=10)
    plt.yticks(np.arange(0.0, 1.1, 0.2), fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(0.02, 0.44), framealpha=0.92, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Threshold tradeoff plot saved to {output_path}")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    output_path: str = "outputs/figures/spam_roc_pr_curve.png"
):
    """
    Generates side-by-side ROC curve and Precision-Recall curve.
    Optimized for print and column-width readability.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    
    prec, rec, _ = precision_recall_curve(y_true, probs)
    avg_prec = average_precision_score(y_true, probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=3.0, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#6b7280", lw=1.8, linestyle="--", label="Random Chance")
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (Ham Misclass.)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Spam Recall)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.92)
    
    # Precision-Recall Curve
    axes[1].plot(rec, prec, color="#2ca02c", lw=3.0, label=f"PR (AP = {avg_prec:.4f})")
    baseline_pr = np.mean(y_true)
    axes[1].axhline(y=baseline_pr, color="#6b7280", linestyle="--", lw=1.8, label=f"Baseline ({baseline_pr:.1%})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Spam Recall (Detection Rate)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Spam Precision (Purity)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=10, framealpha=0.92)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"ROC and PR curves saved to {output_path}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"ROC and PR curves saved to {output_path}")
