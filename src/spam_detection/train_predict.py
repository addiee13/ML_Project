"""
Master Training, Model Selection, Evaluation, and Prediction Pipeline
for Project 1: Spam Email Detection.

Executes the entire end-to-end pipeline:
1. Loads and preprocesses training datasets (train1, train2).
2. Deduplicates records and logs corpus distributions.
3. Performs Stratified 5-Fold Cross Validation across candidate models.
4. Generates model comparison tables (CSV, JSON).
5. Conducts probability threshold tradeoff analysis.
6. Generates figures:
   - Confusion Matrix (outputs/figures/spam_confusion_matrix.png)
   - Threshold Tradeoff Curves (outputs/figures/spam_threshold_tradeoff.png)
   - ROC & Precision-Recall Curves (outputs/figures/spam_roc_pr_curve.png)
7. Performs qualitative error analysis (outputs/tables/spam_error_analysis.csv/.json).
8. Refits the optimal model on 100% of the training data.
9. Predicts on test data (spam_test.csv, 6,447 emails).
10. Validates and saves predictions to:
    - outputs/Dalwai_Spam.txt (0/1)
    - outputs/LastName_Spam.txt (0/1)
    - outputs/Dalwai_Spam_text.txt (ham/spam)
    - outputs/LastName_Spam_text.txt (ham/spam)
    - Dalwai_Spam.txt (root directory)
    - LastName_Spam.txt (root directory)
"""

import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.spam_detection.preprocess import (
    get_combined_clean_training_data,
    load_and_preprocess_test
)
from src.spam_detection.models import (
    get_tfidf_vectorizer,
    get_candidate_models,
    RANDOM_SEED
)
from src.spam_detection.evaluate import (
    evaluate_models_cv,
    compute_threshold_tradeoffs,
    save_comparison_tables,
    plot_confusion_matrix,
    plot_threshold_tradeoff,
    plot_roc_pr_curves
)
from src.spam_detection.error_analysis import generate_error_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================================")
    logger.info("Starting Project 1: Spam Email Detection End-to-End Pipeline")
    logger.info("==================================================================")

    # 1. Load and clean training data
    logger.info("\n>>> Step 1: Loading and preprocessing training datasets...")
    clean_df, stats = get_combined_clean_training_data()
    
    logger.info("Corpus Statistics:")
    logger.info(f" - Train 1 (SMS): {stats['train1_total']} (Ham: {stats['train1_ham']}, Spam: {stats['train1_spam']})")
    logger.info(f" - Train 2 (Email): {stats['train2_total']} (Ham: {stats['train2_ham']}, Spam: {stats['train2_spam']})")
    logger.info(f" - Combined before dedup: {stats['combined_before_dedup']} (Ham: {stats['combined_ham_before_dedup']}, Spam: {stats['combined_spam_before_dedup']})")
    logger.info(f" - Internal duplicates removed: Train1={stats['train1_internal_duplicates']}, Train2={stats['train2_internal_duplicates']}, Cross-overlap={stats['cross_dataset_overlap']}")
    logger.info(f" - Total duplicates removed: {stats['duplicates_removed']}")
    logger.info(f" - Clean training size: {stats['clean_total']} (Ham: {stats['clean_ham']} [{stats['clean_ham_ratio']:.2%}], Spam: {stats['clean_spam']} [{stats['clean_spam_ratio']:.2%}])")
    logger.info(f" - Conflicting labels: {stats['num_conflicting_labels']}")

    X = clean_df['text'].values
    y = clean_df['label'].values

    # 2. Cross-Validation Evaluation across candidate models
    logger.info("\n>>> Step 2: Stratified 5-Fold Cross Validation across candidate models...")
    models = get_candidate_models(random_state=RANDOM_SEED)
    vectorizer_factory = get_tfidf_vectorizer
    
    comparison_df, oof_preds_dict, oof_probs_dict = evaluate_models_cv(
        models=models,
        vectorizer_factory=vectorizer_factory,
        X=X,
        y=y,
        n_splits=5,
        random_state=RANDOM_SEED
    )

    logger.info("\n" + "=" * 90)
    logger.info("MODEL COMPARISON RESULTS (Stratified 5-Fold Cross Validation)")
    logger.info("=" * 90)
    logger.info(
        comparison_df[[
            "Model", "Accuracy_Mean", "Macro_F1_Mean", "Spam_Precision_Mean",
            "Spam_Recall_Mean", "Spam_F1_Mean", "Ham_FPR_Mean"
        ]].to_string(index=False)
    )

    # Save comparison tables
    save_comparison_tables(comparison_df)

    # 3. Model Selection & Probability Tradeoff Analysis
    # Calibrated Linear SVM (C=2.0) delivers top Spam F1 while maintaining low Ham FPR
    best_model_name = "Calibrated Linear SVM (C=2.0)"
    logger.info(f"\n>>> Step 3: Selected optimal model: '{best_model_name}'")
    
    best_oof_preds = oof_preds_dict[best_model_name]
    best_oof_probs = oof_probs_dict[best_model_name]

    # Threshold tradeoff analysis
    tradeoff_df = compute_threshold_tradeoffs(y, best_oof_probs)
    tradeoff_csv = "outputs/tables/spam_threshold_tradeoffs.csv"
    tradeoff_df.to_csv(tradeoff_csv, index=False)
    logger.info(f"Threshold tradeoffs saved to {tradeoff_csv}")
    logger.info("\nThreshold Tradeoff Table:")
    logger.info(tradeoff_df[["Threshold", "Accuracy", "Macro_F1", "Spam_Precision", "Spam_Recall", "Spam_F1", "Ham_FPR", "False_Positives", "False_Negatives"]].to_string(index=False))

    # 4. Generate Figures
    logger.info("\n>>> Step 4: Generating evaluation figures...")
    plot_confusion_matrix(y, best_oof_preds, best_model_name, "outputs/figures/spam_confusion_matrix.png")
    plot_threshold_tradeoff(tradeoff_df, best_model_name, "outputs/figures/spam_threshold_tradeoff.png")
    plot_roc_pr_curves(y, best_oof_probs, best_model_name, "outputs/figures/spam_roc_pr_curve.png")

    # 5. Error Analysis
    logger.info("\n>>> Step 5: Performing qualitative error analysis...")
    generate_error_analysis(
        texts=X,
        y_true=y,
        y_pred=best_oof_preds,
        probs=best_oof_probs,
        output_csv="outputs/tables/spam_error_analysis.csv",
        output_json="outputs/tables/spam_error_analysis.json"
    )

    # 6. Fit Final Model on 100% Data and Predict on Test Set
    logger.info("\n>>> Step 6: Refitting optimal model on 100% of clean labeled training data...")
    final_vec = vectorizer_factory()
    X_full_vec = final_vec.fit_transform(X)
    
    final_model = models[best_model_name]
    final_model.fit(X_full_vec, y)

    # Load test dataset
    logger.info("\n>>> Step 7: Generating predictions for spam_test.csv (6,447 rows)...")
    test_df = load_and_preprocess_test("datasets/Spam Email Detection/spam_test.csv")
    X_test = test_df['text'].values
    X_test_vec = final_vec.transform(X_test)

    test_probs = final_model.predict_proba(X_test_vec)[:, 1]
    
    # We use default threshold 0.50 which achieves peak balanced F1 (0.945)
    test_preds_numeric = (test_probs >= 0.50).astype(int)
    test_preds_text = np.where(test_preds_numeric == 1, "spam", "ham")

    # Verification checks
    expected_rows = 6447
    if len(test_preds_numeric) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, got {len(test_preds_numeric)}")

    ham_pred_count = int((test_preds_numeric == 0).sum())
    spam_pred_count = int((test_preds_numeric == 1).sum())
    logger.info(f"Test Predictions Summary:")
    logger.info(f" - Total Samples: {len(test_preds_numeric):,}")
    logger.info(f" - Predicted Ham (0): {ham_pred_count:,} ({ham_pred_count/expected_rows:.2%})")
    logger.info(f" - Predicted Spam (1): {spam_pred_count:,} ({spam_pred_count/expected_rows:.2%})")

    # 7. Save prediction files in all required locations
    logger.info("\n>>> Step 8: Writing prediction files...")
    
    # Primary numeric predictions (0 / 1)
    file_map_numeric = [
        "outputs/Dalwai_Spam.txt",
        "outputs/LastName_Spam.txt",
        "Dalwai_Spam.txt",
        "LastName_Spam.txt"
    ]
    for path in file_map_numeric:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for pred in test_preds_numeric:
                f.write(f"{pred}\n")
        logger.info(f"Wrote {len(test_preds_numeric)} lines to {path}")

    # Text predictions ('ham' / 'spam') for complete interoperability
    file_map_text = [
        "outputs/Dalwai_Spam_text.txt",
        "outputs/LastName_Spam_text.txt"
    ]
    for path in file_map_text:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for pred in test_preds_text:
                f.write(f"{pred}\n")
        logger.info(f"Wrote {len(test_preds_text)} lines to {path}")

    # Final verification of file line counts
    logger.info("\n>>> Step 9: Verifying generated files integrity...")
    for path in file_map_numeric + file_map_text:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() != ""]
        assert len(lines) == expected_rows, f"Integrity check failed: {path} has {len(lines)} lines, expected {expected_rows}"
        logger.info(f"Verified {path}: exactly {len(lines)} valid non-empty lines.")

    logger.info("\n==================================================================")
    logger.info("Project 1: Spam Email Detection Pipeline Completed Successfully!")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
