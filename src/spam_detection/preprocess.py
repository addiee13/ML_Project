"""
Data loading and preprocessing module for Spam Email Detection.

Handles:
1. Loading spam_train1.csv and combining fragmented text across columns (v2, Unnamed: 2, 3, 4).
2. Loading spam_train2.csv and extracting clean text and numeric labels (label_num).
3. Merging datasets and removing cross-dataset and within-dataset duplicates.
4. Logging preprocessing statistics (counts, class proportions, duplicate count).
5. Loading test dataset spam_test.csv.
"""

import os
import logging
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def load_and_preprocess_train1(filepath: str) -> pd.DataFrame:
    """
    Load spam_train1.csv (SMS Spam Collection format).
    Combines text segments across v2, Unnamed: 2, Unnamed: 3, Unnamed: 4.
    Maps label v1: 'ham' -> 0, 'spam' -> 1.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    text_cols = ['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    def combine_segments(row):
        parts = [
            str(row[c]).strip() for c in text_cols
            if c in row and pd.notnull(row[c]) and str(row[c]).strip() != ''
        ]
        return " ".join(parts).strip()

    df['text'] = df.apply(combine_segments, axis=1)
    df['label'] = df['v1'].str.strip().str.lower().map({'ham': 0, 'spam': 1})

    if df['label'].isnull().any():
        raise ValueError("Encountered unmapped labels in train1 dataset.")

    df['source'] = 'train1'
    return df[['text', 'label', 'source']]


def load_and_preprocess_train2(filepath: str) -> pd.DataFrame:
    """
    Load spam_train2.csv (Email Spam format).
    Extracts 'text' and 'label_num' (0 for ham, 1 for spam).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label_num'].astype(int)
    df['source'] = 'train2'

    return df[['text', 'label', 'source']]


def load_and_preprocess_test(filepath: str) -> pd.DataFrame:
    """
    Load spam_test.csv.
    Extracts 'message' text without labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    df['text'] = df['message'].astype(str).str.strip()
    return df[['text']]


def get_combined_clean_training_data(
    train1_path: str = "datasets/Spam Email Detection/spam_train1.csv",
    train2_path: str = "datasets/Spam Email Detection/spam_train2.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Combines train1 and train2, performs deduplication, and returns clean DataFrame
    along with detailed preprocessing metadata and statistics.
    """
    df1 = load_and_preprocess_train1(train1_path)
    df2 = load_and_preprocess_train2(train2_path)

    stats = {
        "train1_total": len(df1),
        "train1_ham": int((df1['label'] == 0).sum()),
        "train1_spam": int((df1['label'] == 1).sum()),
        "train2_total": len(df2),
        "train2_ham": int((df2['label'] == 0).sum()),
        "train2_spam": int((df2['label'] == 1).sum()),
        "combined_before_dedup": len(df1) + len(df2),
        "combined_ham_before_dedup": int((df1['label'] == 0).sum() + (df2['label'] == 0).sum()),
        "combined_spam_before_dedup": int((df1['label'] == 1).sum() + (df2['label'] == 1).sum()),
    }

    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Check for label conflicts on identical texts
    conflicts = combined_df.groupby('text')['label'].nunique()
    num_conflicts = int((conflicts > 1).sum())
    stats["num_conflicting_labels"] = num_conflicts

    # Count duplicates within each source and across sources
    stats["train1_internal_duplicates"] = int(df1['text'].duplicated().sum())
    stats["train2_internal_duplicates"] = int(df2['text'].duplicated().sum())
    stats["cross_dataset_overlap"] = int(len(set(df1['text']).intersection(set(df2['text']))))

    # Deduplicate based on text
    initial_count = len(combined_df)
    clean_df = combined_df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    duplicates_removed = initial_count - len(clean_df)

    stats["duplicates_removed"] = int(duplicates_removed)
    stats["clean_total"] = len(clean_df)
    stats["clean_ham"] = int((clean_df['label'] == 0).sum())
    stats["clean_spam"] = int((clean_df['label'] == 1).sum())
    stats["clean_ham_ratio"] = float(stats["clean_ham"] / stats["clean_total"])
    stats["clean_spam_ratio"] = float(stats["clean_spam"] / stats["clean_total"])

    return clean_df, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_df, stats = get_combined_clean_training_data()
    print("=== Preprocessing Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    test_df = load_and_preprocess_test("datasets/Spam Email Detection/spam_test.csv")
    print(f"Test dataset loaded: {len(test_df)} samples.")
