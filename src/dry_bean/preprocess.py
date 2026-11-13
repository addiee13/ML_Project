"""
Preprocessing module for Dry Bean Clustering.

This module handles:
1. Loading the dataset from Excel/CSV.
2. Data hygiene checks: missing values, duplicates, sample ID continuity.
3. Feature isolation: strictly excluding 'Class' and preserving 'sample_id'.
4. Feature standardization using StandardScaler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

def load_dry_bean_data(data_path="datasets/Dry Bean Clustering/Dry_Bean_Dataset.xlsx"):
    """
    Loads the Dry Bean dataset and performs validation checks.
    
    Returns:
        df (pd.DataFrame): Raw dataframe.
        X_raw (pd.DataFrame): 16 numerical features.
        sample_ids (pd.Series): sample_id column (1 to 13,611).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_excel(data_path)
    print(f"[Preprocess] Loaded dataset from {data_path} with shape {df.shape}")
    
    # Validation checks
    assert 'sample_id' in df.columns, "sample_id column missing!"
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Feature {col} missing from dataset!"
        
    # Check sample_id
    sample_ids = df['sample_id'].copy()
    assert (sample_ids == range(1, len(df) + 1)).all(), "sample_id is not consecutive 1 to N!"
    
    # Check missing values in features
    X_raw = df[FEATURE_COLUMNS].copy()
    missing_counts = X_raw.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"[Preprocess] Warning: missing values found:\n{missing_counts[missing_counts > 0]}")
    else:
        print("[Preprocess] No missing values found in numerical features (100% complete).")
        
    # Check duplicate rows in features
    duplicates = X_raw.duplicated().sum()
    print(f"[Preprocess] Duplicate feature records count: {duplicates}")
    
    # Ensure Class is excluded
    if 'Class' in df.columns:
        print("[Preprocess] Notice: 'Class' column strictly excluded from all preprocessing and clustering pipelines.")
        
    return df, X_raw, sample_ids

def standardize_features(X_raw):
    """
    Standardizes the 16 numerical features using StandardScaler (mean=0, variance=1).
    
    Args:
        X_raw (pd.DataFrame or np.ndarray): Raw numerical features.
        
    Returns:
        X_scaled (np.ndarray): Standardized feature matrix.
        scaler (StandardScaler): Fitted scaler instance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"[Preprocess] Features standardized: shape={X_scaled.shape}, mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
    return X_scaled, scaler

if __name__ == "__main__":
    df, X_raw, sample_ids = load_dry_bean_data()
    X_scaled, scaler = standardize_features(X_raw)
    print("[Preprocess] Preprocessing module test passed successfully.")
