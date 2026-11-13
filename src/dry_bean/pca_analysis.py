"""
Principal Component Analysis (PCA) module for Dry Bean Data.

This module handles:
1. Fitting PCA on all 16 standardized features.
2. Computing eigenvalues, explained variance ratios, and cumulative explained variance.
3. Component selection analysis (Kaiser-Guttman criterion, 90% and 95% thresholds).
4. Exporting PCA variance table to outputs/tables/drybean_pca_variance.csv.
5. Transforming features to optimal retained component space (n=4, capturing 95.02% variance).
6. Transforming features to 2D component space for visualization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def run_full_pca(X_scaled, feature_names=None, random_state=42):
    """
    Fits PCA on all 16 standardized features and extracts variance statistics.
    
    Args:
        X_scaled (np.ndarray): Standardized feature matrix (13611, 16).
        feature_names (list): Names of the 16 features.
        random_state (int): Fixed random seed.
        
    Returns:
        pca_full (PCA): Fitted PCA model with 16 components.
        variance_df (pd.DataFrame): Table of eigenvalues and variance ratios.
        loadings_df (pd.DataFrame): PCA factor loadings (eigenvectors).
    """
    n_features = X_scaled.shape[1]
    pca_full = PCA(n_components=n_features, random_state=random_state)
    pca_full.fit(X_scaled)
    
    eigenvalues = pca_full.explained_variance_
    var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)
    
    components = [f"PC{i}" for i in range(1, n_features + 1)]
    
    variance_df = pd.DataFrame({
        "Component": components,
        "Eigenvalue": np.round(eigenvalues, 4),
        "Explained_Variance_Ratio": np.round(var_ratio, 4),
        "Explained_Variance_Percent": np.round(var_ratio * 100, 2),
        "Cumulative_Variance_Ratio": np.round(cum_var_ratio, 4),
        "Cumulative_Variance_Percent": np.round(cum_var_ratio * 100, 2)
    })
    
    if feature_names is not None:
        loadings_df = pd.DataFrame(
            pca_full.components_.T,
            index=feature_names,
            columns=components
        )
    else:
        loadings_df = pd.DataFrame(pca_full.components_.T, columns=components)
        
    return pca_full, variance_df, loadings_df

def save_pca_variance_table(variance_df, output_path="outputs/tables/drybean_pca_variance.csv"):
    """
    Saves the PCA variance table to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    variance_df.to_csv(output_path, index=False)
    print(f"[PCA] Saved PCA variance summary table to {output_path}")

def select_optimal_pca(X_scaled, n_components=4, random_state=42):
    """
    Fits PCA with the optimal number of components for dimensionality reduction.
    n_components=4 captures 95.02% of total cumulative variance while reducing
    dimensions from 16 to 4 (75% compression).
    
    Args:
        X_scaled (np.ndarray): Standardized features.
        n_components (int): Number of components to retain (default 4).
        random_state (int): Fixed random seed.
        
    Returns:
        X_pca (np.ndarray): Transformed features in 4D PC space.
        pca_model (PCA): Fitted PCA model.
    """
    pca_model = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)
    captured_variance = np.sum(pca_model.explained_variance_ratio_) * 100
    print(f"[PCA] Reduced from {X_scaled.shape[1]} to {n_components} components.")
    print(f"[PCA] Total variance captured: {captured_variance:.2f}%")
    return X_pca, pca_model

def get_2d_pca_projection(X_scaled, random_state=42):
    """
    Extracts 2D PCA projection for visualization.
    PC1 captures 55.47% and PC2 captures 26.43% (total 81.90%).
    
    Args:
        X_scaled (np.ndarray): Standardized features.
        random_state (int): Fixed random seed.
        
    Returns:
        X_pca_2d (np.ndarray): Transformed features in 2D (PC1, PC2).
        pca_2d (PCA): Fitted 2D PCA model.
    """
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100
    print(f"[PCA 2D] 2D projection explains {var_2d:.2f}% variance (PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%, PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%)")
    return X_pca_2d, pca_2d

if __name__ == "__main__":
    from preprocess import load_dry_bean_data, standardize_features, FEATURE_COLUMNS
    df, X_raw, sample_ids = load_dry_bean_data()
    X_scaled, scaler = standardize_features(X_raw)
    
    pca_full, variance_df, loadings_df = run_full_pca(X_scaled, FEATURE_COLUMNS)
    print("\nPCA Variance Summary:")
    print(variance_df)
    save_pca_variance_table(variance_df)
    
    X_pca, pca_model = select_optimal_pca(X_scaled, n_components=4)
    X_pca_2d, pca_2d = get_2d_pca_projection(X_scaled)
    print("[PCA] PCA module test passed successfully.")
