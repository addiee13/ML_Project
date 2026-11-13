"""
Clustering module for Dry Bean Data.

This module implements the 4 experimental configurations (k = 7):
- Config 1: K-Means on Standardized Features
- Config 2: K-Means on PCA-transformed Features (optimal n=4 components, 95.02% variance)
- Config 3: Method 2 (Gaussian Mixture Model) on Standardized Features
- Config 4: Method 2 (Gaussian Mixture Model) on PCA-transformed Features

All random seeds are fixed (random_state=42) for exact reproducibility.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def train_kmeans(X, n_clusters=7, random_state=42, n_init=10, max_iter=300):
    """
    Fits K-Means clustering algorithm.
    
    Args:
        X (np.ndarray): Input feature matrix.
        n_clusters (int): Number of clusters (k=7).
        random_state (int): Fixed random seed.
        n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        model (KMeans): Fitted KMeans model.
        labels (np.ndarray): Cluster labels (0 to 6).
    """
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return model, labels

def train_gmm(X, n_components=7, random_state=42, covariance_type='full', n_init=5, max_iter=200):
    """
    Fits Gaussian Mixture Model (GMM) clustering algorithm.
    
    Args:
        X (np.ndarray): Input feature matrix.
        n_components (int): Number of mixture components (k=7).
        random_state (int): Fixed random seed.
        covariance_type (str): Covariance matrix parameterization ('full').
        n_init (int): Number of initializations.
        max_iter (int): Maximum EM iterations.
        
    Returns:
        model (GaussianMixture): Fitted GMM model.
        labels (np.ndarray): Hard cluster assignments (0 to 6).
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return model, labels

def train_agglomerative(X, n_clusters=7, linkage='ward'):
    """
    Fits Agglomerative Hierarchical Clustering algorithm.
    
    Args:
        X (np.ndarray): Input feature matrix.
        n_clusters (int): Number of clusters (k=7).
        linkage (str): Linkage criterion ('ward').
        
    Returns:
        model (AgglomerativeClustering): Fitted model.
        labels (np.ndarray): Cluster labels (0 to 6).
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)
    return model, labels

def run_four_configurations(X_scaled, X_pca, k=7, random_state=42):
    """
    Executes the 4 core experimental configurations specified in the project:
    1. K-Means on Standardized Features
    2. K-Means on PCA-transformed Features
    3. GMM on Standardized Features
    4. GMM on PCA-transformed Features
    
    Returns:
        results_dict: Dictionary mapping configuration name to model, labels, and input data.
    """
    results = {}
    
    # Config 1: K-Means on Standardized Features
    print("[Clustering] Fitting Config 1: K-Means on Standardized Features...")
    km_scaled, km_scaled_labels = train_kmeans(X_scaled, n_clusters=k, random_state=random_state)
    results["Config 1: K-Means (Standardized)"] = {
        "model": km_scaled,
        "labels": km_scaled_labels,
        "input_space": "Standardized (16D)",
        "X_algo": X_scaled,
        "algorithm": "K-Means"
    }
    
    # Config 2: K-Means on PCA-transformed Features
    print("[Clustering] Fitting Config 2: K-Means on PCA-transformed Features...")
    km_pca, km_pca_labels = train_kmeans(X_pca, n_clusters=k, random_state=random_state)
    results["Config 2: K-Means (PCA 4D)"] = {
        "model": km_pca,
        "labels": km_pca_labels,
        "input_space": "PCA (4D, 95.02% Var)",
        "X_algo": X_pca,
        "algorithm": "K-Means"
    }
    
    # Config 3: GMM on Standardized Features
    print("[Clustering] Fitting Config 3: GMM on Standardized Features...")
    gmm_scaled, gmm_scaled_labels = train_gmm(X_scaled, n_components=k, random_state=random_state)
    results["Config 3: GMM (Standardized)"] = {
        "model": gmm_scaled,
        "labels": gmm_scaled_labels,
        "input_space": "Standardized (16D)",
        "X_algo": X_scaled,
        "algorithm": "Gaussian Mixture Model"
    }
    
    # Config 4: GMM on PCA-transformed Features
    print("[Clustering] Fitting Config 4: GMM on PCA-transformed Features...")
    gmm_pca, gmm_pca_labels = train_gmm(X_pca, n_components=k, random_state=random_state)
    results["Config 4: GMM (PCA 4D)"] = {
        "model": gmm_pca,
        "labels": gmm_pca_labels,
        "input_space": "PCA (4D, 95.02% Var)",
        "X_algo": X_pca,
        "algorithm": "Gaussian Mixture Model"
    }
    
    print("[Clustering] All 4 configurations fitted successfully.")
    return results

if __name__ == "__main__":
    from preprocess import load_dry_bean_data, standardize_features
    from pca_analysis import select_optimal_pca
    
    df, X_raw, sample_ids = load_dry_bean_data()
    X_scaled, scaler = standardize_features(X_raw)
    X_pca, pca_model = select_optimal_pca(X_scaled, n_components=4)
    
    results = run_four_configurations(X_scaled, X_pca)
    for name, res in results.items():
        print(f"  {name}: {len(np.unique(res['labels']))} clusters, label distribution: {np.bincount(res['labels'])}")
