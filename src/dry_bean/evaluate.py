"""
Evaluation and Profiling module for Dry Bean Clustering.

This module computes:
1. Internal cluster validation metrics:
   - Silhouette Score (higher is better, [-1, 1])
   - Calinski-Harabasz Index (higher is better)
   - Davies-Bouldin Index (lower is better, >= 0)
2. Cluster size distribution (counts and percentages).
3. Comprehensive comparison table across all 4 configurations, exported as CSV and JSON.
4. Rich cluster morphological profiles (Mean, Std, Median across all 16 original features).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def compute_metrics(X, labels):
    """
    Computes internal validation metrics for a clustering solution.
    
    Args:
        X (np.ndarray): Feature representation.
        labels (np.ndarray): Cluster labels.
        
    Returns:
        dict: Silhouette, Calinski-Harabasz, Davies-Bouldin, and cluster sizes.
    """
    sil = float(silhouette_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique_labels, counts)}
    
    return {
        "silhouette_score": round(sil, 4),
        "calinski_harabasz_index": round(ch, 2),
        "davies_bouldin_index": round(db, 4),
        "cluster_sizes": cluster_sizes,
        "n_clusters": len(unique_labels)
    }

def evaluate_all_configurations(clustering_results, X_scaled,
                                csv_path="outputs/tables/drybean_clustering_comparison.csv",
                                json_path="outputs/tables/drybean_clustering_comparison.json"):
    """
    Evaluates all 4 experimental configurations and exports comparison tables.
    
    Args:
        clustering_results (dict): Output from clustering.run_four_configurations.
        X_scaled (np.ndarray): Standardized feature matrix (16D).
        csv_path (str): CSV export destination.
        json_path (str): JSON export destination.
        
    Returns:
        comparison_df (pd.DataFrame): Comparison table.
        comparison_json (dict): Detailed metrics structure.
    """
    records = []
    json_output = {}
    
    for config_name, res in clustering_results.items():
        labels = res["labels"]
        X_algo = res["X_algo"]
        input_space = res["input_space"]
        algo_name = res["algorithm"]
        
        # Metrics in algorithm space
        metrics_algo = compute_metrics(X_algo, labels)
        
        # Metrics in common standardized space for cross-space comparison
        metrics_scaled = compute_metrics(X_scaled, labels)
        
        # Format cluster sizes string (e.g., C0: 519, C1: 3262, ...)
        sizes_str = ", ".join([f"C{k}: {v}" for k, v in sorted(metrics_algo["cluster_sizes"].items())])
        
        record = {
            "Configuration": config_name,
            "Algorithm": algo_name,
            "Feature_Space": input_space,
            "Silhouette_Score": metrics_algo["silhouette_score"],
            "Calinski_Harabasz_Index": metrics_algo["calinski_harabasz_index"],
            "Davies_Bouldin_Index": metrics_algo["davies_bouldin_index"],
            "Silhouette_Score_ScaledSpace": metrics_scaled["silhouette_score"],
            "Calinski_Harabasz_ScaledSpace": metrics_scaled["calinski_harabasz_index"],
            "Davies_Bouldin_ScaledSpace": metrics_scaled["davies_bouldin_index"],
            "Cluster_Sizes": sizes_str
        }
        records.append(record)
        
        json_output[config_name] = {
            "algorithm": algo_name,
            "feature_space": input_space,
            "metrics_in_algorithm_space": metrics_algo,
            "metrics_in_standardized_space": metrics_scaled
        }
        
    comparison_df = pd.DataFrame(records)
    
    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    comparison_df.to_csv(csv_path, index=False)
    print(f"[Evaluate] Saved comparison CSV to {csv_path}")
    
    # Save JSON
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)
    print(f"[Evaluate] Saved comparison JSON to {json_path}")
    
    return comparison_df, json_output

def compute_cluster_profiles(X_raw, labels, output_path="outputs/tables/drybean_cluster_profiles.csv"):
    """
    Computes cluster profiles (Mean, Std, Median) of all 16 original numerical features
    for each of the 7 clusters.
    
    Args:
        X_raw (pd.DataFrame): Original unscaled 16 features.
        labels (np.ndarray): Cluster labels from the selected model.
        output_path (str): Path to save cluster profiles CSV.
        
    Returns:
        profile_df (pd.DataFrame): Aggregated profile table.
    """
    df_temp = X_raw.copy()
    df_temp['Cluster'] = labels
    
    feature_cols = [c for c in X_raw.columns if c != 'Cluster']
    
    # Aggregate mean, std, median
    agg_mean = df_temp.groupby('Cluster')[feature_cols].mean().round(4)
    agg_std = df_temp.groupby('Cluster')[feature_cols].std().round(4)
    agg_median = df_temp.groupby('Cluster')[feature_cols].median().round(4)
    counts = df_temp['Cluster'].value_counts().sort_index()
    percentages = (counts / len(df_temp) * 100).round(2)
    
    # Build clean hierarchical profile table
    rows = []
    for c_id in sorted(df_temp['Cluster'].unique()):
        count_c = counts[c_id]
        pct_c = percentages[c_id]
        
        for feat in feature_cols:
            rows.append({
                "Cluster": f"Cluster {c_id}",
                "Sample_Count": count_c,
                "Sample_Percent": f"{pct_c}%",
                "Feature": feat,
                "Mean": agg_mean.loc[c_id, feat],
                "Std": agg_std.loc[c_id, feat],
                "Median": agg_median.loc[c_id, feat]
            })
            
    profile_df = pd.DataFrame(rows)
    
    # Also save a wide version for convenient inspection: Mean per feature by Cluster
    wide_mean = agg_mean.copy()
    wide_mean['Count'] = counts
    wide_mean['Percent'] = percentages
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile_df.to_csv(output_path, index=False)
    print(f"[Evaluate] Saved comprehensive cluster profiles to {output_path}")
    
    # Also save the wide mean/median profile summary
    wide_path = output_path.replace(".csv", "_summary.csv")
    wide_mean.to_csv(wide_path)
    print(f"[Evaluate] Saved wide profile summary to {wide_path}")
    
    return profile_df, wide_mean

def get_morphological_interpretations():
    """
    Returns botanical and morphological interpretation guide for dry bean clusters.
    """
    return {
        "Size_Axis": "Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter measure overall physical volume and scale.",
        "Elongation_Axis": "AspectRation (Major/Minor) and Eccentricity quantify seed elongation vs roundness. High values indicate long/cylindrical beans; low values indicate spherical beans.",
        "Compactness_Axis": "Compactness and roundness reflect circularity and plumpness. Compactness > 0.88 indicates nearly spherical seeds (e.g. Seker variety).",
        "Boundary_Axis": "Solidity (Area/ConvexArea) and Extent quantify surface smoothness and bounding-box occupancy."
    }

if __name__ == "__main__":
    from preprocess import load_dry_bean_data, standardize_features
    from pca_analysis import select_optimal_pca
    from clustering import run_four_configurations
    
    df, X_raw, sample_ids = load_dry_bean_data()
    X_scaled, scaler = standardize_features(X_raw)
    X_pca, pca_model = select_optimal_pca(X_scaled, n_components=4)
    
    results = run_four_configurations(X_scaled, X_pca)
    comp_df, comp_json = evaluate_all_configurations(results, X_scaled)
    print("\nClustering Comparison Table:")
    print(comp_df[['Configuration', 'Silhouette_Score', 'Calinski_Harabasz_Index', 'Davies_Bouldin_Index']])
    
    # Selected best model: Config 2 (KMeans on PCA 4D)
    best_labels = results["Config 2: K-Means (PCA 4D)"]["labels"]
    profile_df, wide_mean = compute_cluster_profiles(X_raw, best_labels)
    print("\nCluster Profiles Summary (Mean Area & Compactness):")
    print(wide_mean[['Count', 'Percent', 'Area', 'MajorAxisLength', 'AspectRation', 'Compactness']])
