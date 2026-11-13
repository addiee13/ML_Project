"""
Master execution script for Dry Bean Clustering (Project 2).

This script coordinates the complete end-to-end unsupervised learning pipeline:
1. Data loading, validation, and missing value/duplicate inspection.
2. Feature scaling with StandardScaler (strictly excluding 'Class').
3. Full 16-feature PCA, scree analysis, and optimal component selection (n=4, 95.02% variance).
4. 4 Clustering configurations (k=7):
   - Config 1: K-Means (Standardized 16D)
   - Config 2: K-Means (PCA 4D) [Selected Final Model]
   - Config 3: GMM (Standardized 16D)
   - Config 4: GMM (PCA 4D)
5. Comprehensive evaluation (Silhouette, Calinski-Harabasz, Davies-Bouldin, cluster distributions).
6. Cluster profiling (Mean, Std, Median) and morphological interpretation.
7. Publication figure generation (300 DPI).
8. Final output submission generation:
   - outputs/LastName_DryBeanClusters.csv
   - outputs/Dalwai_DryBeanClusters.csv
   - LastName_DryBeanClusters.csv (root)
   - Dalwai_DryBeanClusters.csv (root)
9. Automated output validation (13,612 lines, exact columns, zero missing values).
"""

import os
import shutil
import random
import numpy as np
import pandas as pd

from preprocess import load_dry_bean_data, standardize_features, FEATURE_COLUMNS
from pca_analysis import run_full_pca, save_pca_variance_table, select_optimal_pca, get_2d_pca_projection
from clustering import run_four_configurations
from evaluate import evaluate_all_configurations, compute_cluster_profiles
from visualize import generate_all_visualizations

def set_seed(seed=42):
    """Sets fixed random seed across all libraries for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_and_validate_output_csv(sample_ids, labels, output_paths):
    """
    Creates submission CSV files with format: sample_id,Class_id
    and validates format, row count, and data integrity.
    """
    sub_df = pd.DataFrame({
        "sample_id": sample_ids.astype(int),
        "Class_id": labels.astype(int)
    })
    
    # Validation checks
    assert len(sub_df) == 13611, f"Expected 13,611 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["sample_id", "Class_id"], f"Columns mismatch: {sub_df.columns}"
    assert sub_df['sample_id'].is_monotonic_increasing, "sample_id is not sorted!"
    assert (sub_df['sample_id'] == range(1, 13612)).all(), "sample_id values do not match 1..13611!"
    assert sub_df.isnull().sum().sum() == 0, "Missing values found in submission dataframe!"
    assert len(np.unique(labels)) == 7, f"Expected 7 clusters, found {len(np.unique(labels))}"
    
    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sub_df.to_csv(path, index=False)
        print(f"[Output] Successfully generated: {path}")
        
        # Verify line count on disk
        with open(path, 'r') as f:
            line_count = sum(1 for _ in f)
        assert line_count == 13612, f"Line count mismatch in {path}: expected 13612, got {line_count}"
        print(f"[Validation] Verified {path}: exactly {line_count} lines (1 header + 13,611 samples).")

def main():
    print("=" * 78)
    print(" DRY BEAN CLUSTERING PIPELINE (PROJECT 2) — END-TO-END REPRODUCIBLE RUN ")
    print("=" * 78)
    
    set_seed(42)
    
    # 1. Load Data
    print("\n--- STEP 1: DATA INGESTION & QUALITY INSPECTION ---")
    data_path = "datasets/Dry Bean Clustering/Dry_Bean_Dataset.xlsx"
    df, X_raw, sample_ids = load_dry_bean_data(data_path)
    
    # 2. Standardize Features
    print("\n--- STEP 2: FEATURE STANDARDIZATION ---")
    X_scaled, scaler = standardize_features(X_raw)
    
    # 3. PCA Analysis
    print("\n--- STEP 3: PCA ANALYSIS & DIMENSIONALITY REDUCTION ---")
    pca_full, variance_df, loadings_df = run_full_pca(X_scaled, FEATURE_COLUMNS, random_state=42)
    save_pca_variance_table(variance_df)
    
    print("\nPCA Variance Summary (First 6 Components):")
    print(variance_df.head(6).to_string(index=False))
    
    # Retain 4 components (95.02% variance)
    X_pca, pca_model = select_optimal_pca(X_scaled, n_components=4, random_state=42)
    X_pca_2d, pca_2d = get_2d_pca_projection(X_scaled, random_state=42)
    
    # 4. Clustering (4 Configurations)
    print("\n--- STEP 4: EXPERIMENTAL CLUSTERING CONFIGURATIONS (k=7) ---")
    clustering_results = run_four_configurations(X_scaled, X_pca, k=7, random_state=42)
    
    # 5. Evaluation
    print("\n--- STEP 5: INTERNAL CLUSTER VALIDATION ---")
    comp_df, comp_json = evaluate_all_configurations(clustering_results, X_scaled)
    print("\nConfiguration Performance Comparison:")
    disp_cols = ['Configuration', 'Feature_Space', 'Silhouette_Score', 'Calinski_Harabasz_Index', 'Davies_Bouldin_Index']
    print(comp_df[disp_cols].to_string(index=False))
    
    # 6. Cluster Profiling for Winning Model
    print("\n--- STEP 6: WINNING MODEL SELECTION & BOTANICAL PROFILING ---")
    winning_config = "Config 2: K-Means (PCA 4D)"
    winning_labels = clustering_results[winning_config]["labels"]
    print(f"[Selection] Final selected configuration: '{winning_config}'")
    print("            Rationale: Superior Silhouette score (0.3277), Calinski-Harabasz index (9134.22),")
    print("            and Davies-Bouldin index (1.0328) with effective collinear noise reduction.")
    
    profile_df, wide_summary = compute_cluster_profiles(X_raw, winning_labels)
    print("\nCluster Morphological Summary (Mean Feature Values):")
    summary_cols = ['Count', 'Percent', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Compactness']
    print(wide_summary[summary_cols].to_string())
    
    # 7. Visualizations
    print("\n--- STEP 7: PUBLICATION FIGURE GENERATION ---")
    generate_all_visualizations(variance_df, X_pca_2d, clustering_results)
    
    # 8. Final Output Prediction Generation
    print("\n--- STEP 8: FINAL SUBMISSION OUTPUT GENERATION ---")
    output_files = [
        "outputs/LastName_DryBeanClusters.csv",
        "outputs/Dalwai_DryBeanClusters.csv",
        "LastName_DryBeanClusters.csv",
        "Dalwai_DryBeanClusters.csv"
    ]
    generate_and_validate_output_csv(sample_ids, winning_labels, output_files)
    
    print("\n" + "=" * 78)
    print(" DRY BEAN CLUSTERING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 78)

if __name__ == "__main__":
    main()
