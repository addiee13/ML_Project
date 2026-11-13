"""
Visualization module for Dry Bean Clustering and PCA.

This module generates publication-quality figures (300 DPI):
1. outputs/figures/drybean_pca_variance.png:
   Scree plot with individual and cumulative explained variance,
   Kaiser-Guttman criterion, and 90%/95% threshold lines.
2. outputs/figures/drybean_kmeans_clusters.png:
   2D PCA projection colored by K-Means cluster assignments,
   with centroids, morphology labels, and sample counts.
3. outputs/figures/drybean_method2_clusters.png:
   2D PCA projection colored by Method 2 (GMM) cluster assignments.

Clearly documents the distinction between PCA for 2D visualization (81.90% var)
vs PCA for dimensionality reduction before clustering (4D, 95.02% var).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

CLUSTER_PALETTE = [
    '#2b5c8f',  # Navy Blue
    '#d95f02',  # Burnt Orange
    '#7570b3',  # Purple
    '#1b9e77',  # Teal Green
    '#e7298a',  # Magenta
    '#e6ab02',  # Mustard Yellow
    '#a6761d'   # Bronze Brown
]

def plot_pca_variance(variance_df, output_path="outputs/figures/drybean_pca_variance.png"):
    """
    Plots the PCA scree plot and cumulative explained variance curve.
    
    Args:
        variance_df (pd.DataFrame): Output from pca_analysis.run_full_pca.
        output_path (str): Destination PNG path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    n_comp = len(variance_df)
    x = np.arange(1, n_comp + 1)
    
    # Left: Scree Plot (Eigenvalues & Individual Variance)
    bars = ax1.bar(x, variance_df['Explained_Variance_Percent'], color='#2b5c8f', alpha=0.85, width=0.6, label='Individual % Variance')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Scree Plot: Individual Explained Variance', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x], rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 65)
    
    # Add value annotations on top 5 bars
    for i, bar in enumerate(bars[:5]):
        val = variance_df['Explained_Variance_Percent'].iloc[i]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{val:.1f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a365d')
        
    # Kaiser-Guttman equivalent line (variance of 1 / 16 = 6.25%)
    kaiser_pct = (1.0 / 16.0) * 100
    ax1.axhline(kaiser_pct, color='#d95f02', linestyle='--', linewidth=1.5,
                label=f'Kaiser Criterion (Eigenvalue=1.0, {kaiser_pct:.1f}%)')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Right: Cumulative Explained Variance Curve
    cum_pct = variance_df['Cumulative_Variance_Percent']
    ax2.plot(x, cum_pct, color='#1b9e77', marker='o', linewidth=2.5, markersize=6, label='Cumulative Variance (%)')
    ax2.axhline(90.0, color='#e6ab02', linestyle='--', linewidth=1.5, label='90% Threshold')
    ax2.axhline(95.0, color='#d95f02', linestyle='--', linewidth=1.5, label='95% Threshold')
    
    # Highlight chosen component (n=4)
    ax2.axvline(4, color='#7570b3', linestyle=':', linewidth=2, label='Selected: 4 PCs (95.02%)')
    ax2.scatter([4], [cum_pct.iloc[3]], color='#7570b3', s=120, zorder=5)
    ax2.annotate(f"Optimal Choice: 4 PCs\nCaptures {cum_pct.iloc[3]:.2f}% Variance",
                 xy=(4, cum_pct.iloc[3]), xytext=(5.5, 80),
                 arrowprops=dict(facecolor='#7570b3', shrink=0.08, width=1.5, headwidth=7),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4f8", ec="#7570b3"))
    
    ax2.set_xlabel('Number of Retained Principal Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance Curve', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"PC{i}" for i in x], rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(40, 103)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Dry Bean Dataset (16 Features): Principal Component Analysis (PCA)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualize] Saved PCA variance plot to {output_path}")

def plot_2d_clusters(X_pca_2d, labels, title, output_path, algo_subtitle="", centroids_2d=None, cluster_labels_dict=None):
    """
    Plots 2D PCA projection colored by cluster assignments.
    
    Args:
        X_pca_2d (np.ndarray): 2D coordinates (PC1, PC2).
        labels (np.ndarray): Cluster labels.
        title (str): Main title.
        output_path (str): Save path.
        algo_subtitle (str): Subtitle describing algorithm and configuration.
        centroids_2d (np.ndarray): Optional 2D coordinates of cluster centroids.
        cluster_labels_dict (dict): Optional mapping from cluster ID to description.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    
    unique_labels = sorted(np.unique(labels))
    
    for idx, c in enumerate(unique_labels):
        mask = (labels == c)
        c_count = mask.sum()
        c_pct = c_count / len(labels) * 100
        
        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        label_text = f"Cluster {c}: n={c_count:,} ({c_pct:.1f}%)"
        if cluster_labels_dict and c in cluster_labels_dict:
            label_text = f"Cluster {c} ({cluster_labels_dict[c]}): n={c_count:,} ({c_pct:.1f}%)"
            
        ax.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            c=color, label=label_text, alpha=0.55, s=18, edgecolors='none'
        )
        
    # Plot centroids if provided or compute projected mean
    if centroids_2d is not None:
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            c='black', marker='X', s=160, linewidths=2, edgecolors='white',
            label='Cluster Centroid', zorder=10
        )
    else:
        # Compute empirical centroids in 2D
        for idx, c in enumerate(unique_labels):
            mean_pt = X_pca_2d[labels == c].mean(axis=0)
            ax.scatter(
                mean_pt[0], mean_pt[1],
                c='black', marker='X', s=130, linewidths=1.8, edgecolors='white',
                zorder=10
            )
            
    ax.set_xlabel('Principal Component 1 (55.47% Variance) — [Bean Size & Scale]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Principal Component 2 (26.43% Variance) — [Compactness vs Elongation]', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    
    # Subtitle note distinguishing visualization vs clustering space
    note_text = (
        f"{algo_subtitle}\n"
        "Methodology Note: 2D PCA projection shown for visual display only (81.90% total variance explained).\n"
        "Clustering was performed in optimal 4-dimensional PCA space (capturing 95.02% total variance)."
    )
    plt.figtext(0.5, -0.04, note_text, ha='center', fontsize=9.5, style='italic',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#cccccc", lw=0.8))
    
    ax.legend(loc='best', frameon=True, fontsize=9, markerscale=1.8, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualize] Saved cluster visualization to {output_path}")

def generate_all_visualizations(variance_df, X_pca_2d, results_dict):
    """
    Generates all required project figures:
    - drybean_pca_variance.png
    - drybean_kmeans_clusters.png
    - drybean_method2_clusters.png
    """
    # 1. PCA variance curve
    plot_pca_variance(variance_df, output_path="outputs/figures/drybean_pca_variance.png")
    
    # Morphological descriptors for K-Means clusters (Config 2)
    kmeans_labels = results_dict["Config 2: K-Means (PCA 4D)"]["labels"]
    kmeans_descriptors = {
        0: "Large Elongated",
        1: "Small Oval (Dermason-type)",
        2: "Large Oval (Cali/Barbunya)",
        3: "Small Round (Seker-type)",
        4: "Giant Broad (Bombay-type)",
        5: "Medium Compact (Sira-type)",
        6: "Long Cylindrical (Horoz-type)"
    }
    
    plot_2d_clusters(
        X_pca_2d,
        kmeans_labels,
        title="K-Means Clustering on Dry Bean Dataset (k=7)",
        output_path="outputs/figures/drybean_kmeans_clusters.png",
        algo_subtitle="Configuration: K-Means on PCA-transformed Features (k=7, Silhouette=0.3277, CH=9134.2)",
        cluster_labels_dict=kmeans_descriptors
    )
    
    # 3. Method 2 (GMM) clusters
    gmm_labels = results_dict["Config 4: GMM (PCA 4D)"]["labels"]
    gmm_descriptors = {
        0: "Small Round",
        1: "Large Oval",
        2: "Giant Broad",
        3: "Long Cylindrical",
        4: "Medium Compact",
        5: "Small Oval",
        6: "Medium Elongated"
    }
    plot_2d_clusters(
        X_pca_2d,
        gmm_labels,
        title="Gaussian Mixture Model (GMM) Clustering on Dry Bean Dataset (k=7)",
        output_path="outputs/figures/drybean_method2_clusters.png",
        algo_subtitle="Configuration: Full Covariance GMM on PCA-transformed Features (k=7, Silhouette=0.3020, CH=7618.6)",
        cluster_labels_dict=gmm_descriptors
    )

if __name__ == "__main__":
    from preprocess import load_dry_bean_data, standardize_features, FEATURE_COLUMNS
    from pca_analysis import run_full_pca, select_optimal_pca, get_2d_pca_projection
    from clustering import run_four_configurations
    
    df, X_raw, sample_ids = load_dry_bean_data()
    X_scaled, scaler = standardize_features(X_raw)
    
    pca_full, variance_df, loadings_df = run_full_pca(X_scaled, FEATURE_COLUMNS)
    X_pca, pca_model = select_optimal_pca(X_scaled, n_components=4)
    X_pca_2d, pca_2d = get_2d_pca_projection(X_scaled)
    
    results = run_four_configurations(X_scaled, X_pca)
    generate_all_visualizations(variance_df, X_pca_2d, results)
    print("[Visualize] All visualizations generated successfully.")
