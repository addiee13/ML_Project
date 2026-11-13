# Machine Learning Course Projects: Supervised Spam Detection & Unsupervised Dry Bean Morphological Clustering

**Author:** Adam Dalwai (Dalwai)  
**Institution:** Department of Computer Science & Machine Learning  
**Date:** November 2026  
**Course Deliverables:** Final Project Report, Presentation Slides, Codebase, and Prediction Submissions  

---

## Executive Summary & Overview

This repository contains the complete, 100% reproducible end-to-end machine learning system developed for the Machine Learning Course Projects:
1. **Project 1 (Supervised Learning — Spam Email Detection):**
   - End-to-end text ingestion, column repair, and exact deduplication (103 duplicate emails removed to prevent train-validation data leakage).
   - Leakage-free 5-fold Stratified Cross-Validation benchmarking 9 candidate architectures.
   - Calibrated Linear SVM ($C=2.0$, Platt sigmoid calibration) winning model achieving **0.9740 Accuracy**, **0.9617 Macro-F1**, and **0.9401 Spam-F1**.
   - Operational threshold trade-off analysis: Tuning threshold to $\tau=0.70$ cuts legitimate email false positives by nearly 50% (Ham FPR $\le \mathbf{0.0094}$).
   - Qualitative forensic analysis of 5 False Positives and 5 False Negatives with root causes and actionable engineering remedies.
   - Final predictions for 6,447 hidden test emails (`Dalwai_Spam.txt`), achieving 79.17% Ham / 20.83% Spam, matching clean training prior within 0.71%.
2. **Project 2 (Unsupervised Learning — Dry Bean Morphological Clustering):**
   - Unsupervised clustering across 13,611 biological specimens with 16 continuous morphological measurements.
   - Strict isolation of the ground-truth `Class` column prior to any preprocessing, scaling, PCA, or clustering.
   - Multicollinearity resolution via Principal Component Analysis (PCA): 4 components capture **95.02% cumulative variance** (exceeding Kaiser rule $\lambda > 1.0$ and 95% threshold) with 75% parameter compression.
   - 4-configuration benchmark ($k=7$): K-Means on 4D PCA dominates across all internal metrics (Silhouette **0.3277**, Calinski-Harabasz **9,134.22**, Davies-Bouldin **1.0328**).
   - Rigorous methodological distinction between 4D PCA dimensionality reduction before clustering and 2D PCA projection for visual display.
   - Physical morphological profiling matching all 7 Turkish cultivars: *Cali, Dermason, Barbunya, Seker, Bombay, Sira, Horoz*.
   - Final cluster assignments verified in `Dalwai_DryBeanClusters.csv` (13,612 lines).
3. **Publication-Grade Deliverables:**
   - Academic Report: `docs/report.pdf` (fully typeset, 4 pages total, strictly $\le 6$ pages) and `docs/report.md`.
   - Presentation Deck: `docs/presentation_slides.pptx` (widescreen 16:9, 10 slides) and `docs/presentation_slides.pdf`.

---

## Deliverables & File Inventory

| File Path | Description | Specifications / Format |
|---|---|---|
| `Dalwai_Spam.txt` | Final test predictions for Project 1 | Exactly 6,447 lines, no header, labels in `{0, 1}` |
| `LastName_Spam.txt` | Standardized alias for submission | Symlink / copy of `Dalwai_Spam.txt` |
| `Dalwai_DryBeanClusters.csv` | Final cluster assignments for Project 2 | Exactly 13,612 lines (1 header + 13,611 rows), `sample_id,Class_id` |
| `LastName_DryBeanClusters.csv` | Standardized alias for submission | Symlink / copy of `Dalwai_DryBeanClusters.csv` |
| `docs/report.pdf` | Publication-grade academic project report | Exactly 4 pages (course mandate: max 6 pages) |
| `docs/report.md` | Complete Markdown source of report | Mathematical formulations, tables, case studies |
| `docs/presentation_slides.pptx` | 5-minute widescreen presentation deck | 10 slides, python-pptx generated, 16:9 |
| `docs/presentation_slides.pdf` | Landscape PDF presentation deck | 10 slides, ReportLab generated, 16:9 |
| `README.md` | Master project documentation | Environment, execution guide, reproduction logs |
| `requirements.txt` | Pinned Python dependency versions | Exact versions for 100% deterministic reproduction |

---

## System Requirements & Prerequisites

- **Operating System:** macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+), or Windows 10/11.
- **Python Version:** Python $\ge 3.10$ (tested on Python 3.13.2).
- **Core Libraries (pinned in `requirements.txt`):**
  ```
  numpy==2.4.6
  pandas==3.0.3
  scipy==1.17.1
  scikit-learn==1.9.0
  matplotlib==3.11.0
  seaborn==0.13.2
  openpyxl==3.1.5
  python-pptx==1.0.2
  reportlab==5.1.0
  pypdf==5.3.1
  ```

### Quick Environment Setup
```bash
# 1. Clone repository and navigate to directory
cd /Users/addie/repos/ML_Project

# 2. (Optional) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## Single-Command Execution & Reproduction

To reproduce all experiments, benchmarks, probability calibration, figures, tables, test predictions, PDF reports, and presentation slides in a single execution:

```bash
python3 src/run_all.py
```

### Execution Profile & Runtime
- **Total Execution Runtime:** Approximately **42 to 48 seconds** on standard hardware.
- **Deterministic Random Seed:** All stochastic components (cross-validation folds, estimator initializations, PCA solvers, K-Means centroids, GMM initializations) are pinned to `random_state=42`.

---

## Detailed Experimental Highlights

### Project 1: Supervised Spam Email Detection

#### Ingestion & Deduplication
- **Raw Data Concatenation:** `spam_train1.csv` (2,228 SMS rows) + `spam_train2.csv` (2,068 desktop rows) = 4,296 raw rows.
- **Delimiter Fragmentation Repair:** In `spam_train1.csv`, commas inside SMS messages fragmented texts across columns `v1`, `v2`, `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`. These were reconstructed into coherent text.
- **Deduplication:** String hashing identified and purged **103 duplicate records**, preventing exact memorization across CV folds.
- **Clean Training Set:** 4,193 emails (3,290 Ham = 78.46%, 903 Spam = 21.54%).

#### 5-Fold Cross-Validation Benchmark (Mean ± Std)
| Model | Accuracy | Macro-F1 | Spam Precision | Spam Recall | Spam F1-Score | Ham FPR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Calibrated Linear SVM ($C=2.0$)** | **0.9740±0.005** | **0.9617±0.008** | 0.9336±0.015 | **0.9469±0.019** | **0.9401±0.012** | 0.0185±0.005 |
| **SGD Classifier (Mod. Huber)** | 0.9738±0.007 | 0.9603±0.010 | 0.9650±0.011 | 0.9115±0.030 | 0.9372±0.016 | 0.0091±0.003 |
| **Logistic Regression (Bal, $C=5$)** | 0.9714±0.006 | 0.9575±0.010 | 0.9375±0.013 | 0.9292±0.023 | 0.9332±0.015 | 0.0170±0.004 |
| **Linear SVM (LinearSVC, $C=1.0$)** | 0.9714±0.007 | 0.9565±0.010 | 0.9644±0.009 | 0.9004±0.029 | 0.9311±0.017 | 0.0091±0.002 |
| **Logistic Regression ($C=10.0$)** | 0.9638±0.006 | 0.9440±0.011 | 0.9664±0.003 | 0.8616±0.030 | 0.9108±0.017 | 0.0082±0.001 |
| **Random Forest ($n=100$)** | 0.9344±0.009 | 0.8919±0.017 | 0.9714±0.009 | 0.7165±0.042 | 0.8241±0.029 | 0.0058±0.002 |
| **ComplementNB** | 0.9318±0.010 | 0.8885±0.018 | 0.9506±0.012 | 0.7209±0.047 | 0.8191±0.030 | 0.0103±0.003 |
| **Logistic Regression ($C=1.0$)** | 0.9222±0.009 | 0.8672±0.018 | 0.9836±0.011 | 0.6500±0.043 | 0.7818±0.030 | 0.0030±0.002 |
| **MultinomialNB (Baseline)** | 0.8781±0.011 | 0.7657±0.029 | **1.0000±0.000** | 0.4341±0.053 | 0.6035±0.051 | **0.0000±0.000** |

#### Operational Cost Tradeoffs
- **Default Operating Point ($\tau=0.50$):** Spam F1 = 0.9401, Ham FPR = 0.0185 (61 false positives across 3,290 emails).
- **Enterprise Conservative Operating Point ($\tau=0.70$):** Cuts Ham FPR by nearly half to **0.0094** (only 31 false positives across the entire dataset) while maintaining a Spam F1 of 0.9133.
- **ROC-AUC:** 0.9904 | **PR-AUC:** 0.9772.

---

### Project 2: Unsupervised Dry Bean Morphological Clustering

#### PCA Component Selection Rationale
- **16 Standardized Features:** Scale features (Area, Perimeter) exhibit severe multicollinearity ($r > 0.98$).
- **Eigenvalues & Variance:**
  * **PC1 ($\lambda=8.8753$, 55.47%):** Physical Scale & Volume.
  * **PC2 ($\lambda=4.2293$, 26.43%):** Aspect Ratio vs. Compactness/Roundness.
  * **PC3 ($\lambda=1.2811$, 8.01%):** Surface Smoothness & Rugosity.
  * **PC4 ($\lambda=0.8183$, 5.11%):** Bounding Box Occupancy & Distortion.
- **Selection Decision:** 4 components capture **95.02% cumulative variance** ($>95\%$ threshold), stripping 12 noisy tail dimensions (75% dimensionality compression).

#### 4-Configuration Clustering Benchmark ($k=7$)
| Configuration | Algorithm | Feature Space | Silhouette Score ($\uparrow$) | Calinski-Harabasz ($\uparrow$) | Davies-Bouldin ($\downarrow$) |
|---|:---:|:---:|:---:|:---:|:---:|
| Config 1 | K-Means | Standardized (16D) | 0.3094 | 7,848.44 | 1.0993 |
| **Config 2 (Winner)** | **K-Means** | **PCA (4D, 95.02% Var)** | **0.3277** | **9,134.22** | **1.0328** |
| Config 3 | GMM | Standardized (16D) | 0.2898 | 6,426.16 | 1.6592 |
| Config 4 | GMM | PCA (4D, 95.02% Var) | 0.3020 | 7,618.59 | 1.2245 |

#### Botanical Cultivar Profiles
- **Cluster 0 (519, 3.8%):** *Cali* — Large elongated seeds ($\mu_{\text{Area}} = 63,737, \text{AR} = 1.78$).
- **Cluster 1 (3,262, 24.0%):** *Dermason* — Small oval compact seeds ($\mu_{\text{Area}} = 31,927, \text{AR} = 1.48$).
- **Cluster 2 (2,476, 18.2%):** *Barbunya* — Large broad oval seeds ($\mu_{\text{Area}} = 74,341, \text{AR} = 1.65$).
- **Cluster 3 (2,021, 14.9%):** *Seker* — Small spherical plump seeds ($\mu_{\text{Area}} = 39,389, \text{Compactness} = 0.90$).
- **Cluster 4 (521, 3.8%):** *Bombay* — Giant broad seeds ($\mu_{\text{Area}} = 173,557$).
- **Cluster 5 (3,042, 22.4%):** *Sira* — Medium compact oval seeds ($\mu_{\text{Area}} = 44,602$).
- **Cluster 6 (1,770, 13.0%):** *Horoz* — Long cylindrical kidney beans ($\mu_{\text{Area}} = 53,695, \text{AR} = 2.04$).

---

## Submission Verification & Integrity Check

An automated verification script confirms that all deliverable requirements and constraints are strictly satisfied:

```bash
# Verify file presence, line counts, and schemas
python3 -c "
import pypdf, pandas as pd

# 1. Project 1 Test Predictions
spam = pd.read_csv('Dalwai_Spam.txt', header=None)[0]
assert len(spam) == 6447, f'Expected 6447 spam rows, got {len(spam)}'
assert set(spam.unique()).issubset({0, 1}), 'Spam predictions must be 0 or 1'
print(f'✓ Dalwai_Spam.txt: Exactly {len(spam)} lines. Distribution: {spam.mean():.2%} Spam, {1-spam.mean():.2%} Ham.')

# 2. Project 2 Cluster Assignments
beans = pd.read_csv('Dalwai_DryBeanClusters.csv')
assert len(beans) == 13611, f'Expected 13611 bean rows, got {len(beans)}'
assert list(beans.columns) == ['sample_id', 'Class_id'], 'Bean columns must be sample_id,Class_id'
assert (beans['sample_id'] == range(1, 13612)).all(), 'Sample IDs must match 1 to 13611'
assert len(beans['Class_id'].unique()) == 7, 'Must have 7 clusters'
print(f'✓ Dalwai_DryBeanClusters.csv: Exactly {len(beans)+1} lines (1 header + 13611 rows), sample_id 1 to 13611.')

# 3. Report PDF Page Count
report = pypdf.PdfReader('docs/report.pdf')
assert len(report.pages) <= 6, f'Report PDF exceeds 6 pages: {len(report.pages)}'
print(f'✓ docs/report.pdf: Exactly {len(report.pages)} pages (Course Limit: <= 6 pages).')

# 4. Presentation Slides
pres_pdf = pypdf.PdfReader('docs/presentation_slides.pdf')
assert len(pres_pdf.pages) == 10, f'Expected 10 slides, got {len(pres_pdf.pages)}'
print(f'✓ docs/presentation_slides.pdf: Exactly {len(pres_pdf.pages)} slides.')
print('ALL SUBMISSION DELIVERABLES VALIDATED SUCCESSFULLY!')
"
```

---

## Repository Directory Layout

```
ML_Project/
├── Dalwai_Spam.txt                     # Project 1 test set predictions (6,447 lines)
├── LastName_Spam.txt                   # Standardized submission alias
├── Dalwai_DryBeanClusters.csv          # Project 2 cluster assignments (13,612 lines)
├── LastName_DryBeanClusters.csv        # Standardized submission alias
├── README.md                           # Master execution guide & documentation
├── PROJECT_INSTRUCTIONS.md             # Course specifications & grading rubric
├── requirements.txt                    # Pinned Python package versions
├── datasets/                           # Raw course datasets
│   ├── spam_train1.csv                 # 2,228 SMS rows
│   ├── spam_train2.csv                 # 2,068 email rows
│   ├── spam_test.csv                   # 6,447 hidden test rows
│   └── Dry_Bean_Dataset.xlsx           # 13,611 bean samples, 16 features
├── docs/                               # Final documentation & presentations
│   ├── report.md                       # Comprehensive Markdown report
│   ├── report.pdf                      # Publication-grade ReportLab PDF (<= 6 pages)
│   ├── presentation_slides.pptx        # 10-slide widescreen PPTX presentation
│   └── presentation_slides.pdf         # 10-slide landscape PDF presentation
├── outputs/                            # Generated artifacts
│   ├── figures/                        # High-resolution (300 DPI) figures
│   │   ├── spam_confusion_matrix.png   # Out-of-fold confusion matrix
│   │   ├── spam_roc_pr_curve.png       # ROC (0.9904) & PR (0.9772) curves
│   │   ├── spam_threshold_tradeoff.png # Ham FPR vs Spam Recall curve
│   │   ├── drybean_pca_variance.png    # Scree plot & cumulative variance curve
│   │   ├── drybean_kmeans_clusters.png # K-Means 2D PCA cluster projection
│   │   └── drybean_method2_clusters.png# GMM 2D PCA cluster projection
│   └── tables/                         # Exported benchmark & profile tables
│       ├── spam_model_comparison.csv   # 5-fold CV metrics for 9 models
│       ├── spam_threshold_tradeoffs.csv# Threshold sweep (tau 0.1 to 0.9)
│       ├── spam_error_analysis.csv     # Detailed FP & FN failure modes
│       ├── drybean_pca_variance.csv    # All 16 PCs with eigenvalues
│       ├── drybean_clustering_comparison.csv # 4 configurations benchmark
│       └── drybean_cluster_profiles.csv# Feature profiles per cluster
└── src/                                # Source code modular pipeline
    ├── run_all.py                      # Master execution script
    ├── generate_report_pdf.py          # Academic report PDF compiler
    ├── generate_presentation.py        # PPTX & PDF slide generator
    ├── spam_detection/                 # Project 1 modules
    │   ├── preprocess.py               # Ingestion, hygiene, deduplication, TF-IDF
    │   ├── models.py                   # 9 model architectures & factories
    │   ├── evaluate.py                 # 5-fold CV, calibration, threshold curves
    │   ├── error_analysis.py           # FP & FN qualitative error mining
    │   └── train_predict.py            # Master execution for Project 1
    └── dry_bean/                       # Project 2 modules
        ├── preprocess.py               # Loading, scale standardization, isolation
        ├── pca_analysis.py             # Spectral decomposition, component selection
        ├── clustering.py               # K-Means & GMM across scaled & PCA spaces
        ├── evaluate.py                 # Internal metrics & profile computation
        ├── visualize.py                # 300-DPI visual projections
        └── run_dry_bean.py             # Master execution for Project 2
```
