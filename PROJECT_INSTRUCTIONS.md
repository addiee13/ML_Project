# Machine Learning Course Projects - Specification & Instructions

**Deadline:** December 4th, 11:59 PM  
**Submission Package:** A single ZIP file containing the Project Report (PDF, max 6 pages), Source Code, README, Presentation Slides, and Final Output Predictions (`LastName_Spam.txt`, `LastName_DryBeanClusters.csv`).

---

## Executive Summary & Objectives

These projects cover supervised learning (Spam Email Detection) and unsupervised learning (Clustering of Dry Bean Measurements). The goal is to design rigorous, reproducible experiments, compare methods fairly, analyze results deeply, and justify final model choices.

---

## Project 1: Spam Email Detection (Supervised Learning)

### 1. Objective
Build an end-to-end machine learning system in Python that classifies emails as **Spam** or **Ham** using the provided labeled training datasets, and predict labels for the unlabeled test dataset.

### 2. Datasets
- **Training Dataset 1 (`spam_train1.csv`):** 2,228 emails (1,927 Ham, 301 Spam).
- **Training Dataset 2 (`spam_train2.csv`):** 2,068 emails (1,440 Ham, 628 Spam).
- **Test Dataset (`spam_test.csv`):** 6,447 emails with hidden labels.

#### Rules for Data Handling:
- The two training datasets may be combined.
- Must document and explain how they are combined.
- Must check for and handle duplicate emails before creating validation splits so identical emails do not appear in both train and validation sets (data leakage prevention).
- Follow the label encoding provided with the datasets.
- Do NOT use the hidden test set for model selection or hyperparameter tuning.
- Do NOT use external labeled datasets or pretrained language models (e.g., BERT, RoBERTa, LLMs). Standard ML libraries (e.g., `scikit-learn`, `nltk`) are allowed.

### 3. Experimental Design & Validation
- **Preprocessing:** Document all text cleaning, tokenization, stop-word removal, stemming/lemmatization, etc.
- **Text Representation:** TF-IDF feature extraction or count vectors.
- **Validation Scheme:** Stratified K-fold cross-validation (or a rigorously justified stratified train/val split). The exact same validation splits must be used across all model comparisons.
- **Leakage Prevention:** All learned preprocessing steps (vocabulary construction, TF-IDF weighting, scaling) must be fit **strictly** on the training portion of each fold/split and only transformed on the validation portion.

### 4. Model Comparison
Compare at least **three** models:
1. **Simple Baseline:** TF-IDF with Multinomial Naive Bayes (`MultinomialNB`).
2. **Classifier 2:** E.g., Logistic Regression (`LogisticRegression`) or Support Vector Machine (`LinearSVC` / `SVC`).
3. **Classifier 3:** E.g., Linear SVM or Random Forest / Gradient Boosting.

For each model, report:
- Text representation details.
- Hyperparameters and tuning procedure (e.g., grid search or cross-validated search on training splits).
- Comprehensive validation performance table.
- Clear rationale for the final model selection (higher accuracy alone is **not** sufficient; tradeoff between precision/recall and ham false positive rate is paramount).

### 5. Evaluation Metrics (Validation Data)
Report the following metrics (mean and standard deviation across folds):
- **Accuracy**
- **Macro-F1**
- **Spam Precision**
- **Spam Recall**
- **Spam F1-Score**
- **Ham False Positive Rate (FPR):** $\frac{\text{Ham emails predicted as Spam}}{\text{Total actual Ham emails}}$
- **Confusion Matrix** for the final selected model.
- **Tradeoff Discussion:** Analyze the operational tradeoff between detecting spam and incorrectly blocking legitimate emails (Ham FPR).

### 6. Error Analysis
- Deep-dive qualitative analysis of at least **5 False Positives** (legitimate emails blocked as spam) and **5 False Negatives** (spam emails missed).
- Identify recurring patterns, failure modes, root causes, and propose actionable improvements supported by findings.
- Use strictly out-of-fold / validation predictions.

### 7. Final Predictions
- Refit the selected final configuration using all available labeled training data.
- Predict labels for `spam_test.csv` (6,447 emails).
- Save output as `LastName_Spam.txt`:
  - Exactly one predicted label per line.
  - Matches the original test-data order.
  - No header row.

---

## Project 2: Unsupervised Learning with Dry Bean Data

### 1. Objective
Discover, evaluate, and interpret cluster structures in the Dry Bean dataset containing 13,611 samples and 16 numerical measurement features (12 dimensions and 4 shape forms).

### 2. Dataset
- **File:** `Dry_Bean_Dataset.xlsx` / `Dry_Bean_Dataset.txt`
- **Samples:** 13,611
- **Features:** 16 numerical features (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`).
- **Class Column:** Exclude the `Class` column entirely from preprocessing, PCA, clustering, and parameter selection.

### 3. Data Preparation
- Inspect and document treatment of missing values, duplicate records, outliers/unusual observations, and feature scale disparities.
- Preserve sample identifiers (`sample_id` or original 1-based row index) throughout all steps.

### 4. Clustering Methods & Experimental Configurations
Apply **$k = 7$ clusters** to reflect the 7 varieties.
Compare two distinct clustering algorithms:
1. **Method 1:** K-Means clustering.
2. **Method 2:** A second clustering method covered in class (e.g., Gaussian Mixture Models (GMM) or Agglomerative Hierarchical Clustering).

For **both** methods, evaluate two feature spaces:
- **Approach A:** Standardized features directly (StandardScaler).
- **Approach B:** Standardized features followed by PCA (clustering in the retained principal-component space).

This yields at least **four experimental configurations**:
1. K-Means on Standardized Features
2. K-Means on PCA-reduced Features
3. Method 2 on Standardized Features
4. Method 2 on PCA-reduced Features

### 5. PCA Analysis & Justification
- Explain the criterion for choosing the number of retained principal components (e.g., explained variance ratio threshold, elbow method).
- Report cumulative explained variance for all retained components.
- Discuss how PCA affects cluster quality, geometry, stability, and interpretability.

### 6. Evaluation & Cluster Interpretation
- **Internal Cluster Validation:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz score.
- **Cluster Profiles:** Sample count per cluster, table/plots of original feature means/medians per cluster.
- **Visualization:** 2D PCA projection colored by cluster assignment for each method's selected configuration.
- **Critical Distinction:** Clearly distinguish between PCA used for 2D visualization vs. PCA used as dimensionality reduction prior to clustering.
- **Interpretation:** Physical and morphological interpretation of clusters (e.g., small round beans vs. large elongated beans).
- **Final Selection:** Justify the final clustering configuration based on metrics and interpretability.

### 7. Final Cluster Assignments
- Submit `LastName_DryBeanClusters.csv` with exactly two columns:
  - `sample_id`: Original sample identifier (1 to 13611).
  - `Class_id`: Assigned cluster ID (e.g., 0 to 6, or 1 to 7; noise is -1).
- Every sample in the dataset must have an assignment. Preserve exact order and sample IDs.

---

## Submission Deliverables & Structure

1. **Project Report (`report.pdf`):** Maximum 6 pages covering both projects (excluding references).
2. **Source Code (`src/`):** Clean, modular, reproducible Python scripts/notebooks.
3. **README (`README.md`):** System requirements, dependencies (`requirements.txt`), random seeds, step-by-step reproduction instructions, and expected outputs.
4. **Final Output Files:**
   - `LastName_Spam.txt`
   - `LastName_DryBeanClusters.csv`
5. **Presentation Slides (`presentation_slides.pdf` / `.pptx`):** 5-minute presentation deck covering both projects.
6. **Complete Submission ZIP:** All files packaged into a single archive.

---

## Grading Rubric

| Category | Weight | Key Criteria |
|---|:---:|---|
| **Experimental Design & Validation** | 30% | Appropriate preprocessing, leakage prevention, fair stratified validation, justified parameter selection, sound clustering evaluation. |
| **Comparison & Analysis** | 30% | Rigorous model comparisons, thorough spam error analysis, PCA comparisons, rich cluster interpretation, evidence-based conclusions. |
| **Code & Reproducibility** | 20% | Clean modular code, documented dependencies, fixed random seeds, 100% reproducible execution. |
| **Hidden Test Performance** | 10% | Spam detection performance on hidden test set (Spam F1-score & Ham FPR). |
| **Presentation** | 10% | High-quality slides, clear explanation of both projects, effective figures, strong justification of choices. |
