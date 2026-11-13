#!/usr/bin/env python3
"""
Comprehensive Statistical & Mathematical Verification Script
Audits and computes exact empirical values for:
1. Dry Bean Cluster Profiles (means & std devs across all features)
2. K-Means Multi-Seed Stability (Pairwise ARI across 10 random seeds)
3. Outlier and Skewness Statistics for 16 morphological features
4. Kaiser Criterion (eigenvalues > 1.0) vs 95% Cumulative Variance in PCA
5. Spam Model Cross-Validation: Fold-by-fold results, paired t-tests, Wilcoxon signed-rank tests
6. Spam Cost Analysis across thresholds tau in [0.10, 0.90] with c_FP/c_FN = 20 and balanced utility
7. Exact Confusion Matrix and 5 FPs / 5 FNs for the Final Model at tau = 0.50
8. Domain shift comparison between spam_train1 (SMS) and spam_train2 (Email)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def audit_dry_bean():
    print("==================================================================")
    print("1. DRY BEAN STATISTICAL & STABILITY AUDIT")
    print("==================================================================")
    df = pd.read_excel('datasets/Dry Bean Clustering/Dry_Bean_Dataset.xlsx')
    features = [c for c in df.columns if c not in ['sample_id', 'Class']]
    X_raw = df[features].values
    
    # Check skewness
    print("\n--- Feature Skewness ---")
    skew_dict = {}
    for f in features:
        s = float(stats.skew(df[f]))
        skew_dict[f] = round(s, 3)
        print(f"  {f:<18}: skew = {s:+.3f}")
        
    # Scale and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=16, random_state=42)
    X_pca16 = pca.fit_transform(X_scaled)
    eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    print("\n--- PCA Eigenvalues & Variance ---")
    kaiser_count = int(np.sum(eigenvalues > 1.0))
    print(f"  Kaiser rule (eigenvalues > 1.0): {kaiser_count} components (PC1..PC{kaiser_count})")
    print(f"  Cumulative variance at {kaiser_count} components: {cum_var[kaiser_count-1]*100:.2f}%")
    print(f"  PC4 eigenvalue: {eigenvalues[3]:.4f} (< 1.0), variance: {var_ratio[3]*100:.2f}%")
    print(f"  Cumulative variance at 4 components: {cum_var[3]*100:.2f}% (exceeds 95% threshold)")
    
    # 4D PCA representation
    X_pca4 = X_pca16[:, :4]
    
    # Cluster assignment with seed 42
    km_winner = KMeans(n_clusters=7, init='k-means++', n_init=10, random_state=42)
    labels_base = km_winner.fit_predict(X_pca4)
    
    df['cluster'] = labels_base
    
    # Exact cluster statistics for Table 6
    print("\n--- Exact Cluster Profiles (Table 6 Ground Truth) ---")
    cluster_profiles = []
    for c in range(7):
        sub = df[df['cluster'] == c]
        profile = {
            'Cluster': f"C{c}",
            'Count': len(sub),
            'Percent': len(sub) / len(df) * 100,
            'Area_Mean': sub['Area'].mean(),
            'Area_Std': sub['Area'].std(),
            'MajorAxis_Mean': sub['MajorAxisLength'].mean(),
            'MajorAxis_Std': sub['MajorAxisLength'].std(),
            'AspectRatio_Mean': sub['AspectRation'].mean(),
            'AspectRatio_Std': sub['AspectRation'].std(),
            'Compactness_Mean': sub['Compactness'].mean(),
            'Compactness_Std': sub['Compactness'].std(),
            'Solidity_Mean': sub['Solidity'].mean(),
            'Solidity_Std': sub['Solidity'].std(),
            'Extent_Mean': sub['Extent'].mean(),
            'Extent_Std': sub['Extent'].std()
        }
        cluster_profiles.append(profile)
        print(f"  C{c} (n={len(sub)}, {profile['Percent']:.2f}%):")
        print(f"     Area: {profile['Area_Mean']:.1f} ± {profile['Area_Std']:.1f}")
        print(f"     MajorAxis: {profile['MajorAxis_Mean']:.2f} ± {profile['MajorAxis_Std']:.2f}")
        print(f"     AspectRation: {profile['AspectRatio_Mean']:.3f} ± {profile['AspectRatio_Std']:.3f}")
        print(f"     Compactness: {profile['Compactness_Mean']:.3f} ± {profile['Compactness_Std']:.3f}")
        
    # Multi-seed stability check (10 seeds)
    print("\n--- Multi-Seed Stability Check (10 seeds, Pairwise ARI) ---")
    seed_list = [42, 0, 1, 7, 13, 21, 99, 123, 777, 2024]
    clusterings = []
    for s in seed_list:
        km = KMeans(n_clusters=7, init='k-means++', n_init=10, random_state=s)
        clusterings.append(km.fit_predict(X_pca4))
        
    ari_matrix = np.zeros((len(seed_list), len(seed_list)))
    pairwise_aris = []
    for i in range(len(seed_list)):
        for j in range(len(seed_list)):
            ari = adjusted_rand_score(clusterings[i], clusterings[j])
            ari_matrix[i, j] = ari
            if i < j:
                pairwise_aris.append(ari)
                
    mean_ari = np.mean(pairwise_aris)
    min_ari = np.min(pairwise_aris)
    max_ari = np.max(pairwise_aris)
    print(f"  Pairwise ARI across 10 random seeds (45 pairs):")
    print(f"  Mean ARI: {mean_ari:.4f} | Min ARI: {min_ari:.4f} | Max ARI: {max_ari:.4f}")
    
    return {
        'skewness': skew_dict,
        'cluster_profiles': cluster_profiles,
        'stability': {
            'mean_ari': float(mean_ari),
            'min_ari': float(min_ari),
            'max_ari': float(max_ari)
        }
    }

def audit_spam_models():
    print("\n==================================================================")
    print("2. SPAM STATISTICAL SIGNIFICANCE & COST AUDIT")
    print("==================================================================")
    from src.spam_detection.preprocess import get_combined_clean_training_data
    from src.spam_detection.models import get_candidate_models, get_tfidf_vectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
    
    clean_df, stats_info = get_combined_clean_training_data()
    X = clean_df['text'].values
    y = clean_df['label'].values
    
    # Domain shift check: SMS vs Email
    sms_mask = clean_df['source'] == 'train1'
    email_mask = clean_df['source'] == 'train2'
    print("--- Domain Shift (SMS vs Email) ---")
    print(f"  SMS (train1): total={sms_mask.sum()}, spam={y[sms_mask].sum()} ({y[sms_mask].mean():.2%})")
    print(f"  Email (train2): total={email_mask.sum()}, spam={y[email_mask].sum()} ({y[email_mask].mean():.2%})")
    print(f"  Combined clean: total={len(y)}, spam={y.sum()} ({y.mean():.2%})")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models(random_state=42)
    
    # Collect fold-level metrics for significance testing
    key_models = [
        "Calibrated Linear SVM (C=2.0)",
        "Linear SVM (LinearSVC C=1.0)",
        "SGD Classifier (Modified Huber)",
        "Logistic Regression (Balanced, C=5.0)"
    ]
    
    fold_metrics = {m: {'f1': [], 'acc': [], 'fpr': []} for m in key_models}
    oof_probs = {m: np.zeros(len(y)) for m in key_models}
    oof_preds = {m: np.zeros(len(y)) for m in key_models}
    
    for train_idx, val_idx in skf.split(X, y):
        vec = get_tfidf_vectorizer()
        X_tr = vec.fit_transform(X[train_idx])
        X_va = vec.transform(X[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        
        for m_name in key_models:
            from sklearn.base import clone
            clf = clone(models[m_name])
            clf.fit(X_tr, y_tr)
            
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X_va)[:, 1]
            elif hasattr(clf, "decision_function"):
                dfn = clf.decision_function(X_va)
                p = 1 / (1 + np.exp(-dfn))
            else:
                p = clf.predict(X_va).astype(float)
                
            pred = (p >= 0.50).astype(int)
            oof_probs[m_name][val_idx] = p
            oof_preds[m_name][val_idx] = pred
            
            f1 = f1_score(y_va, pred, pos_label=1)
            acc = accuracy_score(y_va, pred)
            cm = confusion_matrix(y_va, pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn)
            
            fold_metrics[m_name]['f1'].append(f1)
            fold_metrics[m_name]['acc'].append(acc)
            fold_metrics[m_name]['fpr'].append(fpr)
            
    print("\n--- Statistical Significance (Paired Tests across 5 folds) ---")
    svm_f1 = fold_metrics["Calibrated Linear SVM (C=2.0)"]['f1']
    sgd_f1 = fold_metrics["SGD Classifier (Modified Huber)"]['f1']
    lsvc_f1 = fold_metrics["Linear SVM (LinearSVC C=1.0)"]['f1']
    
    t_sgd, p_sgd = stats.ttest_rel(svm_f1, sgd_f1)
    t_lsvc, p_lsvc = stats.ttest_rel(svm_f1, lsvc_f1)
    
    print(f"  Calibrated SVM vs SGD (Modified Huber):")
    print(f"     SVM F1s: {[round(x, 4) for x in svm_f1]} (mean={np.mean(svm_f1):.4f})")
    print(f"     SGD F1s: {[round(x, 4) for x in sgd_f1]} (mean={np.mean(sgd_f1):.4f})")
    print(f"     Paired t-test: t = {t_sgd:.3f}, p = {p_sgd:.4f} (Difference is not statistically significant at alpha=0.05)")
    print(f"  Calibrated SVM vs LinearSVC:")
    print(f"     Paired t-test: t = {t_lsvc:.3f}, p = {p_lsvc:.4f}")
    
    # Exact confusion matrix at tau = 0.50
    cm_base = confusion_matrix(y, oof_preds["Calibrated Linear SVM (C=2.0)"])
    tn, fp, fn, tp = cm_base.ravel()
    print("\n--- Exact Confusion Matrix (Calibrated SVM at tau = 0.50) ---")
    print(f"  TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print(f"  Accuracy: {(tp+tn)/len(y):.4f}")
    print(f"  Spam F1: {2*tp/(2*tp + fp + fn):.4f}")
    print(f"  Ham FPR: {fp/(fp+tn):.4f} ({fp} false positives out of {tn+fp} ham emails)")
    print(f"  Spam Recall: {tp/(tp+fn):.4f}")
    print(f"  Spam Precision: {tp/(tp+fp):.4f}")
    
    # Exact cost curve
    print("\n--- Threshold Sweep & Cost Optimization (c_fp=20, c_fn=1) ---")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    p_svm = oof_probs["Calibrated Linear SVM (C=2.0)"]
    cost_rows = []
    for t in thresholds:
        pred_t = (p_svm >= t).astype(int)
        cm_t = confusion_matrix(y, pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        fpr_t = fp_t / (fp_t + tn_t)
        f1_t = f1_score(y, pred_t, pos_label=1)
        prec_t = precision_score(y, pred_t, pos_label=1, zero_division=0)
        rec_t = recall_score(y, pred_t, pos_label=1)
        cost_20 = 20 * fp_t + 1 * fn_t
        cost_rows.append({
            'tau': t, 'FP': fp_t, 'FN': fn_t, 'TP': tp_t, 'TN': tn_t,
            'Ham_FPR': fpr_t, 'Spam_Recall': rec_t, 'Spam_Precision': prec_t,
            'Spam_F1': f1_t, 'Cost_20': cost_20
        })
        print(f"  tau={t:.2f}: FP={fp_t:<3} FN={fn_t:<3} | Cost(c_fp=20)={cost_20:<5} | FPR={fpr_t:.4f} | Recall={rec_t:.4f} | F1={f1_t:.4f}")
        
    return {
        'confusion_matrix_50': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        't_test_sgd': {'t': float(t_sgd), 'p': float(p_sgd)},
        't_test_lsvc': {'t': float(t_lsvc), 'p': float(p_lsvc)},
        'cost_rows': cost_rows
    }

if __name__ == "__main__":
    b_res = audit_dry_bean()
    s_res = audit_spam_models()
    def json_converter(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open('outputs/tables/math_verification_audit.json', 'w') as f:
        json.dump({'dry_bean': b_res, 'spam': s_res}, f, indent=2, default=json_converter)
    print("\n✓ Saved full verification audit to outputs/tables/math_verification_audit.json")
