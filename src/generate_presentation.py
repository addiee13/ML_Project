#!/usr/bin/env python3
"""
Presentation Deck Generator (PPTX and PDF)
Generates:
1. docs/presentation_slides.pptx (via python-pptx, 16:9 widescreen)
2. docs/presentation_slides.pdf (via ReportLab, 16:9 widescreen landscape)

Structure (10 Slides):
1. Title slide
2. Executive Project Overview
3. Project 1: Data Hygiene, Deduplication & Validation Setup
4. Project 1: Model Benchmarks & Calibrated SVM Selection
5. Project 1: Threshold Tradeoffs & Qualitative Error Analysis
6. Project 2: Dry Bean Morphological PCA (95.02% variance in 4 PCs)
7. Project 2: Clustering Benchmark (K-Means vs GMM on Scaled vs PCA)
8. Project 2: Morphological Cluster Profiles & Botanical Archetypes
9. Project 2: 2D Visual Projections vs 4D Clustering Space
10. Conclusions, Reproducibility & Deliverable Verification
"""

import os
import sys
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ReportLab imports for landscape PDF generation
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# Palette Constants
RGB_NAVY = RGBColor(26, 54, 93)       # #1A365D
RGB_TEAL = RGBColor(13, 148, 136)     # #0D9488
RGB_CHARCOAL = RGBColor(45, 55, 72)   # #2D3748
RGB_LIGHT_BG = RGBColor(248, 250, 252)# #F8FAFC
RGB_CARD_BG = RGBColor(237, 242, 247) # #EDF2F7
RGB_WHITE = RGBColor(255, 255, 255)
RGB_BORDER = RGBColor(203, 213, 224)

HEX_NAVY = colors.HexColor("#1A365D")
HEX_TEAL = colors.HexColor("#0D9488")
HEX_CHARCOAL = colors.HexColor("#2D3748")
HEX_LIGHT_BG = colors.HexColor("#F8FAFC")
HEX_CARD_BG = colors.HexColor("#EDF2F7")
HEX_BORDER = colors.HexColor("#CBD5E0")

def add_header(slide, title_text, category_text="Machine Learning Course Projects"):
    """Adds a standard structured header banner to a slide."""
    # Category tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = RGB_TEAL
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = RGB_NAVY

def add_card(slide, left, top, width, height, bg_color=RGB_CARD_BG):
    """Draws a subtle rounded or rectangular container card."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = RGB_BORDER
    shape.line.width = Pt(1)
    return shape

def generate_pptx_slides(output_path="docs/presentation_slides.pptx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # -----------------------------------------------------------------
    # Slide 1: Title Slide
    # -----------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGB_NAVY
    bg1.line.fill.background()
    
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p1 = tf1.paragraphs[0]
    p1.text = "MACHINE LEARNING COURSE PROJECTS"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = RGB_TEAL
    p1.space_after = Pt(12)
    
    p2 = tf1.add_paragraph()
    p2.text = "Supervised Spam Email Detection &\nUnsupervised Dry Bean Morphological Clustering"
    p2.font.size = Pt(30)
    p2.font.bold = True
    p2.font.color.rgb = RGB_WHITE
    p2.space_after = Pt(24)
    
    p3 = tf1.add_paragraph()
    p3.text = "Author: Adam Dalwai (Dalwai)  |  Department of Computer Science & Machine Learning\nDeliverables: Final Research Report, Submission Predictions, & Presentation Deck"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGB_BORDER

    # -----------------------------------------------------------------
    # Slide 2: Executive Project Overview
    # -----------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Project Overview: Dual Machine Learning Paradigms")
    
    # Left Card: Supervised Spam Detection
    add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb_s2_l = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf2_l = tb_s2_l.text_frame
    tf2_l.word_wrap = True
    
    p = tf2_l.paragraphs[0]
    p.text = "Project 1: Supervised Spam Detection"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGB_NAVY
    p.space_after = Pt(10)
    
    bullets_p1 = [
        "Core Mission: Classify emails as Ham (0) vs. Spam (1) under severe class imbalance (78.5% Ham / 21.5% Spam).",
        "Operational Cost Asymmetry: Blocking authentic business/personal emails (False Positive) is far more disruptive than inbox spam (C_FP >> C_FN).",
        "Key Safeguards: Recombined fragmented SMS columns; purged 103 duplicates; TF-IDF fit strictly on CV training folds (zero data leakage).",
        "Winning Model: Calibrated Linear SVM (C=2.0) with Platt sigmoid scaling achieves 0.9740 Acc, 0.9617 Macro-F1, and 0.9401 Spam-F1.",
        "Operational Threshold: Shifting τ from 0.50 to 0.70 cuts Ham FPR by 50% (0.0185 → 0.0094) while retaining 0.9133 Spam-F1."
    ]
    for b in bullets_p1:
        p = tf2_l.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = RGB_CHARCOAL
        p.space_after = Pt(6)

    # Right Card: Unsupervised Dry Bean Clustering
    add_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_s2_r = s2.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf2_r = tb_s2_r.text_frame
    tf2_r.word_wrap = True
    
    p = tf2_r.paragraphs[0]
    p.text = "Project 2: Dry Bean Morphological Clustering"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGB_NAVY
    p.space_after = Pt(10)
    
    bullets_p2 = [
        "Core Mission: Discover 7 botanical variety clusters across 13,611 biological specimens and 16 morphological features without labels.",
        "Strict Class Isolation: Ground-truth variety column isolated and removed prior to any scaling, PCA, or clustering pipeline.",
        "Curse of Dimensionality & Collinearity: Scale features exhibit r > 0.98. PCA captures 95.02% variance in 4 orthogonal components.",
        "Winning Configuration: K-Means on 4D PCA dominates all configurations (Silhouette 0.3277, Calinski-Harabasz 9,134.22, Davies-Bouldin 1.0328).",
        "Phenotypic Matching: Profiles precisely resolve all 7 cultivars: Cali, Dermason, Barbunya, Seker, Bombay, Sira, Horoz."
    ]
    for b in bullets_p2:
        p = tf2_r.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = RGB_CHARCOAL
        p.space_after = Pt(6)

    # -----------------------------------------------------------------
    # Slide 3: Project 1: Data Hygiene, Deduplication & Validation Setup
    # -----------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Project 1: Data Hygiene, Deduplication & Leakage-Free Design", "Project 1: Supervised Learning")
    
    # 3 Summary Cards
    card_w = Inches(3.64)
    # Card 1: Ingestion
    add_card(s3, Inches(0.8), Inches(1.5), card_w, Inches(5.4))
    tb = s3.shapes.add_textbox(Inches(0.95), Inches(1.7), card_w - Inches(0.3), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Heterogeneous Ingestion"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "Train 1 (SMS Collection): 2,228 rows. Unescaped delimiter commas caused fragmented columns (v1, v2, Unnamed: 2-4). Recombined into unified text.",
        "Train 2 (Desktop Emails): 2,068 rows of formal Enron/SpamAssassin emails.",
        "Direct Concatenation: 4,296 raw email instances combined."
    ]:
        p = tf.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    # Card 2: Deduplication
    add_card(s3, Inches(4.8), Inches(1.5), card_w, Inches(5.4))
    tb = s3.shapes.add_textbox(Inches(4.95), Inches(1.7), card_w - Inches(0.3), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. Leakage-Proof Deduplication"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "103 Duplicate Records Detected: String hashing identified duplicate emails across datasets.",
        "Data Leakage Barrier: Leaving identical emails in the corpus allows exact memorization across train and validation splits.",
        "Final Clean Corpus: 4,193 unique emails.",
        "Class Prior: 3,290 Ham (78.46%) vs. 903 Spam (21.54%). Imbalance ratio 3.64:1."
    ]:
        p = tf.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    # Card 3: Validation & TF-IDF
    add_card(s3, Inches(8.8), Inches(1.5), card_w, Inches(5.4))
    tb = s3.shapes.add_textbox(Inches(8.95), Inches(1.7), card_w - Inches(0.3), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. Leakage-Free Validation"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "5-Fold Stratified CV: Preserves exact 78.5% / 21.5% class ratio in every fold.",
        "Strict Vectorizer Containment: Vocabulary, DF, and IDF weights fit ONLY on fold training splits (80%), then transformed on validation fold (20%).",
        "Sublinear TF-IDF: Unigrams + bigrams, sublinear TF (1 + log(tf)), min_df=2, max_df=0.95 (~14,800 features).",
        "Zero Test Peeking: Hidden test set untouched."
    ]:
        p = tf.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    # -----------------------------------------------------------------
    # Slide 4: Project 1: Model Benchmarks & Calibrated SVM Selection
    # -----------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project 1: Model Benchmarks & Calibrated SVM Selection", "Project 1: Supervised Learning")
    
    # Add Benchmark Table on Left
    table_shape = s4.shapes.add_table(10, 6, Inches(0.8), Inches(1.5), Inches(7.6), Inches(5.3))
    table = table_shape.table
    col_w = [Inches(2.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1)]
    for idx, w in enumerate(col_w):
        table.columns[idx].width = w
        
    headers = ["Model", "Accuracy", "Macro-F1", "Spam Rec.", "Spam F1", "Ham FPR"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = RGB_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = RGB_WHITE
            p.alignment = PP_ALIGN.CENTER
            
    rows = [
        ["Calibrated Linear SVM (C=2.0)", "0.9740±0.005", "0.9617±0.008", "0.9469±0.019", "0.9401±0.012", "0.0185±0.005"],
        ["SGD Classifier (Mod. Huber)", "0.9738±0.007", "0.9603±0.010", "0.9115±0.030", "0.9372±0.016", "0.0091±0.003"],
        ["Logistic Regression (Bal, C=5)", "0.9714±0.006", "0.9575±0.010", "0.9292±0.023", "0.9332±0.015", "0.0170±0.004"],
        ["Linear SVM (LinearSVC, C=1.0)", "0.9714±0.007", "0.9565±0.010", "0.9004±0.029", "0.9311±0.017", "0.0091±0.002"],
        ["Logistic Regression (C=10.0)", "0.9638±0.006", "0.9440±0.011", "0.8616±0.030", "0.9108±0.017", "0.0082±0.001"],
        ["Random Forest (n=100)", "0.9344±0.009", "0.8919±0.017", "0.7165±0.042", "0.8241±0.029", "0.0058±0.002"],
        ["ComplementNB", "0.9318±0.010", "0.8885±0.018", "0.7209±0.047", "0.8191±0.030", "0.0103±0.003"],
        ["Logistic Regression (C=1.0)", "0.9222±0.009", "0.8672±0.018", "0.6500±0.043", "0.7818±0.030", "0.0030±0.002"],
        ["MultinomialNB (Baseline)", "0.8781±0.011", "0.7657±0.029", "0.4341±0.053", "0.6035±0.051", "0.0000±0.000"]
    ]
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RGBColor(230, 246, 245) # highlight winner
            elif i % 2 == 1:
                cell.fill.fore_color.rgb = RGB_CARD_BG
            else:
                cell.fill.fore_color.rgb = RGB_WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(8.5)
                p.font.color.rgb = RGB_NAVY if i == 0 else RGB_CHARCOAL
                if i == 0: p.font.bold = True
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER

    # Right Card: Theoretical Takeaways
    add_card(s4, Inches(8.7), Inches(1.5), Inches(3.8), Inches(5.3))
    tb_r = s4.shapes.add_textbox(Inches(8.85), Inches(1.7), Inches(3.5), Inches(4.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Why Calibrated SVM Wins"
    p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "Baseline Failure: MultinomialNB suffers severe prior overwhelm (P(Ham)=78.5%), missing 56.6% of spam (Recall=0.4341).",
        "Tree Splitting Deficit: Random Forest F1=0.8241; orthogonal decision trees struggle in 14,800-dimensional sparse TF-IDF spaces.",
        "Maximum Margin Power: Linear SVM finds optimal hyperplane separation in high-dimensional text.",
        "Platt Sigmoid Calibration: Fits out-of-fold logistic mapping P(y=1|x) = 1 / (1 + exp(A·f(x) + B)), enabling exact operating threshold tuning."
    ]:
        p = tf_r.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    # -----------------------------------------------------------------
    # Slide 5: Project 1: Threshold Tradeoffs & Qualitative Error Analysis
    # -----------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Project 1: Operating Thresholds & Qualitative Error Analysis", "Project 1: Supervised Learning")
    
    # Left: Embed ROC and Tradeoff Figures
    s5.shapes.add_picture("outputs/figures/spam_roc_pr_curve.png", Inches(0.8), Inches(1.5), width=Inches(5.8))
    s5.shapes.add_picture("outputs/figures/spam_threshold_tradeoff.png", Inches(0.8), Inches(4.3), width=Inches(5.8))
    
    # Right: Error Analysis Cards
    add_card(s5, Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.4))
    tb_err = s5.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.2), Inches(5.0))
    tf_err = tb_err.text_frame
    tf_err.word_wrap = True
    
    p = tf_err.paragraphs[0]
    p.text = "Operating Threshold Tuning (τ)"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(4)
    p = tf_err.add_paragraph()
    p.text = "• Default τ=0.50: Ham FPR = 0.0185 (61 false positives out of 3,290), Spam F1 = 0.9401.\n" \
             "• Conservative τ=0.70: Cuts Ham FPR by 50% to 0.0094 (only 31 false positives) while maintaining Spam F1 = 0.9133."
    p.font.size = Pt(10); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(10)
    
    p = tf_err.add_paragraph()
    p.text = "Qualitative Failure Modes & Remedies"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(4)
    
    err_items = [
        "False Positive #179 (P=0.78, P2P Referral): Referral invite ('fullonsms.com') triggered by 'free sms' bigrams. Remedy: Domain reputation lookup.",
        "False Positive #373 (P=0.76, Intimate SMS): 'sex cuddling' triggered adult spam weights. Remedy: Syntactic conversational dependency parsing.",
        "False Positive #4106 (P=0.84, Health Bulletin): Pharmaceutical safety warning triggered online pharmacy filters. Remedy: Internal DKIM sender verification.",
        "False Negative #193 (P=0.04, Complaint Camouflage): Adversarial support complaint evades filter. Remedy: Link target & attachment entropy inspection.",
        "False Negative #85 (P=0.08, Slang Evasion): 'b fun 2nite call me free'. Remedy: Subword Byte-Pair Encoding & Soundex phonetic normalization."
    ]
    for item in err_items:
        p = tf_err.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(9.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(4)

    # -----------------------------------------------------------------
    # Slide 6: Project 2: Dry Bean Morphological PCA (95.02% Variance)
    # -----------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Project 2: Dry Bean Morphological PCA (95.02% Variance in 4 PCs)", "Project 2: Unsupervised Learning")
    
    # Left: Scree Plot Image
    s6.shapes.add_picture("outputs/figures/drybean_pca_variance.png", Inches(0.8), Inches(1.5), width=Inches(6.4))
    
    # Right: PCA Loadings & Math Rationale
    add_card(s6, Inches(7.5), Inches(1.5), Inches(5.0), Inches(5.4))
    tb_pca = s6.shapes.add_textbox(Inches(7.7), Inches(1.7), Inches(4.6), Inches(5.0))
    tf_pca = tb_pca.text_frame
    tf_pca.word_wrap = True
    
    p = tf_pca.paragraphs[0]
    p.text = "Component Selection & Morphology"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    
    pca_bullets = [
        "Data Hygiene: 13,611 biological specimens, 16 features, 0 missing values. Class column strictly isolated prior to scaling.",
        "StandardScaler: Eliminates scale disparities (Area ~60k vs ShapeFactor2 ~0.0017).",
        "Kaiser-Guttman Rule (λ > 1.0): Selects 3 components capturing 89.90% variance.",
        "Optimal 4-Component Choice: Captures 95.02% cumulative variance (>95% threshold) with 75% parameter compression (16D → 4D).",
        "PC1 (55.5%): Seed Scale & Physical Volume (Area, Perimeter, EquivDiameter).",
        "PC2 (26.4%): Aspect Ratio vs. Roundness (AspectRation, Compactness).",
        "PC3 (8.0%): Surface Smoothness & Rugosity (Solidity, ShapeFactor4).",
        "PC4 (5.1%): Bounding Box Occupancy & Distortion (Extent)."
    ]
    for b in pca_bullets:
        p = tf_pca.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(4)

    # -----------------------------------------------------------------
    # Slide 7: Project 2: Clustering Benchmark (K-Means vs GMM)
    # -----------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Project 2: Clustering Benchmark (K-Means vs. GMM on Scaled vs. PCA)", "Project 2: Unsupervised Learning")
    
    # Comparison Table
    table_shape7 = s7.shapes.add_table(5, 6, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.2))
    table7 = table_shape7.table
    col_w7 = [Inches(1.8), Inches(1.8), Inches(2.3), Inches(1.8), Inches(2.0), Inches(2.0)]
    for idx, w in enumerate(col_w7):
        table7.columns[idx].width = w
        
    hdr7 = ["Configuration", "Algorithm", "Feature Space", "Silhouette (↑)", "Calinski-Harabasz (↑)", "Davies-Bouldin (↓)"]
    for j, h in enumerate(hdr7):
        cell = table7.cell(0, j)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = RGB_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGB_WHITE
            p.alignment = PP_ALIGN.CENTER
            
    rows7 = [
        ["Config 1", "K-Means", "Standardized (16D)", "0.3094", "7,848.44", "1.0993"],
        ["Config 2 (Winner)", "K-Means", "PCA (4D, 95.02% Var)", "0.3277", "9,134.22", "1.0328"],
        ["Config 3", "GMM", "Standardized (16D)", "0.2898", "6,426.16", "1.6592"],
        ["Config 4", "GMM", "PCA (4D, 95.02% Var)", "0.3020", "7,618.59", "1.2245"]
    ]
    for i, row in enumerate(rows7):
        for j, val in enumerate(row):
            cell = table7.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            if i == 1:
                cell.fill.fore_color.rgb = RGBColor(230, 246, 245)
            elif i % 2 == 0:
                cell.fill.fore_color.rgb = RGB_CARD_BG
            else:
                cell.fill.fore_color.rgb = RGB_WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9.5)
                p.font.color.rgb = RGB_NAVY if i == 1 else RGB_CHARCOAL
                if i == 1: p.font.bold = True
                p.alignment = PP_ALIGN.CENTER

    # 2 Explanatory Cards below table
    add_card(s7, Inches(0.8), Inches(4.0), Inches(5.7), Inches(2.9))
    tb_c1 = s7.shapes.add_textbox(Inches(1.0), Inches(4.15), Inches(5.3), Inches(2.6))
    tf_c1 = tb_c1.text_frame; tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "Why K-Means on 4D PCA Dominates"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(6)
    for txt in [
        "Sweeps All Validation Metrics: Highest Silhouette (0.3277), highest Calinski-Harabasz (9,134.22, a 16.4% gain), lowest Davies-Bouldin (1.0328).",
        "Noise Stripping: Discarding the 12 isotropic tail dimensions restores Euclidean distance contrast (mitigating the curse of dimensionality).",
        "Cross-Space Validation: Projecting labels back into 16D space yields Sil 0.3076 and CH 7,817.26, confirming true physical cluster cohesion."
    ]:
        p = tf_c1.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(9.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(4)

    add_card(s7, Inches(6.8), Inches(4.0), Inches(5.733), Inches(2.9))
    tb_c2 = s7.shapes.add_textbox(Inches(7.0), Inches(4.15), Inches(5.3), Inches(2.6))
    tf_c2 = tb_c2.text_frame; tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "Why GMM Struggles on This Geometry"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(6)
    for txt in [
        "Over-Parameterization: In 16D space, fitting 7 full covariance matrices requires estimating 7 × (16×17/2) = 952 covariance parameters.",
        "Covariance Degeneracy: High sample-to-dimension ratio in small clusters induces component collapse and diffuse probabilities.",
        "K-Means Stability: Orthogonal 4D PCA satisfies spherical Voronoi assumptions far better than raw non-orthogonal space."
    ]:
        p = tf_c2.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(9.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(4)

    # -----------------------------------------------------------------
    # Slide 8: Project 2: Morphological Profiles & Botanical Archetypes
    # -----------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Project 2: Morphological Cluster Profiles & Botanical Archetypes", "Project 2: Unsupervised Learning")
    
    # Table of Cluster Profiles
    table_shape8 = s8.shapes.add_table(8, 7, Inches(0.8), Inches(1.5), Inches(11.733), Inches(3.6))
    table8 = table_shape8.table
    col_w8 = [Inches(1.0), Inches(1.4), Inches(1.4), Inches(1.4), Inches(1.4), Inches(1.4), Inches(3.733)]
    for idx, w in enumerate(col_w8):
        table8.columns[idx].width = w
        
    hdr8 = ["Cluster", "Count (%)", "Area (μ)", "Aspect Ratio", "Compactness", "Solidity", "Botanical Variety Archetype"]
    for j, h in enumerate(hdr8):
        cell = table8.cell(0, j)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = RGB_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(9.5); p.font.bold = True; p.font.color.rgb = RGB_WHITE
            p.alignment = PP_ALIGN.CENTER
            
    rows8 = [
        ["Cluster 0", "519 (3.8%)", "63,737", "1.779", "0.745", "0.973", "Cali: Large elongated seeds, medium circularity"],
        ["Cluster 1", "3,262 (24.0%)", "31,927", "1.476", "0.823", "0.988", "Dermason: Small oval, highly compact, dense commercial seed"],
        ["Cluster 2", "2,476 (18.2%)", "74,341", "1.651", "0.778", "0.985", "Barbunya: Large broad oval seeds, speckled market class"],
        ["Cluster 3", "2,021 (14.9%)", "39,389", "1.235", "0.900", "0.990", "Seker: Small round/spherical seeds, maximum plumpness"],
        ["Cluster 4", "521 (3.8%)", "173,557", "1.585", "0.793", "0.987", "Bombay: Giant broad seeds, massive volume (>170k Area)"],
        ["Cluster 5", "3,042 (22.4%)", "44,602", "1.576", "0.796", "0.988", "Sira: Medium compact oval seeds, general-purpose variety"],
        ["Cluster 6", "1,770 (13.0%)", "53,695", "2.039", "0.699", "0.987", "Horoz: Long cylindrical kidney bean, maximum elongation"]
    ]
    for i, row in enumerate(rows8):
        for j, val in enumerate(row):
            cell = table8.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGB_CARD_BG if i % 2 == 1 else RGB_WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(8.5)
                p.font.color.rgb = RGB_CHARCOAL
                if j == 0 or j == 6: p.font.bold = True
                p.alignment = PP_ALIGN.LEFT if j == 6 else PP_ALIGN.CENTER

    # Morphological Continuum Card below table
    add_card(s8, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.6))
    tb_cont = s8.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.333), Inches(1.4))
    tf_cont = tb_cont.text_frame; tf_cont.word_wrap = True
    p = tf_cont.paragraphs[0]
    p.text = "Biological Phenotypic Continuity"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(3)
    p = tf_cont.add_paragraph()
    p.text = "• The Scale Axis: Cluster 4 (Bombay) isolates giant seeds (Area 173k, 3x commercial size). Clusters 0 & 2 capture large market classes (Cali & Barbunya).\n" \
             "• The Shape Spectrum: Clusters trace a smooth biological transition from spherical (Seker, AR 1.23, Compactness 0.90) → oval (Dermason & Sira) → long cylindrical (Horoz, AR 2.04, Compactness 0.70)."
    p.font.size = Pt(9.5); p.font.color.rgb = RGB_CHARCOAL

    # -----------------------------------------------------------------
    # Slide 9: Project 2: 2D Visual Projections vs 4D Clustering Space
    # -----------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Project 2: 2D Visual Projections vs. 4D Clustering Space", "Project 2: Unsupervised Learning")
    
    # Side-by-side cluster figures
    s9.shapes.add_picture("outputs/figures/drybean_kmeans_clusters.png", Inches(0.8), Inches(1.5), width=Inches(5.7))
    s9.shapes.add_picture("outputs/figures/drybean_method2_clusters.png", Inches(6.8), Inches(1.5), width=Inches(5.7))
    
    # Caption Box below
    add_card(s9, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.0))
    tb_c9 = s9.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.333), Inches(0.9))
    tf_c9 = tb_c9.text_frame; tf_c9.word_wrap = True
    p = tf_c9.paragraphs[0]
    p.text = "CRITICAL METHODOLOGICAL DISTINCTION: 4D CLUSTERING SPACE vs. 2D DISPLAY PROJECTION"
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = RGB_TEAL; p.space_after = Pt(2)
    p = tf_c9.add_paragraph()
    p.text = "• Clustering was executed in 4-Dimensional PCA Space (capturing 95.02% cumulative variance, retaining PC3 smoothness and PC4 extent).\n" \
             "• Visualizations plot samples on PC1 & PC2 only (capturing 81.90% variance). Apparent 2D overlaps are cleanly separated along PC3 and PC4 in 4D."
    p.font.size = Pt(9); p.font.color.rgb = RGB_CHARCOAL

    # -----------------------------------------------------------------
    # Slide 10: Conclusions, Reproducibility & Deliverable Verification
    # -----------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Conclusions, Reproducibility & Deliverable Verification", "Executive Summary")
    
    # Card 1: Deliverables Verification
    add_card(s10, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb10_l = s10.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf10_l = tb10_l.text_frame; tf10_l.word_wrap = True
    p = tf10_l.paragraphs[0]
    p.text = "Submission Files & Verification"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "Dalwai_Spam.txt: Exactly 6,447 lines, no header, values in {0, 1}. Test distribution (79.17% Ham / 20.83% Spam) aligns with clean training prior within 0.71%.",
        "Dalwai_DryBeanClusters.csv: Exactly 13,612 lines (1 header + 13,611 rows), schema 'sample_id,Class_id'. Exactly preserves original 1-based indices.",
        "docs/report.pdf: Publication-grade academic report formatted strictly within the 6-page maximum limit (4 pages total, fully typeset).",
        "docs/presentation_slides.pptx & .pdf: Complete 10-slide widescreen presentation deck covering both projects."
    ]:
        p = tf10_l.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    # Card 2: Technical Insights
    add_card(s10, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4))
    tb10_r = s10.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf10_r = tb10_r.text_frame; tf10_r.word_wrap = True
    p = tf10_r.paragraphs[0]
    p.text = "Key Cross-Paradigm Learnings"
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = RGB_NAVY; p.space_after = Pt(8)
    for txt in [
        "Representation Dictates Algorithm: Sparse text (D ≈ 14,800) naturally favors linear hyperplanes (SVM); dense collinear biological features (D=16) require orthogonal PCA before clustering.",
        "Cost Asymmetry Trumps Raw Accuracy: In spam filtering, probability calibration allows surgical operational threshold tuning (slashing false positives by 50%).",
        "Strict Leakage Barriers: In supervised learning, fit vectorizers strictly on fold training splits; in unsupervised learning, isolate ground-truth class columns completely.",
        "100% Reproducible Execution: Pipeline runs end-to-end via 'python3 src/run_all.py' in ~42 seconds with fixed random_state=42."
    ]:
        p = tf10_r.add_paragraph(); p.text = f"• {txt}"; p.font.size = Pt(10.5); p.font.color.rgb = RGB_CHARCOAL; p.space_after = Pt(6)

    prs.save(output_path)
    print(f"[Presentation] Saved PPTX presentation to {output_path}")

def generate_pdf_slides(output_path="docs/presentation_slides.pdf"):
    """
    Renders presentation slides to landscape 16:9 PDF using ReportLab.
    Ensures docs/presentation_slides.pdf is generated natively and reproducibly.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 16:9 landscape dimensions: 11 x 6.1875 inches (792 x 445.5 pt)
    page_w = 792
    page_h = 445.5
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=(page_w, page_h),
        leftMargin=28,
        rightMargin=28,
        topMargin=20,
        bottomMargin=20
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=HEX_NAVY,
        spaceAfter=2
    )
    
    category_style = ParagraphStyle(
        'SlideCategory',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10.5,
        textColor=HEX_TEAL,
        spaceAfter=3
    )
    
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=HEX_NAVY,
        spaceAfter=5
    )
    
    slide_body = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=HEX_CHARCOAL,
        spaceAfter=3.5
    )
    
    tbl_hdr = ParagraphStyle('TblHdr', fontName='Helvetica-Bold', fontSize=7.0, leading=8.5, textColor=colors.white, alignment=1)
    tbl_cell = ParagraphStyle('TblCell', fontName='Helvetica', fontSize=6.5, leading=8.0, textColor=HEX_CHARCOAL, alignment=1)
    tbl_cell_left = ParagraphStyle('TblCellLeft', parent=tbl_cell, alignment=0)
    tbl_cell_bold = ParagraphStyle('TblCellBold', parent=tbl_cell, fontName='Helvetica-Bold')

    story = []
    
    def slide_header_block(title, cat="MACHINE LEARNING COURSE PROJECTS"):
        return [
            Paragraph(cat.upper(), category_style),
            Paragraph(title, title_style),
            HRFlowable(width="100%", thickness=1, color=HEX_NAVY, spaceAfter=8, spaceBefore=2)
        ]
        
    # Slide 1: Title
    story.append(Spacer(1, 40))
    t_box_data = [[
        Paragraph("<font color='#0D9488'><b>MACHINE LEARNING COURSE PROJECTS</b></font><br/><br/>"
                  "<font color='#1A365D' size='22'><b>Supervised Spam Email Detection &amp;<br/>Unsupervised Dry Bean Morphological Clustering</b></font><br/><br/>"
                  "<font color='#4A5568' size='10'><b>Author:</b> Adam Dalwai (Dalwai) &nbsp;|&nbsp; <b>Institution:</b> Department of Computer Science &amp; Machine Learning<br/>"
                  "<b>Submission Deliverables:</b> Research Report (PDF/MD), Final Predictions, Presentation Deck</font>",
                  ParagraphStyle('TitleP', alignment=1, leading=16))
    ]]
    t1 = Table(t_box_data, colWidths=[720])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HEX_CARD_BG),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 25),
        ('BOTTOMPADDING', (0,0), (-1,-1), 25),
        ('GRID', (0,0), (-1,-1), 1.5, HEX_BORDER),
    ]))
    story.append(t1)
    story.append(PageBreak())
    
    # Slide 2: Executive Overview
    story.extend(slide_header_block("Executive Project Overview: Dual Machine Learning Paradigms"))
    left_c = (
        "<b>Project 1: Supervised Spam Detection</b><br/><br/>"
        "• <b>Objective:</b> Binary classification (Ham vs. Spam) under severe label skew (78.5% Ham / 21.5% Spam).<br/>"
        "• <b>Cost Asymmetry:</b> Blocking legitimate email (Ham FPR) is 20x more costly than delivering spam (C_FP >> C_FN).<br/>"
        "• <b>Hygiene &amp; Leakage:</b> Purged 103 cross-dataset duplicates; TF-IDF fit strictly on fold training splits.<br/>"
        "• <b>Winner:</b> Calibrated Linear SVM (C=2.0) with Platt scaling (0.9740 Acc, 0.9617 Macro-F1, 0.9401 Spam-F1).<br/>"
        "• <b>Threshold Tuning:</b> Shifting τ to 0.70 cuts Ham FPR by 50% (0.0185 → 0.0094) while preserving 0.9133 F1."
    )
    right_c = (
        "<b>Project 2: Dry Bean Morphological Clustering</b><br/><br/>"
        "• <b>Objective:</b> Unsupervised clustering across M=13,611 biological specimens and 16 morphological features.<br/>"
        "• <b>Class Isolation:</b> Ground-truth varieties completely excluded from preprocessing, PCA, and clustering.<br/>"
        "• <b>Collinearity &amp; PCA:</b> Scale features exhibit r > 0.98. 4 PCs capture 95.02% cumulative variance.<br/>"
        "• <b>Winner:</b> K-Means on 4D PCA dominates across all metrics (Silhouette 0.3277, Calinski-Harabasz 9,134.22).<br/>"
        "• <b>Botanical Profiles:</b> Precisely matches 7 cultivars: Cali, Dermason, Barbunya, Seker, Bombay, Sira, Horoz."
    )
    t2 = Table([[Paragraph(left_c, slide_body), Paragraph(right_c, slide_body)]], colWidths=[360, 360])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HEX_CARD_BG),
        ('GRID', (0,0), (-1,-1), 1, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(t2)
    story.append(PageBreak())
    
    # Slide 3: Project 1 Data Hygiene
    story.extend(slide_header_block("Project 1: Data Hygiene, Deduplication & Leakage-Free Design", "PROJECT 1: SUPERVISED LEARNING"))
    c1 = "<b>1. Ingestion &amp; Repair</b><br/><br/>• Train 1 (SMS): 2,228 rows. Unescaped delimiter commas caused fragmented columns (v1 through Unnamed: 4). Reconciled into clean text.<br/>• Train 2 (Desktop): 2,068 rows of formal Enron emails.<br/>• Raw Total: 4,296 instances."
    c2 = "<b>2. Exact Deduplication</b><br/><br/>• 103 Duplicates Removed: String hashing identified duplicate records across datasets.<br/>• Leakage Firewall: Eliminates exact email memorization between train and validation folds.<br/>• Clean Corpus: 4,193 unique emails (78.46% Ham / 21.54% Spam)."
    c3 = "<b>3. Leakage-Free Design</b><br/><br/>• 5-Fold Stratified CV: Preserves 78.5% class balance.<br/>• Strict Containment: TF-IDF vocabulary, DF, and IDF weights fit ONLY on fold training splits.<br/>• Sublinear TF-IDF: Bigrams, sublinear TF (1+log(tf)), min_df=2 (~14,800 active features)."
    t3 = Table([[Paragraph(c1, slide_body), Paragraph(c2, slide_body), Paragraph(c3, slide_body)]], colWidths=[240, 240, 240])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HEX_CARD_BG),
        ('GRID', (0,0), (-1,-1), 1, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(t3)
    story.append(PageBreak())
    
    # Slide 4: Model Benchmarks
    story.extend(slide_header_block("Project 1: Model Benchmarks & Calibrated SVM Selection", "PROJECT 1: SUPERVISED LEARNING"))
    tbl4_data = [
        [Paragraph("<b>Model Architecture</b>", tbl_hdr), Paragraph("<b>Accuracy</b>", tbl_hdr), Paragraph("<b>Macro-F1</b>", tbl_hdr), Paragraph("<b>Spam Rec.</b>", tbl_hdr), Paragraph("<b>Spam F1</b>", tbl_hdr), Paragraph("<b>Ham FPR</b>", tbl_hdr)],
        [Paragraph("<b>Calibrated Linear SVM (C=2.0)</b>", tbl_cell_bold), Paragraph("<b>0.9740±0.005</b>", tbl_cell_bold), Paragraph("<b>0.9617±0.008</b>", tbl_cell_bold), Paragraph("<b>0.9469±0.019</b>", tbl_cell_bold), Paragraph("<b>0.9401±0.012</b>", tbl_cell_bold), Paragraph("0.0185±0.005", tbl_cell)],
        [Paragraph("SGD Classifier (Mod. Huber)", tbl_cell_left), Paragraph("0.9738±0.007", tbl_cell), Paragraph("0.9603±0.010", tbl_cell), Paragraph("0.9115±0.030", tbl_cell), Paragraph("0.9372±0.016", tbl_cell), Paragraph("0.0091±0.003", tbl_cell)],
        [Paragraph("Logistic Regression (Bal, C=5)", tbl_cell_left), Paragraph("0.9714±0.006", tbl_cell), Paragraph("0.9575±0.010", tbl_cell), Paragraph("0.9292±0.023", tbl_cell), Paragraph("0.9332±0.015", tbl_cell), Paragraph("0.0170±0.004", tbl_cell)],
        [Paragraph("Linear SVM (LinearSVC, C=1.0)", tbl_cell_left), Paragraph("0.9714±0.007", tbl_cell), Paragraph("0.9565±0.010", tbl_cell), Paragraph("0.9004±0.029", tbl_cell), Paragraph("0.9311±0.017", tbl_cell), Paragraph("0.0091±0.002", tbl_cell)],
        [Paragraph("Random Forest (n=100)", tbl_cell_left), Paragraph("0.9344±0.009", tbl_cell), Paragraph("0.8919±0.017", tbl_cell), Paragraph("0.7165±0.042", tbl_cell), Paragraph("0.8241±0.029", tbl_cell), Paragraph("0.0058±0.002", tbl_cell)],
        [Paragraph("MultinomialNB (Baseline)", tbl_cell_left), Paragraph("0.8781±0.011", tbl_cell), Paragraph("0.7657±0.029", tbl_cell), Paragraph("0.4341±0.053", tbl_cell), Paragraph("0.6035±0.051", tbl_cell), Paragraph("0.0000±0.000", tbl_cell)]
    ]
    t4 = Table(tbl4_data, colWidths=[200, 95, 95, 95, 95, 95])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HEX_NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, HEX_LIGHT_BG])
    ]))
    story.append(t4)
    story.append(Spacer(1, 8))
    t4_exp = Paragraph(
        "<b>Key Takeaways:</b> Baseline MultinomialNB fails due to prior overwhelm (Recall 0.4341, misses 56.6% spam). "
        "Random Forest struggles in sparse 14,800-dim TF-IDF space (F1=0.8241). "
        "<b>Calibrated Linear SVM (C=2.0)</b> dominates, using Platt sigmoid calibration to map margins to authentic posterior probabilities P(y=1|x).",
        slide_body
    )
    story.append(t4_exp)
    story.append(PageBreak())
    
    # Slide 5: Thresholds & Error Analysis
    story.extend(slide_header_block("Project 1: Threshold Tradeoffs & Qualitative Error Analysis", "PROJECT 1: SUPERVISED LEARNING"))
    s5_left = [
        Image("outputs/figures/spam_roc_pr_curve.png", width=3.8*inch, height=1.6*inch),
        Image("outputs/figures/spam_threshold_tradeoff.png", width=3.8*inch, height=1.6*inch)
    ]
    s5_right_txt = (
        "<b>Operating Point Selection:</b><br/>"
        "• Default τ=0.50: Ham FPR = 0.0185 (61 false positives out of 3,290), Spam F1 = 0.9401.<br/>"
        "• Conservative τ=0.70: Slashes Ham FPR by 50% to 0.0094 (only 31 false positives) with Spam F1 = 0.9133.<br/><br/>"
        "<b>Qualitative Failure Modes &amp; Remedies:</b><br/>"
        "• <i>FP #179 (P=0.78, P2P Referral):</i> Referral invite triggered by 'free sms' bigram. Remedy: Domain reputation scoring.<br/>"
        "• <i>FP #373 (P=0.76, Intimate SMS):</i> 'sex cuddling' triggered adult weights. Remedy: Syntactic conversational parsing.<br/>"
        "• <i>FP #4106 (P=0.84, Health Bulletin):</i> Clinical terms mimic pharmacy spam. Remedy: Internal DKIM sender check.<br/>"
        "• <i>FN #193 (P=0.04, Complaint Camouflage):</i> Formal register hides phishing link. Remedy: URL destination inspection.<br/>"
        "• <i>FN #85 (P=0.08, Slang Evasion):</i> 'b fun 2nite call me free'. Remedy: Subword BPE &amp; Soundex normalization."
    )
    t5 = Table([[s5_left, Paragraph(s5_right_txt, slide_body)]], colWidths=[310, 420])
    t5.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t5)
    story.append(PageBreak())
    
    # Slide 6: Dry Bean PCA
    story.extend(slide_header_block("Project 2: Dry Bean Morphological PCA (95.02% Variance in 4 PCs)", "PROJECT 2: UNSUPERVISED LEARNING"))
    s6_left = Image("outputs/figures/drybean_pca_variance.png", width=4.6*inch, height=3.2*inch)
    s6_right_txt = (
        "<b>Dimensionality Reduction Rationale:</b><br/><br/>"
        "• <b>Dataset:</b> 13,611 samples, 16 features, 0 missing values. Class column strictly isolated.<br/>"
        "• <b>StandardScaler:</b> Normalizes Area (~60k) and ShapeFactor2 (~0.0017) to zero mean, unit variance.<br/>"
        "• <b>Multicollinearity Resolution:</b> Scale features exhibit Pearson r > 0.98. PCA isolates orthogonal physical axes.<br/>"
        "• <b>Kaiser Rule (λ > 1.0):</b> Retains 3 components (89.90% variance).<br/>"
        "• <b>Selected 4 PCs (95.02%):</b> Surpasses the 95% engineering threshold, compressing dimensions by 75% while stripping isotropic noise.<br/><br/>"
        "<b>Morphological Eigenvector Loadings:</b><br/>"
        "• <b>PC1 (55.5%):</b> Size &amp; Physical Scale (Area, Perimeter, EquivDiameter).<br/>"
        "• <b>PC2 (26.4%):</b> Aspect Ratio vs. Compactness (AspectRation, Compactness).<br/>"
        "• <b>PC3 (8.0%):</b> Surface Rugosity &amp; Smoothness (Solidity, ShapeFactor4).<br/>"
        "• <b>PC4 (5.1%):</b> Bounding Box Occupancy (Extent)."
    )
    t6 = Table([[s6_left, Paragraph(s6_right_txt, slide_body)]], colWidths=[340, 390])
    t6.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t6)
    story.append(PageBreak())
    
    # Slide 7: Clustering Benchmark
    story.extend(slide_header_block("Project 2: Clustering Benchmark (K-Means vs. GMM on Scaled vs. PCA)", "PROJECT 2: UNSUPERVISED LEARNING"))
    tbl7_data = [
        [Paragraph("<b>Configuration</b>", tbl_hdr), Paragraph("<b>Algorithm</b>", tbl_hdr), Paragraph("<b>Feature Space</b>", tbl_hdr), Paragraph("<b>Silhouette (↑)</b>", tbl_hdr), Paragraph("<b>Calinski-Harabasz (↑)</b>", tbl_hdr), Paragraph("<b>Davies-Bouldin (↓)</b>", tbl_hdr)],
        [Paragraph("Config 1", tbl_cell), Paragraph("K-Means", tbl_cell), Paragraph("Standardized (16D)", tbl_cell), Paragraph("0.3094", tbl_cell), Paragraph("7,848.44", tbl_cell), Paragraph("1.0993", tbl_cell)],
        [Paragraph("<b>Config 2 (Winner)</b>", tbl_cell_bold), Paragraph("<b>K-Means</b>", tbl_cell_bold), Paragraph("<b>PCA (4D, 95.02%)</b>", tbl_cell_bold), Paragraph("<b>0.3277</b>", tbl_cell_bold), Paragraph("<b>9,134.22</b>", tbl_cell_bold), Paragraph("<b>1.0328</b>", tbl_cell_bold)],
        [Paragraph("Config 3", tbl_cell), Paragraph("GMM", tbl_cell), Paragraph("Standardized (16D)", tbl_cell), Paragraph("0.2898", tbl_cell), Paragraph("6,426.16", tbl_cell), Paragraph("1.6592", tbl_cell)],
        [Paragraph("Config 4", tbl_cell), Paragraph("GMM", tbl_cell), Paragraph("PCA (4D, 95.02%)", tbl_cell), Paragraph("0.3020", tbl_cell), Paragraph("7,618.59", tbl_cell), Paragraph("1.2245", tbl_cell)]
    ]
    t7 = Table(tbl7_data, colWidths=[110, 100, 160, 110, 130, 120])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HEX_NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, HEX_LIGHT_BG])
    ]))
    story.append(t7)
    story.append(Spacer(1, 10))
    c7_l = "<b>Why K-Means on 4D PCA Dominates:</b><br/>• Dominates all 3 validation metrics: highest Silhouette (0.3277), highest Calinski-Harabasz (9,134.22), lowest Davies-Bouldin (1.0328).<br/>• Stripping the 12 isotropic noise dimensions mitigates Euclidean distance degradation.<br/>• Cross-Space Validation: Projecting labels back to 16D space yields Sil 0.3076, CH 7,817.26, confirming true physical cluster cohesion."
    c7_r = "<b>Why GMM Struggles:</b><br/>• Over-Parameterization: In 16D space, fitting 7 full covariance matrices requires estimating 7 × (16×17/2) = 952 parameters.<br/>• Component Collapse: Severe covariance estimation error creates diffuse, fragmented probabilities.<br/>• Voronoi Geometry: Orthogonal 4D PCA satisfies spherical assumptions much better than raw non-orthogonal space."
    t7_exp = Table([[Paragraph(c7_l, slide_body), Paragraph(c7_r, slide_body)]], colWidths=[360, 360])
    t7_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HEX_CARD_BG),
        ('GRID', (0,0), (-1,-1), 1, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(t7_exp)
    story.append(PageBreak())
    
    # Slide 8: Morphological Profiles
    story.extend(slide_header_block("Project 2: Morphological Cluster Profiles & Botanical Archetypes", "PROJECT 2: UNSUPERVISED LEARNING"))
    tbl8_data = [
        [Paragraph("<b>Cluster</b>", tbl_hdr), Paragraph("<b>Count (%)</b>", tbl_hdr), Paragraph("<b>Area (μ)</b>", tbl_hdr), Paragraph("<b>Aspect Ratio</b>", tbl_hdr), Paragraph("<b>Compactness</b>", tbl_hdr), Paragraph("<b>Botanical Variety Archetype &amp; Morphology</b>", tbl_hdr)],
        [Paragraph("<b>0</b>", tbl_cell_bold), Paragraph("519 (3.8%)", tbl_cell), Paragraph("63,737", tbl_cell), Paragraph("1.779", tbl_cell), Paragraph("0.745", tbl_cell), Paragraph("<b>Cali:</b> Large elongated seeds, medium circularity", tbl_cell_left)],
        [Paragraph("<b>1</b>", tbl_cell_bold), Paragraph("3,262 (24.0%)", tbl_cell), Paragraph("31,927", tbl_cell), Paragraph("1.476", tbl_cell), Paragraph("0.823", tbl_cell), Paragraph("<b>Dermason:</b> Small oval, highly compact, dense commercial seed", tbl_cell_left)],
        [Paragraph("<b>2</b>", tbl_cell_bold), Paragraph("2,476 (18.2%)", tbl_cell), Paragraph("74,341", tbl_cell), Paragraph("1.651", tbl_cell), Paragraph("0.778", tbl_cell), Paragraph("<b>Barbunya:</b> Large broad oval seeds, speckled market class", tbl_cell_left)],
        [Paragraph("<b>3</b>", tbl_cell_bold), Paragraph("2,021 (14.9%)", tbl_cell), Paragraph("39,389", tbl_cell), Paragraph("1.235", tbl_cell), Paragraph("0.900", tbl_cell), Paragraph("<b>Seker:</b> Small round/spherical seeds, maximum plumpness", tbl_cell_left)],
        [Paragraph("<b>4</b>", tbl_cell_bold), Paragraph("521 (3.8%)", tbl_cell), Paragraph("173,557", tbl_cell), Paragraph("1.585", tbl_cell), Paragraph("0.793", tbl_cell), Paragraph("<b>Bombay:</b> Giant broad seeds, massive volume (>170k Area)", tbl_cell_left)],
        [Paragraph("<b>5</b>", tbl_cell_bold), Paragraph("3,042 (22.4%)", tbl_cell), Paragraph("44,602", tbl_cell), Paragraph("1.576", tbl_cell), Paragraph("0.796", tbl_cell), Paragraph("<b>Sira:</b> Medium compact oval seeds, general-purpose variety", tbl_cell_left)],
        [Paragraph("<b>6</b>", tbl_cell_bold), Paragraph("1,770 (13.0%)", tbl_cell), Paragraph("53,695", tbl_cell), Paragraph("2.039", tbl_cell), Paragraph("0.699", tbl_cell), Paragraph("<b>Horoz:</b> Long cylindrical kidney bean, maximum elongation", tbl_cell_left)]
    ]
    t8 = Table(tbl8_data, colWidths=[55, 75, 75, 75, 75, 375])
    t8.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HEX_NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, HEX_LIGHT_BG])
    ]))
    story.append(t8)
    story.append(Spacer(1, 8))
    t8_exp = Paragraph(
        "<b>Phenotypic Continuum:</b> Clusters trace a clear physical spectrum from giant seeds (Bombay, Area 173k) to small round seeds (Seker, Compactness 0.90) "
        "and highly elongated kidney beans (Horoz, Aspect Ratio 2.04), confirming authentic botanical variety discovery without labels.",
        slide_body
    )
    story.append(t8_exp)
    story.append(PageBreak())
    
    # Slide 9: 2D Visual Projections vs 4D Clustering
    story.extend(slide_header_block("Project 2: 2D Visual Projections vs. 4D Clustering Space", "PROJECT 2: UNSUPERVISED LEARNING"))
    s9_figs = [
        Image("outputs/figures/drybean_kmeans_clusters.png", width=4.5*inch, height=3.0*inch),
        Image("outputs/figures/drybean_method2_clusters.png", width=4.5*inch, height=3.0*inch)
    ]
    t9 = Table([[s9_figs[0], s9_figs[1]]], colWidths=[360, 360])
    t9.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t9)
    story.append(Spacer(1, 4))
    t9_note = Paragraph(
        "<b>CRITICAL METHODOLOGICAL DISTINCTION:</b> Clustering was executed in <b>4-Dimensional PCA Space</b> (95.02% variance, preserving PC3 smoothness and PC4 extent). "
        "Visualizations project points onto <b>2D space</b> (81.90% variance) for human inspection. Apparent overlaps in 2D are cleanly separated along PC3 and PC4 in 4D.",
        slide_body
    )
    story.append(t9_note)
    story.append(PageBreak())
    
    # Slide 10: Conclusions & Deliverable Verification
    story.extend(slide_header_block("Conclusions, Reproducibility & Deliverable Verification", "EXECUTIVE SUMMARY"))
    c10_l = (
        "<b>Submission Files &amp; Verification:</b><br/><br/>"
        "• <b>Dalwai_Spam.txt:</b> Exactly 6,447 lines, no header. Test predictions yield 79.17% Ham / 20.83% Spam, matching clean training prior within 0.71%.<br/>"
        "• <b>Dalwai_DryBeanClusters.csv:</b> Exactly 13,612 lines (1 header + 13,611 rows), schema 'sample_id,Class_id'. Exactly preserves 1-based indices.<br/>"
        "• <b>docs/report.pdf:</b> Maximum 6 pages requirement strictly satisfied (4 pages total, fully typeset academic report).<br/>"
        "• <b>docs/presentation_slides.pptx &amp; .pdf:</b> Complete 10-slide widescreen presentation deck."
    )
    c10_r = (
        "<b>Key Cross-Paradigm Learnings:</b><br/><br/>"
        "• <b>Representation Dictates Algorithm:</b> Sparse text (D ≈ 14,800) naturally favors linear hyperplanes (SVM); dense collinear biological features (D=16) require orthogonal PCA before clustering.<br/>"
        "• <b>Cost Asymmetry:</b> In spam filtering, probability calibration enables operational threshold tuning, cutting false positives by 50%.<br/>"
        "• <b>Data Leakage Firewalls:</b> Fit vectorizers strictly on fold training splits; isolate ground-truth class columns completely in unsupervised tasks.<br/>"
        "• <b>100% Reproducible:</b> Single-command execution (python3 src/run_all.py, ~42s runtime, seed=42)."
    )
    t10 = Table([[Paragraph(c10_l, slide_body), Paragraph(c10_r, slide_body)]], colWidths=[360, 360])
    t10.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HEX_CARD_BG),
        ('GRID', (0,0), (-1,-1), 1, HEX_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(t10)
    
    doc.build(story)
    print(f"[Presentation] Saved PDF presentation to {output_path}")

if __name__ == "__main__":
    generate_pptx_slides("docs/presentation_slides.pptx")
    generate_pdf_slides("docs/presentation_slides.pdf")
