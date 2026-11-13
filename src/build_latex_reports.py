#!/usr/bin/env python3
"""
Build and compile 5 distinct publication-grade LaTeX report variants.
All variants strictly satisfy the <= 6 pages limit, incorporate all verified empirical numbers,
exact Table 6 sample standard deviations, descriptive morphological archetypes,
consistent operating thresholds (tau*=0.50 baseline, tau=0.70 constrained),
paired t-tests, multi-seed stability (ARI=0.9978), and classical NLP remedies.
"""

import os
import subprocess
import pymupdf

BASE_DIR = "/Users/addie/repos/ML_Project"

def compile_variant(tex_dir):
    print(f"\n>>> Compiling LaTeX in: {tex_dir}")
    os.system(f"ln -s {BASE_DIR}/outputs/figures {tex_dir}/figures 2>/dev/null || true")
    res = subprocess.run("tectonic report.tex", shell=True, cwd=tex_dir, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation error in {tex_dir}:\n{res.stderr}")
        return False, 0
    pdf_path = os.path.join(tex_dir, "report.pdf")
    doc = pymupdf.open(pdf_path)
    pages = len(doc)
    print(f"Compiled successfully! Total pages: {pages}")
    return True, pages

# ==============================================================================
# VARIANT 2: NATURE / SCIENCE EDITORIAL (PALATINO, FOREST GREEN, DEEP OCHRE)
# ==============================================================================
def write_variant2():
    out_dir = os.path.join(BASE_DIR, "docs/latex_reports/variant2_nature")
    os.makedirs(out_dir, exist_ok=True)
    
    tex = r"""\documentclass[9.5pt,twocolumn,letterpaper]{article}

\usepackage[
  top=0.65in,
  bottom=0.65in,
  left=0.65in,
  right=0.65in,
  columnsep=0.22in
]{geometry}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathpazo}
\usepackage{microtype}

\usepackage{amsmath,mathtools,bm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{siunitx}
\sisetup{detect-all, tight-spacing=true}

\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[font=small,labelfont=bf,skip=3pt]{caption}
\usepackage{float}

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}
\setlength{\floatsep}{4pt plus 1pt minus 1pt}
\setlength{\textfloatsep}{5pt plus 1pt minus 1pt}
\setlength{\intextsep}{4pt plus 1pt minus 1pt}

\usepackage{xcolor}
\definecolor{forestgreen}{RGB}{25,75,45}
\definecolor{deepochre}{RGB}{180,105,25}
\definecolor{darkslate}{RGB}{35,45,55}
\usepackage[
  colorlinks=true,
  linkcolor=forestgreen,
  citecolor=deepochre,
  urlcolor=forestgreen
]{hyperref}

\usepackage[compact]{titlesec}
\titleformat{\section}{\large\bfseries\color{forestgreen}}{\thesection.}{0.4em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{forestgreen}}{\thesubsection.}{0.35em}{}
\titleformat{\subsubsection}{\small\bfseries\itshape\color{darkslate}}{\thesubsubsection.}{0.3em}{}
\titlespacing{\section}{0pt}{4.5pt plus 1pt minus 1pt}{2pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsection}{0pt}{3.5pt plus 1pt minus 1pt}{1.5pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsubsection}{0pt}{2pt plus 0.5pt minus 0.5pt}{1pt}

\usepackage{enumitem}
\setlist{nosep, leftmargin=1.0em, labelsep=0.3em}

\graphicspath{{figures/}{../../../outputs/figures/}{outputs/figures/}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bZ}{\mathbf{Z}}
\newcommand{\bSigma}{\mathbf{\Sigma}}
\newcommand{\bLambda}{\mathbf{\Lambda}}
\newcommand{\bV}{\mathbf{V}}
\newcommand{\indic}{\mathbb{I}}

\title{\vspace{-0.3in}\textbf{\Large Empirical Evaluation of Supervised Spam Filtering and Unsupervised Morphological Clustering}\vspace{-0.08in}}
\author{\textbf{Adam Dalwai} \\ Department of Computer Science \\ \texttt{adamdalwai1@gmail.com}}
\date{\vspace{-0.15in}\small November 2026\vspace{-0.1in}}

\begin{document}

\maketitle

\begin{abstract}
\noindent \textbf{We present an empirical study covering supervised text classification and unsupervised geometric clustering.} In Project~1, we develop an asymmetric cost-sensitive spam filter over a combined corpus of 4,193 deduplicated messages. Using stratified 5-fold cross-validation with fold-isolated TF-IDF extraction, we evaluate nine candidate architectures. Calibrated Linear Support Vector Machines ($C=2.0$) achieve top balanced performance: Accuracy $0.9740 \pm 0.0053$, Macro-F1 $0.9617 \pm 0.0079$, and Spam F1 $0.9401 \pm 0.0123$. While uncalibrated SGD and LinearSVC attain lower Ham False Positive Rates at default thresholds (0.0091 vs 0.0185), paired statistical tests show no significant F1 difference ($p=0.579$), and Platt probability calibration enables post-hoc operational threshold tuning. Under risk-constrained optimization ($\tau=0.70$), Ham FPR is reduced by 49.2\% to 0.0094 (31 false positives out of 3,290) while sustaining Spam F1 at 0.9133. Out-of-fold error analysis examines five false positives and five false negatives at the decision boundary, proposing NLP feature remedies. Predictions on 6,447 unlabeled test instances match clean training priors within 0.71\%. In Project~2, we perform unsupervised clustering on 13,611 biological specimens characterized by 16 morphological features with ground-truth class labels strictly isolated. Principal Component Analysis demonstrates that four dimensions capture 95.02\% of cumulative variance, compared with three dimensions under the Kaiser rule (89.90\%). Benchmarking four configurations ($k=7$) shows that K-Means on 4D PCA features yields a Silhouette score of 0.3277, Calinski--Harabasz index of 9,134.22, and Davies--Bouldin index of 1.0328, with pairwise Adjusted Rand Index across 10 random seeds averaging 0.9978. Morphological profiling characterizes seven distinct geometric archetypes.
\end{abstract}

\section{Introduction and Problem Formulations}
Supervised text categorization and unsupervised geometric clustering serve distinct objectives in applied machine learning. Supervised learning requires strict data hygiene, fold-level feature isolation to prevent leakage, and threshold calibration under asymmetric operational error costs. Unsupervised clustering requires multi-scale normalization, covariance decomposition, and internal metric validation independent of external reference labels.

We evaluate both tasks using canonical benchmarks: (1)~binary spam detection over combined SMS and desktop email corpora, and (2)~morphological cluster discovery across high-dimensional geometric measurements of dry bean seeds. All pipelines are implemented in Python (v3.13.2) utilizing \texttt{scikit-learn} (v1.9.0), \texttt{numpy} (v2.4.6), \texttt{pandas} (v3.0.3), and \texttt{scipy} (v1.17.1), with all stochastic operations pinned to $\text{seed}=42$ for deterministic reproducibility.

\section{Project 1: Spam Email Detection}

\subsection{Corpus Ingestion, Cleaning, and Deduplication}
The supervised learning dataset combines two heterogeneous sources:
\begin{itemize}
  \item \textbf{Training Dataset 1 (SMS):} 2,228 short mobile messages (1,927 Ham, 301 Spam; 13.51\% spam prevalence). Comma delimiter fragmentation across columns \texttt{v2}, \texttt{Unnamed: 2}, \texttt{Unnamed: 3}, and \texttt{Unnamed: 4} was repaired by programmatic text reconstruction.
  \item \textbf{Training Dataset 2 (Email):} 2,068 desktop emails (1,440 Ham, 628 Spam; 30.37\% spam prevalence) with full message bodies and integer labels.
  \item \textbf{Hidden Test Dataset:} 6,447 unlabeled instances for final inference.
\end{itemize}

Direct union yields 4,296 raw instances. Exact string hashing identified and removed \textbf{103 duplicate records} (64 in Dataset~1, 39 in Dataset~2, 0 cross-corpus collisions; 0 conflicting labels). Purging duplicates established a clean training corpus of \textbf{4,193 unique instances} (3,290 Ham [78.46\%], 903 Spam [21.54\%], imbalance ratio 3.64:1). Deduplication prior to partitioning ensures identical texts cannot co-occur across training and validation splits.

\subsection{Experimental Design, Leakage Prevention, and Models}
Text was tokenized using sublinear TF-IDF scaling:
\begin{equation}
\text{TF-IDF}(t, d) = \left(1 + \log(\text{tf}(t, d))\right) \cdot \log\left(\frac{1 + N}{1 + \text{df}(t)}\right)
\end{equation}
with unigram and bigram extraction ($n \in \{1, 2\}$), document frequency threshold $\text{min\_df}=2$, sublinear term weighting, and accent stripping, producing $\approx 14,800$ sparse features.

\textbf{Fold Isolation Protocol:} In each fold of Stratified 5-Fold Cross-Validation, the TF-IDF vectorizer was fit strictly on the training partition ($X_{\text{train}}$) and applied to the validation partition ($X_{\text{val}}$) via \texttt{transform}. Test instances were never accessed during vocabulary construction. Table~\ref{tab:model_specs} details the hyperparameter configurations across all nine evaluated architectures.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l X}
\toprule
\textbf{Model Architecture} & \textbf{Hyperparameter Specification \& Tuning Space} \\
\midrule
MultinomialNB (Baseline) & $\alpha=1.0$, uniform class prior (uncalibrated) \\
ComplementNB & $\alpha=1.0$, norm=False (designed for imbalanced text) \\
Logistic Regression ($C=1.0$) & $C=1.0$, L2 penalty, lbfgs solver, max\_iter=1000 \\
Logistic Regression ($C=10.0$) & $C=10.0$, low regularization, lbfgs solver \\
Logistic Regression (Balanced) & $C=5.0$, \texttt{class\_weight='balanced'}, lbfgs solver \\
Linear SVM (LinearSVC) & $C=1.0$, squared hinge loss, L2 penalty \\
\textbf{Calibrated Linear SVM} & LinearSVC ($C=2.0$) with Platt sigmoid calibration (5-fold internal CV) \\
Random Forest ($n=100$) & 100 estimators, Gini criterion, max\_features='sqrt' \\
SGD Classifier (Mod. Huber) & Modified Huber loss, $\alpha=10^{-4}$, elasticnet ($\rho=0.15$) \\
\bottomrule
\end{tabularx}
\caption{Hyperparameter configurations across nine evaluated spam classifiers.}
\label{tab:model_specs}
\end{table}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.06}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c@{}}
\toprule
\textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Spam Precision} & \textbf{Spam Recall} & \textbf{Spam F1} & \textbf{Ham FPR} \\
\midrule
\textbf{Calibrated Linear SVM ($C=2.0$)} & \textbf{0.9740$\pm$0.0053} & \textbf{0.9617$\pm$0.0079} & 0.9336$\pm$0.0155 & \textbf{0.9469$\pm$0.0186} & \textbf{0.9401$\pm$0.0123} & 0.0185$\pm$0.0046 \\
SGD Classifier (Modified Huber) & 0.9738$\pm$0.0066 & 0.9603$\pm$0.0103 & 0.9650$\pm$0.0115 & 0.9115$\pm$0.0303 & 0.9372$\pm$0.0164 & 0.0091$\pm$0.0032 \\
Logistic Regression (Balanced, $C=5.0$) & 0.9714$\pm$0.0064 & 0.9575$\pm$0.0096 & 0.9375$\pm$0.0132 & 0.9292$\pm$0.0235 & 0.9332$\pm$0.0152 & 0.0170$\pm$0.0038 \\
Linear SVM (LinearSVC, $C=1.0$) & 0.9714$\pm$0.0066 & 0.9565$\pm$0.0104 & 0.9644$\pm$0.0088 & 0.9004$\pm$0.0293 & 0.9311$\pm$0.0167 & 0.0091$\pm$0.0024 \\
Logistic Regression ($C=10.0$) & 0.9638$\pm$0.0065 & 0.9440$\pm$0.0106 & 0.9664$\pm$0.0030 & 0.8616$\pm$0.0301 & 0.9108$\pm$0.0172 & 0.0082$\pm$0.0007 \\
Random Forest ($n=100$, Gini) & 0.9344$\pm$0.0093 & 0.8919$\pm$0.0170 & 0.9714$\pm$0.0090 & 0.7165$\pm$0.0421 & 0.8241$\pm$0.0286 & 0.0058$\pm$0.0018 \\
Complement Naive Bayes & 0.9318$\pm$0.0096 & 0.8885$\pm$0.0180 & 0.9506$\pm$0.0121 & 0.7209$\pm$0.0468 & 0.8191$\pm$0.0303 & 0.0103$\pm$0.0026 \\
Logistic Regression ($C=1.0$, L2) & 0.9222$\pm$0.0087 & 0.8672$\pm$0.0175 & 0.9836$\pm$0.0111 & 0.6500$\pm$0.0435 & 0.7818$\pm$0.0300 & 0.0030$\pm$0.0021 \\
Multinomial Naive Bayes (Baseline) & 0.8781$\pm$0.0113 & 0.7657$\pm$0.0288 & \textbf{1.0000$\pm$0.0000} & 0.4341$\pm$0.0527 & 0.6035$\pm$0.0515 & \textbf{0.0000$\pm$0.0000} \\
\bottomrule
\end{tabular*}
\caption{Stratified 5-Fold Cross-Validation Performance Comparison across Nine Model Architectures (Reported as $\mu \pm \sigma$).}
\label{tab:spam_benchmark}
\end{table*}

\subsection{Cross-Validation Results and Statistical Comparison}
Table~\ref{tab:spam_benchmark} reports cross-validated performance. MultinomialNB achieved 100\% precision but suffered severe under-detection (Spam Recall $0.4341 \pm 0.0527$), missing 56.6\% of spam messages. This occurs because the 3.64:1 class prior heavily suppresses uncalibrated posterior probabilities.

Calibrated Linear SVM ($C=2.0$) attained the highest mean Spam F1 ($0.9401 \pm 0.0123$) and Spam Recall ($0.9469 \pm 0.0186$). However, SGD Classifier with Modified Huber loss achieved a comparable Spam F1 ($0.9372 \pm 0.0164$) with a lower Ham FPR ($0.0091 \pm 0.0032$). A paired two-tailed $t$-test across the five cross-validation folds reveals that the F1 difference between Calibrated SVM and SGD is not statistically significant ($t = 0.603, p = 0.579$). Similarly, LinearSVC ($C=1.0$) shows no statistically significant F1 shortfall ($t = 1.513, p = 0.205$).

\textbf{Rationale for Model Selection:} Calibrated Linear SVM is selected not due to an unassailable F1 margin, but because Platt sigmoid scaling via \texttt{CalibratedClassifierCV} fits parameters $A, B \in \R$ on internal 5-fold cross-validation margins:
\begin{equation}
P(y=1|\bx) = \frac{1}{1 + \exp\left(A \cdot f(\bx) + B\right)}
\end{equation}
producing true posterior probabilities. Uncalibrated LinearSVC yields arbitrary margins $f(\bx) \in (-\infty, \infty)$ that cannot be thresholded on probability scale, while SGD's surrogate loss produces poorer log-loss calibration. Continuous probability outputs are indispensable for operational threshold adjustments under asymmetric costs.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\columnwidth]{figures/spam_confusion_matrix.png}
\caption{Out-of-fold Confusion Matrix for the selected Calibrated Linear SVM ($C=2.0$) at operating threshold $\tau^*=0.50$ (Left: Counts; Right: Normalized Rates).}
\label{fig:confusion_matrix}
\end{figure}

\subsection{Operational Decision Thresholds and Cost Analysis}
Let $c_{FP}$ and $c_{FN}$ denote operational penalties for False Positives (blocking legitimate communications) and False Negatives (spam leakage). Total cost is modeled as $\text{Cost}(\tau) = c_{FP} \cdot FP(\tau) + c_{FN} \cdot FN(\tau)$. With $c_{FP}/c_{FN} \approx 20$, misclassifying ham is 20 times more severe. Table~\ref{tab:threshold_tradeoffs} evaluates out-of-fold metrics across $\tau \in [0.10, 0.90]$.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{c c c c c c r r r}
\toprule
\textbf{$\tau$} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & \textbf{Spam F1} & \textbf{Ham FPR} & \textbf{FP} & \textbf{FN} & \textbf{Cost ($c_{FP}=20$)} \\
\midrule
0.10 & 0.8595 & 0.6059 & \textbf{0.9945} & 0.7530 & 0.1775 & 584 & \textbf{5} & 11,685 \\
0.20 & 0.9294 & 0.7579 & 0.9878 & 0.8577 & 0.0866 & 285 & 11 & 5,711 \\
0.30 & 0.9556 & 0.8431 & 0.9756 & 0.9045 & 0.0498 & 164 & 22 & 3,302 \\
0.40 & 0.9680 & 0.8984 & 0.9601 & 0.9283 & 0.0298 & 98 & 36 & 1,996 \\
\textbf{0.50} & \textbf{0.9740} & 0.9334 & 0.9468 & \textbf{0.9401} & 0.0185 & 61 & 48 & 1,268 \\
0.60 & 0.9721 & 0.9517 & 0.9169 & 0.9340 & 0.0128 & 42 & 75 & 915 \\
\textbf{0.70} & 0.9645 & \textbf{0.9620} & 0.8693 & 0.9133 & \textbf{0.0094} & \textbf{31} & 118 & 738 \\
0.80 & 0.9392 & 0.9751 & 0.7364 & 0.8391 & 0.0052 & 17 & 238 & \textbf{578} \\
0.90 & 0.9046 & 0.9809 & 0.5681 & 0.7195 & 0.0030 & 10 & 390 & 590 \\
\bottomrule
\end{tabularx}
\caption{Calibrated Linear SVM performance and asymmetric cost across decision thresholds $\tau$ on out-of-fold validation data.}
\label{tab:threshold_tradeoffs}
\end{table}

\textbf{Cost Tradeoff Discussion:} Pure linear cost minimization under $c_{FP}/c_{FN}=20$ reaches its global minimum at $\tau=0.80$ ($\text{Cost}=578$). However, operating at $\tau=0.80$ causes Spam Recall to collapse from 94.68\% to 73.64\% (238 spam leaks). In production, engineering deployments adopt risk-constrained optimization:
\begin{equation}
\min_{\tau} \text{Ham FPR}(\tau) \quad \text{subject to} \quad \text{Spam Recall}(\tau) \ge 0.85
\end{equation}
Under this criterion, $\tau=0.70$ is the optimal operating point, reducing Ham FPR by 49.2\% to \textbf{0.0094} (only 31 false positives out of 3,290 legitimate emails) while maintaining a high Spam F1 of 0.9133. For standard balanced utility, $\tau^*=0.50$ provides maximum F1 (0.9401) and Accuracy (0.9740). Figure~\ref{fig:confusion_matrix} plots the confusion matrix at $\tau^*=0.50$ (TN: 3,229, FP: 61, FN: 48, TP: 855).

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_roc_pr_curve.png}
  \caption{ROC \& PR Curves}
  \label{subfig:roc_pr}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_threshold_tradeoff.png}
  \caption{Tradeoffs vs. $\tau$}
  \label{subfig:threshold}
\end{subfigure}
\caption{Discrimination diagnostics for Calibrated Linear SVM: (a) ROC-AUC 0.9904, PR-AUC 0.9772; (b) Precision, Recall, F1, and Ham FPR response across $\tau$.}
\label{fig:spam_diagnostics}
\end{figure}

\subsection{Qualitative Error Analysis and Actionable Remedies}
Table~\ref{tab:error_forensics} presents five False Positives and five False Negatives from out-of-fold validation predictions evaluated at operating threshold $\tau^*=0.50$. All five FPs satisfy $P(\text{Spam}) \ge 0.50$ and all five FNs satisfy $P(\text{Spam}) < 0.50$, ensuring complete consistency with the reported confusion matrix.

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{l c c X X}
\toprule
\textbf{Error Case \& ID} & \textbf{Truth} & $\bm{P(\text{Spam})}$ & \textbf{Failure Mechanism \& Key Text Elements} & \textbf{Actionable Engineering Remedy (Classical NLP)} \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Positives (Ham Misclassified as Spam at $\tau^*=0.50$) --- Total Out-of-Fold Count: 61}}} \\
FP 1 (\#179) & Ham $\to$ Spam & 0.7771 & Opt-in referral SMS: \texttt{"free sms", "contact list", "fullonsms.com"}. High positive weights on promotional tokens trigger false alarm. & Domain token downweighting; distinguish domain hostnames from promotional URL paths via regex feature extraction. \\
FP 2 (\#373) & Ham $\to$ Spam & 0.7613 & Intimate personal SMS: \texttt{"looking forward to all the sex cuddling.. Only two more sleeps"}. Adult unigram co-occurs with commercial adult spam. & Windowed co-occurrence features linking explicit unigrams with first-person personal pronouns (\texttt{"I"}, \texttt{"we"}, \texttt{"sleeps"}). \\
FP 3 (\#4106) & Ham $\to$ Spam & 0.8388 & Corporate memo on FDA drug recall: \texttt{"danger of phenylpropanolamine", "alka-seltzer"}. Body terms match online pharmacy spam. & Forwarded message zoning; identify and downweight quoted historical forward headers (\texttt{"- - original message - -"}) relative to new text. \\
FP 4 (\#25) & Ham $\to$ Spam & 0.7691 & Sparse ultra-short SMS: \texttt{"Just normal only here :)"}. Rare token combinations yield high variance in sparse feature space. & Length-calibrated shrinkage prior for short messages ($<5$ words), requiring multiple corroborating spam tokens before overriding ham prior. \\
FP 5 (\#534) & Ham $\to$ Spam & 0.6543 & Conversational query with non-standard syntax: \texttt{"No message..no responce..what happend?"}. Ellipses and typo penalized by bias term. & Punctuation sequence normalization (\texttt{".."} $\to$ \texttt{"."}) and character $n$-gram hashing ($n \in [3, 5]$) for colloquial orthography. \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Negatives (Spam Escaping Filter at $\tau^*=0.50$) --- Total Out-of-Fold Count: 48}}} \\
FN 1 (\#193) & Spam $\to$ Ham & 0.0451 & Adversarial customer complaint mimicry: \texttt{"ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS... L8ER GOT MEGA BILL"}. & Regex feature detectors for UK premium-rate dial codes (\texttt{"0870"}, \texttt{"0871"}) and non-standard currency symbols (\texttt{"£"}, \texttt{"\$\$"}). \\
FN 2 (\#85) & Spam $\to$ Ham & 0.0755 & Phonetic adult chatline evasion: \texttt{"Im nasty and have a thing 4 filthyguys... Txt XXX SLO"}. Creative spelling evades word vocabulary. & Subword character $n$-grams ($n \in [3, 5]$) to capture phonetic slang patterns despite non-standard spacing and abbreviations. \\
FN 3 (\#404) & Spam $\to$ Ham & 0.4229 & Financial pump-and-dump: \texttt{"Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK... Nasdaq Symbol CDGT"}. & Financial pattern regex: uppercase stock ticker detector (\texttt{"Symbol [A-Z]\{3,5\}"}) combined with promotional yield claims (\texttt{"UP OVER [0-9]+\%"}). \\
FN 4 (\#468) & Spam $\to$ Ham & 0.1855 & Conversational lure: \texttt{"Hi this is Amy, we will be sending you a free phone number in a couple of days..."}. Casual opening masks lure. & Positional transition bigrams capturing abrupt shifts from informal greeting openings to urgent commercial calls-to-action. \\
FN 5 (\#332) & Spam $\to$ Ham & 0.2154 & Legal disclaimer mimicry: \texttt{"0A\$NETWORKS allow companies to bill for SMS, so they are responsible for their suppliers..."}. & Disclaimer token density ratio: measure proportion of regulatory/boilerplate unigrams relative to document length and unique vocabulary. \\
\bottomrule
\end{tabularx}
\caption{Qualitative Analysis of Representative False Positives and False Negatives at Operating Threshold $\tau^*=0.50$.}
\label{tab:error_forensics}
\end{table*}

\subsection{Domain Shift Analysis and Hidden Test Set Predictions}
The training corpus exhibits substantial domain shift: SMS texts (train1, 2,164 clean rows) average 15.2 words with 13.22\% spam prevalence, whereas desktop emails (train2, 2,029 clean rows) average 184.6 words with 30.41\% spam prevalence. TF-IDF sublinear scaling and bigram extraction normalize length disparities.

The Calibrated Linear SVM ($C=2.0$) model was refit on all 4,193 clean training instances. Inference on the 6,447 unlabeled test instances at $\tau^*=0.50$ produced \textbf{5,104 Ham (79.17\%)} and \textbf{1,343 Spam (20.83\%)}. This matches the clean training prior (78.46\% Ham / 21.54\% Spam) within \textbf{0.71\%}, confirming balanced generalization without distributional collapse. Output files were generated and verified: \texttt{Dalwai\_Spam.txt} (exactly 6,447 lines, no header, labels in $\{0, 1\}$).

\section{Project 2: Dry Bean Morphological Clustering}

\begin{figure*}[t]
\centering
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_pca_variance.png}
  \caption{PCA Scree \& Variance}
  \label{subfig:pca_scree}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_kmeans_clusters.png}
  \caption{K-Means ($k=7$) 2D Projection}
  \label{subfig:kmeans_2d}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_method2_clusters.png}
  \caption{GMM ($k=7$) 2D Projection}
  \label{subfig:gmm_2d}
\end{subfigure}
\caption{Dry bean unsupervised clustering diagnostics: (a) Scree plot and cumulative variance; (b) K-Means ($k=7$) clusters projected on PC1--PC2; (c) GMM ($k=7$) clusters projected on PC1--PC2.}
\label{fig:drybean_banner}
\end{figure*}

\subsection{Data Hygiene, Skewness, and Feature Scaling}
The Dry Bean dataset contains 13,611 biological specimens with 16 continuous morphological measurements (12 geometric dimensions and 4 shape factors).

\textbf{Hygiene Verification:} Numerical features exhibit zero missing values. An audit identified 68 duplicate feature vectors; all 13,611 rows were preserved with monotonic 1-based indexing (\texttt{sample\_id} 1 to 13,611) per submission requirements. The \texttt{Class} column was unobserved and strictly excluded from all preprocessing, scaling, PCA, and clustering.

\textbf{Skewness and Outlier Analysis:} Feature distributions exhibit severe right-skewness: \texttt{Area} ($+2.953$), \texttt{ConvexArea} ($+2.941$), \texttt{MinorAxisLength} ($+2.238$), \texttt{EquivDiameter} ($+1.949$), and \texttt{Perimeter} ($+1.626$). Conversely, \texttt{ShapeFactor4} ($-2.759$) and \texttt{Solidity} ($-2.550$) display negative skewness. High area values ($>170{,}000\,\text{px}^2$) represent macroscopic seed varieties (Cluster 4) rather than measurement errors. We apply \texttt{StandardScaler} ($z = \frac{x - \mu}{\sigma}$) rather than nonlinear log-transforms to preserve the physical geometric aspect ratios across coordinate axes.

\subsection{PCA Dimensionality Selection: Kaiser Rule vs. 95\% Variance}
Let $\bZ \in \R^{13611 \times 16}$ denote the standardized feature matrix. We perform eigen-decomposition on the sample covariance matrix $\bSigma = \frac{1}{N-1}\bZ^T\bZ = \bV\bLambda\bV^T$ ($N=13,611$). Table~\ref{tab:pca_variance} reports the eigenvalue spectrum.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l r r r X}
\toprule
\textbf{PC} & \textbf{Eigenval.} & \textbf{Var (\%)} & \textbf{Cum (\%)} & \textbf{Morphological Interpretation} \\
\midrule
PC1 & 8.8753 & 55.47 & 55.47 & Scale \& Volume (\texttt{Area}: +0.33) \\
PC2 & 4.2293 & 26.43 & 81.90 & Aspect Ratio vs. Roundness (\texttt{AspectR}: +0.40) \\
PC3 & 1.2811 & 8.01 & 89.90 & Boundary Smoothness (\texttt{Solidity}: +0.46) \\
PC4 & 0.8183 & 5.11 & \textbf{95.02} & Bounding Occupancy (\texttt{Extent}: +0.72) \\
\midrule
PC5 & 0.4383 & 2.74 & 97.76 & Secondary boundary harmonics \\
PC6--16 & $<0.19$ & $<1.15$ & 100.00 & Residual collinear measurement noise \\
\bottomrule
\end{tabularx}
\caption{Eigenvalue spectrum and variance decomposition across 16 morphological features.}
\label{tab:pca_variance}
\end{table}

\textbf{Dimensionality Selection Criteria:}
\begin{itemize}
  \item \textbf{Kaiser-Guttman Rule ($\lambda_i > 1.0$):} Retains exactly 3 components ($\lambda_1=8.88, \lambda_2=4.23, \lambda_3=1.28$), explaining 89.90\% of variance.
  \item \textbf{95\% Variance Threshold:} Retains 4 components, reaching \textbf{95.02\%} cumulative variance. Although PC4's eigenvalue ($\lambda_4=0.8183$) is below 1.0, it captures bounding box occupancy (\texttt{Extent} loading $+0.72$), which separates compact seeds from boundary irregularities. Retaining 4 PCs reduces dimensions by 75\% (16D $\to$ 4D) while capturing 95\% of total information.
\end{itemize}

\subsection{Four-Configuration Clustering Benchmark ($k=7$)}
Table~\ref{tab:clustering_benchmark} compares four configurations across three internal validation metrics.

\textbf{Metric Interpretation Across Dimensionalities:}
\begin{itemize}
  \item In algorithm space, \textbf{Config 2 (K-Means on 4D PCA)} attains the highest Silhouette score (0.3277), highest Calinski--Harabasz index (9,134.22), and lowest Davies--Bouldin index (1.0328).
  \item Comparing metrics across different dimensional spaces requires care: distances concentrate differently in 16D vs 4D. Projecting cluster assignments back into original 16D standardized space yields a Silhouette score of \textbf{0.3076} (virtually identical to 16D K-Means at 0.3094) and Calinski--Harabasz of \textbf{7,817.26}. Thus, 4D PCA achieves comparable cluster structure in unreduced space while providing 75\% dimensionality reduction and faster convergence.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l l l c c c c@{}}
\toprule
\textbf{Configuration} & \textbf{Algorithm} & \textbf{Feature Space} & \makecell[c]{\textbf{Silhouette}\\\textbf{($\uparrow$)}} & \makecell[c]{\textbf{Calinski--}\\\textbf{Harabasz ($\uparrow$)}} & \makecell[c]{\textbf{Davies--}\\\textbf{Bouldin ($\downarrow$)}} & \makecell[c]{\textbf{Cluster Size Distribution}\\\textbf{($k=7$)}} \\
\midrule
Config 1 & K-Means & Standardized (16D) & 0.3094 & 7,848.44 & 1.0993 & 3111, 541, 3157, 520, 2480, 2033, 1769 \\
\textbf{Config 2 (Winner)} & \textbf{K-Means} & \textbf{PCA (4D, 95.0\% Var)} & \textbf{0.3277} & \textbf{9,134.22} & \textbf{1.0328} & 519, 3262, 2476, 2021, 521, 3042, 1770 \\
Config 3 & GMM & Standardized (16D) & 0.2898 & 6,426.16 & 1.6592 & 3191, 693, 2836, 520, 2514, 1984, 1873 \\
Config 4 & GMM & PCA (4D, 95.0\% Var) & 0.3020 & 7,618.59 & 1.2245 & 1875, 2699, 522, 1591, 2547, 3546, 831 \\
\bottomrule
\end{tabular*}
\caption{Internal Cluster Validation Comparison across Four Experimental Configurations ($k=7$) on 13,611 Dry Bean Specimens.}
\label{tab:clustering_benchmark}
\end{table*}

\subsection{Cluster Stability across Multi-Seed Initializations}
To evaluate whether K-Means partitions are sensitive to initialization, we executed Config~2 across 10 random seeds ($s \in \{0, 1, 7, 13, 21, 42, 99, 123, 777, 2024\}$) with $k\text{-means++}$ seeding ($n_{\text{init}}=10$). The pairwise Adjusted Rand Index (ARI) across all 45 seed combinations yielded:
\begin{equation}
\text{Mean ARI} = \mathbf{0.9978} \quad (\text{range: } [0.9939, 1.0000])
\end{equation}
An average ARI of 0.9978 confirms that the four-dimensional PCA representation provides an exceptionally convex, stable objective landscape, with K-Means converging to the identical partition structure regardless of random seed.

\subsection{Dimensionality Reduction vs. 2D Visual Display}
We emphasize the distinction between PCA for \textbf{Dimensionality Reduction Prior to Clustering} vs. \textbf{2D Visualization}:
\begin{itemize}
  \item \textbf{4D Clustering Space:} K-Means operates in $\R^4$ capturing \textbf{95.02\%} variance, retaining boundary rugosity (PC3) and bounding box occupancy (PC4).
  \item \textbf{2D Visual Projection:} Figures~\ref{fig:drybean_banner}(b) and \ref{fig:drybean_banner}(c) plot clusters on PC1--PC2, capturing \textbf{81.90\%} variance. The remaining 13.12\% variance operates orthogonally; apparent overlaps in 2D are separated along PC3 and PC4.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}c r r c c c c l@{}}
\toprule
\textbf{Cluster} & \textbf{Count} & \textbf{Share} & \textbf{Area ($\text{px}^2$)} & \makecell[c]{\textbf{Major Axis}\\\textbf{Length ($\text{px}$)}} & \makecell[c]{\textbf{Aspect}\\\textbf{Ratio}} & \textbf{Compactness} & \textbf{Morphological Cluster Archetype} \\
\midrule
C0 & 519 & 3.81\% & 63,737 $\pm$ 14,259 & 380.65 $\pm$ 46.79 & 1.779 $\pm$ 0.168 & 0.745 $\pm$ 0.037 & \textbf{Medium-Large Elongated Seeds} \\
C1 & 3,262 & 23.97\% & 31,927 $\pm$ 4,667 & 244.47 $\pm$ 18.23 & 1.475 $\pm$ 0.079 & 0.823 $\pm$ 0.022 & \textbf{Small Compact Elliptical Seeds} \\
C2 & 2,476 & 18.19\% & 74,341 $\pm$ 9,213 & 395.73 $\pm$ 32.13 & 1.651 $\pm$ 0.135 & 0.778 $\pm$ 0.034 & \textbf{Large Broad Oval Seeds} \\
C3 & 2,021 & 14.85\% & 39,389 $\pm$ 5,270 & 248.55 $\pm$ 18.85 & 1.235 $\pm$ 0.064 & 0.900 $\pm$ 0.024 & \textbf{Small Compact Spherical Seeds} \\
C4 & 521 & 3.83\% & 173,557 $\pm$ 23,289 & 593.25 $\pm$ 53.21 & 1.585 $\pm$ 0.119 & 0.793 $\pm$ 0.031 & \textbf{Giant Macroscopic Seeds} \\
C5 & 3,042 & 22.35\% & 44,602 $\pm$ 5,387 & 299.11 $\pm$ 19.40 & 1.576 $\pm$ 0.088 & 0.796 $\pm$ 0.022 & \textbf{Medium Compact Oval Seeds} \\
C6 & 1,770 & 13.00\% & 53,695 $\pm$ 7,229 & 373.64 $\pm$ 28.91 & 2.039 $\pm$ 0.122 & 0.699 $\pm$ 0.021 & \textbf{Long Cylindrical Kidney Seeds} \\
\bottomrule
\end{tabular*}
\caption{Empirical morphological profile summary across the seven discovered dry bean clusters (Config 2: K-Means on 4D PCA, Reported as exact sample $\mu \pm \sigma$).}
\label{tab:cluster_profiles}
\end{table*}

\subsection{Morphological Cluster Profiles and Archetype Characterization}
Table~\ref{tab:cluster_profiles} summarizes the physical characteristics of the seven discovered clusters from Config~2 with exact empirical sample standard deviations. Because true botanical labels were unobserved, we interpret the clusters as data-driven morphological archetypes:
\begin{itemize}
  \item \textbf{C0 (519, 3.81\%): Medium-Large Elongated.} Mean area $63{,}737 \pm 14{,}259$ with high aspect ratio ($1.779 \pm 0.168$).
  \item \textbf{C1 (3,262, 23.97\%): Small Compact Elliptical.} Smallest average area ($31{,}927 \pm 4{,}667$) and high compactness ($0.823 \pm 0.022$).
  \item \textbf{C2 (2,476, 18.19\%): Large Broad Oval.} Substantial seed area ($74{,}341 \pm 9{,}213$) with moderate elongation ($1.651 \pm 0.135$).
  \item \textbf{C3 (2,021, 14.85\%): Small Compact Spherical.} Lowest aspect ratio ($1.235 \pm 0.064$) and highest compactness ($0.900 \pm 0.024$).
  \item \textbf{C4 (521, 3.83\%): Giant Macroscopic.} Distinctive giant seed profile ($173{,}557 \pm 23{,}289$), more than double the area of all other clusters.
  \item \textbf{C5 (3,042, 22.35\%): Medium Compact Oval.} Intermediate size ($44{,}602 \pm 5{,}387$) and standard elongation ($1.576 \pm 0.088$).
  \item \textbf{C6 (1,770, 13.00\%): Long Cylindrical Kidney.} Highest aspect ratio ($2.039 \pm 0.122$) and lowest compactness ($0.699 \pm 0.021$).
\end{itemize}

\subsection{Submission Cluster Assignment File Verification}
Final cluster assignments from Config~2 were saved as \texttt{Dalwai\_DryBeanClusters.csv} (and \texttt{LastName\_DryBeanClusters.csv}). The file contains exactly 13,612 lines (header \texttt{sample\_id,Class\_id} plus 13,611 rows), strictly preserves 1-based indexing (\texttt{sample\_id} 1..13611), and assigns all instances to one of seven clusters ($\text{Class\_id} \in \{0, \dots, 6\}$) with zero missing entries.

\section{Discussion and Methodological Synthesis}
The dual investigations demonstrate core machine learning principles:
\begin{enumerate}
  \item \textbf{Data Leakage Prevention:} Strictly fitting vectorizers inside cross-validation training partitions prevents optimistic bias. In unsupervised tasks, isolating target labels ensures true blind discovery.
  \item \textbf{Calibrated Probability Estimation:} Accuracy alone fails under asymmetric operational costs. Platt probability calibration enables formal risk-constrained threshold tuning ($\tau=0.70$), halving false positives at minor recall penalty.
  \item \textbf{Spectral Dimensionality Reduction:} Standardizing features and applying PCA reduces dimensions by 75\% while retaining 95.02\% variance, yielding superior cluster stability ($\text{mean ARI}=0.9978$).
  \item \textbf{Internal Metric Geometry:} Metric values across different dimensional spaces must be interpreted relative to subspace geometry; validating 4D clusters in unreduced 16D space verifies physical cluster cohesion.
\end{enumerate}

\section{Conclusion}
This study presented an empirical evaluation of supervised spam detection and unsupervised dry bean clustering. For Project~1, Calibrated Linear SVM ($C=2.0$) achieved balanced performance (Spam F1: 0.9401, Accuracy: 0.9740), with probability calibration enabling operational cost control. For Project~2, K-Means on 4D PCA features achieved optimal cluster separation (Silhouette: 0.3277, Calinski--Harabasz: 9,134.22) and high initialization stability (ARI: 0.9978), identifying seven morphological seed archetypes. Both pipelines are fully reproducible with fixed random seeds.

\bibliographystyle{IEEEtran}
\begin{thebibliography}{1}
\bibitem{koklu2020multiclass}
M.~Koklu and I.~A. Ozkan, ``Multiclass classification of dry beans using computer vision and machine learning techniques,'' \emph{Computers and Electronics in Agriculture}, vol. 174, p. 105507, 2020.
\bibitem{platt1999probabilistic}
J.~Platt, ``Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,'' \emph{Advances in Large Margin Classifiers}, vol.~10, no.~3, pp. 61--74, 1999.
\bibitem{rousseeuw1987silhouettes}
P.~J. Rousseeuw, ``Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,'' \emph{Journal of Computational and Applied Mathematics}, vol.~20, pp. 53--65, 1987.
\bibitem{calinski1974dendrite}
T.~Cali{\'n}ski and J.~Harabasz, ``A dendrite method for cluster analysis,'' \emph{Communications in Statistics-theory and Methods}, vol.~3, no.~1, pp. 1--27, 1974.
\end{thebibliography}

\end{document}
"""
    with open(os.path.join(out_dir, "report.tex"), "w") as f:
        f.write(tex)
    return compile_variant(out_dir)

# ==============================================================================
# VARIANT 3: ICLR COMPUTATIONAL (INDIGO & CYAN, SANS HEADINGS, MODERN)
# ==============================================================================
def write_variant3():
    out_dir = os.path.join(BASE_DIR, "docs/latex_reports/variant3_iclr")
    os.makedirs(out_dir, exist_ok=True)
    
    tex = r"""\documentclass[9.5pt,twocolumn,letterpaper]{article}

\usepackage[
  top=0.65in,
  bottom=0.65in,
  left=0.65in,
  right=0.65in,
  columnsep=0.22in
]{geometry}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}

\usepackage{amsmath,mathtools,bm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{siunitx}
\sisetup{detect-all, tight-spacing=true}

\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[font=small,labelfont=bf,skip=3pt]{caption}
\usepackage{float}

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}
\setlength{\floatsep}{4pt plus 1pt minus 1pt}
\setlength{\textfloatsep}{5pt plus 1pt minus 1pt}
\setlength{\intextsep}{4pt plus 1pt minus 1pt}

\usepackage{xcolor}
\definecolor{indigoblue}{RGB}{30,27,75}
\definecolor{cyanaccent}{RGB}{14,116,144}
\definecolor{darkslate}{RGB}{51,65,85}
\usepackage[
  colorlinks=true,
  linkcolor=indigoblue,
  citecolor=cyanaccent,
  urlcolor=indigoblue
]{hyperref}

\usepackage[compact]{titlesec}
\titleformat{\section}{\large\sffamily\bfseries\color{indigoblue}}{\thesection.}{0.4em}{}
\titleformat{\subsection}{\normalsize\sffamily\bfseries\color{cyanaccent}}{\thesubsection.}{0.35em}{}
\titleformat{\subsubsection}{\small\sffamily\bfseries\color{darkslate}}{\thesubsubsection.}{0.3em}{}
\titlespacing{\section}{0pt}{4.5pt plus 1pt minus 1pt}{2pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsection}{0pt}{3.5pt plus 1pt minus 1pt}{1.5pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsubsection}{0pt}{2pt plus 0.5pt minus 0.5pt}{1pt}

\usepackage{enumitem}
\setlist{nosep, leftmargin=1.0em, labelsep=0.3em}

\graphicspath{{figures/}{../../../outputs/figures/}{outputs/figures/}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bZ}{\mathbf{Z}}
\newcommand{\bSigma}{\mathbf{\Sigma}}
\newcommand{\bLambda}{\mathbf{\Lambda}}
\newcommand{\bV}{\mathbf{V}}
\newcommand{\indic}{\mathbb{I}}

\title{\vspace{-0.3in}\textbf{\Large Empirical Evaluation of Supervised Spam Filtering and Unsupervised Morphological Clustering}\vspace{-0.08in}}
\author{\textbf{Adam Dalwai} \\ Department of Computer Science \\ \texttt{adamdalwai1@gmail.com}}
\date{\vspace{-0.15in}\small November 2026\vspace{-0.1in}}

\begin{document}

\maketitle

\begin{abstract}
\noindent We present an empirical study covering supervised text classification and unsupervised geometric clustering. In Project~1, we develop an asymmetric cost-sensitive spam filter over a combined corpus of 4,193 deduplicated messages. Using stratified 5-fold cross-validation with fold-isolated TF-IDF extraction, we evaluate nine candidate architectures. Calibrated Linear Support Vector Machines ($C=2.0$) achieve top balanced performance: Accuracy $0.9740 \pm 0.0053$, Macro-F1 $0.9617 \pm 0.0079$, and Spam F1 $0.9401 \pm 0.0123$. While uncalibrated SGD and LinearSVC attain lower Ham False Positive Rates at default thresholds (0.0091 vs 0.0185), paired statistical tests show no significant F1 difference ($p=0.579$), and Platt probability calibration enables post-hoc operational threshold tuning. Under risk-constrained optimization ($\tau=0.70$), Ham FPR is reduced by 49.2\% to 0.0094 (31 false positives out of 3,290) while sustaining Spam F1 at 0.9133. Out-of-fold error analysis examines five false positives and five false negatives at the decision boundary, proposing NLP feature remedies. Predictions on 6,447 unlabeled test instances match clean training priors within 0.71\%. In Project~2, we perform unsupervised clustering on 13,611 biological specimens characterized by 16 morphological features with ground-truth class labels strictly isolated. Principal Component Analysis demonstrates that four dimensions capture 95.02\% of cumulative variance, compared with three dimensions under the Kaiser rule (89.90\%). Benchmarking four configurations ($k=7$) shows that K-Means on 4D PCA features yields a Silhouette score of 0.3277, Calinski--Harabasz index of 9,134.22, and Davies--Bouldin index of 1.0328, with pairwise Adjusted Rand Index across 10 random seeds averaging 0.9978. Morphological profiling characterizes seven distinct geometric archetypes.
\end{abstract}

\section{Introduction and Problem Formulations}
Supervised text categorization and unsupervised geometric clustering serve distinct objectives in applied machine learning. Supervised learning requires strict data hygiene, fold-level feature isolation to prevent leakage, and threshold calibration under asymmetric operational error costs. Unsupervised clustering requires multi-scale normalization, covariance decomposition, and internal metric validation independent of external reference labels.

We evaluate both tasks using canonical benchmarks: (1)~binary spam detection over combined SMS and desktop email corpora, and (2)~morphological cluster discovery across high-dimensional geometric measurements of dry bean seeds. All pipelines are implemented in Python (v3.13.2) utilizing \texttt{scikit-learn} (v1.9.0), \texttt{numpy} (v2.4.6), \texttt{pandas} (v3.0.3), and \texttt{scipy} (v1.17.1), with all stochastic operations pinned to $\text{seed}=42$ for deterministic reproducibility.

\section{Project 1: Spam Email Detection}

\subsection{Corpus Ingestion, Cleaning, and Deduplication}
The supervised learning dataset combines two heterogeneous sources:
\begin{itemize}
  \item \textbf{Training Dataset 1 (SMS):} 2,228 short mobile messages (1,927 Ham, 301 Spam; 13.51\% spam prevalence). Comma delimiter fragmentation across columns \texttt{v2}, \texttt{Unnamed: 2}, \texttt{Unnamed: 3}, and \texttt{Unnamed: 4} was repaired by programmatic text reconstruction.
  \item \textbf{Training Dataset 2 (Email):} 2,068 desktop emails (1,440 Ham, 628 Spam; 30.37\% spam prevalence) with full message bodies and integer labels.
  \item \textbf{Hidden Test Dataset:} 6,447 unlabeled instances for final inference.
\end{itemize}

Direct union yields 4,296 raw instances. Exact string hashing identified and removed \textbf{103 duplicate records} (64 in Dataset~1, 39 in Dataset~2, 0 cross-corpus collisions; 0 conflicting labels). Purging duplicates established a clean training corpus of \textbf{4,193 unique instances} (3,290 Ham [78.46\%], 903 Spam [21.54\%], imbalance ratio 3.64:1). Deduplication prior to partitioning ensures identical texts cannot co-occur across training and validation splits.

\subsection{Experimental Design, Leakage Prevention, and Models}
Text was tokenized using sublinear TF-IDF scaling:
\begin{equation}
\text{TF-IDF}(t, d) = \left(1 + \log(\text{tf}(t, d))\right) \cdot \log\left(\frac{1 + N}{1 + \text{df}(t)}\right)
\end{equation}
with unigram and bigram extraction ($n \in \{1, 2\}$), document frequency threshold $\text{min\_df}=2$, sublinear term weighting, and accent stripping, producing $\approx 14,800$ sparse features.

\textbf{Fold Isolation Protocol:} In each fold of Stratified 5-Fold Cross-Validation, the TF-IDF vectorizer was fit strictly on the training partition ($X_{\text{train}}$) and applied to the validation partition ($X_{\text{val}}$) via \texttt{transform}. Test instances were never accessed during vocabulary construction. Table~\ref{tab:model_specs} details the hyperparameter configurations across all nine evaluated architectures.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l X}
\toprule
\textbf{Model Architecture} & \textbf{Hyperparameter Specification \& Tuning Space} \\
\midrule
MultinomialNB (Baseline) & $\alpha=1.0$, uniform class prior (uncalibrated) \\
ComplementNB & $\alpha=1.0$, norm=False (designed for imbalanced text) \\
Logistic Regression ($C=1.0$) & $C=1.0$, L2 penalty, lbfgs solver, max\_iter=1000 \\
Logistic Regression ($C=10.0$) & $C=10.0$, low regularization, lbfgs solver \\
Logistic Regression (Balanced) & $C=5.0$, \texttt{class\_weight='balanced'}, lbfgs solver \\
Linear SVM (LinearSVC) & $C=1.0$, squared hinge loss, L2 penalty \\
\textbf{Calibrated Linear SVM} & LinearSVC ($C=2.0$) with Platt sigmoid calibration (5-fold internal CV) \\
Random Forest ($n=100$) & 100 estimators, Gini criterion, max\_features='sqrt' \\
SGD Classifier (Mod. Huber) & Modified Huber loss, $\alpha=10^{-4}$, elasticnet ($\rho=0.15$) \\
\bottomrule
\end{tabularx}
\caption{Hyperparameter configurations across nine evaluated spam classifiers.}
\label{tab:model_specs}
\end{table}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.06}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c@{}}
\toprule
\textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Spam Precision} & \textbf{Spam Recall} & \textbf{Spam F1} & \textbf{Ham FPR} \\
\midrule
\textbf{Calibrated Linear SVM ($C=2.0$)} & \textbf{0.9740$\pm$0.0053} & \textbf{0.9617$\pm$0.0079} & 0.9336$\pm$0.0155 & \textbf{0.9469$\pm$0.0186} & \textbf{0.9401$\pm$0.0123} & 0.0185$\pm$0.0046 \\
SGD Classifier (Modified Huber) & 0.9738$\pm$0.0066 & 0.9603$\pm$0.0103 & 0.9650$\pm$0.0115 & 0.9115$\pm$0.0303 & 0.9372$\pm$0.0164 & 0.0091$\pm$0.0032 \\
Logistic Regression (Balanced, $C=5.0$) & 0.9714$\pm$0.0064 & 0.9575$\pm$0.0096 & 0.9375$\pm$0.0132 & 0.9292$\pm$0.0235 & 0.9332$\pm$0.0152 & 0.0170$\pm$0.0038 \\
Linear SVM (LinearSVC, $C=1.0$) & 0.9714$\pm$0.0066 & 0.9565$\pm$0.0104 & 0.9644$\pm$0.0088 & 0.9004$\pm$0.0293 & 0.9311$\pm$0.0167 & 0.0091$\pm$0.0024 \\
Logistic Regression ($C=10.0$) & 0.9638$\pm$0.0065 & 0.9440$\pm$0.0106 & 0.9664$\pm$0.0030 & 0.8616$\pm$0.0301 & 0.9108$\pm$0.0172 & 0.0082$\pm$0.0007 \\
Random Forest ($n=100$, Gini) & 0.9344$\pm$0.0093 & 0.8919$\pm$0.0170 & 0.9714$\pm$0.0090 & 0.7165$\pm$0.0421 & 0.8241$\pm$0.0286 & 0.0058$\pm$0.0018 \\
Complement Naive Bayes & 0.9318$\pm$0.0096 & 0.8885$\pm$0.0180 & 0.9506$\pm$0.0121 & 0.7209$\pm$0.0468 & 0.8191$\pm$0.0303 & 0.0103$\pm$0.0026 \\
Logistic Regression ($C=1.0$, L2) & 0.9222$\pm$0.0087 & 0.8672$\pm$0.0175 & 0.9836$\pm$0.0111 & 0.6500$\pm$0.0435 & 0.7818$\pm$0.0300 & 0.0030$\pm$0.0021 \\
Multinomial Naive Bayes (Baseline) & 0.8781$\pm$0.0113 & 0.7657$\pm$0.0288 & \textbf{1.0000$\pm$0.0000} & 0.4341$\pm$0.0527 & 0.6035$\pm$0.0515 & \textbf{0.0000$\pm$0.0000} \\
\bottomrule
\end{tabular*}
\caption{Stratified 5-Fold Cross-Validation Performance Comparison across Nine Model Architectures (Reported as $\mu \pm \sigma$).}
\label{tab:spam_benchmark}
\end{table*}

\subsection{Cross-Validation Results and Statistical Comparison}
Table~\ref{tab:spam_benchmark} reports cross-validated performance. MultinomialNB achieved 100\% precision but suffered severe under-detection (Spam Recall $0.4341 \pm 0.0527$), missing 56.6\% of spam messages. This occurs because the 3.64:1 class prior heavily suppresses uncalibrated posterior probabilities.

Calibrated Linear SVM ($C=2.0$) attained the highest mean Spam F1 ($0.9401 \pm 0.0123$) and Spam Recall ($0.9469 \pm 0.0186$). However, SGD Classifier with Modified Huber loss achieved a comparable Spam F1 ($0.9372 \pm 0.0164$) with a lower Ham FPR ($0.0091 \pm 0.0032$). A paired two-tailed $t$-test across the five cross-validation folds reveals that the F1 difference between Calibrated SVM and SGD is not statistically significant ($t = 0.603, p = 0.579$). Similarly, LinearSVC ($C=1.0$) shows no statistically significant F1 shortfall ($t = 1.513, p = 0.205$).

\textbf{Rationale for Model Selection:} Calibrated Linear SVM is selected not due to an unassailable F1 margin, but because Platt sigmoid scaling via \texttt{CalibratedClassifierCV} fits parameters $A, B \in \R$ on internal 5-fold cross-validation margins:
\begin{equation}
P(y=1|\bx) = \frac{1}{1 + \exp\left(A \cdot f(\bx) + B\right)}
\end{equation}
producing true posterior probabilities. Uncalibrated LinearSVC yields arbitrary margins $f(\bx) \in (-\infty, \infty)$ that cannot be thresholded on probability scale, while SGD's surrogate loss produces poorer log-loss calibration. Continuous probability outputs are indispensable for operational threshold adjustments under asymmetric costs.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\columnwidth]{figures/spam_confusion_matrix.png}
\caption{Out-of-fold Confusion Matrix for the selected Calibrated Linear SVM ($C=2.0$) at operating threshold $\tau^*=0.50$ (Left: Counts; Right: Normalized Rates).}
\label{fig:confusion_matrix}
\end{figure}

\subsection{Operational Decision Thresholds and Cost Analysis}
Let $c_{FP}$ and $c_{FN}$ denote operational penalties for False Positives (blocking legitimate communications) and False Negatives (spam leakage). Total cost is modeled as $\text{Cost}(\tau) = c_{FP} \cdot FP(\tau) + c_{FN} \cdot FN(\tau)$. With $c_{FP}/c_{FN} \approx 20$, misclassifying ham is 20 times more severe. Table~\ref{tab:threshold_tradeoffs} evaluates out-of-fold metrics across $\tau \in [0.10, 0.90]$.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{c c c c c c r r r}
\toprule
\textbf{$\tau$} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & \textbf{Spam F1} & \textbf{Ham FPR} & \textbf{FP} & \textbf{FN} & \textbf{Cost ($c_{FP}=20$)} \\
\midrule
0.10 & 0.8595 & 0.6059 & \textbf{0.9945} & 0.7530 & 0.1775 & 584 & \textbf{5} & 11,685 \\
0.20 & 0.9294 & 0.7579 & 0.9878 & 0.8577 & 0.0866 & 285 & 11 & 5,711 \\
0.30 & 0.9556 & 0.8431 & 0.9756 & 0.9045 & 0.0498 & 164 & 22 & 3,302 \\
0.40 & 0.9680 & 0.8984 & 0.9601 & 0.9283 & 0.0298 & 98 & 36 & 1,996 \\
\textbf{0.50} & \textbf{0.9740} & 0.9334 & 0.9468 & \textbf{0.9401} & 0.0185 & 61 & 48 & 1,268 \\
0.60 & 0.9721 & 0.9517 & 0.9169 & 0.9340 & 0.0128 & 42 & 75 & 915 \\
\textbf{0.70} & 0.9645 & \textbf{0.9620} & 0.8693 & 0.9133 & \textbf{0.0094} & \textbf{31} & 118 & 738 \\
0.80 & 0.9392 & 0.9751 & 0.7364 & 0.8391 & 0.0052 & 17 & 238 & \textbf{578} \\
0.90 & 0.9046 & 0.9809 & 0.5681 & 0.7195 & 0.0030 & 10 & 390 & 590 \\
\bottomrule
\end{tabularx}
\caption{Calibrated Linear SVM performance and asymmetric cost across decision thresholds $\tau$ on out-of-fold validation data.}
\label{tab:threshold_tradeoffs}
\end{table}

\textbf{Cost Tradeoff Discussion:} Pure linear cost minimization under $c_{FP}/c_{FN}=20$ reaches its global minimum at $\tau=0.80$ ($\text{Cost}=578$). However, operating at $\tau=0.80$ causes Spam Recall to collapse from 94.68\% to 73.64\% (238 spam leaks). In production, engineering deployments adopt risk-constrained optimization:
\begin{equation}
\min_{\tau} \text{Ham FPR}(\tau) \quad \text{subject to} \quad \text{Spam Recall}(\tau) \ge 0.85
\end{equation}
Under this criterion, $\tau=0.70$ is the optimal operating point, reducing Ham FPR by 49.2\% to \textbf{0.0094} (only 31 false positives out of 3,290 legitimate emails) while maintaining a high Spam F1 of 0.9133. For standard balanced utility, $\tau^*=0.50$ provides maximum F1 (0.9401) and Accuracy (0.9740). Figure~\ref{fig:confusion_matrix} plots the confusion matrix at $\tau^*=0.50$ (TN: 3,229, FP: 61, FN: 48, TP: 855).

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_roc_pr_curve.png}
  \caption{ROC \& PR Curves}
  \label{subfig:roc_pr}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_threshold_tradeoff.png}
  \caption{Tradeoffs vs. $\tau$}
  \label{subfig:threshold}
\end{subfigure}
\caption{Discrimination diagnostics for Calibrated Linear SVM: (a) ROC-AUC 0.9904, PR-AUC 0.9772; (b) Precision, Recall, F1, and Ham FPR response across $\tau$.}
\label{fig:spam_diagnostics}
\end{figure}

\subsection{Qualitative Error Analysis and Actionable Remedies}
Table~\ref{tab:error_forensics} presents five False Positives and five False Negatives from out-of-fold validation predictions evaluated at operating threshold $\tau^*=0.50$. All five FPs satisfy $P(\text{Spam}) \ge 0.50$ and all five FNs satisfy $P(\text{Spam}) < 0.50$, ensuring complete consistency with the reported confusion matrix.

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{l c c X X}
\toprule
\textbf{Error Case \& ID} & \textbf{Truth} & $\bm{P(\text{Spam})}$ & \textbf{Failure Mechanism \& Key Text Elements} & \textbf{Actionable Engineering Remedy (Classical NLP)} \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Positives (Ham Misclassified as Spam at $\tau^*=0.50$) --- Total Out-of-Fold Count: 61}}} \\
FP 1 (\#179) & Ham $\to$ Spam & 0.7771 & Opt-in referral SMS: \texttt{"free sms", "contact list", "fullonsms.com"}. High positive weights on promotional tokens trigger false alarm. & Domain token downweighting; distinguish domain hostnames from promotional URL paths via regex feature extraction. \\
FP 2 (\#373) & Ham $\to$ Spam & 0.7613 & Intimate personal SMS: \texttt{"looking forward to all the sex cuddling.. Only two more sleeps"}. Adult unigram co-occurs with commercial adult spam. & Windowed co-occurrence features linking explicit unigrams with first-person personal pronouns (\texttt{"I"}, \texttt{"we"}, \texttt{"sleeps"}). \\
FP 3 (\#4106) & Ham $\to$ Spam & 0.8388 & Corporate memo on FDA drug recall: \texttt{"danger of phenylpropanolamine", "alka-seltzer"}. Body terms match online pharmacy spam. & Forwarded message zoning; identify and downweight quoted historical forward headers (\texttt{"- - original message - -"}) relative to new text. \\
FP 4 (\#25) & Ham $\to$ Spam & 0.7691 & Sparse ultra-short SMS: \texttt{"Just normal only here :)"}. Rare token combinations yield high variance in sparse feature space. & Length-calibrated shrinkage prior for short messages ($<5$ words), requiring multiple corroborating spam tokens before overriding ham prior. \\
FP 5 (\#534) & Ham $\to$ Spam & 0.6543 & Conversational query with non-standard syntax: \texttt{"No message..no responce..what happend?"}. Ellipses and typo penalized by bias term. & Punctuation sequence normalization (\texttt{".."} $\to$ \texttt{"."}) and character $n$-gram hashing ($n \in [3, 5]$) for colloquial orthography. \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Negatives (Spam Escaping Filter at $\tau^*=0.50$) --- Total Out-of-Fold Count: 48}}} \\
FN 1 (\#193) & Spam $\to$ Ham & 0.0451 & Adversarial customer complaint mimicry: \texttt{"ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS... L8ER GOT MEGA BILL"}. & Regex feature detectors for UK premium-rate dial codes (\texttt{"0870"}, \texttt{"0871"}) and non-standard currency symbols (\texttt{"£"}, \texttt{"\$\$"}). \\
FN 2 (\#85) & Spam $\to$ Ham & 0.0755 & Phonetic adult chatline evasion: \texttt{"Im nasty and have a thing 4 filthyguys... Txt XXX SLO"}. Creative spelling evades word vocabulary. & Subword character $n$-grams ($n \in [3, 5]$) to capture phonetic slang patterns despite non-standard spacing and abbreviations. \\
FN 3 (\#404) & Spam $\to$ Ham & 0.4229 & Financial pump-and-dump: \texttt{"Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK... Nasdaq Symbol CDGT"}. & Financial pattern regex: uppercase stock ticker detector (\texttt{"Symbol [A-Z]\{3,5\}"}) combined with promotional yield claims (\texttt{"UP OVER [0-9]+\%"}). \\
FN 4 (\#468) & Spam $\to$ Ham & 0.1855 & Conversational lure: \texttt{"Hi this is Amy, we will be sending you a free phone number in a couple of days..."}. Casual opening masks lure. & Positional transition bigrams capturing abrupt shifts from informal greeting openings to urgent commercial calls-to-action. \\
FN 5 (\#332) & Spam $\to$ Ham & 0.2154 & Legal disclaimer mimicry: \texttt{"0A\$NETWORKS allow companies to bill for SMS, so they are responsible for their suppliers..."}. & Disclaimer token density ratio: measure proportion of regulatory/boilerplate unigrams relative to document length and unique vocabulary. \\
\bottomrule
\end{tabularx}
\caption{Qualitative Analysis of Representative False Positives and False Negatives at Operating Threshold $\tau^*=0.50$.}
\label{tab:error_forensics}
\end{table*}

\subsection{Domain Shift Analysis and Hidden Test Set Predictions}
The training corpus exhibits substantial domain shift: SMS texts (train1, 2,164 clean rows) average 15.2 words with 13.22\% spam prevalence, whereas desktop emails (train2, 2,029 clean rows) average 184.6 words with 30.41\% spam prevalence. TF-IDF sublinear scaling and bigram extraction normalize length disparities.

The Calibrated Linear SVM ($C=2.0$) model was refit on all 4,193 clean training instances. Inference on the 6,447 unlabeled test instances at $\tau^*=0.50$ produced \textbf{5,104 Ham (79.17\%)} and \textbf{1,343 Spam (20.83\%)}. This matches the clean training prior (78.46\% Ham / 21.54\% Spam) within \textbf{0.71\%}, confirming balanced generalization without distributional collapse. Output files were generated and verified: \texttt{Dalwai\_Spam.txt} (exactly 6,447 lines, no header, labels in $\{0, 1\}$).

\section{Project 2: Dry Bean Morphological Clustering}

\begin{figure*}[t]
\centering
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_pca_variance.png}
  \caption{PCA Scree \& Variance}
  \label{subfig:pca_scree}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_kmeans_clusters.png}
  \caption{K-Means ($k=7$) 2D Projection}
  \label{subfig:kmeans_2d}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_method2_clusters.png}
  \caption{GMM ($k=7$) 2D Projection}
  \label{subfig:gmm_2d}
\end{subfigure}
\caption{Dry bean unsupervised clustering diagnostics: (a) Scree plot and cumulative variance; (b) K-Means ($k=7$) clusters projected on PC1--PC2; (c) GMM ($k=7$) clusters projected on PC1--PC2.}
\label{fig:drybean_banner}
\end{figure*}

\subsection{Data Hygiene, Skewness, and Feature Scaling}
The Dry Bean dataset contains 13,611 biological specimens with 16 continuous morphological measurements (12 geometric dimensions and 4 shape factors).

\textbf{Hygiene Verification:} Numerical features exhibit zero missing values. An audit identified 68 duplicate feature vectors; all 13,611 rows were preserved with monotonic 1-based indexing (\texttt{sample\_id} 1 to 13,611) per submission requirements. The \texttt{Class} column was unobserved and strictly excluded from all preprocessing, scaling, PCA, and clustering.

\textbf{Skewness and Outlier Analysis:} Feature distributions exhibit severe right-skewness: \texttt{Area} ($+2.953$), \texttt{ConvexArea} ($+2.941$), \texttt{MinorAxisLength} ($+2.238$), \texttt{EquivDiameter} ($+1.949$), and \texttt{Perimeter} ($+1.626$). Conversely, \texttt{ShapeFactor4} ($-2.759$) and \texttt{Solidity} ($-2.550$) display negative skewness. High area values ($>170{,}000\,\text{px}^2$) represent macroscopic seed varieties (Cluster 4) rather than measurement errors. We apply \texttt{StandardScaler} ($z = \frac{x - \mu}{\sigma}$) rather than nonlinear log-transforms to preserve the physical geometric aspect ratios across coordinate axes.

\subsection{PCA Dimensionality Selection: Kaiser Rule vs. 95\% Variance}
Let $\bZ \in \R^{13611 \times 16}$ denote the standardized feature matrix. We perform eigen-decomposition on the sample covariance matrix $\bSigma = \frac{1}{N-1}\bZ^T\bZ = \bV\bLambda\bV^T$ ($N=13,611$). Table~\ref{tab:pca_variance} reports the eigenvalue spectrum.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l r r r X}
\toprule
\textbf{PC} & \textbf{Eigenval.} & \textbf{Var (\%)} & \textbf{Cum (\%)} & \textbf{Morphological Interpretation} \\
\midrule
PC1 & 8.8753 & 55.47 & 55.47 & Scale \& Volume (\texttt{Area}: +0.33) \\
PC2 & 4.2293 & 26.43 & 81.90 & Aspect Ratio vs. Roundness (\texttt{AspectR}: +0.40) \\
PC3 & 1.2811 & 8.01 & 89.90 & Boundary Smoothness (\texttt{Solidity}: +0.46) \\
PC4 & 0.8183 & 5.11 & \textbf{95.02} & Bounding Occupancy (\texttt{Extent}: +0.72) \\
\midrule
PC5 & 0.4383 & 2.74 & 97.76 & Secondary boundary harmonics \\
PC6--16 & $<0.19$ & $<1.15$ & 100.00 & Residual collinear measurement noise \\
\bottomrule
\end{tabularx}
\caption{Eigenvalue spectrum and variance decomposition across 16 morphological features.}
\label{tab:pca_variance}
\end{table}

\textbf{Dimensionality Selection Criteria:}
\begin{itemize}
  \item \textbf{Kaiser-Guttman Rule ($\lambda_i > 1.0$):} Retains exactly 3 components ($\lambda_1=8.88, \lambda_2=4.23, \lambda_3=1.28$), explaining 89.90\% of variance.
  \item \textbf{95\% Variance Threshold:} Retains 4 components, reaching \textbf{95.02\%} cumulative variance. Although PC4's eigenvalue ($\lambda_4=0.8183$) is below 1.0, it captures bounding box occupancy (\texttt{Extent} loading $+0.72$), which separates compact seeds from boundary irregularities. Retaining 4 PCs reduces dimensions by 75\% (16D $\to$ 4D) while capturing 95\% of total information.
\end{itemize}

\subsection{Four-Configuration Clustering Benchmark ($k=7$)}
Table~\ref{tab:clustering_benchmark} compares four configurations across three internal validation metrics.

\textbf{Metric Interpretation Across Dimensionalities:}
\begin{itemize}
  \item In algorithm space, \textbf{Config 2 (K-Means on 4D PCA)} attains the highest Silhouette score (0.3277), highest Calinski--Harabasz index (9,134.22), and lowest Davies--Bouldin index (1.0328).
  \item Comparing metrics across different dimensional spaces requires care: distances concentrate differently in 16D vs 4D. Projecting cluster assignments back into original 16D standardized space yields a Silhouette score of \textbf{0.3076} (virtually identical to 16D K-Means at 0.3094) and Calinski--Harabasz of \textbf{7,817.26}. Thus, 4D PCA achieves comparable cluster structure in unreduced space while providing 75\% dimensionality reduction and faster convergence.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l l l c c c c@{}}
\toprule
\textbf{Configuration} & \textbf{Algorithm} & \textbf{Feature Space} & \makecell[c]{\textbf{Silhouette}\\\textbf{($\uparrow$)}} & \makecell[c]{\textbf{Calinski--}\\\textbf{Harabasz ($\uparrow$)}} & \makecell[c]{\textbf{Davies--}\\\textbf{Bouldin ($\downarrow$)}} & \makecell[c]{\textbf{Cluster Size Distribution}\\\textbf{($k=7$)}} \\
\midrule
Config 1 & K-Means & Standardized (16D) & 0.3094 & 7,848.44 & 1.0993 & 3111, 541, 3157, 520, 2480, 2033, 1769 \\
\textbf{Config 2 (Winner)} & \textbf{K-Means} & \textbf{PCA (4D, 95.0\% Var)} & \textbf{0.3277} & \textbf{9,134.22} & \textbf{1.0328} & 519, 3262, 2476, 2021, 521, 3042, 1770 \\
Config 3 & GMM & Standardized (16D) & 0.2898 & 6,426.16 & 1.6592 & 3191, 693, 2836, 520, 2514, 1984, 1873 \\
Config 4 & GMM & PCA (4D, 95.0\% Var) & 0.3020 & 7,618.59 & 1.2245 & 1875, 2699, 522, 1591, 2547, 3546, 831 \\
\bottomrule
\end{tabular*}
\caption{Internal Cluster Validation Comparison across Four Experimental Configurations ($k=7$) on 13,611 Dry Bean Specimens.}
\label{tab:clustering_benchmark}
\end{table*}

\subsection{Cluster Stability across Multi-Seed Initializations}
To evaluate whether K-Means partitions are sensitive to initialization, we executed Config~2 across 10 random seeds ($s \in \{0, 1, 7, 13, 21, 42, 99, 123, 777, 2024\}$) with $k\text{-means++}$ seeding ($n_{\text{init}}=10$). The pairwise Adjusted Rand Index (ARI) across all 45 seed combinations yielded:
\begin{equation}
\text{Mean ARI} = \mathbf{0.9978} \quad (\text{range: } [0.9939, 1.0000])
\end{equation}
An average ARI of 0.9978 confirms that the four-dimensional PCA representation provides an exceptionally convex, stable objective landscape, with K-Means converging to the identical partition structure regardless of random seed.

\subsection{Dimensionality Reduction vs. 2D Visual Display}
We emphasize the distinction between PCA for \textbf{Dimensionality Reduction Prior to Clustering} vs. \textbf{2D Visualization}:
\begin{itemize}
  \item \textbf{4D Clustering Space:} K-Means operates in $\R^4$ capturing \textbf{95.02\%} variance, retaining boundary rugosity (PC3) and bounding box occupancy (PC4).
  \item \textbf{2D Visual Projection:} Figures~\ref{fig:drybean_banner}(b) and \ref{fig:drybean_banner}(c) plot clusters on PC1--PC2, capturing \textbf{81.90\%} variance. The remaining 13.12\% variance operates orthogonally; apparent overlaps in 2D are separated along PC3 and PC4.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}c r r c c c c l@{}}
\toprule
\textbf{Cluster} & \textbf{Count} & \textbf{Share} & \textbf{Area ($\text{px}^2$)} & \makecell[c]{\textbf{Major Axis}\\\textbf{Length ($\text{px}$)}} & \makecell[c]{\textbf{Aspect}\\\textbf{Ratio}} & \textbf{Compactness} & \textbf{Morphological Cluster Archetype} \\
\midrule
C0 & 519 & 3.81\% & 63,737 $\pm$ 14,259 & 380.65 $\pm$ 46.79 & 1.779 $\pm$ 0.168 & 0.745 $\pm$ 0.037 & \textbf{Medium-Large Elongated Seeds} \\
C1 & 3,262 & 23.97\% & 31,927 $\pm$ 4,667 & 244.47 $\pm$ 18.23 & 1.475 $\pm$ 0.079 & 0.823 $\pm$ 0.022 & \textbf{Small Compact Elliptical Seeds} \\
C2 & 2,476 & 18.19\% & 74,341 $\pm$ 9,213 & 395.73 $\pm$ 32.13 & 1.651 $\pm$ 0.135 & 0.778 $\pm$ 0.034 & \textbf{Large Broad Oval Seeds} \\
C3 & 2,021 & 14.85\% & 39,389 $\pm$ 5,270 & 248.55 $\pm$ 18.85 & 1.235 $\pm$ 0.064 & 0.900 $\pm$ 0.024 & \textbf{Small Compact Spherical Seeds} \\
C4 & 521 & 3.83\% & 173,557 $\pm$ 23,289 & 593.25 $\pm$ 53.21 & 1.585 $\pm$ 0.119 & 0.793 $\pm$ 0.031 & \textbf{Giant Macroscopic Seeds} \\
C5 & 3,042 & 22.35\% & 44,602 $\pm$ 5,387 & 299.11 $\pm$ 19.40 & 1.576 $\pm$ 0.088 & 0.796 $\pm$ 0.022 & \textbf{Medium Compact Oval Seeds} \\
C6 & 1,770 & 13.00\% & 53,695 $\pm$ 7,229 & 373.64 $\pm$ 28.91 & 2.039 $\pm$ 0.122 & 0.699 $\pm$ 0.021 & \textbf{Long Cylindrical Kidney Seeds} \\
\bottomrule
\end{tabular*}
\caption{Empirical morphological profile summary across the seven discovered dry bean clusters (Config 2: K-Means on 4D PCA, Reported as exact sample $\mu \pm \sigma$).}
\label{tab:cluster_profiles}
\end{table*}

\subsection{Morphological Cluster Profiles and Archetype Characterization}
Table~\ref{tab:cluster_profiles} summarizes the physical characteristics of the seven discovered clusters from Config~2 with exact empirical sample standard deviations. Because true botanical labels were unobserved, we interpret the clusters as data-driven morphological archetypes:
\begin{itemize}
  \item \textbf{C0 (519, 3.81\%): Medium-Large Elongated.} Mean area $63{,}737 \pm 14{,}259$ with high aspect ratio ($1.779 \pm 0.168$).
  \item \textbf{C1 (3,262, 23.97\%): Small Compact Elliptical.} Smallest average area ($31{,}927 \pm 4{,}667$) and high compactness ($0.823 \pm 0.022$).
  \item \textbf{C2 (2,476, 18.19\%): Large Broad Oval.} Substantial seed area ($74{,}341 \pm 9{,}213$) with moderate elongation ($1.651 \pm 0.135$).
  \item \textbf{C3 (2,021, 14.85\%): Small Compact Spherical.} Lowest aspect ratio ($1.235 \pm 0.064$) and highest compactness ($0.900 \pm 0.024$).
  \item \textbf{C4 (521, 3.83\%): Giant Macroscopic.} Distinctive giant seed profile ($173{,}557 \pm 23{,}289$), more than double the area of all other clusters.
  \item \textbf{C5 (3,042, 22.35\%): Medium Compact Oval.} Intermediate size ($44{,}602 \pm 5{,}387$) and standard elongation ($1.576 \pm 0.088$).
  \item \textbf{C6 (1,770, 13.00\%): Long Cylindrical Kidney.} Highest aspect ratio ($2.039 \pm 0.122$) and lowest compactness ($0.699 \pm 0.021$).
\end{itemize}

\subsection{Submission Cluster Assignment File Verification}
Final cluster assignments from Config~2 were saved as \texttt{Dalwai\_DryBeanClusters.csv} (and \texttt{LastName\_DryBeanClusters.csv}). The file contains exactly 13,612 lines (header \texttt{sample\_id,Class\_id} plus 13,611 rows), strictly preserves 1-based indexing (\texttt{sample\_id} 1..13611), and assigns all instances to one of seven clusters ($\text{Class\_id} \in \{0, \dots, 6\}$) with zero missing entries.

\section{Discussion and Methodological Synthesis}
The dual investigations demonstrate core machine learning principles:
\begin{enumerate}
  \item \textbf{Data Leakage Prevention:} Strictly fitting vectorizers inside cross-validation training partitions prevents optimistic bias. In unsupervised tasks, isolating target labels ensures true blind discovery.
  \item \textbf{Calibrated Probability Estimation:} Accuracy alone fails under asymmetric operational costs. Platt probability calibration enables formal risk-constrained threshold tuning ($\tau=0.70$), halving false positives at minor recall penalty.
  \item \textbf{Spectral Dimensionality Reduction:} Standardizing features and applying PCA reduces dimensions by 75\% while retaining 95.02\% variance, yielding superior cluster stability ($\text{mean ARI}=0.9978$).
  \item \textbf{Internal Metric Geometry:} Metric values across different dimensional spaces must be interpreted relative to subspace geometry; validating 4D clusters in unreduced 16D space verifies physical cluster cohesion.
\end{enumerate}

\section{Conclusion}
This study presented an empirical evaluation of supervised spam detection and unsupervised dry bean clustering. For Project~1, Calibrated Linear SVM ($C=2.0$) achieved balanced performance (Spam F1: 0.9401, Accuracy: 0.9740), with probability calibration enabling operational cost control. For Project~2, K-Means on 4D PCA features achieved optimal cluster separation (Silhouette: 0.3277, Calinski--Harabasz: 9,134.22) and high initialization stability (ARI: 0.9978), identifying seven morphological seed archetypes. Both pipelines are fully reproducible with fixed random seeds.

\bibliographystyle{IEEEtran}
\begin{thebibliography}{1}
\bibitem{koklu2020multiclass}
M.~Koklu and I.~A. Ozkan, ``Multiclass classification of dry beans using computer vision and machine learning techniques,'' \emph{Computers and Electronics in Agriculture}, vol. 174, p. 105507, 2020.
\bibitem{platt1999probabilistic}
J.~Platt, ``Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,'' \emph{Advances in Large Margin Classifiers}, vol.~10, no.~3, pp. 61--74, 1999.
\bibitem{rousseeuw1987silhouettes}
P.~J. Rousseeuw, ``Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,'' \emph{Journal of Computational and Applied Mathematics}, vol.~20, pp. 53--65, 1987.
\bibitem{calinski1974dendrite}
T.~Cali{\'n}ski and J.~Harabasz, ``A dendrite method for cluster analysis,'' \emph{Communications in Statistics-theory and Methods}, vol.~3, no.~1, pp. 1--27, 1974.
\end{thebibliography}

\end{document}
"""
    with open(os.path.join(out_dir, "report.tex"), "w") as f:
        f.write(tex)
    return compile_variant(out_dir)

# ==============================================================================
# VARIANT 4: OXFORD / SIAM MONOGRAPH (CHARTER / ONE-COLUMN / CRIMSON RED)
# ==============================================================================
def write_variant4():
    out_dir = os.path.join(BASE_DIR, "docs/latex_reports/variant4_monograph")
    os.makedirs(out_dir, exist_ok=True)
    
    tex = r"""\documentclass[10pt,onecolumn,letterpaper]{article}

\usepackage[
  top=0.75in,
  bottom=0.75in,
  left=0.85in,
  right=0.85in
]{geometry}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}

\usepackage{amsmath,mathtools,bm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{siunitx}
\sisetup{detect-all, tight-spacing=true}

\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\usepackage{float}

\usepackage{xcolor}
\definecolor{crimsonred}{RGB}{105,20,35}
\definecolor{darkcharcoal}{RGB}{40,40,40}
\definecolor{antiquegold}{RGB}{160,100,10}
\usepackage[
  colorlinks=true,
  linkcolor=crimsonred,
  citecolor=antiquegold,
  urlcolor=crimsonred
]{hyperref}

\usepackage[compact]{titlesec}
\titleformat{\section}{\Large\bfseries\color{crimsonred}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\bfseries\color{crimsonred}}{\thesubsection.}{0.4em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\itshape\color{darkcharcoal}}{\thesubsubsection.}{0.3em}{}
\titlespacing{\section}{0pt}{6pt plus 1pt minus 1pt}{3pt plus 1pt minus 1pt}
\titlespacing{\subsection}{0pt}{5pt plus 1pt minus 1pt}{2pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsubsection}{0pt}{3pt plus 0.5pt minus 0.5pt}{1pt}

\usepackage{enumitem}
\setlist{nosep, leftmargin=1.2em, labelsep=0.4em}

\graphicspath{{figures/}{../../../outputs/figures/}{outputs/figures/}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bZ}{\mathbf{Z}}
\newcommand{\bSigma}{\mathbf{\Sigma}}
\newcommand{\bLambda}{\mathbf{\Lambda}}
\newcommand{\bV}{\mathbf{V}}
\newcommand{\indic}{\mathbb{I}}

\title{\vspace{-0.3in}\textbf{\Large Empirical Evaluation of Supervised Spam Filtering and Morphological Clustering}\vspace{-0.08in}}
\author{\textbf{Adam Dalwai} \\ Department of Computer Science \\ \texttt{adamdalwai1@gmail.com}}
\date{\vspace{-0.15in}\small November 2026\vspace{-0.1in}}

\begin{document}

\maketitle

\begin{abstract}
\noindent We present an empirical study covering supervised text classification and unsupervised geometric clustering. In Project~1, we develop an asymmetric cost-sensitive spam filter over a combined corpus of 4,193 deduplicated messages. Using stratified 5-fold cross-validation with fold-isolated TF-IDF extraction, we evaluate nine candidate architectures. Calibrated Linear Support Vector Machines ($C=2.0$) achieve top balanced performance: Accuracy $0.9740 \pm 0.0053$, Macro-F1 $0.9617 \pm 0.0079$, and Spam F1 $0.9401 \pm 0.0123$. While uncalibrated SGD and LinearSVC attain lower Ham False Positive Rates at default thresholds (0.0091 vs 0.0185), paired statistical tests show no significant F1 difference ($p=0.579$), and Platt probability calibration enables post-hoc operational threshold tuning. Under risk-constrained optimization ($\tau=0.70$), Ham FPR is reduced by 49.2\% to 0.0094 (31 false positives out of 3,290) while sustaining Spam F1 at 0.9133. Out-of-fold error analysis examines five false positives and five false negatives at the decision boundary, proposing NLP feature remedies. Predictions on 6,447 unlabeled test instances match clean training priors within 0.71\%. In Project~2, we perform unsupervised clustering on 13,611 biological specimens characterized by 16 morphological features with ground-truth class labels strictly isolated. Principal Component Analysis demonstrates that four dimensions capture 95.02\% of cumulative variance, compared with three dimensions under the Kaiser rule (89.90\%). Benchmarking four configurations ($k=7$) shows that K-Means on 4D PCA features yields a Silhouette score of 0.3277, Calinski--Harabasz index of 9,134.22, and Davies--Bouldin index of 1.0328, with pairwise Adjusted Rand Index across 10 random seeds averaging 0.9978. Morphological profiling characterizes seven distinct geometric archetypes.
\end{abstract}

\section{Introduction and Problem Formulations}
Supervised text categorization and unsupervised geometric clustering serve distinct objectives in applied machine learning. Supervised learning requires strict data hygiene, fold-level feature isolation to prevent leakage, and threshold calibration under asymmetric operational error costs. Unsupervised clustering requires multi-scale normalization, covariance decomposition, and internal metric validation independent of external reference labels.

We evaluate both tasks using canonical benchmarks: (1)~binary spam detection over combined SMS and desktop email corpora, and (2)~morphological cluster discovery across high-dimensional geometric measurements of dry bean seeds. All pipelines are implemented in Python (v3.13.2) utilizing \texttt{scikit-learn} (v1.9.0), \texttt{numpy} (v2.4.6), \texttt{pandas} (v3.0.3), and \texttt{scipy} (v1.17.1), with all stochastic operations pinned to $\text{seed}=42$ for deterministic reproducibility.

\section{Project 1: Spam Email Detection}

\subsection{Corpus Ingestion, Cleaning, and Deduplication}
The supervised learning dataset combines two heterogeneous sources:
\begin{itemize}
  \item \textbf{Training Dataset 1 (SMS):} 2,228 short mobile messages (1,927 Ham, 301 Spam; 13.51\% spam prevalence). Comma delimiter fragmentation across columns \texttt{v2}, \texttt{Unnamed: 2}, \texttt{Unnamed: 3}, and \texttt{Unnamed: 4} was repaired by programmatic text reconstruction.
  \item \textbf{Training Dataset 2 (Email):} 2,068 desktop emails (1,440 Ham, 628 Spam; 30.37\% spam prevalence) with full message bodies and integer labels.
  \item \textbf{Hidden Test Dataset:} 6,447 unlabeled instances for final inference.
\end{itemize}

Direct union yields 4,296 raw instances. Exact string hashing identified and removed \textbf{103 duplicate records} (64 in Dataset~1, 39 in Dataset~2, 0 cross-corpus collisions; 0 conflicting labels). Purging duplicates established a clean training corpus of \textbf{4,193 unique instances} (3,290 Ham [78.46\%], 903 Spam [21.54\%], imbalance ratio 3.64:1). Deduplication prior to partitioning ensures identical texts cannot co-occur across training and validation splits.

\subsection{Experimental Design, Leakage Prevention, and Models}
Text was tokenized using sublinear TF-IDF scaling:
\begin{equation}
\text{TF-IDF}(t, d) = \left(1 + \log(\text{tf}(t, d))\right) \cdot \log\left(\frac{1 + N}{1 + \text{df}(t)}\right)
\end{equation}
with unigram and bigram extraction ($n \in \{1, 2\}$), document frequency threshold $\text{min\_df}=2$, sublinear term weighting, and accent stripping, producing $\approx 14,800$ sparse features.

\textbf{Fold Isolation Protocol:} In each fold of Stratified 5-Fold Cross-Validation, the TF-IDF vectorizer was fit strictly on the training partition ($X_{\text{train}}$) and applied to the validation partition ($X_{\text{val}}$) via \texttt{transform}. Test instances were never accessed during vocabulary construction. Table~\ref{tab:model_specs} details the hyperparameter configurations across all nine evaluated architectures.

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\textwidth}{l X}
\toprule
\textbf{Model Architecture} & \textbf{Hyperparameter Specification \& Tuning Space} \\
\midrule
MultinomialNB (Baseline) & $\alpha=1.0$, uniform class prior (uncalibrated) \\
ComplementNB & $\alpha=1.0$, norm=False (designed for imbalanced text) \\
Logistic Regression ($C=1.0$) & $C=1.0$, L2 penalty, lbfgs solver, max\_iter=1000 \\
Logistic Regression ($C=10.0$) & $C=10.0$, low regularization, lbfgs solver \\
Logistic Regression (Balanced) & $C=5.0$, \texttt{class\_weight='balanced'}, lbfgs solver \\
Linear SVM (LinearSVC) & $C=1.0$, squared hinge loss, L2 penalty \\
\textbf{Calibrated Linear SVM} & LinearSVC ($C=2.0$) with Platt sigmoid calibration (5-fold internal CV) \\
Random Forest ($n=100$) & 100 estimators, Gini criterion, max\_features='sqrt' \\
SGD Classifier (Mod. Huber) & Modified Huber loss, $\alpha=10^{-4}$, elasticnet ($\rho=0.15$) \\
\bottomrule
\end{tabularx}
\caption{Hyperparameter configurations across nine evaluated spam classifiers.}
\label{tab:model_specs}
\end{table}

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\textwidth}{@{}X c c c c c c@{}}
\toprule
\textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Spam Precision} & \textbf{Spam Recall} & \textbf{Spam F1} & \textbf{Ham FPR} \\
\midrule
\textbf{Calibrated Linear SVM ($C=2.0$)} & \textbf{0.9740$\pm$0.0053} & \textbf{0.9617$\pm$0.0079} & 0.9336$\pm$0.0155 & \textbf{0.9469$\pm$0.0186} & \textbf{0.9401$\pm$0.0123} & 0.0185$\pm$0.0046 \\
SGD Classifier (Modified Huber) & 0.9738$\pm$0.0066 & 0.9603$\pm$0.0103 & 0.9650$\pm$0.0115 & 0.9115$\pm$0.0303 & 0.9372$\pm$0.0164 & 0.0091$\pm$0.0032 \\
Logistic Regression (Balanced, $C=5.0$) & 0.9714$\pm$0.0064 & 0.9575$\pm$0.0096 & 0.9375$\pm$0.0132 & 0.9292$\pm$0.0235 & 0.9332$\pm$0.0152 & 0.0170$\pm$0.0038 \\
Linear SVM (LinearSVC, $C=1.0$) & 0.9714$\pm$0.0066 & 0.9565$\pm$0.0104 & 0.9644$\pm$0.0088 & 0.9004$\pm$0.0293 & 0.9311$\pm$0.0167 & 0.0091$\pm$0.0024 \\
Logistic Regression ($C=10.0$) & 0.9638$\pm$0.0065 & 0.9440$\pm$0.0106 & 0.9664$\pm$0.0030 & 0.8616$\pm$0.0301 & 0.9108$\pm$0.0172 & 0.0082$\pm$0.0007 \\
Random Forest ($n=100$, Gini) & 0.9344$\pm$0.0093 & 0.8919$\pm$0.0170 & 0.9714$\pm$0.0090 & 0.7165$\pm$0.0421 & 0.8241$\pm$0.0286 & 0.0058$\pm$0.0018 \\
Complement Naive Bayes & 0.9318$\pm$0.0096 & 0.8885$\pm$0.0180 & 0.9506$\pm$0.0121 & 0.7209$\pm$0.0468 & 0.8191$\pm$0.0303 & 0.0103$\pm$0.0026 \\
Logistic Regression ($C=1.0$, L2) & 0.9222$\pm$0.0087 & 0.8672$\pm$0.0175 & 0.9836$\pm$0.0111 & 0.6500$\pm$0.0435 & 0.7818$\pm$0.0300 & 0.0030$\pm$0.0021 \\
Multinomial Naive Bayes (Baseline) & 0.8781$\pm$0.0113 & 0.7657$\pm$0.0288 & \textbf{1.0000$\pm$0.0000} & 0.4341$\pm$0.0527 & 0.6035$\pm$0.0515 & \textbf{0.0000$\pm$0.0000} \\
\bottomrule
\end{tabularx}
\caption{Stratified 5-Fold Cross-Validation Performance Comparison across Nine Model Architectures (Reported as $\mu \pm \sigma$).}
\label{tab:spam_benchmark}
\end{table}

\subsection{Cross-Validation Results and Statistical Comparison}
Table~\ref{tab:spam_benchmark} reports cross-validated performance. MultinomialNB achieved 100\% precision but suffered severe under-detection (Spam Recall $0.4341 \pm 0.0527$), missing 56.6\% of spam messages. This occurs because the 3.64:1 class prior heavily suppresses uncalibrated posterior probabilities.

Calibrated Linear SVM ($C=2.0$) attained the highest mean Spam F1 ($0.9401 \pm 0.0123$) and Spam Recall ($0.9469 \pm 0.0186$). However, SGD Classifier with Modified Huber loss achieved a comparable Spam F1 ($0.9372 \pm 0.0164$) with a lower Ham FPR ($0.0091 \pm 0.0032$). A paired two-tailed $t$-test across the five cross-validation folds reveals that the F1 difference between Calibrated SVM and SGD is not statistically significant ($t = 0.603, p = 0.579$). Similarly, LinearSVC ($C=1.0$) shows no statistically significant F1 shortfall ($t = 1.513, p = 0.205$).

\textbf{Rationale for Model Selection:} Calibrated Linear SVM is selected not due to an unassailable F1 margin, but because Platt sigmoid scaling via \texttt{CalibratedClassifierCV} fits parameters $A, B \in \R$ on internal 5-fold cross-validation margins:
\begin{equation}
P(y=1|\bx) = \frac{1}{1 + \exp\left(A \cdot f(\bx) + B\right)}
\end{equation}
producing true posterior probabilities. Uncalibrated LinearSVC yields arbitrary margins $f(\bx) \in (-\infty, \infty)$ that cannot be thresholded on probability scale, while SGD's surrogate loss produces poorer log-loss calibration. Continuous probability outputs are indispensable for operational threshold adjustments under asymmetric costs.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.34\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_confusion_matrix.png}
  \caption{Confusion Matrix ($\tau^*=0.50$).}
  \label{fig:confusion_matrix}
\end{minipage}
\hfill
\begin{minipage}{0.62\textwidth}
  \centering
  \includegraphics[width=0.48\textwidth]{figures/spam_roc_pr_curve.png}
  \hfill
  \includegraphics[width=0.48\textwidth]{figures/spam_threshold_tradeoff.png}
  \caption{ROC/PR (AUC 0.9904) and Metric Tradeoffs vs. $\tau$.}
  \label{fig:spam_diagnostics}
\end{minipage}
\end{figure}

\subsection{Operational Decision Thresholds and Cost Analysis}
Let $c_{FP}$ and $c_{FN}$ denote operational penalties for False Positives and False Negatives. Total cost is modeled as $\text{Cost}(\tau) = c_{FP} \cdot FP(\tau) + c_{FN} \cdot FN(\tau)$. With $c_{FP}/c_{FN} \approx 20$, misclassifying ham is 20 times more severe. Table~\ref{tab:threshold_tradeoffs} evaluates out-of-fold metrics across $\tau \in [0.10, 0.90]$.

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\textwidth}{@{\extracolsep{\fill}}c c c c c c r r r@{}}
\toprule
\textbf{$\tau$} & \textbf{Accuracy} & \textbf{Spam Prec.} & \textbf{Spam Rec.} & \textbf{Spam F1} & \textbf{Ham FPR} & \textbf{FP} & \textbf{FN} & \textbf{Cost ($c_{FP}=20$)} \\
\midrule
0.10 & 0.8595 & 0.6059 & \textbf{0.9945} & 0.7530 & 0.1775 & 584 & \textbf{5} & 11,685 \\
0.20 & 0.9294 & 0.7579 & 0.9878 & 0.8577 & 0.0866 & 285 & 11 & 5,711 \\
0.30 & 0.9556 & 0.8431 & 0.9756 & 0.9045 & 0.0498 & 164 & 22 & 3,302 \\
0.40 & 0.9680 & 0.8984 & 0.9601 & 0.9283 & 0.0298 & 98 & 36 & 1,996 \\
\textbf{0.50} & \textbf{0.9740} & 0.9334 & 0.9468 & \textbf{0.9401} & 0.0185 & 61 & 48 & 1,268 \\
0.60 & 0.9721 & 0.9517 & 0.9169 & 0.9340 & 0.0128 & 42 & 75 & 915 \\
\textbf{0.70} & 0.9645 & \textbf{0.9620} & 0.8693 & 0.9133 & \textbf{0.0094} & \textbf{31} & 118 & 738 \\
0.80 & 0.9392 & 0.9751 & 0.7364 & 0.8391 & 0.0052 & 17 & 238 & \textbf{578} \\
0.90 & 0.9046 & 0.9809 & 0.5681 & 0.7195 & 0.0030 & 10 & 390 & 590 \\
\bottomrule
\end{tabularx}
\caption{Calibrated Linear SVM performance and asymmetric cost across decision thresholds $\tau$ on out-of-fold validation data.}
\label{tab:threshold_tradeoffs}
\end{table}

\textbf{Cost Tradeoff Discussion:} Pure linear cost minimization under $c_{FP}/c_{FN}=20$ reaches its global minimum at $\tau=0.80$ ($\text{Cost}=578$). However, operating at $\tau=0.80$ causes Spam Recall to collapse from 94.68\% to 73.64\% (238 spam leaks). In production, engineering deployments adopt risk-constrained optimization:
\begin{equation}
\min_{\tau} \text{Ham FPR}(\tau) \quad \text{subject to} \quad \text{Spam Recall}(\tau) \ge 0.85
\end{equation}
Under this criterion, $\tau=0.70$ is the optimal operating point, reducing Ham FPR by 49.2\% to \textbf{0.0094} (only 31 false positives out of 3,290 legitimate emails) while maintaining a high Spam F1 of 0.9133. For standard balanced utility, $\tau^*=0.50$ provides maximum F1 (0.9401) and Accuracy (0.9740).

\subsection{Qualitative Error Analysis and Actionable Remedies}
Table~\ref{tab:error_forensics} presents five False Positives and five False Negatives from out-of-fold validation predictions evaluated at operating threshold $\tau^*=0.50$. All five FPs satisfy $P(\text{Spam}) \ge 0.50$ and all five FNs satisfy $P(\text{Spam}) < 0.50$, ensuring complete consistency with the reported confusion matrix.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{l c c X X}
\toprule
\textbf{Error Case \& ID} & \textbf{Truth} & $\bm{P(\text{Spam})}$ & \textbf{Failure Mechanism \& Key Text Elements} & \textbf{Actionable Engineering Remedy (Classical NLP)} \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Positives (Ham Misclassified as Spam at $\tau^*=0.50$) --- Total Out-of-Fold Count: 61}}} \\
FP 1 (\#179) & Ham $\to$ Spam & 0.7771 & Opt-in referral SMS: \texttt{"free sms", "contact list", "fullonsms.com"}. High positive weights on promotional tokens trigger false alarm. & Domain token downweighting; distinguish domain hostnames from promotional URL paths via regex feature extraction. \\
FP 2 (\#373) & Ham $\to$ Spam & 0.7613 & Intimate personal SMS: \texttt{"looking forward to all the sex cuddling.. Only two more sleeps"}. Adult unigram co-occurs with commercial adult spam. & Windowed co-occurrence features linking explicit unigrams with first-person personal pronouns (\texttt{"I"}, \texttt{"we"}, \texttt{"sleeps"}). \\
FP 3 (\#4106) & Ham $\to$ Spam & 0.8388 & Corporate memo on FDA drug recall: \texttt{"danger of phenylpropanolamine", "alka-seltzer"}. Body terms match online pharmacy spam. & Forwarded message zoning; identify and downweight quoted historical forward headers (\texttt{"- - original message - -"}) relative to new text. \\
FP 4 (\#25) & Ham $\to$ Spam & 0.7691 & Sparse ultra-short SMS: \texttt{"Just normal only here :)"}. Rare token combinations yield high variance in sparse feature space. & Length-calibrated shrinkage prior for short messages ($<5$ words), requiring multiple corroborating spam tokens before overriding ham prior. \\
FP 5 (\#534) & Ham $\to$ Spam & 0.6543 & Conversational query with non-standard syntax: \texttt{"No message..no responce..what happend?"}. Ellipses and typo penalized by bias term. & Punctuation sequence normalization (\texttt{".."} $\to$ \texttt{"."}) and character $n$-gram hashing ($n \in [3, 5]$) for colloquial orthography. \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Negatives (Spam Escaping Filter at $\tau^*=0.50$) --- Total Out-of-Fold Count: 48}}} \\
FN 1 (\#193) & Spam $\to$ Ham & 0.0451 & Adversarial customer complaint mimicry: \texttt{"ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS... L8ER GOT MEGA BILL"}. & Regex feature detectors for UK premium-rate dial codes (\texttt{"0870"}, \texttt{"0871"}) and non-standard currency symbols (\texttt{"£"}, \texttt{"\$\$"}). \\
FN 2 (\#85) & Spam $\to$ Ham & 0.0755 & Phonetic adult chatline evasion: \texttt{"Im nasty and have a thing 4 filthyguys... Txt XXX SLO"}. Creative spelling evades word vocabulary. & Subword character $n$-grams ($n \in [3, 5]$) to capture phonetic slang patterns despite non-standard spacing and abbreviations. \\
FN 3 (\#404) & Spam $\to$ Ham & 0.4229 & Financial pump-and-dump: \texttt{"Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK... Nasdaq Symbol CDGT"}. & Financial pattern regex: uppercase stock ticker detector (\texttt{"Symbol [A-Z]\{3,5\}"}) combined with promotional yield claims (\texttt{"UP OVER [0-9]+\%"}). \\
FN 4 (\#468) & Spam $\to$ Ham & 0.1855 & Conversational lure: \texttt{"Hi this is Amy, we will be sending you a free phone number in a couple of days..."}. Casual opening masks lure. & Positional transition bigrams capturing abrupt shifts from informal greeting openings to urgent commercial calls-to-action. \\
FN 5 (\#332) & Spam $\to$ Ham & 0.2154 & Legal disclaimer mimicry: \texttt{"0A\$NETWORKS allow companies to bill for SMS, so they are responsible for their suppliers..."}. & Disclaimer token density ratio: measure proportion of regulatory/boilerplate unigrams relative to document length and unique vocabulary. \\
\bottomrule
\end{tabularx}
\caption{Qualitative Analysis of Representative False Positives and False Negatives at Operating Threshold $\tau^*=0.50$.}
\label{tab:error_forensics}
\end{table}

\subsection{Domain Shift Analysis and Hidden Test Set Predictions}
The training corpus exhibits substantial domain shift: SMS texts (train1, 2,164 clean rows) average 15.2 words with 13.22\% spam prevalence, whereas desktop emails (train2, 2,029 clean rows) average 184.6 words with 30.41\% spam prevalence. TF-IDF sublinear scaling and bigram extraction normalize length disparities.

The Calibrated Linear SVM ($C=2.0$) model was refit on all 4,193 clean training instances. Inference on the 6,447 unlabeled test instances at $\tau^*=0.50$ produced \textbf{5,104 Ham (79.17\%)} and \textbf{1,343 Spam (20.83\%)}. This matches the clean training prior (78.46\% Ham / 21.54\% Spam) within \textbf{0.71\%}, confirming balanced generalization without distributional collapse. Output files were generated and verified: \texttt{Dalwai\_Spam.txt} (exactly 6,447 lines, no header, labels in $\{0, 1\}$).

\section{Project 2: Dry Bean Morphological Clustering}

\subsection{Data Hygiene, Skewness, and Feature Scaling}
The Dry Bean dataset contains 13,611 biological specimens with 16 continuous morphological measurements (12 geometric dimensions and 4 shape factors).

\textbf{Hygiene Verification:} Numerical features exhibit zero missing values. An audit identified 68 duplicate feature vectors; all 13,611 rows were preserved with monotonic 1-based indexing (\texttt{sample\_id} 1 to 13,611) per submission requirements. The \texttt{Class} column was unobserved and strictly excluded from all preprocessing, scaling, PCA, and clustering.

\textbf{Skewness and Outlier Analysis:} Feature distributions exhibit severe right-skewness: \texttt{Area} ($+2.953$), \texttt{ConvexArea} ($+2.941$), \texttt{MinorAxisLength} ($+2.238$), \texttt{EquivDiameter} ($+1.949$), and \texttt{Perimeter} ($+1.626$). Conversely, \texttt{ShapeFactor4} ($-2.759$) and \texttt{Solidity} ($-2.550$) display negative skewness. High area values ($>170{,}000\,\text{px}^2$) represent macroscopic seed varieties (Cluster 4) rather than measurement errors. We apply \texttt{StandardScaler} ($z = \frac{x - \mu}{\sigma}$) rather than nonlinear log-transforms to preserve the physical geometric aspect ratios across coordinate axes.

\subsection{PCA Dimensionality Selection: Kaiser Rule vs. 95\% Variance}
Let $\bZ \in \R^{13611 \times 16}$ denote the standardized feature matrix. We perform eigen-decomposition on the sample covariance matrix $\bSigma = \frac{1}{N-1}\bZ^T\bZ = \bV\bLambda\bV^T$ ($N=13,611$). Table~\ref{tab:pca_variance} reports the eigenvalue spectrum.

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_pca_variance.png}
  \caption{PCA Scree \& Variance}
  \label{subfig:pca_scree}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_kmeans_clusters.png}
  \caption{K-Means ($k=7$) 2D Projection}
  \label{subfig:kmeans_2d}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_method2_clusters.png}
  \caption{GMM ($k=7$) 2D Projection}
  \label{subfig:gmm_2d}
\end{subfigure}
\caption{Dry bean unsupervised clustering diagnostics: (a) Scree plot and cumulative variance; (b) K-Means ($k=7$) clusters projected on PC1--PC2; (c) GMM ($k=7$) clusters projected on PC1--PC2.}
\label{fig:drybean_banner}
\end{figure}

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\textwidth}{l r r r X}
\toprule
\textbf{PC} & \textbf{Eigenvalue} & \textbf{Variance (\%)} & \textbf{Cumulative (\%)} & \textbf{Morphological Interpretation} \\
\midrule
PC1 & 8.8753 & 55.47 & 55.47 & Scale \& Volume (\texttt{Area}: +0.33) \\
PC2 & 4.2293 & 26.43 & 81.90 & Aspect Ratio vs. Roundness (\texttt{AspectR}: +0.40) \\
PC3 & 1.2811 & 8.01 & 89.90 & Boundary Smoothness (\texttt{Solidity}: +0.46) \\
PC4 & 0.8183 & 5.11 & \textbf{95.02} & Bounding Occupancy (\texttt{Extent}: +0.72) \\
\midrule
PC5 & 0.4383 & 2.74 & 97.76 & Secondary boundary harmonics \\
PC6--16 & $<0.19$ & $<1.15$ & 100.00 & Residual collinear measurement noise \\
\bottomrule
\end{tabularx}
\caption{Eigenvalue spectrum and variance decomposition across 16 morphological features.}
\label{tab:pca_variance}
\end{table}

\textbf{Dimensionality Selection Criteria:} The Kaiser-Guttman rule ($\lambda_i > 1.0$) retains 3 components explaining 89.90\% variance. In contrast, the 95\% variance threshold retains 4 components, capturing \textbf{95.02\%} cumulative variance. Retaining PC4 captures bounding box occupancy (\texttt{Extent} loading $+0.72$), separating compact seeds from irregular contours while reducing dimensionality by 75\% (16D $\to$ 4D).

\subsection{Four-Configuration Clustering Benchmark ($k=7$)}
Table~\ref{tab:clustering_benchmark} compares four configurations across three internal validation metrics.

\textbf{Metric Interpretation Across Dimensionalities:} Config 2 (K-Means on 4D PCA) attains the highest Silhouette score (0.3277), highest Calinski--Harabasz index (9,134.22), and lowest Davies--Bouldin index (1.0328). Projecting cluster assignments back into original 16D standardized space yields a Silhouette score of \textbf{0.3076} (virtually identical to 16D K-Means at 0.3094) and Calinski--Harabasz of \textbf{7,817.26}. Thus, 4D PCA achieves comparable cluster structure in unreduced space while providing 75\% dimensionality reduction and faster convergence.

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabularx}{\textwidth}{l l l c c c X}
\toprule
\textbf{Configuration} & \textbf{Algorithm} & \textbf{Feature Space} & \makecell[c]{\textbf{Silhouette}\\\textbf{($\uparrow$)}} & \makecell[c]{\textbf{Calinski--}\\\textbf{Harabasz ($\uparrow$)}} & \makecell[c]{\textbf{Davies--}\\\textbf{Bouldin ($\downarrow$)}} & \makecell[c]{\textbf{Cluster Size Distribution}\\\textbf{($k=7$)}} \\
\midrule
Config 1 & K-Means & Standardized (16D) & 0.3094 & 7,848.44 & 1.0993 & 3111, 541, 3157, 520, 2480, 2033, 1769 \\
\textbf{Config 2 (Winner)} & \textbf{K-Means} & \textbf{PCA (4D, 95.0\% Var)} & \textbf{0.3277} & \textbf{9,134.22} & \textbf{1.0328} & 519, 3262, 2476, 2021, 521, 3042, 1770 \\
Config 3 & GMM & Standardized (16D) & 0.2898 & 6,426.16 & 1.6592 & 3191, 693, 2836, 520, 2514, 1984, 1873 \\
Config 4 & GMM & PCA (4D, 95.0\% Var) & 0.3020 & 7,618.59 & 1.2245 & 1875, 2699, 522, 1591, 2547, 3546, 831 \\
\bottomrule
\end{tabularx}
\caption{Internal Cluster Validation Comparison across Four Experimental Configurations ($k=7$) on 13,611 Dry Bean Specimens.}
\label{tab:clustering_benchmark}
\end{table}

\subsection{Cluster Stability across Multi-Seed Initializations}
To evaluate initialization sensitivity, we executed Config~2 across 10 random seeds ($s \in \{0, 1, 7, 13, 21, 42, 99, 123, 777, 2024\}$) with $k\text{-means++}$ seeding ($n_{\text{init}}=10$). The pairwise Adjusted Rand Index (ARI) across all 45 seed combinations yielded:
\begin{equation}
\text{Mean ARI} = \mathbf{0.9978} \quad (\text{range: } [0.9939, 1.0000])
\end{equation}
An average ARI of 0.9978 confirms that the four-dimensional PCA representation provides an exceptionally convex, stable objective landscape, converging to the identical partition structure regardless of random seed.

\subsection{Dimensionality Reduction vs. 2D Visual Display}
We emphasize the distinction between PCA for \textbf{Dimensionality Reduction Prior to Clustering} vs. \textbf{2D Visualization}: K-Means operates in $\R^4$ capturing \textbf{95.02\%} variance, retaining boundary rugosity (PC3) and bounding box occupancy (PC4). In contrast, Figures~\ref{fig:drybean_banner}(b) and \ref{fig:drybean_banner}(c) plot clusters on PC1--PC2, capturing \textbf{81.90\%} variance; apparent overlaps in 2D are separated along PC3 and PC4.

\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabularx}{\textwidth}{c r r c c c X}
\toprule
\textbf{Cluster} & \textbf{Count} & \textbf{Share} & \textbf{Area ($\text{px}^2$)} & \makecell[c]{\textbf{Major Axis}\\\textbf{Length ($\text{px}$)}} & \makecell[c]{\textbf{Aspect}\\\textbf{Ratio}} & \textbf{Morphological Cluster Archetype} \\
\midrule
C0 & 519 & 3.81\% & 63,737 $\pm$ 14,259 & 380.65 $\pm$ 46.79 & 1.779 $\pm$ 0.168 & \textbf{Medium-Large Elongated Seeds} \\
C1 & 3,262 & 23.97\% & 31,927 $\pm$ 4,667 & 244.47 $\pm$ 18.23 & 1.475 $\pm$ 0.079 & \textbf{Small Compact Elliptical Seeds} \\
C2 & 2,476 & 18.19\% & 74,341 $\pm$ 9,213 & 395.73 $\pm$ 32.13 & 1.651 $\pm$ 0.135 & \textbf{Large Broad Oval Seeds} \\
C3 & 2,021 & 14.85\% & 39,389 $\pm$ 5,270 & 248.55 $\pm$ 18.85 & 1.235 $\pm$ 0.064 & \textbf{Small Compact Spherical Seeds} \\
C4 & 521 & 3.83\% & 173,557 $\pm$ 23,289 & 593.25 $\pm$ 53.21 & 1.585 $\pm$ 0.119 & \textbf{Giant Macroscopic Seeds} \\
C5 & 3,042 & 22.35\% & 44,602 $\pm$ 5,387 & 299.11 $\pm$ 19.40 & 1.576 $\pm$ 0.088 & \textbf{Medium Compact Oval Seeds} \\
C6 & 1,770 & 13.00\% & 53,695 $\pm$ 7,229 & 373.64 $\pm$ 28.91 & 2.039 $\pm$ 0.122 & \textbf{Long Cylindrical Kidney Seeds} \\
\bottomrule
\end{tabularx}
\caption{Empirical morphological profile summary across the seven discovered dry bean clusters (Config 2: K-Means on 4D PCA, Reported as exact sample $\mu \pm \sigma$).}
\label{tab:cluster_profiles}
\end{table}

\subsection{Morphological Cluster Profiles and Archetype Characterization}
Table~\ref{tab:cluster_profiles} summarizes the physical characteristics of the seven discovered clusters from Config~2 with exact empirical sample standard deviations. Because true botanical labels were unobserved, we interpret the clusters as data-driven morphological archetypes:
\begin{itemize}
  \item \textbf{C0 (519, 3.81\%): Medium-Large Elongated.} Mean area $63{,}737 \pm 14{,}259$ with high aspect ratio ($1.779 \pm 0.168$).
  \item \textbf{C1 (3,262, 23.97\%): Small Compact Elliptical.} Smallest average area ($31{,}927 \pm 4{,}667$) and high compactness ($0.823 \pm 0.022$).
  \item \textbf{C2 (2,476, 18.19\%): Large Broad Oval.} Substantial seed area ($74{,}341 \pm 9{,}213$) with moderate elongation ($1.651 \pm 0.135$).
  \item \textbf{C3 (2,021, 14.85\%): Small Compact Spherical.} Lowest aspect ratio ($1.235 \pm 0.064$) and highest compactness ($0.900 \pm 0.024$).
  \item \textbf{C4 (521, 3.83\%): Giant Macroscopic.} Distinctive giant seed profile ($173{,}557 \pm 23{,}289$), more than double the area of all other clusters.
  \item \textbf{C5 (3,042, 22.35\%): Medium Compact Oval.} Intermediate size ($44{,}602 \pm 5{,}387$) and standard elongation ($1.576 \pm 0.088$).
  \item \textbf{C6 (1,770, 13.00\%): Long Cylindrical Kidney.} Highest aspect ratio ($2.039 \pm 0.122$) and lowest compactness ($0.699 \pm 0.021$).
\end{itemize}

\subsection{Submission Cluster Assignment File Verification}
Final cluster assignments from Config~2 were saved as \texttt{Dalwai\_DryBeanClusters.csv} (and \texttt{LastName\_DryBeanClusters.csv}). The file contains exactly 13,612 lines (header \texttt{sample\_id,Class\_id} plus 13,611 rows), strictly preserves 1-based indexing (\texttt{sample\_id} 1..13611), and assigns all instances to one of seven clusters ($\text{Class\_id} \in \{0, \dots, 6\}$) with zero missing entries.

\section{Discussion and Methodological Synthesis}
The dual investigations demonstrate core machine learning principles:
\begin{enumerate}
  \item \textbf{Data Leakage Prevention:} Strictly fitting vectorizers inside cross-validation training partitions prevents optimistic bias. In unsupervised tasks, isolating target labels ensures true blind discovery.
  \item \textbf{Calibrated Probability Estimation:} Accuracy alone fails under asymmetric operational costs. Platt probability calibration enables formal risk-constrained threshold tuning ($\tau=0.70$), halving false positives at minor recall penalty.
  \item \textbf{Spectral Dimensionality Reduction:} Standardizing features and applying PCA reduces dimensions by 75\% while retaining 95.02\% variance, yielding superior cluster stability ($\text{mean ARI}=0.9978$).
  \item \textbf{Internal Metric Geometry:} Metric values across different dimensional spaces must be interpreted relative to subspace geometry; validating 4D clusters in unreduced 16D space verifies physical cluster cohesion.
\end{enumerate}

\section{Conclusion}
This study presented an empirical evaluation of supervised spam detection and unsupervised dry bean clustering. For Project~1, Calibrated Linear SVM ($C=2.0$) achieved balanced performance (Spam F1: 0.9401, Accuracy: 0.9740), with probability calibration enabling operational cost control. For Project~2, K-Means on 4D PCA features achieved optimal cluster separation (Silhouette: 0.3277, Calinski--Harabasz: 9,134.22) and high initialization stability (ARI: 0.9978), identifying seven morphological seed archetypes. Both pipelines are fully reproducible with fixed random seeds.

\bibliographystyle{IEEEtran}
\begin{thebibliography}{1}
\bibitem{koklu2020multiclass}
M.~Koklu and I.~A. Ozkan, ``Multiclass classification of dry beans using computer vision and machine learning techniques,'' \emph{Computers and Electronics in Agriculture}, vol. 174, p. 105507, 2020.
\bibitem{platt1999probabilistic}
J.~Platt, ``Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,'' \emph{Advances in Large Margin Classifiers}, vol.~10, no.~3, pp. 61--74, 1999.
\bibitem{rousseeuw1987silhouettes}
P.~J. Rousseeuw, ``Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,'' \emph{Journal of Computational and Applied Mathematics}, vol.~20, pp. 53--65, 1987.
\bibitem{calinski1974dendrite}
T.~Cali{\'n}ski and J.~Harabasz, ``A dendrite method for cluster analysis,'' \emph{Communications in Statistics-theory and Methods}, vol.~3, no.~1, pp. 1--27, 1974.
\end{thebibliography}

\end{document}
"""
    with open(os.path.join(out_dir, "report.tex"), "w") as f:
        f.write(tex)
    return compile_variant(out_dir)

# ==============================================================================
# VARIANT 5: DEEPMIND / GOOGLE INDUSTRIAL (SANS-SERIF, COBALT & EMERALD)
# ==============================================================================
def write_variant5():
    out_dir = os.path.join(BASE_DIR, "docs/latex_reports/variant5_deepmind")
    os.makedirs(out_dir, exist_ok=True)
    
    tex = r"""\documentclass[9.5pt,twocolumn,letterpaper]{article}

\usepackage[
  top=0.65in,
  bottom=0.65in,
  left=0.65in,
  right=0.65in,
  columnsep=0.22in
]{geometry}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}

\usepackage{amsmath,mathtools,bm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{siunitx}
\sisetup{detect-all, tight-spacing=true}

\usepackage{graphicx}
\usepackage{subcaption}
\usepackage[font=small,labelfont=bf,skip=3pt]{caption}
\usepackage{float}

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}
\setlength{\floatsep}{4pt plus 1pt minus 1pt}
\setlength{\textfloatsep}{5pt plus 1pt minus 1pt}
\setlength{\intextsep}{4pt plus 1pt minus 1pt}

\usepackage{xcolor}
\definecolor{cobaltblue}{RGB}{29,78,216}
\definecolor{deepobsidian}{RGB}{15,23,42}
\definecolor{emeraldsignal}{RGB}{16,185,129}
\usepackage[
  colorlinks=true,
  linkcolor=cobaltblue,
  citecolor=emeraldsignal,
  urlcolor=cobaltblue
]{hyperref}

\usepackage[compact]{titlesec}
\titleformat{\section}{\large\bfseries\color{cobaltblue}}{\thesection.}{0.4em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{deepobsidian}}{\thesubsection.}{0.35em}{}
\titleformat{\subsubsection}{\small\bfseries\itshape\color{cobaltblue}}{\thesubsubsection.}{0.3em}{}
\titlespacing{\section}{0pt}{4.5pt plus 1pt minus 1pt}{2pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsection}{0pt}{3.5pt plus 1pt minus 1pt}{1.5pt plus 0.5pt minus 0.5pt}
\titlespacing{\subsubsection}{0pt}{2pt plus 0.5pt minus 0.5pt}{1pt}

\usepackage{enumitem}
\setlist{nosep, leftmargin=1.0em, labelsep=0.3em}

\graphicspath{{figures/}{../../../outputs/figures/}{outputs/figures/}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bZ}{\mathbf{Z}}
\newcommand{\bSigma}{\mathbf{\Sigma}}
\newcommand{\bLambda}{\mathbf{\Lambda}}
\newcommand{\bV}{\mathbf{V}}
\newcommand{\indic}{\mathbb{I}}

\title{\vspace{-0.3in}\textbf{\Large Empirical Evaluation of Supervised Spam Filtering and Unsupervised Morphological Clustering}\vspace{-0.08in}}
\author{\textbf{Adam Dalwai} \\ Department of Computer Science \\ \texttt{adamdalwai1@gmail.com}}
\date{\vspace{-0.15in}\small November 2026\vspace{-0.1in}}

\begin{document}

\maketitle

\begin{abstract}
\noindent We present an empirical study covering supervised text classification and unsupervised geometric clustering. In Project~1, we develop an asymmetric cost-sensitive spam filter over a combined corpus of 4,193 deduplicated messages. Using stratified 5-fold cross-validation with fold-isolated TF-IDF extraction, we evaluate nine candidate architectures. Calibrated Linear Support Vector Machines ($C=2.0$) achieve top balanced performance: Accuracy $0.9740 \pm 0.0053$, Macro-F1 $0.9617 \pm 0.0079$, and Spam F1 $0.9401 \pm 0.0123$. While uncalibrated SGD and LinearSVC attain lower Ham False Positive Rates at default thresholds (0.0091 vs 0.0185), paired statistical tests show no significant F1 difference ($p=0.579$), and Platt probability calibration enables post-hoc operational threshold tuning. Under risk-constrained optimization ($\tau=0.70$), Ham FPR is reduced by 49.2\% to 0.0094 (31 false positives out of 3,290) while sustaining Spam F1 at 0.9133. Out-of-fold error analysis examines five false positives and five false negatives at the decision boundary, proposing NLP feature remedies. Predictions on 6,447 unlabeled test instances match clean training priors within 0.71\%. In Project~2, we perform unsupervised clustering on 13,611 biological specimens characterized by 16 morphological features with ground-truth class labels strictly isolated. Principal Component Analysis demonstrates that four dimensions capture 95.02\% of cumulative variance, compared with three dimensions under the Kaiser rule (89.90\%). Benchmarking four configurations ($k=7$) shows that K-Means on 4D PCA features yields a Silhouette score of 0.3277, Calinski--Harabasz index of 9,134.22, and Davies--Bouldin index of 1.0328, with pairwise Adjusted Rand Index across 10 random seeds averaging 0.9978. Morphological profiling characterizes seven distinct geometric archetypes.
\end{abstract}

\section{Introduction and Problem Formulations}
Supervised text categorization and unsupervised geometric clustering serve distinct objectives in applied machine learning. Supervised learning requires strict data hygiene, fold-level feature isolation to prevent leakage, and threshold calibration under asymmetric operational error costs. Unsupervised clustering requires multi-scale normalization, covariance decomposition, and internal metric validation independent of external reference labels.

We evaluate both tasks using canonical benchmarks: (1)~binary spam detection over combined SMS and desktop email corpora, and (2)~morphological cluster discovery across high-dimensional geometric measurements of dry bean seeds. All pipelines are implemented in Python (v3.13.2) utilizing \texttt{scikit-learn} (v1.9.0), \texttt{numpy} (v2.4.6), \texttt{pandas} (v3.0.3), and \texttt{scipy} (v1.17.1), with all stochastic operations pinned to $\text{seed}=42$ for deterministic reproducibility.

\section{Project 1: Spam Email Detection}

\subsection{Corpus Ingestion, Cleaning, and Deduplication}
The supervised learning dataset combines two heterogeneous sources:
\begin{itemize}
  \item \textbf{Training Dataset 1 (SMS):} 2,228 short mobile messages (1,927 Ham, 301 Spam; 13.51\% spam prevalence). Comma delimiter fragmentation across columns \texttt{v2}, \texttt{Unnamed: 2}, \texttt{Unnamed: 3}, and \texttt{Unnamed: 4} was repaired by programmatic text reconstruction.
  \item \textbf{Training Dataset 2 (Email):} 2,068 desktop emails (1,440 Ham, 628 Spam; 30.37\% spam prevalence) with full message bodies and integer labels.
  \item \textbf{Hidden Test Dataset:} 6,447 unlabeled instances for final inference.
\end{itemize}

Direct union yields 4,296 raw instances. Exact string hashing identified and removed \textbf{103 duplicate records} (64 in Dataset~1, 39 in Dataset~2, 0 cross-corpus collisions; 0 conflicting labels). Purging duplicates established a clean training corpus of \textbf{4,193 unique instances} (3,290 Ham [78.46\%], 903 Spam [21.54\%], imbalance ratio 3.64:1). Deduplication prior to partitioning ensures identical texts cannot co-occur across training and validation splits.

\subsection{Experimental Design, Leakage Prevention, and Models}
Text was tokenized using sublinear TF-IDF scaling:
\begin{equation}
\text{TF-IDF}(t, d) = \left(1 + \log(\text{tf}(t, d))\right) \cdot \log\left(\frac{1 + N}{1 + \text{df}(t)}\right)
\end{equation}
with unigram and bigram extraction ($n \in \{1, 2\}$), document frequency threshold $\text{min\_df}=2$, sublinear term weighting, and accent stripping, producing $\approx 14,800$ sparse features.

\textbf{Fold Isolation Protocol:} In each fold of Stratified 5-Fold Cross-Validation, the TF-IDF vectorizer was fit strictly on the training partition ($X_{\text{train}}$) and applied to the validation partition ($X_{\text{val}}$) via \texttt{transform}. Test instances were never accessed during vocabulary construction. Table~\ref{tab:model_specs} details the hyperparameter configurations across all nine evaluated architectures.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l X}
\toprule
\textbf{Model Architecture} & \textbf{Hyperparameter Specification \& Tuning Space} \\
\midrule
MultinomialNB (Baseline) & $\alpha=1.0$, uniform class prior (uncalibrated) \\
ComplementNB & $\alpha=1.0$, norm=False (designed for imbalanced text) \\
Logistic Regression ($C=1.0$) & $C=1.0$, L2 penalty, lbfgs solver, max\_iter=1000 \\
Logistic Regression ($C=10.0$) & $C=10.0$, low regularization, lbfgs solver \\
Logistic Regression (Balanced) & $C=5.0$, \texttt{class\_weight='balanced'}, lbfgs solver \\
Linear SVM (LinearSVC) & $C=1.0$, squared hinge loss, L2 penalty \\
\textbf{Calibrated Linear SVM} & LinearSVC ($C=2.0$) with Platt sigmoid calibration (5-fold internal CV) \\
Random Forest ($n=100$) & 100 estimators, Gini criterion, max\_features='sqrt' \\
SGD Classifier (Mod. Huber) & Modified Huber loss, $\alpha=10^{-4}$, elasticnet ($\rho=0.15$) \\
\bottomrule
\end{tabularx}
\caption{Hyperparameter configurations across nine evaluated spam classifiers.}
\label{tab:model_specs}
\end{table}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.06}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c@{}}
\toprule
\textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Spam Precision} & \textbf{Spam Recall} & \textbf{Spam F1} & \textbf{Ham FPR} \\
\midrule
\textbf{Calibrated Linear SVM ($C=2.0$)} & \textbf{0.9740$\pm$0.0053} & \textbf{0.9617$\pm$0.0079} & 0.9336$\pm$0.0155 & \textbf{0.9469$\pm$0.0186} & \textbf{0.9401$\pm$0.0123} & 0.0185$\pm$0.0046 \\
SGD Classifier (Modified Huber) & 0.9738$\pm$0.0066 & 0.9603$\pm$0.0103 & 0.9650$\pm$0.0115 & 0.9115$\pm$0.0303 & 0.9372$\pm$0.0164 & 0.0091$\pm$0.0032 \\
Logistic Regression (Balanced, $C=5.0$) & 0.9714$\pm$0.0064 & 0.9575$\pm$0.0096 & 0.9375$\pm$0.0132 & 0.9292$\pm$0.0235 & 0.9332$\pm$0.0152 & 0.0170$\pm$0.0038 \\
Linear SVM (LinearSVC, $C=1.0$) & 0.9714$\pm$0.0066 & 0.9565$\pm$0.0104 & 0.9644$\pm$0.0088 & 0.9004$\pm$0.0293 & 0.9311$\pm$0.0167 & 0.0091$\pm$0.0024 \\
Logistic Regression ($C=10.0$) & 0.9638$\pm$0.0065 & 0.9440$\pm$0.0106 & 0.9664$\pm$0.0030 & 0.8616$\pm$0.0301 & 0.9108$\pm$0.0172 & 0.0082$\pm$0.0007 \\
Random Forest ($n=100$, Gini) & 0.9344$\pm$0.0093 & 0.8919$\pm$0.0170 & 0.9714$\pm$0.0090 & 0.7165$\pm$0.0421 & 0.8241$\pm$0.0286 & 0.0058$\pm$0.0018 \\
Complement Naive Bayes & 0.9318$\pm$0.0096 & 0.8885$\pm$0.0180 & 0.9506$\pm$0.0121 & 0.7209$\pm$0.0468 & 0.8191$\pm$0.0303 & 0.0103$\pm$0.0026 \\
Logistic Regression ($C=1.0$, L2) & 0.9222$\pm$0.0087 & 0.8672$\pm$0.0175 & 0.9836$\pm$0.0111 & 0.6500$\pm$0.0435 & 0.7818$\pm$0.0300 & 0.0030$\pm$0.0021 \\
Multinomial Naive Bayes (Baseline) & 0.8781$\pm$0.0113 & 0.7657$\pm$0.0288 & \textbf{1.0000$\pm$0.0000} & 0.4341$\pm$0.0527 & 0.6035$\pm$0.0515 & \textbf{0.0000$\pm$0.0000} \\
\bottomrule
\end{tabular*}
\caption{Stratified 5-Fold Cross-Validation Performance Comparison across Nine Model Architectures (Reported as $\mu \pm \sigma$).}
\label{tab:spam_benchmark}
\end{table*}

\subsection{Cross-Validation Results and Statistical Comparison}
Table~\ref{tab:spam_benchmark} reports cross-validated performance. MultinomialNB achieved 100\% precision but suffered severe under-detection (Spam Recall $0.4341 \pm 0.0527$), missing 56.6\% of spam messages. This occurs because the 3.64:1 class prior heavily suppresses uncalibrated posterior probabilities.

Calibrated Linear SVM ($C=2.0$) attained the highest mean Spam F1 ($0.9401 \pm 0.0123$) and Spam Recall ($0.9469 \pm 0.0186$). However, SGD Classifier with Modified Huber loss achieved a comparable Spam F1 ($0.9372 \pm 0.0164$) with a lower Ham FPR ($0.0091 \pm 0.0032$). A paired two-tailed $t$-test across the five cross-validation folds reveals that the F1 difference between Calibrated SVM and SGD is not statistically significant ($t = 0.603, p = 0.579$). Similarly, LinearSVC ($C=1.0$) shows no statistically significant F1 shortfall ($t = 1.513, p = 0.205$).

\textbf{Rationale for Model Selection:} Calibrated Linear SVM is selected not due to an unassailable F1 margin, but because Platt sigmoid scaling via \texttt{CalibratedClassifierCV} fits parameters $A, B \in \R$ on internal 5-fold cross-validation margins:
\begin{equation}
P(y=1|\bx) = \frac{1}{1 + \exp\left(A \cdot f(\bx) + B\right)}
\end{equation}
producing true posterior probabilities. Uncalibrated LinearSVC yields arbitrary margins $f(\bx) \in (-\infty, \infty)$ that cannot be thresholded on probability scale, while SGD's surrogate loss produces poorer log-loss calibration. Continuous probability outputs are indispensable for operational threshold adjustments under asymmetric costs.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\columnwidth]{figures/spam_confusion_matrix.png}
\caption{Out-of-fold Confusion Matrix for the selected Calibrated Linear SVM ($C=2.0$) at operating threshold $\tau^*=0.50$ (Left: Counts; Right: Normalized Rates).}
\label{fig:confusion_matrix}
\end{figure}

\subsection{Operational Decision Thresholds and Cost Analysis}
Let $c_{FP}$ and $c_{FN}$ denote operational penalties for False Positives and False Negatives. Total cost is modeled as $\text{Cost}(\tau) = c_{FP} \cdot FP(\tau) + c_{FN} \cdot FN(\tau)$. With $c_{FP}/c_{FN} \approx 20$, misclassifying ham is 20 times more severe. Table~\ref{tab:threshold_tradeoffs} evaluates out-of-fold metrics across $\tau \in [0.10, 0.90]$.

\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{c c c c c c r r r}
\toprule
\textbf{$\tau$} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Recall} & \textbf{Spam F1} & \textbf{Ham FPR} & \textbf{FP} & \textbf{FN} & \textbf{Cost ($c_{FP}=20$)} \\
\midrule
0.10 & 0.8595 & 0.6059 & \textbf{0.9945} & 0.7530 & 0.1775 & 584 & \textbf{5} & 11,685 \\
0.20 & 0.9294 & 0.7579 & 0.9878 & 0.8577 & 0.0866 & 285 & 11 & 5,711 \\
0.30 & 0.9556 & 0.8431 & 0.9756 & 0.9045 & 0.0498 & 164 & 22 & 3,302 \\
0.40 & 0.9680 & 0.8984 & 0.9601 & 0.9283 & 0.0298 & 98 & 36 & 1,996 \\
\textbf{0.50} & \textbf{0.9740} & 0.9334 & 0.9468 & \textbf{0.9401} & 0.0185 & 61 & 48 & 1,268 \\
0.60 & 0.9721 & 0.9517 & 0.9169 & 0.9340 & 0.0128 & 42 & 75 & 915 \\
\textbf{0.70} & 0.9645 & \textbf{0.9620} & 0.8693 & 0.9133 & \textbf{0.0094} & \textbf{31} & 118 & 738 \\
0.80 & 0.9392 & 0.9751 & 0.7364 & 0.8391 & 0.0052 & 17 & 238 & \textbf{578} \\
0.90 & 0.9046 & 0.9809 & 0.5681 & 0.7195 & 0.0030 & 10 & 390 & 590 \\
\bottomrule
\end{tabularx}
\caption{Calibrated Linear SVM performance and asymmetric cost across decision thresholds $\tau$ on out-of-fold validation data.}
\label{tab:threshold_tradeoffs}
\end{table}

\textbf{Cost Tradeoff Discussion:} Pure linear cost minimization under $c_{FP}/c_{FN}=20$ reaches its global minimum at $\tau=0.80$ ($\text{Cost}=578$). However, operating at $\tau=0.80$ causes Spam Recall to collapse from 94.68\% to 73.64\% (238 spam leaks). In production, engineering deployments adopt risk-constrained optimization:
\begin{equation}
\min_{\tau} \text{Ham FPR}(\tau) \quad \text{subject to} \quad \text{Spam Recall}(\tau) \ge 0.85
\end{equation}
Under this criterion, $\tau=0.70$ is the optimal operating point, reducing Ham FPR by 49.2\% to \textbf{0.0094} (only 31 false positives out of 3,290 legitimate emails) while maintaining a high Spam F1 of 0.9133. For standard balanced utility, $\tau^*=0.50$ provides maximum F1 (0.9401) and Accuracy (0.9740). Figure~\ref{fig:confusion_matrix} plots the confusion matrix at $\tau^*=0.50$ (TN: 3,229, FP: 61, FN: 48, TP: 855).

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_roc_pr_curve.png}
  \caption{ROC \& PR Curves}
  \label{subfig:roc_pr}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\columnwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/spam_threshold_tradeoff.png}
  \caption{Tradeoffs vs. $\tau$}
  \label{subfig:threshold}
\end{subfigure}
\caption{Discrimination diagnostics for Calibrated Linear SVM: (a) ROC-AUC 0.9904, PR-AUC 0.9772; (b) Precision, Recall, F1, and Ham FPR response across $\tau$.}
\label{fig:spam_diagnostics}
\end{figure}

\subsection{Qualitative Error Analysis and Actionable Remedies}
Table~\ref{tab:error_forensics} presents five False Positives and five False Negatives from out-of-fold validation predictions evaluated at operating threshold $\tau^*=0.50$. All five FPs satisfy $P(\text{Spam}) \ge 0.50$ and all five FNs satisfy $P(\text{Spam}) < 0.50$, ensuring complete consistency with the reported confusion matrix.

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{l c c X X}
\toprule
\textbf{Error Case \& ID} & \textbf{Truth} & $\bm{P(\text{Spam})}$ & \textbf{Failure Mechanism \& Key Text Elements} & \textbf{Actionable Engineering Remedy (Classical NLP)} \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Positives (Ham Misclassified as Spam at $\tau^*=0.50$) --- Total Out-of-Fold Count: 61}}} \\
FP 1 (\#179) & Ham $\to$ Spam & 0.7771 & Opt-in referral SMS: \texttt{"free sms", "contact list", "fullonsms.com"}. High positive weights on promotional tokens trigger false alarm. & Domain token downweighting; distinguish domain hostnames from promotional URL paths via regex feature extraction. \\
FP 2 (\#373) & Ham $\to$ Spam & 0.7613 & Intimate personal SMS: \texttt{"looking forward to all the sex cuddling.. Only two more sleeps"}. Adult unigram co-occurs with commercial adult spam. & Windowed co-occurrence features linking explicit unigrams with first-person personal pronouns (\texttt{"I"}, \texttt{"we"}, \texttt{"sleeps"}). \\
FP 3 (\#4106) & Ham $\to$ Spam & 0.8388 & Corporate memo on FDA drug recall: \texttt{"danger of phenylpropanolamine", "alka-seltzer"}. Body terms match online pharmacy spam. & Forwarded message zoning; identify and downweight quoted historical forward headers (\texttt{"- - original message - -"}) relative to new text. \\
FP 4 (\#25) & Ham $\to$ Spam & 0.7691 & Sparse ultra-short SMS: \texttt{"Just normal only here :)"}. Rare token combinations yield high variance in sparse feature space. & Length-calibrated shrinkage prior for short messages ($<5$ words), requiring multiple corroborating spam tokens before overriding ham prior. \\
FP 5 (\#534) & Ham $\to$ Spam & 0.6543 & Conversational query with non-standard syntax: \texttt{"No message..no responce..what happend?"}. Ellipses and typo penalized by bias term. & Punctuation sequence normalization (\texttt{".."} $\to$ \texttt{"."}) and character $n$-gram hashing ($n \in [3, 5]$) for colloquial orthography. \\
\midrule
\multicolumn{5}{l}{\textit{\textbf{False Negatives (Spam Escaping Filter at $\tau^*=0.50$) --- Total Out-of-Fold Count: 48}}} \\
FN 1 (\#193) & Spam $\to$ Ham & 0.0451 & Adversarial customer complaint mimicry: \texttt{"ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS... L8ER GOT MEGA BILL"}. & Regex feature detectors for UK premium-rate dial codes (\texttt{"0870"}, \texttt{"0871"}) and non-standard currency symbols (\texttt{"£"}, \texttt{"\$\$"}). \\
FN 2 (\#85) & Spam $\to$ Ham & 0.0755 & Phonetic adult chatline evasion: \texttt{"Im nasty and have a thing 4 filthyguys... Txt XXX SLO"}. Creative spelling evades word vocabulary. & Subword character $n$-grams ($n \in [3, 5]$) to capture phonetic slang patterns despite non-standard spacing and abbreviations. \\
FN 3 (\#404) & Spam $\to$ Ham & 0.4229 & Financial pump-and-dump: \texttt{"Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK... Nasdaq Symbol CDGT"}. & Financial pattern regex: uppercase stock ticker detector (\texttt{"Symbol [A-Z]\{3,5\}"}) combined with promotional yield claims (\texttt{"UP OVER [0-9]+\%"}). \\
FN 4 (\#468) & Spam $\to$ Ham & 0.1855 & Conversational lure: \texttt{"Hi this is Amy, we will be sending you a free phone number in a couple of days..."}. Casual opening masks lure. & Positional transition bigrams capturing abrupt shifts from informal greeting openings to urgent commercial calls-to-action. \\
FN 5 (\#332) & Spam $\to$ Ham & 0.2154 & Legal disclaimer mimicry: \texttt{"0A\$NETWORKS allow companies to bill for SMS, so they are responsible for their suppliers..."}. & Disclaimer token density ratio: measure proportion of regulatory/boilerplate unigrams relative to document length and unique vocabulary. \\
\bottomrule
\end{tabularx}
\caption{Qualitative Analysis of Representative False Positives and False Negatives at Operating Threshold $\tau^*=0.50$.}
\label{tab:error_forensics}
\end{table*}

\subsection{Domain Shift Analysis and Hidden Test Set Predictions}
The training corpus exhibits substantial domain shift: SMS texts (train1, 2,164 clean rows) average 15.2 words with 13.22\% spam prevalence, whereas desktop emails (train2, 2,029 clean rows) average 184.6 words with 30.41\% spam prevalence. TF-IDF sublinear scaling and bigram extraction normalize length disparities.

The Calibrated Linear SVM ($C=2.0$) model was refit on all 4,193 clean training instances. Inference on the 6,447 unlabeled test instances at $\tau^*=0.50$ produced \textbf{5,104 Ham (79.17\%)} and \textbf{1,343 Spam (20.83\%)}. This matches the clean training prior (78.46\% Ham / 21.54\% Spam) within \textbf{0.71\%}, confirming balanced generalization without distributional collapse. Output files were generated and verified: \texttt{Dalwai\_Spam.txt} (exactly 6,447 lines, no header, labels in $\{0, 1\}$).

\section{Project 2: Dry Bean Morphological Clustering}

\begin{figure*}[t]
\centering
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_pca_variance.png}
  \caption{PCA Scree \& Variance}
  \label{subfig:pca_scree}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_kmeans_clusters.png}
  \caption{K-Means ($k=7$) 2D Projection}
  \label{subfig:kmeans_2d}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
  \centering
  \includegraphics[width=\textwidth]{figures/drybean_method2_clusters.png}
  \caption{GMM ($k=7$) 2D Projection}
  \label{subfig:gmm_2d}
\end{subfigure}
\caption{Dry bean unsupervised clustering diagnostics: (a) Scree plot and cumulative variance; (b) K-Means ($k=7$) clusters projected on PC1--PC2; (c) GMM ($k=7$) clusters projected on PC1--PC2.}
\label{fig:drybean_banner}
\end{figure*}

\subsection{Data Hygiene, Skewness, and Feature Scaling}
The Dry Bean dataset contains 13,611 biological specimens with 16 continuous morphological measurements (12 geometric dimensions and 4 shape factors).

\textbf{Hygiene Verification:} Numerical features exhibit zero missing values. An audit identified 68 duplicate feature vectors; all 13,611 rows were preserved with monotonic 1-based indexing (\texttt{sample\_id} 1 to 13,611) per submission requirements. The \texttt{Class} column was unobserved and strictly excluded from all preprocessing, scaling, PCA, and clustering.

\textbf{Skewness and Outlier Analysis:} Feature distributions exhibit severe right-skewness: \texttt{Area} ($+2.953$), \texttt{ConvexArea} ($+2.941$), \texttt{MinorAxisLength} ($+2.238$), \texttt{EquivDiameter} ($+1.949$), and \texttt{Perimeter} ($+1.626$). Conversely, \texttt{ShapeFactor4} ($-2.759$) and \texttt{Solidity} ($-2.550$) display negative skewness. High area values ($>170{,}000\,\text{px}^2$) represent macroscopic seed varieties (Cluster 4) rather than measurement errors. We apply \texttt{StandardScaler} ($z = \frac{x - \mu}{\sigma}$) rather than nonlinear log-transforms to preserve the physical geometric aspect ratios across coordinate axes.

\subsection{PCA Dimensionality Selection: Kaiser Rule vs. 95\% Variance}
Let $\bZ \in \R^{13611 \times 16}$ denote the standardized feature matrix. We perform eigen-decomposition on the sample covariance matrix $\bSigma = \frac{1}{N-1}\bZ^T\bZ = \bV\bLambda\bV^T$ ($N=13,611$). Table~\ref{tab:pca_variance} reports the eigenvalue spectrum.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabularx}{\columnwidth}{l r r r X}
\toprule
\textbf{PC} & \textbf{Eigenval.} & \textbf{Var (\%)} & \textbf{Cum (\%)} & \textbf{Morphological Interpretation} \\
\midrule
PC1 & 8.8753 & 55.47 & 55.47 & Scale \& Volume (\texttt{Area}: +0.33) \\
PC2 & 4.2293 & 26.43 & 81.90 & Aspect Ratio vs. Roundness (\texttt{AspectR}: +0.40) \\
PC3 & 1.2811 & 8.01 & 89.90 & Boundary Smoothness (\texttt{Solidity}: +0.46) \\
PC4 & 0.8183 & 5.11 & \textbf{95.02} & Bounding Occupancy (\texttt{Extent}: +0.72) \\
\midrule
PC5 & 0.4383 & 2.74 & 97.76 & Secondary boundary harmonics \\
PC6--16 & $<0.19$ & $<1.15$ & 100.00 & Residual collinear measurement noise \\
\bottomrule
\end{tabularx}
\caption{Eigenvalue spectrum and variance decomposition across 16 morphological features.}
\label{tab:pca_variance}
\end{table}

\textbf{Dimensionality Selection Criteria:}
\begin{itemize}
  \item \textbf{Kaiser-Guttman Rule ($\lambda_i > 1.0$):} Retains exactly 3 components ($\lambda_1=8.88, \lambda_2=4.23, \lambda_3=1.28$), explaining 89.90\% of variance.
  \item \textbf{95\% Variance Threshold:} Retains 4 components, reaching \textbf{95.02\%} cumulative variance. Although PC4's eigenvalue ($\lambda_4=0.8183$) is below 1.0, it captures bounding box occupancy (\texttt{Extent} loading $+0.72$), which separates compact seeds from boundary irregularities. Retaining 4 PCs reduces dimensions by 75\% (16D $\to$ 4D) while capturing 95\% of total information.
\end{itemize}

\subsection{Four-Configuration Clustering Benchmark ($k=7$)}
Table~\ref{tab:clustering_benchmark} compares four configurations across three internal validation metrics.

\textbf{Metric Interpretation Across Dimensionalities:}
\begin{itemize}
  \item In algorithm space, \textbf{Config 2 (K-Means on 4D PCA)} attains the highest Silhouette score (0.3277), highest Calinski--Harabasz index (9,134.22), and lowest Davies--Bouldin index (1.0328).
  \item Comparing metrics across different dimensional spaces requires care: distances concentrate differently in 16D vs 4D. Projecting cluster assignments back into original 16D standardized space yields a Silhouette score of \textbf{0.3076} (virtually identical to 16D K-Means at 0.3094) and Calinski--Harabasz of \textbf{7,817.26}. Thus, 4D PCA achieves comparable cluster structure in unreduced space while providing 75\% dimensionality reduction and faster convergence.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l l l c c c c@{}}
\toprule
\textbf{Configuration} & \textbf{Algorithm} & \textbf{Feature Space} & \makecell[c]{\textbf{Silhouette}\\\textbf{($\uparrow$)}} & \makecell[c]{\textbf{Calinski--}\\\textbf{Harabasz ($\uparrow$)}} & \makecell[c]{\textbf{Davies--}\\\textbf{Bouldin ($\downarrow$)}} & \makecell[c]{\textbf{Cluster Size Distribution}\\\textbf{($k=7$)}} \\
\midrule
Config 1 & K-Means & Standardized (16D) & 0.3094 & 7,848.44 & 1.0993 & 3111, 541, 3157, 520, 2480, 2033, 1769 \\
\textbf{Config 2 (Winner)} & \textbf{K-Means} & \textbf{PCA (4D, 95.0\% Var)} & \textbf{0.3277} & \textbf{9,134.22} & \textbf{1.0328} & 519, 3262, 2476, 2021, 521, 3042, 1770 \\
Config 3 & GMM & Standardized (16D) & 0.2898 & 6,426.16 & 1.6592 & 3191, 693, 2836, 520, 2514, 1984, 1873 \\
Config 4 & GMM & PCA (4D, 95.0\% Var) & 0.3020 & 7,618.59 & 1.2245 & 1875, 2699, 522, 1591, 2547, 3546, 831 \\
\bottomrule
\end{tabular*}
\caption{Internal Cluster Validation Comparison across Four Experimental Configurations ($k=7$) on 13,611 Dry Bean Specimens.}
\label{tab:clustering_benchmark}
\end{table*}

\subsection{Cluster Stability across Multi-Seed Initializations}
To evaluate whether K-Means partitions are sensitive to initialization, we executed Config~2 across 10 random seeds ($s \in \{0, 1, 7, 13, 21, 42, 99, 123, 777, 2024\}$) with $k\text{-means++}$ seeding ($n_{\text{init}}=10$). The pairwise Adjusted Rand Index (ARI) across all 45 seed combinations yielded:
\begin{equation}
\text{Mean ARI} = \mathbf{0.9978} \quad (\text{range: } [0.9939, 1.0000])
\end{equation}
An average ARI of 0.9978 confirms that the four-dimensional PCA representation provides an exceptionally convex, stable objective landscape, with K-Means converging to the identical partition structure regardless of random seed.

\subsection{Dimensionality Reduction vs. 2D Visual Display}
We emphasize the distinction between PCA for \textbf{Dimensionality Reduction Prior to Clustering} vs. \textbf{2D Visualization}:
\begin{itemize}
  \item \textbf{4D Clustering Space:} K-Means operates in $\R^4$ capturing \textbf{95.02\%} variance, retaining boundary rugosity (PC3) and bounding box occupancy (PC4).
  \item \textbf{2D Visual Projection:} Figures~\ref{fig:drybean_banner}(b) and \ref{fig:drybean_banner}(c) plot clusters on PC1--PC2, capturing \textbf{81.90\%} variance. The remaining 13.12\% variance operates orthogonally; apparent overlaps in 2D are separated along PC3 and PC4.
\end{itemize}

\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}c r r c c c c l@{}}
\toprule
\textbf{Cluster} & \textbf{Count} & \textbf{Share} & \textbf{Area ($\text{px}^2$)} & \makecell[c]{\textbf{Major Axis}\\\textbf{Length ($\text{px}$)}} & \makecell[c]{\textbf{Aspect}\\\textbf{Ratio}} & \textbf{Compactness} & \textbf{Morphological Cluster Archetype} \\
\midrule
C0 & 519 & 3.81\% & 63,737 $\pm$ 14,259 & 380.65 $\pm$ 46.79 & 1.779 $\pm$ 0.168 & 0.745 $\pm$ 0.037 & \textbf{Medium-Large Elongated Seeds} \\
C1 & 3,262 & 23.97\% & 31,927 $\pm$ 4,667 & 244.47 $\pm$ 18.23 & 1.475 $\pm$ 0.079 & 0.823 $\pm$ 0.022 & \textbf{Small Compact Elliptical Seeds} \\
C2 & 2,476 & 18.19\% & 74,341 $\pm$ 9,213 & 395.73 $\pm$ 32.13 & 1.651 $\pm$ 0.135 & 0.778 $\pm$ 0.034 & \textbf{Large Broad Oval Seeds} \\
C3 & 2,021 & 14.85\% & 39,389 $\pm$ 5,270 & 248.55 $\pm$ 18.85 & 1.235 $\pm$ 0.064 & 0.900 $\pm$ 0.024 & \textbf{Small Compact Spherical Seeds} \\
C4 & 521 & 3.83\% & 173,557 $\pm$ 23,289 & 593.25 $\pm$ 53.21 & 1.585 $\pm$ 0.119 & 0.793 $\pm$ 0.031 & \textbf{Giant Macroscopic Seeds} \\
C5 & 3,042 & 22.35\% & 44,602 $\pm$ 5,387 & 299.11 $\pm$ 19.40 & 1.576 $\pm$ 0.088 & 0.796 $\pm$ 0.022 & \textbf{Medium Compact Oval Seeds} \\
C6 & 1,770 & 13.00\% & 53,695 $\pm$ 7,229 & 373.64 $\pm$ 28.91 & 2.039 $\pm$ 0.122 & 0.699 $\pm$ 0.021 & \textbf{Long Cylindrical Kidney Seeds} \\
\bottomrule
\end{tabular*}
\caption{Empirical morphological profile summary across the seven discovered dry bean clusters (Config 2: K-Means on 4D PCA, Reported as exact sample $\mu \pm \sigma$).}
\label{tab:cluster_profiles}
\end{table*}

\subsection{Morphological Cluster Profiles and Archetype Characterization}
Table~\ref{tab:cluster_profiles} summarizes the physical characteristics of the seven discovered clusters from Config~2 with exact empirical sample standard deviations. Because true botanical labels were unobserved, we interpret the clusters as data-driven morphological archetypes:
\begin{itemize}
  \item \textbf{C0 (519, 3.81\%): Medium-Large Elongated.} Mean area $63{,}737 \pm 14{,}259$ with high aspect ratio ($1.779 \pm 0.168$).
  \item \textbf{C1 (3,262, 23.97\%): Small Compact Elliptical.} Smallest average area ($31{,}927 \pm 4{,}667$) and high compactness ($0.823 \pm 0.022$).
  \item \textbf{C2 (2,476, 18.19\%): Large Broad Oval.} Substantial seed area ($74{,}341 \pm 9{,}213$) with moderate elongation ($1.651 \pm 0.135$).
  \item \textbf{C3 (2,021, 14.85\%): Small Compact Spherical.} Lowest aspect ratio ($1.235 \pm 0.064$) and highest compactness ($0.900 \pm 0.024$).
  \item \textbf{C4 (521, 3.83\%): Giant Macroscopic.} Distinctive giant seed profile ($173{,}557 \pm 23{,}289$), more than double the area of all other clusters.
  \item \textbf{C5 (3,042, 22.35\%): Medium Compact Oval.} Intermediate size ($44{,}602 \pm 5{,}387$) and standard elongation ($1.576 \pm 0.088$).
  \item \textbf{C6 (1,770, 13.00\%): Long Cylindrical Kidney.} Highest aspect ratio ($2.039 \pm 0.122$) and lowest compactness ($0.699 \pm 0.021$).
\end{itemize}

\subsection{Submission Cluster Assignment File Verification}
Final cluster assignments from Config~2 were saved as \texttt{Dalwai\_DryBeanClusters.csv} (and \texttt{LastName\_DryBeanClusters.csv}). The file contains exactly 13,612 lines (header \texttt{sample\_id,Class\_id} plus 13,611 rows), strictly preserves 1-based indexing (\texttt{sample\_id} 1..13611), and assigns all instances to one of seven clusters ($\text{Class\_id} \in \{0, \dots, 6\}$) with zero missing entries.

\section{Discussion and Methodological Synthesis}
The dual investigations demonstrate core machine learning principles:
\begin{enumerate}
  \item \textbf{Data Leakage Prevention:} Strictly fitting vectorizers inside cross-validation training partitions prevents optimistic bias. In unsupervised tasks, isolating target labels ensures true blind discovery.
  \item \textbf{Calibrated Probability Estimation:} Accuracy alone fails under asymmetric operational costs. Platt probability calibration enables formal risk-constrained threshold tuning ($\tau=0.70$), halving false positives at minor recall penalty.
  \item \textbf{Spectral Dimensionality Reduction:} Standardizing features and applying PCA reduces dimensions by 75\% while retaining 95.02\% variance, yielding superior cluster stability ($\text{mean ARI}=0.9978$).
  \item \textbf{Internal Metric Geometry:} Metric values across different dimensional spaces must be interpreted relative to subspace geometry; validating 4D clusters in unreduced 16D space verifies physical cluster cohesion.
\end{enumerate}

\section{Conclusion}
This study presented an empirical evaluation of supervised spam detection and unsupervised dry bean clustering. For Project~1, Calibrated Linear SVM ($C=2.0$) achieved balanced performance (Spam F1: 0.9401, Accuracy: 0.9740), with probability calibration enabling operational cost control. For Project~2, K-Means on 4D PCA features achieved optimal cluster separation (Silhouette: 0.3277, Calinski--Harabasz: 9,134.22) and high initialization stability (ARI: 0.9978), identifying seven morphological seed archetypes. Both pipelines are fully reproducible with fixed random seeds.

\bibliographystyle{IEEEtran}
\begin{thebibliography}{1}
\bibitem{koklu2020multiclass}
M.~Koklu and I.~A. Ozkan, ``Multiclass classification of dry beans using computer vision and machine learning techniques,'' \emph{Computers and Electronics in Agriculture}, vol. 174, p. 105507, 2020.
\bibitem{platt1999probabilistic}
J.~Platt, ``Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,'' \emph{Advances in Large Margin Classifiers}, vol.~10, no.~3, pp. 61--74, 1999.
\bibitem{rousseeuw1987silhouettes}
P.~J. Rousseeuw, ``Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,'' \emph{Journal of Computational and Applied Mathematics}, vol.~20, pp. 53--65, 1987.
\bibitem{calinski1974dendrite}
T.~Cali{\'n}ski and J.~Harabasz, ``A dendrite method for cluster analysis,'' \emph{Communications in Statistics-theory and Methods}, vol.~3, no.~1, pp. 1--27, 1974.
\end{thebibliography}

\end{document}
"""
    with open(os.path.join(out_dir, "report.tex"), "w") as f:
        f.write(tex)
    return compile_variant(out_dir)

def main():
    print("=================================================================")
    print("Building and Compiling all 5 LaTeX Report Variants")
    print("=================================================================")
    results = {}
    
    # Variant 1: IEEE Two-Column
    v1_dir = os.path.join(BASE_DIR, "docs/latex_reports/variant1_ieee")
    s1, p1 = compile_variant(v1_dir)
    results["Variant 1 (IEEE Two-Column)"] = (s1, p1)
    
    # Variant 2: Nature Editorial
    s2, p2 = write_variant2()
    results["Variant 2 (Nature Editorial)"] = (s2, p2)
    
    # Variant 3: ICLR Modern
    s3, p3 = write_variant3()
    results["Variant 3 (ICLR Computational)"] = (s3, p3)
    
    # Variant 4: Oxford Monograph
    s4, p4 = write_variant4()
    results["Variant 4 (Oxford Monograph)"] = (s4, p4)
    
    # Variant 5: DeepMind Tech Report
    s5, p5 = write_variant5()
    results["Variant 5 (DeepMind Industrial)"] = (s5, p5)
    
    print("\n=================================================================")
    print("LaTeX Report Variants Compilation Summary:")
    for name, (succ, p) in results.items():
        status = "PASSED" if succ and p <= 6 else "FAILED/OVERFLOW"
        print(f" - {name}: {p} Pages [{status}]")
    print("=================================================================")

if __name__ == "__main__":
    main()
