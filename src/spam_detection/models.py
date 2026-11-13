"""
Model definitions and vectorizer factory for Spam Email Detection.

Provides:
- TfidfVectorizer constructor with optimized text representation parameters.
- Dictionary of candidate models spanning multiple distinct model families:
  1. Baseline: Multinomial Naive Bayes (MultinomialNB)
  2. Complement Naive Bayes (ComplementNB)
  3. Logistic Regression (C=1.0, default L2 regularizer)
  4. Logistic Regression (C=10.0, tuned regularizer)
  5. Logistic Regression (C=5.0, class_weight='balanced')
  6. Linear Support Vector Machine (LinearSVC C=1.0)
  7. Calibrated Linear Support Vector Machine (LinearSVC C=2.0 via CalibratedClassifierCV)
  8. Random Forest Classifier (n_estimators=100, random_state=42)
  9. Stochastic Gradient Descent Classifier (loss='modified_huber', smooth hinge loss)
"""

from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


RANDOM_SEED = 42


def get_tfidf_vectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=None,
    strip_accents="unicode"
) -> TfidfVectorizer:
    """
    Constructs a TF-IDF vectorizer.
    Using sublinear_tf=True dampens the effect of repetitive words in long emails.
    Using ngram_range=(1, 2) captures key multi-word phrases (e.g. 'claim prize', 'call free').
    min_df=2 removes single-occurrence typos without losing critical discriminative terms.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        strip_accents=strip_accents
    )


def get_candidate_models(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate classification models for comparative evaluation.
    All models with stochastic components use the specified random_state.
    """
    models = {
        "MultinomialNB (Baseline)": MultinomialNB(
            alpha=1.0
        ),
        "ComplementNB": ComplementNB(
            alpha=1.0
        ),
        "Logistic Regression (C=1.0)": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "Logistic Regression (C=10.0)": LogisticRegression(
            C=10.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "Logistic Regression (Balanced, C=5.0)": LogisticRegression(
            C=5.0,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Linear SVM (LinearSVC C=1.0)": LinearSVC(
            C=1.0,
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            max_iter=2000,
            random_state=random_state
        ),
        "Calibrated Linear SVM (C=2.0)": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=2.0,
                penalty="l2",
                loss="squared_hinge",
                dual="auto",
                max_iter=2000,
                random_state=random_state
            ),
            method="sigmoid",
            cv=3
        ),
        "Random Forest (n=100)": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "SGD Classifier (Modified Huber)": SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            random_state=random_state
        )
    }
    return models
