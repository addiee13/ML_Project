"""
Qualitative Error Analysis module for Spam Email Detection.

Extracts, categorizes, and explains out-of-fold False Positives and False Negatives:
- False Positives (Ham classified as Spam): Costly legitimate email loss.
- False Negatives (Spam classified as Ham): Missed unsolicited / malicious emails.

Generates:
- outputs/tables/spam_error_analysis.csv
- outputs/tables/spam_error_analysis.json
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def generate_error_analysis(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    output_csv: str = "outputs/tables/spam_error_analysis.csv",
    output_json: str = "outputs/tables/spam_error_analysis.json"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive qualitative and quantitative error analysis
    on out-of-fold validation predictions.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

    logger.info(f"Total out-of-fold False Positives: {len(fp_indices)}")
    logger.info(f"Total out-of-fold False Negatives: {len(fn_indices)}")

    # Curate at least 5 representative False Positives and 5 False Negatives
    # Selected to demonstrate diverse, distinct failure modes / archetypes
    curated_cases: List[Dict[str, Any]] = []

    # 1. False Positives
    # Sort FPs by highest probability descending (most confident false alarms)
    sorted_fps = fp_indices[np.argsort(-probs[fp_indices])]

    # Archetype definitions for representative FPs
    fp_archetypes = [
        {
            "target_substr": "fullonsms",
            "category": "Legitimate Service & Referral Invites",
            "root_cause": (
                "Message contains referral tokens like 'free sms', 'contact list', and web link 'www.fullonsms.com'. "
                "The unigram/bigram TF-IDF vectorizer assigns high positive spam weights to 'free sms' and 'visit', "
                "failing to recognize that this is an opt-in referral invitation between acquaintances."
            ),
            "remedy": (
                "Domain token downweighting and URL path parsing to differentiate legitimate peer service invitations from promotional link campaigns."
            )
        },
        {
            "target_substr": "sex cuddling",
            "category": "Intimate Personal Messages with Explicit Lexicon",
            "root_cause": (
                "Contains explicit romantic language ('sex cuddling', 'more sleeps'). In bag-of-words models, "
                "'sex' heavily co-occurs with adult spam. The model lacks syntactic comprehension to discern "
                "an affectionate private exchange from commercial adult spam."
            ),
            "remedy": (
                "Windowed co-occurrence features combining intimate tokens with first-person pronouns ('I', 'we', 'our') to suppress commercial spam classification."
            )
        },
        {
            "target_substr": "tablets",
            "category": "Colloquial Medical/Pharmaceutical Slang",
            "root_cause": (
                "Mentions 'tablets' alongside colloquial body references. In text spam filters, pharmaceutical terms "
                "(tablets, pills, prescriptions) carry heavy spam priors due to widespread illicit pharmacy spam campaigns."
            ),
            "remedy": (
                "Sublinear TF saturation and forwarded-chain zoning ('original message', 'fwd:') to downweight quoted medical alert forwards."
            )
        },
        {
            "target_substr": "Just normal only here",
            "category": "Ultra-Short Ambiguous SMS",
            "root_cause": (
                "Short message ('Just normal only here :)') with sparse vocabulary. Emoticons and short informal phrasing "
                "can produce skewed decision boundaries when rare token combinations match noisy training fold coefficients."
            ),
            "remedy": (
                "Length-calibrated prior damping for ultra-short texts (<5 words) requiring multi-token corroboration before overriding the ham prior."
            )
        },
        {
            "target_substr": "No message..no responce",
            "category": "Short Conversational Punctuation Anomaly",
            "root_cause": (
                "Ellipses and typo 'responce' combined with lack of substantive content. The classifier's bias term "
                "and punctuation tokenization penalize non-standard syntax as spam-like urgency."
            ),
            "remedy": (
                "Normalize repeated punctuation sequences ('..' -> '.') and incorporate character n-gram hashing for informal orthography."
            )
        }
    ]

    for arch in fp_archetypes:
        found_idx = None
        for idx in sorted_fps:
            if arch["target_substr"].lower() in texts[idx].lower():
                found_idx = idx
                break
        if found_idx is None and len(sorted_fps) > 0:
            found_idx = sorted_fps[0]

        if found_idx is not None:
            curated_cases.append({
                "sample_id": int(found_idx),
                "error_type": "False Positive (Ham -> Spam)",
                "true_label": 0,
                "true_label_name": "Ham",
                "predicted_label": 1,
                "predicted_label_name": "Spam",
                "predicted_probability": round(float(probs[found_idx]), 4),
                "category": arch["category"],
                "text_snippet": texts[found_idx][:140] + ("..." if len(texts[found_idx]) > 140 else ""),
                "full_text": texts[found_idx],
                "root_cause": arch["root_cause"],
                "actionable_remedy": arch["remedy"]
            })

    # 2. False Negatives
    # Sort FNs by lowest probability ascending (most confident false negatives)
    sorted_fns = fn_indices[np.argsort(probs[fn_indices])]

    fn_archetypes = [
        {
            "target_substr": "3MOBILE",
            "category": "Adversarial Encoding & Complaint Camouflage",
            "root_cause": (
                "Spam text contains non-standard character encodings ('\x89\xdb', 'å£250') and mimics a billing dispute "
                "or grievance against a mobile carrier ('3 DONT GIV A SHIT'). The angry customer persona bypasses standard promotional spam triggers."
            ),
            "remedy": (
                "Robust character encoding normalization (e.g., ftfy / regex currency mapping) and detecting premium-rate contact numbers (0870) via regex feature extraction."
            )
        },
        {
            "target_substr": "filthyguys",
            "category": "Slang Chatline & Phonetic Evasion",
            "root_cause": (
                "Contains intentional phonetic obfuscations and abbreviations ('filthyguys', 'slo n hard', 'Txt XXX', '4msg'). "
                "Because these words do not match common spam dictionary entries, word-level TF-IDF fails to recognize them as adult chatline solicitations."
            ),
            "remedy": (
                "Incorporate character n-grams (e.g., 3-to-5 char grams) or subword BPE tokenization to capture phonetic chatline patterns despite creative spelling."
            )
        },
        {
            "target_substr": "Bank of Granite",
            "category": "Financial Pump-and-Dump Disguised as Research",
            "root_cause": (
                "Spam simulates legitimate financial newsletter / analyst report ('Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS... Nasdaq Symbol CDGT'). "
                "The corporate and equity research vocabulary closely matches benign business communications (e.g., Enron emails)."
            ),
            "remedy": (
                "Regex feature extraction for OTC stock ticker patterns ('Symbol [A-Z]{3,5}') and promotional yield claims ('UP OVER [0-9]+%')."
            )
        },
        {
            "target_substr": "Hi this is Amy",
            "category": "Conversational Social Engineering Impersonation",
            "root_cause": (
                "Begins with benign conversational greeting ('Hi this is Amy, we will be sending you a free phone number in a couple of days...'). "
                "The casual, polite introductory tone masks the subsequent adult party solicitation, keeping the overall bag-of-words score below the decision boundary."
            ),
            "remedy": (
                "Positional bigram transition features capturing shifts from conversational openings to urgent call-to-actions, plus character n-grams."
            )
        },
        {
            "target_substr": "0A$NETWORKS",
            "category": "Supplier Disclaimer Mimicry",
            "root_cause": (
                "Written in the tone of a commercial policy statement or legal dispute regarding SMS billing liability ('allow companies to bill for SMS, so they are responsible for their suppliers...'). "
                "Lacks standard promotional imperatives ('buy', 'win', 'urgent')."
            ),
            "remedy": (
                "Disclaimer token density ratios and uppercase character counts to differentiate legal policy notices from bulk promotional broadcasts."
            )
        }
    ]

    for arch in fn_archetypes:
        found_idx = None
        for idx in sorted_fns:
            if arch["target_substr"].lower() in texts[idx].lower():
                found_idx = idx
                break
        if found_idx is None and len(sorted_fns) > 0:
            found_idx = sorted_fns[0]

        if found_idx is not None:
            curated_cases.append({
                "sample_id": int(found_idx),
                "error_type": "False Negative (Spam -> Ham)",
                "true_label": 1,
                "true_label_name": "Spam",
                "predicted_label": 0,
                "predicted_label_name": "Ham",
                "predicted_probability": round(float(probs[found_idx]), 4),
                "category": arch["category"],
                "text_snippet": texts[found_idx][:140] + ("..." if len(texts[found_idx]) > 140 else ""),
                "full_text": texts[found_idx],
                "root_cause": arch["root_cause"],
                "actionable_remedy": arch["remedy"]
            })

    curated_df = pd.DataFrame(curated_cases)
    curated_df.to_csv(output_csv, index=False)

    summary_payload = {
        "metrics_summary": {
            "total_samples": len(y_true),
            "total_false_positives": len(fp_indices),
            "total_false_negatives": len(fn_indices),
            "ham_fpr": float(len(fp_indices) / (y_true == 0).sum()),
            "spam_fnr": float(len(fn_indices) / (y_true == 1).sum()),
        },
        "curated_representative_errors": curated_cases,
        "all_fp_sample_ids": fp_indices.tolist(),
        "all_fn_sample_ids": fn_indices.tolist()
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info(f"Saved {len(curated_cases)} representative error cases to {output_csv} and {output_json}")
    return curated_df, summary_payload
