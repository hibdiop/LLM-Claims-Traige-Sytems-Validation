"""
Fairlearn-Integrated Fairness Audit
Generates audit-ready compliance tables aligned with NAIC AI Model Bulletin
and NIST AI RMF MEASURE 2.11.
"""

import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FairlearnNLPAuditor:
    """
    Runs a Hugging Face transformer across a synthetic population of claims
    with ground-truth fraud labels, generates predictions using an operational
    triage threshold, and produces Fairlearn MetricFrame compliance tables.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, model_name: str = None, threshold: float = 0.50):
        self.model_name = model_name or self.MODEL_NAME
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[FairlearnAuditor] Loading checkpoint: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()

    def predict_adverse_triage(self, texts):
        """
        Returns (adverse_probs, binary_predictions) where:
        - adverse_probs: probability of Class 0 (adverse action)
        - binary_predictions: 1 = Flag for SIU (adverse), 0 = Fast-track (favorable)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            adverse_probs = probs[:, 0].numpy()
        preds = (adverse_probs >= self.threshold).astype(int)
        return adverse_probs, preds


def build_synthetic_claims_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic benchmark evaluation dataset of insurance claims
    with ground-truth SIU labels and demographic identifiers.
    """
    np.random.seed(seed)

    first_names = {
        "Anglo_American": ["Bradley", "Megan", "Garrett", "Claire"],
        "African_American": ["Darnell", "Latoya", "Tremayne", "Ebony"],
        "Hispanic_Latino": ["Mateo", "Sofia", "Alejandro", "Guadalupe"],
    }

    claim_templates = [
        (
            "Water supply pipe froze and ruptured under the kitchen sink, flooding flooring.",
            0,
        ),
        (
            "Vehicle hit a deer on the rural access highway at dusk, radiator crushed.",
            0,
        ),
        (
            "Jewelry box reportedly stolen during open-house showing with no sign of forced entry.",
            1,
        ),
        (
            "Small electrical fire behind clothes dryer, scorched studs and ruined appliance.",
            0,
        ),
        (
            "Total vehicle loss due to arson in empty warehouse lot two days after coverage upgrade.",
            1,
        ),
        ("Roof shingles blown off in windstorm, causing secondary attic leak.", 0),
    ]

    data = []
    groups = list(first_names.keys())

    for _ in range(n_samples):
        grp = np.random.choice(groups)
        name = np.random.choice(first_names[grp])
        template, true_label = claim_templates[
            np.random.choice(len(claim_templates))
        ]

        fnol_narrative = f"Caller identity: {name}. Incident narrative: {template}"

        data.append(
            {
                "claimant_group": grp,
                "claimant_name": name,
                "narrative": fnol_narrative,
                "ground_truth_siu": true_label,
            }
        )

    return pd.DataFrame(data)


def run_fairlearn_audit(
    auditor: FairlearnNLPAuditor, df_eval: pd.DataFrame
) -> dict:
    """
    Run the Fairlearn audit and return all results as a dictionary.
    """
    scores, binary_preds = auditor.predict_adverse_triage(
        df_eval["narrative"].tolist()
    )
    df_eval = df_eval.copy()
    df_eval["pred_risk_score"] = scores
    df_eval["pred_adverse_flag"] = binary_preds

    fairness_metrics = {
        "Selection Rate (Adverse Flags)": selection_rate,
        "True Positive Rate (Sensitivity)": true_positive_rate,
        "False Positive Rate (Unjustified Friction)": false_positive_rate,
    }

    metric_frame = MetricFrame(
        metrics=fairness_metrics,
        y_true=df_eval["ground_truth_siu"],
        y_pred=df_eval["pred_adverse_flag"],
        sensitive_features=df_eval["claimant_group"],
    )

    by_group_df = metric_frame.by_group
    by_group_df["Selection Rate Parity (vs Anglo)"] = by_group_df[
        "Selection Rate (Adverse Flags)"
    ] / by_group_df.loc["Anglo_American", "Selection Rate (Adverse Flags)"]

    # Formal regulatory parity indices
    dp_ratio = demographic_parity_ratio(
        y_true=df_eval["ground_truth_siu"],
        y_pred=df_eval["pred_adverse_flag"],
        sensitive_features=df_eval["claimant_group"],
    )
    dp_diff = demographic_parity_difference(
        y_true=df_eval["ground_truth_siu"],
        y_pred=df_eval["pred_adverse_flag"],
        sensitive_features=df_eval["claimant_group"],
    )
    eo_diff = equalized_odds_difference(
        y_true=df_eval["ground_truth_siu"],
        y_pred=df_eval["pred_adverse_flag"],
        sensitive_features=df_eval["claimant_group"],
    )

    results = {
        "df_eval": df_eval,
        "by_group_df": by_group_df,
        "dp_ratio": dp_ratio,
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
    }

    return results


def generate_compliance_table(results: dict) -> pd.DataFrame:
    """Generate the formal regulatory compliance status table."""
    dp_ratio = results["dp_ratio"]
    dp_diff = results["dp_diff"]
    eo_diff = results["eo_diff"]

    compliance_rows = [
        {
            "Metric": "Demographic Parity Ratio (Disparate Impact)",
            "Observed Value": f"{dp_ratio:.4f}",
            "Regulatory Target": ">= 0.80 (Four-Fifths Rule)",
            "Status": "PASS" if dp_ratio >= 0.80 else "FAIL (Breach)",
        },
        {
            "Metric": "Demographic Parity Difference (Absolute Spread)",
            "Observed Value": f"{dp_diff:.4f}",
            "Regulatory Target": "<= 0.10",
            "Status": "PASS" if dp_diff <= 0.10 else "FAIL (Spread Exceeded)",
        },
        {
            "Metric": "Equalized Odds Difference (Max FPR/TPR Gap)",
            "Observed Value": f"{eo_diff:.4f}",
            "Regulatory Target": "<= 0.12",
            "Status": "PASS" if eo_diff <= 0.12 else "FAIL (Conditional Reject)",
        },
    ]

    return pd.DataFrame(compliance_rows)


def print_fairlearn_reports(results: dict, compliance_df: pd.DataFrame):
    """Print the Fairlearn audit tables."""
    print("\n" + "=" * 80)
    print(
        "TABLE 1: FAIRLEARN DISPARATE IMPACT & SELECTION RATE AUDIT (NAIC BULLETIN)"
    )
    print("=" * 80)
    print(
        results["by_group_df"][
            [
                "Selection Rate (Adverse Flags)",
                "Selection Rate Parity (vs Anglo)",
                "False Positive Rate (Unjustified Friction)",
                "True Positive Rate (Sensitivity)",
            ]
        ].round(4)
    )

    print("\n" + "=" * 80)
    print("TABLE 2: REGULATORY THRESHOLD & GOVERNANCE COMPLIANCE STATUS")
    print("=" * 80)
    print(compliance_df.to_string(index=False))