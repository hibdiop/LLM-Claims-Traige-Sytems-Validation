"""
Challenge 2: Adversarial Syntactic Robustness & Evasion
Testing model stability against slight phrasing tweaks.
"""

import pandas as pd
from src.validator import VendorNLPTriageValidator


def run_adversarial_syntax_test(validator: VendorNLPTriageValidator) -> pd.DataFrame:
    """
    Tests whether the model's risk score can be manipulated by superficial
    phrasing changes while the objective claim facts remain identical.
    """
    adversarial_pairs = [
        (
            "Direct Statement",
            "Water leak ruined my hardwood floor and drywall completely.",
        ),
        (
            "Polite/Flattering Syntax",
            "Dear wonderful team, fortunately a minor burst pipe only ruined the hardwood floor and drywall.",
        ),
        (
            "Jargon/Obfuscation",
            "Substantial hydraulic breach occurred, resulting in structural cellulose degradation.",
        ),
        (
            "Minimizing Language",
            "Just a tiny bit of water got on the floor, nothing to worry about really.",
        ),
        (
            "Catastrophizing Language",
            "Total devastation! The entire house is ruined and we may never recover!",
        ),
    ]

    records = []
    for label, text in adversarial_pairs:
        score = validator.get_risk_scores([text])[0]
        records.append(
            {
                "Variation_Type": label,
                "Input_Text": text,
                "Risk_Score": score,
                "Binary_Decision": "ESCALATE" if score >= 0.50 else "FAST_TRACK",
            }
        )

    return pd.DataFrame(records)


def print_adversarial_report(df_adversarial: pd.DataFrame):
    """Print formatted adversarial testing report."""
    print("\n" + "=" * 80)
    print("CHALLENGE 2: ADVERSARIAL SYNTACTIC ROBUSTNESS TEST")
    print("=" * 80)

    for _, row in df_adversarial.iterrows():
        print(f"\n[{row['Variation_Type']}]")
        print(f"  Risk Score: {row['Risk_Score']:.4f}")
        print(f"  Decision:   {row['Binary_Decision']}")
        print(f"  Input:      '{row['Input_Text'][:80]}...'")

    # Identify max swing
    max_score = df_adversarial["Risk_Score"].max()
    min_score = df_adversarial["Risk_Score"].min()
    swing = max_score - min_score

    print(f"\n>> Maximum Score Swing: {swing:.4f}")
    if swing > 0.30:
        print(
            ">> FINDING: Model is highly sensitive to superficial syntax changes."
        )
        print(
            ">> CONCLUSION: Model keys on sentiment markers, not objective claim facts."
        )