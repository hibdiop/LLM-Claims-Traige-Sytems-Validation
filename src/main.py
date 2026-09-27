"""
TMG 2nd-Line Independent Model Validation - Main Orchestrator

This script runs the complete validation suite against a pre-trained
Hugging Face NLP triage model, simulating a vendor black-box audit.
"""

import os
import sys
import json
from datetime import datetime

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validator import VendorNLPTriageValidator
from src.bias_testing import (
    run_demographic_perturbation_test,
    generate_bias_summary,
    print_bias_report,
)
from src.adversarial_testing import (
    run_adversarial_syntax_test,
    print_adversarial_report,
)
from src.agentic_guardrails import (
    run_guardrail_verification,
    print_guardrail_report,
)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def main():
    print("=" * 80)
    print("TMG 2ND-LINE INDEPENDENT MODEL VALIDATION")
    print("Vendor NLP Triage Engine - Challenge Suite")
    print(f"Run Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    ensure_output_dirs()

    # Initialize validator
    validator = VendorNLPTriageValidator()

    # ================================================================
    # CHALLENGE 1: Demographic Perturbation Testing
    # ================================================================
    df_fairness = run_demographic_perturbation_test(validator)
    bias_findings = generate_bias_summary(df_fairness)
    print_bias_report(df_fairness, bias_findings)

    # Save outputs
    df_fairness.to_csv("outputs/bias_perturbation_results.csv", index=False)

    # ================================================================
    # CHALLENGE 2: Adversarial Syntax Testing
    # ================================================================
    df_adversarial = run_adversarial_syntax_test(validator)
    print_adversarial_report(df_adversarial)

    # Save outputs
    df_adversarial.to_csv("outputs/adversarial_results.csv", index=False)

    # ================================================================
    # CHALLENGE 3: Agentic Guardrail Verification
    # ================================================================
    df_guardrails = run_guardrail_verification()
    print_guardrail_report(df_guardrails)

    # Save outputs
    df_guardrails.to_csv("outputs/guardrail_results.csv", index=False)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Bias Perturbation Max Spread: {bias_findings['max_spread']:.4f}")
    print(f"Bias Threshold Exceeded: {bias_findings['threshold_exceeded']}")
    print(
        f"Adversarial Max Swing: {df_adversarial['Risk_Score'].max() - df_adversarial['Risk_Score'].min():.4f}"
    )
    print(
        f"Guardrail Compliance: {len(df_guardrails[df_guardrails['Compliance'] == 'PASS'])}/{len(df_guardrails)}"
    )
    print("\nOutputs saved to ./outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()