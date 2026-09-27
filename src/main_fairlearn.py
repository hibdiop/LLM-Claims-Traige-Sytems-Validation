"""
Main orchestrator for the Fairlearn-integrated fairness audit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fairlearn_audit import (
    FairlearnNLPAuditor,
    build_synthetic_claims_dataset,
    run_fairlearn_audit,
    generate_compliance_table,
    print_fairlearn_reports,
)


def main():
    print("=" * 80)
    print("TMG 2ND-LINE FAIRLEARN-INTEGRATED FAIRNESS AUDIT")
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)

    # Build synthetic population
    df_eval = build_synthetic_claims_dataset(n_samples=300, seed=42)
    print(f"\nSynthetic evaluation dataset: {len(df_eval)} claims")
    print(f"Ground-truth SIU rate: {df_eval['ground_truth_siu'].mean():.2%}")

    # Initialize auditor and run
    auditor = FairlearnNLPAuditor(threshold=0.50)
    results = run_fairlearn_audit(auditor, df_eval)

    # Generate compliance table
    compliance_df = generate_compliance_table(results)

    # Print reports
    print_fairlearn_reports(results, compliance_df)

    # Save outputs
    results["df_eval"].to_csv("outputs/fairlearn_eval_results.csv", index=False)
    results["by_group_df"].to_csv("outputs/fairlearn_by_group.csv")
    compliance_df.to_csv("outputs/fairlearn_compliance.csv", index=False)

    print("\n>> Fairlearn outputs saved to ./outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()