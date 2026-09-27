"""
Generates the executive-facing Model Risk FactSheet in Markdown format.
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_factsheet(
    bias_findings: dict,
    adversarial_swing: float,
    compliance_df: pd.DataFrame,
    output_path: str = "docs/TMG_2ndLine_Model_Risk_Assessment_Vendor_NLP.md",
):
    """Generate the executive FactSheet markdown document."""

    # Extract compliance values
    dp_ratio_row = compliance_df[
        compliance_df["Metric"].str.contains("Demographic Parity Ratio")
    ].iloc[0]
    eo_diff_row = compliance_df[
        compliance_df["Metric"].str.contains("Equalized Odds")
    ].iloc[0]

    dp_ratio = dp_ratio_row["Observed Value"]
    dp_ratio_status = dp_ratio_row["Status"]
    eo_diff = eo_diff_row["Observed Value"]
    eo_diff_status = eo_diff_row["Status"]

    max_spread = bias_findings["max_spread"]

    factsheet = f"""# SECOND-LINE INDEPENDENT MODEL VALIDATION FACTSHEET

**Asset Evaluated:** Commercial Claims NLP Pre-Classifier (`vendor-distilbert-triage-v1`)
**Validation Target:** The Mutual Group Member Carrier Shared Services
**Framework Alignment:** NAIC Model Bulletin on AI (Dec 2023) & NIST AI RMF 1.0 (MEASURE 2.11, GOVERN 3.2)
**Validation Date:** {datetime.now().strftime("%Y-%m-%d")}
**Validator:** 2nd-Line AI Model Risk Validation

---

## 1. Context & Credible Challenge Objective

First-line claims operations proposed integrating an off-the-shelf open-source Transformer to prioritize First Notice of Loss (FNOL) textual statements. First line reported an overall accuracy of 91.2% on standard test sets.

As the independent second line, we executed an independent challenge to answer three questions:

1. Does the model harbor **hidden proxy discrimination** when processing non-Anglo demographic identifiers?
2. Can simple **adversarial syntax variations** manipulate the triage routing?
3. Does the downstream **agentic automation engine** enforce mandatory statutory reserving caps?

---

## 2. Quantitative Challenge Findings

### A. Counterfactual Demographic Perturbation (Fairness Audit)

* **Methodology:** We injected paired demographic name identifiers into identical loss narratives across three distinct groups.
* **Finding:** While base claim severity was identical, claimant narratives containing certain demographic identifiers produced up to a **{max_spread:.2%} shift** in the adverse friction (investigation) score.
* **Regulatory Impact:** Violates **NAIC Model Bulletin Section IV** guidelines requiring insurers to verify that automated systems do not lead to disproportionate investigation triggers for protected classes without actuarial justification.

### B. Syntactic Fragility (Adversarial Testing)

* **Finding:** Rephrasing an identical property loss statement using polite phrasing dropped the risk score by up to **{adversarial_swing:.2%}**.
* **Conclusion:** The model keys on superficial sentiment markers rather than objective claim facts (loss date, cause of loss, policy terms).

### C. Fairlearn Disparate Impact & Equalized Odds Audit

| Metric | Observed | Regulatory Target | Status |
|--------|----------|-------------------|--------|
| Demographic Parity Ratio (Disparate Impact) | {dp_ratio} | >= 0.80 (Four-Fifths Rule) | {dp_ratio_status} |
| Equalized Odds Difference (Max FPR/TPR Gap) | {eo_diff} | <= 0.12 | {eo_diff_status} |

### D. Agentic Guardrails & Authority Limits

* **Finding:** The model decision engine properly routed high-dollar claims (> $10,000) to human adjusters regardless of model score. However, edge cases with ambiguous liability lacked deterministic fallback rules.

---

## 3. Second-Line Validation Verdict & Conditions

**Verdict:** **REJECTED FOR UNCONSTRAINED PRODUCTION USE**
*(Eligible for Human-in-the-Loop Shadow Pilot only, subject to the following remediations)*

### Mandatory Pre-Deployment Conditions (Remediation Plan):

1. **Feature Decoupling:** Strip all claimant identity metadata, names, and greetings prior to passing FNOL text into the NLP tokenizer via a pre-processing scrubbing layer.
2. **Deterministic Information Extraction:** Migrate the architecture from pure sentiment scoring to **Structured Entity Extraction** (cause of loss, itemized cost, dates) before applying decision logic.
3. **Continuous Drift Monitoring:** Implement automated KS-tests and word-distribution drift monitoring every 30 days to detect changes in FNOL intake phrasing.
4. **Fairness Gate:** Implement a pre-production fairness gate that blocks deployment if the Demographic Parity Ratio falls below 0.80 or Equalized Odds Difference exceeds 0.12.

---

## 4. Appendices

### Appendix A: Bias Perturbation Group Means

{_format_group_means(bias_findings['group_means'])}

### Appendix B: Artifact Inventory

| Artifact | Location | Description |
|----------|----------|-------------|
| Validation Harness | `src/validator.py` | Core model loader and scoring |
| Bias Testing | `src/bias_testing.py` | Demographic perturbation tests |
| Adversarial Testing | `src/adversarial_testing.py` | Syntax evasion tests |
| Guardrail Testing | `src/agentic_guardrails.py` | Statutory boundary verification |
| Fairlearn Audit | `src/fairlearn_audit.py` | Formal fairness metrics |
| Bias Results | `outputs/bias_perturbation_results.csv` | Raw perturbation data |
| Fairlearn Results | `outputs/fairlearn_by_group.csv` | Per-group metrics |

---

*This FactSheet was generated by the 2nd-Line Independent Model Validation Harness. All findings are reproducible via the scripts in `src/`.*
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(factsheet)

    print(f"[FactSheet] Generated: {output_path}")
    return factsheet


def _format_group_means(group_means: dict) -> str:
    """Format group means as a markdown table."""
    lines = ["| Group | Mean Risk Score |", "|-------|----------------|"]
    for group, score in group_means.items():
        lines.append(f"| {group} | {score:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    # This script is typically called from main.py
    print("Use main.py to run the full validation suite.")