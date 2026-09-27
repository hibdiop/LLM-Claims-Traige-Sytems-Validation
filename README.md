# 2nd-Line Independent Model Validation

**A validation harness for auditing pre-trained NLP/LLM models used in insurance claims triage.**

Integrating **NAIC AI Model Bulletin (Dec 2023)** and **NIST AI RMF 1.0** to independently challenge a vendor-supplied NLP model before production deployment. 

---

## The Problem

First-line claims operations proposed integrating an off-the-shelf open-source Transformer to prioritize First Notice of Loss (FNOL) textual statements. The vendor reported 91.2% accuracy on standard test sets.

As the independent second line, we asked three questions:

1. Does the model harbor **hidden proxy discrimination** when processing non-Anglo demographic identifiers?
2. Can simple **adversarial syntax variations** manipulate the triage routing?
3. Does the downstream **agentic automation engine** enforce mandatory statutory reserving caps?

---

## The Solution

A complete validation harness that:

- Loads a pre-trained Hugging Face checkpoint (`distilbert-base-uncased-finetuned-sst-2-english`) and treats it as a vendor black-box.
- Runs **counterfactual demographic perturbation tests** to detect proxy bias.
- Runs **adversarial syntax tests** to measure robustness against evasion.
- Verifies **agentic guardrails** against statutory delegation caps.
- Produces **Fairlearn Disparate Impact and Equalized Odds** metrics for audit-defensible compliance.
- Generates an executive-facing **Model Risk FactSheet**.

---

## Key Findings

| Challenge | Finding | Regulatory Impact |
|-----------|---------|-------------------|
| Demographic Perturbation | Up to 14.2% risk score variance driven purely by claimant name | NAIC Model Bulletin Section IV violation |
| Adversarial Syntax | Risk score dropped from 0.94 to 0.31 via polite phrasing | Model keys on sentiment, not claim facts |
| Fairlearn Disparate Impact | Demographic Parity Ratio = 0.78 (below 0.80 threshold) | Four-Fifths Rule breach |
| Agentic Guardrails | High-value claims properly routed to human review | PASS |

**Verdict:** REJECTED FOR UNCONSTRAINED PRODUCTION USE. Eligible for Human-in-the-Loop Shadow Pilot only, subject to remediation.

---

## Project Structure
```text
tmg-nlp-validation/
├── src/
│   ├── validator.py              # Core model loader and scoring
│   ├── bias_testing.py           # Demographic perturbation tests
│   ├── adversarial_testing.py    # Syntax evasion tests
│   ├── agentic_guardrails.py     # Statutory boundary verification
│   ├── fairlearn_audit.py        # Formal fairness metrics
│   ├── generate_factsheet.py     # Executive artifact generator
│   └── main.py                   # Full pipeline orchestrator
│
├── requirements.txt
├── .gitignore
└── README.md
```



---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/tmg-nlp-validation.git
cd tmg-nlp-validation

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full validation suite
python src/main.py
```

## Framework Alignment

| Framework | Requirement | How This Project Addresses It |
|---|---|---|
| NAIC AI Model Bulletin Section IV | Test for unfair discrimination | Counterfactual demographic perturbation + Fairlearn metrics |
| NIST AI RMF MEASURE 2.11 | Evaluate fairness and bias | Disparate Impact Ratio, Equalized Odds Difference |
| NIST AI RMF GOVERN 3.2 | Document risk decisions | Executive FactSheet with remediation conditions |


##Technical Stack 
- Python 3.9+
- Hugging Face Transformers — Pre-trained model loading
- PyTorch — Model inference
- Fairlearn — Fairness metrics and MetricFrame
- Pandas / NumPy — Data manipulation
- scikit-learn — Metric utilities
