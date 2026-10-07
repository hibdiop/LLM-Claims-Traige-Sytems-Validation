# 2nd-Line Independent Model Validation

**A validation framework for auditing pre-trained NLP/LLM models used in insurance claims triage.**

Integrating **NAIC AI Model Bulletin (Dec 2023)** and **NIST AI RMF 1.0** to independently challenge a vendor-supplied NLP model before production deployment. 



## The Problem

When a customer files an insurance claim, known as the First Notice of Loss (FNOL), the claims operations team must decide how quickly to handle it and where to send it. To speed this up, the business team wanted to implement an off-the-shelf, open-source AI language model from a third-party vendor to automatically read customer statements and prioritize urgent or high-risk claims.

**The vendor --->** claimed a 91.2% accuracy rate on standard benchmark tests. However, high accuracy on canned benchmarks does not guarantee the model is fair, secure, or compliant with insurance law.

**As the independent risk audit team (second-line model validation) --->**  our job is to challenge the vendor's claims before real customers are affected. 

We investigated three critical failure modes:

-  ***Hidden Demographic Bias:*** Does the AI unintentionally penalize claimants based on names or cultural markers—such as treating identical claims differently just because of a non-Western name?

-  ***Vulnerability to Manipulation:*** Can a user trick the model into misclassifying a claim simply by changing the wording, tone, or phrasing (e.g., using overly polite text to mask severe damage)?

- ***Guardrail Violations:*** When the AI triggers automated downstream actions (like reserving payout funds), does the system strictly obey legal limits and safety caps, or does it risk making unauthorized financial commitments?



## The Solution

An end-to-end validation framework designed to stress-test the vendor's AI before it reaches production:

- ***Independent "Black-Box" Testing:*** Evaluates a standard pre-trained AI language model (distilbert-base-uncased-finetuned-sst-2-english) without relying on the vendor's internal claims or assumptions.

- ***Name & Demographic Bias Testing:*** Swaps demographic markers (such as ethnic names) across identical insurance claims to verify whether the AI treats similar customers unequally (counterfactual perturbation).

- ***Manipulation & Evasion Stress-Tests:*** Rewrites claims using polite or altered phrasing (adversarial syntax) to see if the AI can be tricked into underestimating damage severity.

- ***Automated Guardrail Verification:*** Tests whether downstream automated actions respect legal and operational limits—ensuring high-value decisions are always flagged for mandatory human review.

- ***Regulatory Fairness Metrics:*** Uses industry-standard fairness algorithms (Fairlearn Disparate Impact and Equalized Odds) to generate defensible mathematical proof of compliance with insurance regulations.

- ***Executive Model Risk FactSheet:*** Compiles all findings, risk scores, and deployment recommendations into a clear summary for leadership and regulators.



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
nlp-validation/
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

## Framework Alignment

| Framework | Requirement | How This Project Addresses It |
|---|---|---|
| NAIC AI Model Bulletin Section IV | Test for unfair discrimination | Counterfactual demographic perturbation + Fairlearn metrics |
| NIST AI RMF MEASURE 2.11 | Evaluate fairness and bias | Disparate Impact Ratio, Equalized Odds Difference |
| NIST AI RMF GOVERN 3.2 | Document risk decisions | Executive FactSheet with remediation conditions |


## Technical Stack 
- **Python 3.9+:** The core programming language used to build and orchestrate the entire testing pipeline.
- **Hugging Face Transformers:** A standard AI library used to download and load the vendor's pre-trained language model into our testing environment.
- **PyTorch:** The underlying computation engine that runs the model and generates risk scores from the claim text.
- **Fairlearn:** A specialized toolkit used to calculate mathematically grounded bias and fairness metrics across demographic groups.
- **Pandas / NumPy:** Data organization libraries used to structure, clean, and manipulate the batches of test claims and results.
- **scikit-learn:** A machine learning utility library used to compute baseline performance, error rates, and evaluation statistics.
