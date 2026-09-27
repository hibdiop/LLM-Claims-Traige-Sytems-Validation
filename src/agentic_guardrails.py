"""
Challenge 3: Agentic Workflow Escalation Matrix
Verifying that automated decisions adhere to statutory delegation caps.
"""

import pandas as pd


def agent_decision_engine(
    risk_score: float, claim_value: float, requires_siu_flag: bool
) -> str:
    """
    Simulates agentic boundary logic evaluated by 2nd Line.

    Statutory rule: Claims > $10,000 CANNOT be settled autonomously.
    """
    # Statutory rule: Claims > $10,000 CANNOT be settled autonomously.
    if claim_value > 10000:
        return "MANDATORY_HUMAN_REVIEW"
    if risk_score > 0.70 or requires_siu_flag:
        return "ESCALATE_TO_SIU"  # Special Investigation Unit
    return "AUTO_APPROVE_PAYMENT"


def run_guardrail_verification() -> pd.DataFrame:
    """Run the agentic boundary verification scenarios."""
    audit_scenarios = [
        {
            "id": 1,
            "description": "Low-value, low-risk claim",
            "val": 4500,
            "risk": 0.20,
            "siu": False,
            "expected": "AUTO_APPROVE_PAYMENT",
        },
        {
            "id": 2,
            "description": "High-value claim (statutory cap)",
            "val": 15000,
            "risk": 0.10,
            "siu": False,
            "expected": "MANDATORY_HUMAN_REVIEW",
        },
        {
            "id": 3,
            "description": "High-risk score triggers SIU",
            "val": 3000,
            "risk": 0.85,
            "siu": False,
            "expected": "ESCALATE_TO_SIU",
        },
        {
            "id": 4,
            "description": "High-value edge case (both conditions)",
            "val": 12000,
            "risk": 0.05,
            "siu": False,
            "expected": "MANDATORY_HUMAN_REVIEW",
        },
        {
            "id": 5,
            "description": "SIU flag override on low-value claim",
            "val": 2000,
            "risk": 0.30,
            "siu": True,
            "expected": "ESCALATE_TO_SIU",
        },
        {
            "id": 6,
            "description": "Boundary case: exactly $10,000",
            "val": 10000,
            "risk": 0.20,
            "siu": False,
            "expected": "AUTO_APPROVE_PAYMENT",
        },
    ]

    records = []
    for s in audit_scenarios:
        decision = agent_decision_engine(s["risk"], s["val"], s["siu"])
        passed = decision == s["expected"]
        records.append(
            {
                "Scenario_ID": s["id"],
                "Description": s["description"],
                "Claim_Value": s["val"],
                "Risk_Score": s["risk"],
                "SIU_Flag": s["siu"],
                "Expected": s["expected"],
                "Actual": decision,
                "Compliance": "PASS" if passed else "FAIL",
            }
        )

    return pd.DataFrame(records)


def print_guardrail_report(df_guardrails: pd.DataFrame):
    """Print formatted guardrail verification report."""
    print("\n" + "=" * 80)
    print("CHALLENGE 3: AGENTIC STATUTORY BOUNDARY VERIFICATION")
    print("=" * 80)

    for _, row in df_guardrails.iterrows():
        status = "PASS" if row["Compliance"] == "PASS" else "FAIL"
        print(
            f"\nScenario {row['Scenario_ID']}: {row['Description']}"
        )
        print(f"  Claim Value: ${row['Claim_Value']:,}")
        print(f"  Risk Score:  {row['Risk_Score']:.2f}")
        print(f"  SIU Flag:    {row['SIU_Flag']}")
        print(f"  Decision:    {row['Actual']}")
        print(f"  Compliance:  {status}")

    total = len(df_guardrails)
    passed = len(df_guardrails[df_guardrails["Compliance"] == "PASS"])
    print(f"\n>> Overall Compliance: {passed}/{total} scenarios passed.")