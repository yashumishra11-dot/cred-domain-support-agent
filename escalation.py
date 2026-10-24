"""
Cred Domain Support Agent - Escalation Engine & Status Lookup Tool
Track: Banking & FinTech (Cred)
Part 2 - Task 6: Designed Escalation Score and Loan Application Status Lookup
"""

from typing import Dict, Any, Optional
import numpy as np
from dataset import LOAN_APPLICATIONS

# Lookup dictionary for fast O(1) retrieval
_APPLICATIONS_BY_ID = {app["record_id"]: app for app in LOAN_APPLICATIONS}


def calculate_dataset_percentiles() -> Dict[str, float]:
    """Computes distribution percentiles for days_since_created across the dataset."""
    days = [app["days_since_created"] for app in LOAN_APPLICATIONS]
    return {
        "p50": float(np.percentile(days, 50)),
        "p75": float(np.percentile(days, 75)),
        "p80": float(np.percentile(days, 80)),
        "p90": float(np.percentile(days, 90)),
        "min": float(min(days)),
        "max": float(max(days)),
    }


def compute_escalation_score(flagged_for_fraud_review: bool, days_since_created: int) -> float:
    """
    Computes a designed escalation score in [0, 1].
    
    Formula:
        escalation_score = (W_FRAUD * fraud_signal) + (W_RECENCY * recency_signal)
        
    Where:
        - fraud_signal = 1.0 if flagged_for_fraud_review else 0.0
        - recency_signal = min(days_since_created / 30.0, 1.0)
        - W_FRAUD = 0.65
        - W_RECENCY = 0.35
        - W_FRAUD + W_RECENCY = 1.00
        
    Justification & Threshold:
        - Fraud review is a critical compliance trigger: any flagged application automatically receives
          a baseline score of at least 0.650, which immediately clears our recommended escalation threshold (0.60).
        - For non-fraud applications, aging introduces operational risk (unresolved customer backlog).
          The 80th percentile of days_since_created in our dataset is 24 days (normalized recency = 24/30 = 0.80).
          0.35 * 0.80 = 0.280.
        - Recommended Escalation Threshold = 0.50:
          * Any fraud-flagged application (score >= 0.65) is automatically escalated to Senior Risk & Compliance.
          * Clean applications with normal processing time (score < 0.35) proceed through standard automated resolution.
    """
    w_fraud = 0.65
    w_recency = 0.35
    
    fraud_signal = 1.0 if flagged_for_fraud_review else 0.0
    recency_signal = min(max(days_since_created, 0) / 30.0, 1.0)
    
    score = (w_fraud * fraud_signal) + (w_recency * recency_signal)
    return round(float(score), 4)


RECOMMENDED_ESCALATION_THRESHOLD = 0.50


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """
    Looks up a loan application record by ID and computes the designed escalation score.
    
    Args:
        record_id: The unique identifier of the loan application (e.g. 'CRD-APP-1001')
        
    Returns:
        dict: Contains record_id, status, loan_amount_inr, category, days_since_created,
              flagged_for_fraud_review, escalation_score, requires_escalation, and message.
    """
    record_id_clean = record_id.strip().upper()
    app = _APPLICATIONS_BY_ID.get(record_id_clean)
    
    if not app:
        return {
            "error": True,
            "message": f"Application record '{record_id}' not found in the lending database.",
            "record_id": record_id,
            "status": "Unknown",
            "loan_amount_inr": 0,
            "escalation_score": 0.0,
            "requires_escalation": False
        }
        
    score = compute_escalation_score(
        flagged_for_fraud_review=app["flagged_for_fraud_review"],
        days_since_created=app["days_since_created"]
    )
    
    requires_escalation = score >= RECOMMENDED_ESCALATION_THRESHOLD
    
    return {
        "error": False,
        "record_id": app["record_id"],
        "category": app["category"],
        "status": app["status"],
        "loan_amount_inr": app["loan_amount_inr"],
        "days_since_created": app["days_since_created"],
        "flagged_for_fraud_review": app["flagged_for_fraud_review"],
        "escalation_score": score,
        "requires_escalation": requires_escalation,
        "message": (
            f"Application {app['record_id']} ({app['category']}) is currently '{app['status']}' "
            f"for INR {app['loan_amount_inr']:,}. Escalation score: {score:.4f} "
            f"({'ESCALATION REQUIRED' if requires_escalation else 'Standard Handling'})."
        )
    }


if __name__ == "__main__":
    percentiles = calculate_dataset_percentiles()
    print("Dataset Recency Distribution Percentiles:")
    for k, v in percentiles.items():
        print(f"  {k}: {v}")
        
    print("\nSample Status Lookups:")
    # Test first 3 applications
    for sample in LOAN_APPLICATIONS[:3]:
        res = check_loan_application_status(sample["record_id"])
        print(f"- {res['record_id']} | Status: {res['status']:12} | Amt: INR {res['loan_amount_inr']:>9,} | "
              f"Fraud: {str(res['flagged_for_fraud_review']):5} | Days: {res['days_since_created']:2} | "
              f"Escalation Score: {res['escalation_score']:.4f} | Escalate: {res['requires_escalation']}")
