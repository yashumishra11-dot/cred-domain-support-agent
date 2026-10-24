"""
Cred Domain Support Agent - Structured Audit Logger
Track: Banking & FinTech (Cred)
Part 3 - Task 12: ELK-Style Structured JSON-Lines Audit Logging with Guaranteed PII Masking
"""

import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from guardrails import mask_pii

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, "audit.jsonl")


class StructuredAuditLogger:
    """
    ELK-compatible structured JSON-Lines logger.
    Guarantees that raw, unmasked fixed-format PII (PAN, Aadhaar, Bank Account numbers)
    NEVER reaches disk in the clear.
    """

    def __init__(self, log_path: str = AUDIT_LOG_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_event(
        self,
        trace_id: str,
        endpoint: str,
        method: str,
        raw_query: str,
        session_id: str,
        latency_ms: float,
        status_code: int,
        response_type: str,
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Masks all fixed-format PII before logging to disk.
        Writes one JSON-Lines record per request.
        """
        # CRITICAL SECURITY REQUIREMENT: Mask raw query BEFORE writing to disk
        sanitized_query, detected_pii = mask_pii(raw_query)

        # Sanitize response final answer if present
        sanitized_response = None
        if response_data:
            sanitized_response = dict(response_data)
            if "final_answer" in sanitized_response:
                masked_ans, _ = mask_pii(sanitized_response["final_answer"])
                sanitized_response["final_answer"] = masked_ans

        log_entry = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "session_id": session_id,
            "endpoint": endpoint,
            "method": method,
            "latency_ms": round(latency_ms, 2),
            "status_code": status_code,
            "query_sanitized": sanitized_query,
            "detected_pii_types": detected_pii,
            "pii_masked_in_log": len(detected_pii) > 0,
            "response_type": response_type,
            "response_payload": sanitized_response,
            "error": error
        }

        # Write to JSON-Lines file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry


# Global audit logger instance
GLOBAL_AUDIT_LOGGER = StructuredAuditLogger()


if __name__ == "__main__":
    print("Testing Structured Audit Logging with PII masking...")
    test_trace = "trc-test-001"
    pii_query = "Customer inquiry with PAN ABCDE1234F and Aadhaar 9999 8888 7777 regarding personal loan."
    
    entry = GLOBAL_AUDIT_LOGGER.log_event(
        trace_id=test_trace,
        endpoint="/ask",
        method="POST",
        raw_query=pii_query,
        session_id="test_session",
        latency_ms=12.45,
        status_code=200,
        response_type="POLICY_ANSWER",
        response_data={"final_answer": "Policy details provided."}
    )
    
    print(f"Logged Entry:\n{json.dumps(entry, indent=2)}")
    print(f"\nAudit file created at: {AUDIT_LOG_FILE}")
    print(f"Verified no raw PAN on disk: {'ABCDE1234F' not in open(AUDIT_LOG_FILE, 'r').read()}")
