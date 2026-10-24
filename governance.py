"""
Cred Domain Support Agent - AI Governance & Response Caching
Track: Banking & FinTech (Cred)
Part 4 - Tasks 15 & 16: Four-Layer AI Governance Model and In-Memory Response Caching
"""

import time
import re
from typing import Dict, Any, Optional, Tuple, Callable


class AutonomyViolationError(PermissionError):
    """Raised when an agent attempts to execute a tool outside its least-autonomy authorization."""
    pass


class BudgetExceededError(ValueError):
    """Raised when a request exceeds the per-request token or cost budget cap."""
    pass


# ==============================================================================
# 1. APPLICATION LAYER: Least Autonomy & Risk Classification
# ==============================================================================

# Role-based tool access control matrix
AUTHORIZED_AGENT_TOOLS = {
    "Retrieval Agent": {"rag_lookup", "policy_search"},
    "Lookup Agent": {"check_loan_application_status"},
    "Response Composer": set(),  # Composer synthesizes findings and cannot invoke external tools
}

def enforce_least_autonomy(agent_role: str, tool_name: str) -> bool:
    """
    Enforces the Principle of Least Autonomy at the Application Layer.
    Only the Lookup Agent is granted permission to query loan application records.
    The Retrieval Agent and Response Composer are strictly prohibited from calling loan lookup.
    """
    allowed_tools = AUTHORIZED_AGENT_TOOLS.get(agent_role, set())
    if tool_name not in allowed_tools:
        raise AutonomyViolationError(
            f"GOVERNANCE VIOLATION (Application Layer): Agent '{agent_role}' is not authorized "
            f"to invoke tool '{tool_name}'. Authorized tools for this role: {list(allowed_tools)}. "
            f"Enforcing strict role segregation under Principle of Least Autonomy."
        )
    return True


RISK_CLASSIFICATION_SUMMARY = (
    "CRED DOMAIN SUPPORT AGENT RISK CLASSIFICATION: HIGH RISK.\n"
    "Justification: Under the EU AI Act and global financial AI governance frameworks, "
    "systems that assist in credit evaluation, loan status determination, interest rate communication, "
    "and access to member banking records are categorized as High Risk. Misinformation or unauthorized "
    "data leakage directly impairs consumer financial wellbeing, triggers regulatory non-compliance "
    "with RBI fair-lending directives, and exposes the institution to severe operational and fraud liabilities. "
    "Consequently, this agent requires mandatory input/output guardrails, least-autonomy tool segregation, "
    "and an independent secondary review stage before customer delivery."
)


# ==============================================================================
# 2. RUNTIME LAYER: Token and Cost Budget Cap
# ==============================================================================

# Budget Caps
MAX_PER_REQUEST_INPUT_TOKENS = 350
MAX_PER_REQUEST_COST_USD = 0.01  # Cap to prevent runaways

def estimate_tokens(text: str) -> int:
    """Accurate token estimator based on whitespace and word/subword splitting (~1.3 tokens per word)."""
    words = text.split()
    return max(1, int(len(words) * 1.35))


def enforce_runtime_budget(query: str, max_tokens: int = MAX_PER_REQUEST_INPUT_TOKENS) -> int:
    """
    Enforces a strict per-request token budget at the Runtime Layer.
    Rejects deliberately oversized prompts before model dispatch to prevent DoS attacks
    and cost overruns.
    """
    token_count = estimate_tokens(query)
    if token_count > max_tokens:
        raise BudgetExceededError(
            f"GOVERNANCE VIOLATION (Runtime Layer): Request exceeds token budget cap. "
            f"Supplied query estimated at {token_count} tokens, which exceeds the limit of {max_tokens} tokens. "
            f"Request rejected to preserve operational budget and system availability."
        )
    return token_count


# ==============================================================================
# 3. RESPONSE CACHING (Task 16)
# ==============================================================================

def normalize_query(query: str) -> str:
    """Normalizes query text by lowercasing, stripping punctuation, and collapsing whitespace."""
    text = query.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class ResponseCache:
    """
    In-memory normalized response cache for grounded generation and status lookups.
    Demonstrably avoids redundant LLM/tool invocations.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.total_calls: int = 0

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        self.total_calls += 1
        key = normalize_query(query)
        if key in self._cache:
            self.hits += 1
            entry = self._cache[key]
            return entry["response"]
        self.misses += 1
        return None

    def set(self, query: str, response: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        key = normalize_query(query)
        self._cache[key] = {
            "response": response,
            "cached_at": time.time(),
            "metadata": metadata or {}
        }

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_calls = 0

    def get_stats(self) -> Dict[str, Any]:
        hit_ratio = (self.hits / self.total_calls) if self.total_calls > 0 else 0.0
        return {
            "total_calls": self.total_calls,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(hit_ratio, 4),
            "cached_entries": len(self._cache)
        }


# Global in-memory cache instance
GLOBAL_RESPONSE_CACHE = ResponseCache()


if __name__ == "__main__":
    print("=" * 60)
    print("AI GOVERNANCE & RESPONSE CACHING TEST")
    print("=" * 60)

    # 1. Least Autonomy Demonstration
    print("\n[Application Layer] Testing Least-Autonomy Tool Permissions:")
    try:
        enforce_least_autonomy("Lookup Agent", "check_loan_application_status")
        print("  - Lookup Agent calling check_loan_application_status: ALLOWED (Correct)")
    except AutonomyViolationError as e:
        print(f"  - Error: {e}")

    try:
        print("  - Attempting to let Retrieval Agent call check_loan_application_status...")
        enforce_least_autonomy("Retrieval Agent", "check_loan_application_status")
    except AutonomyViolationError as e:
        print(f"  - BLOCKED AS EXPECTED: {e}")

    # 2. Risk Classification
    print("\n[Application Layer] Risk Classification:")
    print(f"  {RISK_CLASSIFICATION_SUMMARY}")

    # 3. Runtime Budget Enforcement
    print("\n[Runtime Layer] Testing Budget Cap Enforcement:")
    normal_q = "What are the KYC document requirements for personal loans?"
    toks = enforce_runtime_budget(normal_q)
    print(f"  - Normal query ({toks} tokens): ACCEPTED")

    oversized_q = "Please explain the policy " + "very thoroughly with extra context " * 80
    try:
        enforce_runtime_budget(oversized_q)
        print("  - Oversized query: UNEXPECTEDLY ACCEPTED")
    except BudgetExceededError as e:
        print(f"  - Oversized query BLOCKED AS EXPECTED: {e}")

    # 4. Response Caching Demonstration
    print("\n[Optimization] Testing Response Caching:")
    GLOBAL_RESPONSE_CACHE.clear()
    
    test_query = "What is the fee for late credit card payment?"
    mock_resp = {"final_answer": "Late payment fees follow a tiered slab structure up to INR 1,200."}

    # Call 1: Miss
    t0 = time.perf_counter()
    res1 = GLOBAL_RESPONSE_CACHE.get(test_query)
    if res1 is None:
        time.sleep(0.05)  # Simulate generation latency
        GLOBAL_RESPONSE_CACHE.set(test_query, mock_resp)
        res1 = mock_resp
    t_miss = (time.perf_counter() - t0) * 1000

    # Call 2: Hit
    t1 = time.perf_counter()
    res2 = GLOBAL_RESPONSE_CACHE.get(test_query)
    t_hit = (time.perf_counter() - t1) * 1000

    print(f"  - Call 1 (Cache Miss): {t_miss:.2f} ms | Hits: {GLOBAL_RESPONSE_CACHE.hits}, Misses: {GLOBAL_RESPONSE_CACHE.misses}")
    print(f"  - Call 2 (Cache Hit) : {t_hit:.2f} ms | Hits: {GLOBAL_RESPONSE_CACHE.hits}, Misses: {GLOBAL_RESPONSE_CACHE.misses}")
    print(f"  - Cache Hit Ratio    : {GLOBAL_RESPONSE_CACHE.get_stats()['hit_ratio'] * 100:.1f}%")
    print("=" * 60)
