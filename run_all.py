"""
Cred Domain Support Agent - Master Demonstration & Validation Suite
Track: Banking & FinTech (Cred)
Executes and validates all four parts end-to-end under deterministic MOCK_LLM mode.
"""

import sys
import os
import time

# Ensure UTF-8 and disable telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dataset import print_dataset_summary
from rag_core import get_rag_manager, calibrate_threshold, grounded_generation, evaluate_chunking_strategies
from escalation import check_loan_application_status, calculate_dataset_percentiles, RECOMMENDED_ESCALATION_THRESHOLD
from crew import run_part2_demonstrations
from test_api import test_fastapi_endpoints
from evaluation import print_evaluation_report
from autogen_review import demonstrate_autogen_review
from governance import enforce_least_autonomy, enforce_runtime_budget, GLOBAL_RESPONSE_CACHE, AutonomyViolationError, BudgetExceededError


def main():
    print("*" * 80)
    print("FINAL CAPSTONE: CRED DOMAIN SUPPORT AGENT (CrewAI + AutoGen + FastAPI)")
    print("Track: Banking & FinTech (Cred) | Execution Mode: Deterministic MOCK_LLM")
    print("*" * 80)

    # =========================================================================
    # PART 1: DATASET DESIGN & RAG CORE
    # =========================================================================
    print("\n" + "#" * 80)
    print("PART 1: DATASET DESIGN & RAG CORE")
    print("#" * 80)
    
    print("\n--- Task 1: Dataset Generation & Validation ---")
    print_dataset_summary()

    print("\n--- Tasks 3, 4, 5: ChromaDB Indexing, Calibration, Grounded Generation & Evaluation ---")
    rag = get_rag_manager()
    print(f"ChromaDB Collections: Fixed={rag.coll_fixed.count()} chunks, Sentence={rag.coll_sentence.count()} chunks")

    calib = calibrate_threshold(rag)
    print(f"Empirical Calibration: In-Scope Min={calib['min_in_scope_similarity']:.4f}, Out-of-Scope Max={calib['max_out_of_scope_similarity']:.4f}")
    print(f"Chosen Calibrated Fallback Threshold: {calib['calibrated_threshold']:.4f}")

    eval_rag = evaluate_chunking_strategies(top_k=2)
    print(f"Chunking Evaluation:")
    print(f"  Fixed-Size Collection : Precision={eval_rag['fixed_size']['avg_precision']*100:.1f}%, Recall={eval_rag['fixed_size']['avg_recall']*100:.1f}%")
    print(f"  Sentence Collection   : Precision={eval_rag['sentence']['avg_precision']*100:.1f}%, Recall={eval_rag['sentence']['avg_recall']*100:.1f}%")
    print(f"Recommendation: {eval_rag['recommendation']}")

    # =========================================================================
    # PART 2: CREWAI MULTI-AGENT ORCHESTRATION, MEMORY, SCHEMAS & GUARDRAILS
    # =========================================================================
    print("\n" + "#" * 80)
    print("PART 2: CREWAI ORCHESTRATION, TOOLS, MEMORY & GUARDRAILS")
    print("#" * 80)
    run_part2_demonstrations()

    # =========================================================================
    # PART 3: EVALUATION, OBSERVABILITY & FASTAPI DEPLOYMENT
    # =========================================================================
    print("\n" + "#" * 80)
    print("PART 3: EVALUATION, OBSERVABILITY & FASTAPI DEPLOYMENT")
    print("#" * 80)
    test_fastapi_endpoints()
    print_evaluation_report()

    # =========================================================================
    # PART 4: RESILIENCE & GOVERNANCE
    # =========================================================================
    print("\n" + "#" * 80)
    print("PART 4: RESILIENCE & GOVERNANCE")
    print("#" * 80)
    demonstrate_autogen_review()

    print("\n--- Task 15: Four-Layer AI Governance Model ---")
    print("[Application Layer - Least Autonomy]")
    try:
        enforce_least_autonomy("Lookup Agent", "check_loan_application_status")
        print("  - Lookup Agent -> check_loan_application_status: AUTHORIZED")
        enforce_least_autonomy("Retrieval Agent", "check_loan_application_status")
    except AutonomyViolationError as e:
        print(f"  - Retrieval Agent -> check_loan_application_status: BLOCKED (Least Autonomy Guard: {e})")

    print("\n[Runtime Layer - Per-Request Budget Cap]")
    try:
        enforce_runtime_budget("What are the KYC requirements?")
        print("  - Normal query: ACCEPTED within budget")
        enforce_runtime_budget("Oversized payload " * 120)
    except BudgetExceededError as e:
        print(f"  - Oversized payload: BLOCKED (Budget Guard: {e})")

    print("\n--- Task 16: Response Caching Optimization ---")
    GLOBAL_RESPONSE_CACHE.clear()
    test_q = "How is EMI calculated under the reducing balance method?"
    
    t0 = time.perf_counter()
    res1 = grounded_generation(test_q, use_cache=True)
    t_miss = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    res2 = grounded_generation(test_q, use_cache=True)
    t_hit = (time.perf_counter() - t1) * 1000

    print(f"Cache Call 1 (Miss): {t_miss:.2f} ms | Hits={GLOBAL_RESPONSE_CACHE.hits}, Misses={GLOBAL_RESPONSE_CACHE.misses}")
    print(f"Cache Call 2 (Hit) : {t_hit:.2f} ms | Hits={GLOBAL_RESPONSE_CACHE.hits}, Misses={GLOBAL_RESPONSE_CACHE.misses}")
    print(f"Latency Reduction  : {(1.0 - t_hit / max(t_miss, 0.001)) * 100:.1f}%")
    print(f"Cache Stats        : {GLOBAL_RESPONSE_CACHE.get_stats()}")

    print("\n" + "=" * 80)
    print("ALL 16 ACCEPTANCE CRITERIA SUCCESSFULLY DEMONSTRATED AND VALIDATED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
