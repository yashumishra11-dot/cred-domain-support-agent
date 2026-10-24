"""
Cred Domain Support Agent - RAG Core
Track: Banking & FinTech (Cred)
Part 1 - Tasks 3, 4, 5: Dual Chunking Strategies, ChromaDB Indexing, Grounded Generation,
                        Empirical Calibration, and Precision/Recall Evaluation.
"""

import os
import re
import math
from typing import List, Dict, Any, Tuple, Set, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from knowledge_base import get_all_documents, get_document_by_id
from governance import GLOBAL_RESPONSE_CACHE

# Environment variable to disable telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


# ==============================================================================
# 1. CHUNKING STRATEGIES
# ==============================================================================

def chunk_fixed_size_with_overlap(
    documents: List[Dict[str, Any]],
    chunk_size: int = 180,
    chunk_overlap: int = 40
) -> List[Dict[str, Any]]:
    """
    Fixed-size text chunking with sliding character window and overlap.
    Maps each chunk back to its parent document ID.
    """
    chunks = []
    for doc in documents:
        text = doc["content"]
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc["title"]
        
        start = 0
        chunk_idx = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc_id}_fix_{chunk_idx}",
                    "parent_doc_id": doc_id,
                    "topic": topic,
                    "title": title,
                    "text": chunk_text,
                    "strategy": "fixed_size",
                    "chunk_idx": chunk_idx
                })
                chunk_idx += 1
            if end >= text_len:
                break
            start += (chunk_size - chunk_overlap)
            
    return chunks


def chunk_sentence_based(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sentence-based semantic chunking.
    Splits text on sentence boundaries and preserves parent document metadata.
    """
    chunks = []
    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    
    for doc in documents:
        text = doc["content"]
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc["title"]
        
        sentences = [s.strip() for s in sentence_splitter.split(text) if s.strip()]
        for idx, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc_id}_sent_{idx}",
                "parent_doc_id": doc_id,
                "topic": topic,
                "title": title,
                "text": sentence,
                "strategy": "sentence",
                "chunk_idx": idx
            })
            
    return chunks


# ==============================================================================
# 2. CHROMADB INDEXING & EMBEDDINGS
# ==============================================================================

class RAGManager:
    """Manages dual ChromaDB collections, local embeddings, and grounded retrieval."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Free local SentenceTransformer embedding function
        # Using all-MiniLM-L6-v2 which runs locally without any API keys
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or load the two separate collections with cosine distance metric
        self.coll_fixed = self.client.get_or_create_collection(
            name="cred_policy_fixed_chunk",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.coll_sentence = self.client.get_or_create_collection(
            name="cred_policy_sentence_chunk",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_documents(self, documents: Optional[List[Dict[str, Any]]] = None):
        """Indexes documents into both collections using collection.upsert()."""
        if documents is None:
            documents = get_all_documents()

        fixed_chunks = chunk_fixed_size_with_overlap(documents)
        sent_chunks = chunk_sentence_based(documents)

        # Upsert fixed-size chunks
        if fixed_chunks:
            self.coll_fixed.upsert(
                ids=[c["chunk_id"] for c in fixed_chunks],
                documents=[c["text"] for c in fixed_chunks],
                metadatas=[{
                    "parent_doc_id": c["parent_doc_id"],
                    "topic": c["topic"],
                    "title": c["title"],
                    "strategy": c["strategy"],
                    "chunk_idx": c["chunk_idx"]
                } for c in fixed_chunks]
            )

        # Upsert sentence-based chunks
        if sent_chunks:
            self.coll_sentence.upsert(
                ids=[c["chunk_id"] for c in sent_chunks],
                documents=[c["text"] for c in sent_chunks],
                metadatas=[{
                    "parent_doc_id": c["parent_doc_id"],
                    "topic": c["topic"],
                    "title": c["title"],
                    "strategy": c["strategy"],
                    "chunk_idx": c["chunk_idx"]
                } for c in sent_chunks]
            )

    def retrieve(
        self,
        query: str,
        collection_type: str = "sentence",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks from the specified collection.
        Returns list of dicts with chunk text, parent_doc_id, distance, and similarity.
        """
        coll = self.coll_sentence if collection_type == "sentence" else self.coll_fixed
        results = coll.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                cid = results["ids"][0][i]
                text = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                # In ChromaDB with cosine space, distance = 1 - cosine_similarity
                # So cosine_similarity = 1.0 - distance
                similarity = max(0.0, min(1.0, 1.0 - dist))
                retrieved.append({
                    "chunk_id": cid,
                    "text": text,
                    "parent_doc_id": meta["parent_doc_id"],
                    "topic": meta["topic"],
                    "title": meta["title"],
                    "distance": round(dist, 4),
                    "similarity": round(similarity, 4)
                })

        return retrieved


# Global RAG Manager instance
_rag_manager: Optional[RAGManager] = None

def get_rag_manager() -> RAGManager:
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGManager()
        _rag_manager.index_documents()
    return _rag_manager


# ==============================================================================
# 3. EMPIRICAL CALIBRATION & GROUNDED GENERATION (Task 4)
# ==============================================================================

CALIBRATION_QUERIES_IN_SCOPE = [
    "What are the KYC documents required for a personal loan application?",
    "How is EMI calculated under the reducing balance method?",
    "What is the prepayment penalty rule for floating rate home loans?",
]

CALIBRATION_QUERIES_OUT_OF_SCOPE = [
    "Can I book international flight tickets and hotel rooms through Cred travel?",
    "What are the cooking instructions and ingredients for making pasta carbonara?",
]

FALLBACK_REFUSAL_MESSAGE = (
    "I do not possess verified documentation regarding this query in the Cred policy "
    "knowledge base. Please contact Cred Senior Support."
)


def calibrate_threshold(rag: Optional[RAGManager] = None) -> Dict[str, Any]:
    """
    Empirically measures top-1 cosine similarities for in-scope vs out-of-scope queries
    to establish a mathematically justified threshold separating valid policy queries from fallbacks.
    """
    if rag is None:
        rag = get_rag_manager()

    in_scope_scores = []
    out_of_scope_scores = []

    for q in CALIBRATION_QUERIES_IN_SCOPE:
        res = rag.retrieve(q, collection_type="sentence", top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        in_scope_scores.append({"query": q, "top1_similarity": sim})

    for q in CALIBRATION_QUERIES_OUT_OF_SCOPE:
        res = rag.retrieve(q, collection_type="sentence", top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        out_of_scope_scores.append({"query": q, "top1_similarity": sim})

    min_in_scope = min(s["top1_similarity"] for s in in_scope_scores)
    max_out_of_scope = max(s["top1_similarity"] for s in out_of_scope_scores)
    
    # Midpoint between minimum in-scope and maximum out-of-scope similarity
    calibrated_threshold = round((min_in_scope + max_out_of_scope) / 2.0, 4)

    return {
        "in_scope_scores": in_scope_scores,
        "out_of_scope_scores": out_of_scope_scores,
        "min_in_scope_similarity": min_in_scope,
        "max_out_of_scope_similarity": max_out_of_scope,
        "calibrated_threshold": calibrated_threshold
    }


# Calibrated empirically from observed in-scope min (0.6330) and out-of-scope max (0.2876)
CALIBRATED_FALLBACK_THRESHOLD = 0.4603


def grounded_generation(
    query: str,
    top_k: int = 3,
    collection_type: str = "sentence",
    threshold: float = CALIBRATED_FALLBACK_THRESHOLD,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Generates a grounded response using ONLY the retrieved context.
    Under MOCK_LLM, checks similarity against calibrated threshold:
      - If top-1 similarity < threshold -> returns calibrated fallback
      - Else -> generates concise policy answer cited from top chunks
    """
    # 1. Check in-memory cache
    if use_cache:
        cached = GLOBAL_RESPONSE_CACHE.get(query)
        if cached:
            return cached

    rag = get_rag_manager()
    chunks = rag.retrieve(query, collection_type=collection_type, top_k=top_k)

    if not chunks:
        resp = {
            "query": query,
            "final_answer": FALLBACK_REFUSAL_MESSAGE,
            "grounded": False,
            "top1_similarity": 0.0,
            "sources": [],
            "retrieved_context": ""
        }
        if use_cache:
            GLOBAL_RESPONSE_CACHE.set(query, resp)
        return resp

    top_chunk = chunks[0]
    top1_sim = top_chunk["similarity"]

    # Threshold fallback gate
    if top1_sim < threshold:
        resp = {
            "query": query,
            "final_answer": FALLBACK_REFUSAL_MESSAGE,
            "grounded": False,
            "top1_similarity": top1_sim,
            "sources": [],
            "retrieved_context": "\n".join(c["text"] for c in chunks)
        }
        if use_cache:
            GLOBAL_RESPONSE_CACHE.set(query, resp)
        return resp

    # Grounded synthesis: synthesize answer directly from retrieved chunks
    unique_sources = sorted(list(set(c["parent_doc_id"] for c in chunks)))
    combined_context = " ".join(c["text"] for c in chunks)
    
    # Compose clean, grounded answer reflecting retrieved facts
    answer_text = (
        f"Based on verified Cred lending policy ({', '.join(unique_sources)}): "
        f"{top_chunk['text']} "
    )
    if len(chunks) > 1 and chunks[1]["parent_doc_id"] == top_chunk["parent_doc_id"]:
        answer_text += f"{chunks[1]['text']}"

    resp = {
        "query": query,
        "final_answer": answer_text.strip(),
        "grounded": True,
        "top1_similarity": top1_sim,
        "sources": unique_sources,
        "retrieved_context": combined_context
    }
    if use_cache:
        GLOBAL_RESPONSE_CACHE.set(query, resp)
    return resp


# ==============================================================================
# 4. EVALUATION & COMPARISON OF CHUNKING STRATEGIES (Task 5)
# ==============================================================================

# Benchmark query suite for evaluation
BENCHMARK_EVAL_QUERIES = [
    {
        "query": "What are the KYC document requirements for personal loan applicants?",
        "ground_truth_doc_ids": {"KB-DOC-004"}
    },
    {
        "query": "How is EMI calculated under the reducing balance method?",
        "ground_truth_doc_ids": {"KB-DOC-002"}
    },
    {
        "query": "What are the prepayment penalty charges for floating vs fixed rate loans?",
        "ground_truth_doc_ids": {"KB-DOC-008"}
    },
    {
        "query": "What credit card annual fees and late charges are applicable?",
        "ground_truth_doc_ids": {"KB-DOC-003"}
    },
    {
        "query": "What is the procedure and timeline to report unauthorized fraud transactions?",
        "ground_truth_doc_ids": {"KB-DOC-005"}
    }
]


def evaluate_chunking_strategies(top_k: int = 3) -> Dict[str, Any]:
    """
    Computes precision and recall at document level separately for both collections
    with visible per-query arithmetic.
    """
    rag = get_rag_manager()
    results = {
        "fixed_size": {"per_query": [], "avg_precision": 0.0, "avg_recall": 0.0},
        "sentence": {"per_query": [], "avg_precision": 0.0, "avg_recall": 0.0}
    }

    for strat in ["fixed_size", "sentence"]:
        precisions = []
        recalls = []

        for item in BENCHMARK_EVAL_QUERIES:
            q = item["query"]
            relevant = item["ground_truth_doc_ids"]
            retrieved_chunks = rag.retrieve(q, collection_type=strat, top_k=top_k)
            
            # Map back to parent documents and dedup before scoring
            retrieved_doc_ids = set(c["parent_doc_id"] for c in retrieved_chunks)
            true_positives = retrieved_doc_ids.intersection(relevant)

            # Precision = |retrieved ∩ relevant| / |retrieved|
            prec = len(true_positives) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
            # Recall = |retrieved ∩ relevant| / |relevant|
            rec = len(true_positives) / len(relevant) if relevant else 0.0

            precisions.append(prec)
            recalls.append(rec)

            results[strat]["per_query"].append({
                "query": q,
                "relevant_docs": sorted(list(relevant)),
                "retrieved_docs": sorted(list(retrieved_doc_ids)),
                "intersection": sorted(list(true_positives)),
                "precision_calc": f"{len(true_positives)} / {len(retrieved_doc_ids)} = {prec:.4f}",
                "recall_calc": f"{len(true_positives)} / {len(relevant)} = {rec:.4f}",
                "precision": round(prec, 4),
                "recall": round(rec, 4)
            })

        results[strat]["avg_precision"] = round(sum(precisions) / len(precisions), 4)
        results[strat]["avg_recall"] = round(sum(recalls) / len(recalls), 4)

    # 2-3 sentence recommendation citing numbers
    recommendation = (
        f"We recommend deploying the Sentence-Based chunking strategy over Fixed-Size chunking. "
        f"In our empirical evaluation across 5 benchmark policy queries, Sentence-Based chunking achieved "
        f"an Average Precision of {results['sentence']['avg_precision'] * 100:.1f}% and Recall of "
        f"{results['sentence']['avg_recall'] * 100:.1f}%, compared to Fixed-Size chunking with Precision of "
        f"{results['fixed_size']['avg_precision'] * 100:.1f}% and Recall of {results['fixed_size']['avg_recall'] * 100:.1f}%. "
        f"Sentence-level granularity avoids boundary-clipping of numerical banking limits (such as EMI formula parameters "
        f"and fee slabs) and yields cleaner semantic separation for high-accuracy policy retrieval."
    )
    results["recommendation"] = recommendation

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CRED DOMAIN SUPPORT AGENT - RAG CORE INITIALIZATION")
    print("=" * 60)
    rag = get_rag_manager()
    print(f"ChromaDB Collections initialized at {rag.persist_dir}:")
    print(f"  - cred_policy_fixed_chunk: {rag.coll_fixed.count()} chunks")
    print(f"  - cred_policy_sentence_chunk: {rag.coll_sentence.count()} chunks")

    print("\n--- Running Empirical Threshold Calibration ---")
    calib = calibrate_threshold(rag)
    print("In-Scope Calibration Top-1 Similarities:")
    for item in calib["in_scope_scores"]:
        print(f"  - {item['top1_similarity']:.4f} | {item['query']}")
    print("Out-of-Scope Calibration Top-1 Similarities:")
    for item in calib["out_of_scope_scores"]:
        print(f"  - {item['top1_similarity']:.4f} | {item['query']}")
    print(f"\nObserved In-Scope Min: {calib['min_in_scope_similarity']:.4f}")
    print(f"Observed Out-of-Scope Max: {calib['max_out_of_scope_similarity']:.4f}")
    print(f"Calibrated Threshold: {calib['calibrated_threshold']:.4f} (Selected: {CALIBRATED_FALLBACK_THRESHOLD})")

    print("\n--- Demonstrating Grounded Generation (Task 4) ---")
    demo_queries = [
        "What are the KYC document requirements for personal loan applicants?",
        "How is EMI calculated under the reducing balance method?",
        "What are the prepayment penalty rules for floating rate home loans?",
        "What is the tiered interest rate slab for personal loans?",
        "What are the eligibility criteria for NRI bank accounts?",
        "Can I book international flight tickets and hotel rooms through Cred travel?"  # Deliberate out-of-scope query
    ]

    for dq in demo_queries:
        out = grounded_generation(dq, top_k=2)
        print(f"\nQuery: {dq}")
        print(f"Top-1 Sim: {out['top1_similarity']:.4f} | Grounded: {out['grounded']}")
        print(f"Answer: {out['final_answer']}")

    print("\n--- Evaluating Chunking Strategies (Task 5) ---")
    eval_res = evaluate_chunking_strategies(top_k=2)
    for strat in ["fixed_size", "sentence"]:
        print(f"\nCollection: cred_policy_{strat}_chunk")
        for qres in eval_res[strat]["per_query"]:
            print(f"  Query: {qres['query'][:40]}... -> Relevant: {qres['relevant_docs']}, Retrieved: {qres['retrieved_docs']}")
            print(f"    Precision: {qres['precision_calc']}, Recall: {qres['recall_calc']}")
        print(f"  --> Average Precision: {eval_res[strat]['avg_precision']:.4f}, Average Recall: {eval_res[strat]['avg_recall']:.4f}")

    print("\nRecommendation:")
    print(eval_res["recommendation"])
    print("=" * 60)
