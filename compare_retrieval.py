"""CLI tool to compare raw vector retrieval vs neural re-ranking side-by-side."""

import sys
from insurance_assistant.vector_store import load_vector_store, get_retriever
from insurance_assistant.reranker import rerank_documents
from insurance_assistant import config

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is the waiting period for pre-existing diseases?"

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    print("\n[1/3] Loading Qdrant Cloud Vector Store...")
    vs = load_vector_store()
    retriever = get_retriever(vs, k=config.top_n_candidates)

    print(f"[2/3] Retrieving top {config.top_n_candidates} candidate chunks from Qdrant Cloud...")
    candidates = retriever.invoke(query)
    print(f"      Retrieved {len(candidates)} candidate chunks.")

    print("\n" + "-" * 70)
    print("STAGE 1: RAW VECTOR SEARCH RESULTS (Before Re-ranking: Top 5)")
    print("-" * 70)
    for i, doc in enumerate(candidates[:5]):
        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 110:
            snippet = snippet[:110] + "..."
        print(f"Rank {i+1}: {snippet}")

    print("\n[3/3] Running Neural Cross-Encoder (Jina Reranker v2)...")
    reranked = rerank_documents(query, candidates, top_k=config.top_k_results)

    print("\n" + "-" * 70)
    print("STAGE 2: NEURAL RE-RANKED RESULTS (After Cross-Encoder: Top 5)")
    print("-" * 70)
    for i, doc in enumerate(reranked):
        score = doc.metadata.get("relevance_score", 0.0)
        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 110:
            snippet = snippet[:110] + "..."
        print(f"Rank {i+1} [Score: {score:.4f}]: {snippet}")

    print("\n" + "=" * 70)
    print("Comparison complete.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

