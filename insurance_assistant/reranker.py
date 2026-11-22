"""Neural Cross-Encoder Re-ranker module using Jina Reranker API."""

import httpx
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = config.top_k_results,
    max_per_section: int = 2,
) -> list:
    """Rerank a list of LangChain Document objects using Jina Neural Cross-Encoder with section diversity."""
    if not documents:
        return []

    if not config.ENABLE_RERANKING:
        logger.info("Re-ranking is disabled; returning top %d candidate(s)", top_k)
        return documents[:top_k]

    if not config.jina_key:
        logger.warning("JINA_API_KEY is not set; bypassing re-ranking and returning top %d candidate(s)", top_k)
        return documents[:top_k]

    # Cap candidates to conserve Jina API tokens (max 14 chunks)
    candidates = documents[:14]
    # Send balanced prefixes (650 chars) to Jina cross-encoder to capture deep exclusions (e.g. DNS at char 573)
    # while retaining the full LangChain Document text for delivery to the LLM
    doc_texts = [doc.page_content[:650] for doc in candidates]

    headers = {
        "Authorization": f"Bearer {config.jina_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.reranker_model_name,
        "query": query,
        "documents": doc_texts,
        "top_n": min(len(candidates), max(top_k, 8)),
    }

    logger.info(
        "Re-ranking %d candidate chunk(s) using %s (query: '%s', target top_k=%d, max_per_section=%d, token-optimized)",
        len(candidates),
        config.reranker_model_name,
        query,
        top_k,
        max_per_section,
    )

    try:
        response = httpx.post(
            JINA_RERANK_URL,
            headers=headers,
            json=payload,
            timeout=5.0,
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # Apply Section Diversity to prevent single-table crowding (e.g. Day Care Annexure)
            selected = []
            deferred = []
            section_counts = {}

            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)
                if idx is not None and 0 <= idx < len(candidates):
                    doc = candidates[idx]
                    doc.metadata["relevance_score"] = score
                    sec = doc.metadata.get("section", "DEFAULT")
                    count = section_counts.get(sec, 0)
                    if count < max_per_section:
                        selected.append(doc)
                        section_counts[sec] = count + 1
                    else:
                        deferred.append(doc)
                    if len(selected) == top_k:
                        break

            # If top_k not yet reached, fill remaining slots from deferred candidates
            if len(selected) < top_k and deferred:
                selected.extend(deferred[: top_k - len(selected)])

            logger.info("Successfully re-ranked and selected top %d diverse chunk(s)", len(selected))
            return selected

        logger.warning(
            "Jina Reranker API returned status %d (%s); falling back to raw candidates",
            response.status_code,
            response.text,
        )
        return documents[:top_k]
    except Exception as e:
        logger.warning("Error during neural re-ranking (%s); falling back to raw candidates", e)
        return documents[:top_k]

