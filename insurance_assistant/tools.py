from langchain.tools import tool
from insurance_assistant import config
from insurance_assistant.reranker import rerank_documents
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def create_search_tool(retriever):
    """Create and return a policy search tool configured with the given retriever and neural re-ranker."""
    logger.info("Creating policy search tool")

    @tool
    def search_insurance_policy(question: str) -> str:
        """Search the insurance policy documents to find relevant information for the user query."""
        logger.info("Executing policy search for query: %s", question)
        matching_chunks = retriever.invoke(question)
        logger.info("Retrieved %d candidate chunk(s) from vector store", len(matching_chunks))
        if not matching_chunks:
            return "No relevant policy clauses or coverage details found in the policy documents."

        # Deduplicate candidates across retriever results
        seen = set()
        unique_chunks = []
        for chunk in matching_chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)

        if config.ENABLE_RERANKING:
            unique_chunks = rerank_documents(question, unique_chunks, top_k=config.top_k_results)
            logger.info("Filtered to %d re-ranked chunk(s)", len(unique_chunks))

        return "\n\n".join(chunk.page_content for chunk in unique_chunks)

    return search_insurance_policy
