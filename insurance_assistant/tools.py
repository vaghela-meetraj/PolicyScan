from langchain.tools import tool
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def create_search_tool(retriever):
    """Create and return a policy search tool configured with the given retriever."""
    logger.info("Creating policy search tool")

    @tool
    def search_insurance_policy(question: str) -> str:
        """Search the insurance policy documents to find relevant information for the user query."""
        logger.info("Executing policy search for query: %s", question)
        matching_chunks = retriever.invoke(question)
        logger.info("Retrieved %d matching chunk(s)", len(matching_chunks))
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)

    return search_insurance_policy
