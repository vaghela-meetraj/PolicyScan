from langchain.tools import tool

def create_search_tool(retriever):
    """Create and return a policy search tool configured with the given retriever."""

    @tool
    def search_insurance_policy(question: str) -> str:
        """Search the insurance policy documents to find relevant information for the user query."""
        matching_chunks = retriever.invoke(question)
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)

    return search_insurance_policy
