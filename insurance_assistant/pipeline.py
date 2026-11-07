from insurance_assistant import config
from insurance_assistant.document_loader import load_document
from insurance_assistant.splitter import split_into_chunks
from insurance_assistant.embeddings import get_embedding_model
from insurance_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever,
)
from insurance_assistant.tools import create_search_tool
from insurance_assistant.llm import get_llm
from insurance_assistant.agent import create_insurance_agent
from insurance_assistant.tracing import check_langsmith_tracing
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)


def build_vector_store_for_document(file_path: str = config.data_path):
    """Load an existing vector store or build and persist a new one from the document."""
    if vector_store_exists():
        logger.info("Vector store index found. Loading existing vector store from disk")
        return load_vector_store()

    logger.info("Vector store index not found. Building from document: %s", file_path)
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    logger.info("Vector store built and persisted successfully")

    return vector_store


def build_insurance_assistant(file_path: str = config.data_path):
    """Initialize and return the complete insurance assistant agent pipeline."""
    logger.info("Initializing insurance assistant pipeline")
    config.check_api_keys()
    check_langsmith_tracing()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)
    llm = get_llm()
    agent = create_insurance_agent(llm, [search_tool])
    logger.info("Insurance assistant pipeline ready")

    return agent


def ask(agent, question: str) -> str:
    """Send a question to the insurance assistant agent and return the response."""
    logger.info("Asking assistant question: %s", question)
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )
    answer = response["messages"][-1].content
    logger.info("Received assistant response (%d characters)", len(answer))
    logger.info("Received assistant response: %s", answer)
    return answer
