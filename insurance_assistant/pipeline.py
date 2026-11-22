from insurance_assistant import config
from insurance_assistant.document_loader import load_document
from insurance_assistant.splitter import split_into_chunks
from insurance_assistant.embeddings import get_embedding_model
from insurance_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    delete_vector_store_collection,
    get_retriever,
)
from insurance_assistant.tools import create_search_tool
from insurance_assistant.llm import get_llm
from insurance_assistant.agent import create_insurance_agent
from insurance_assistant.tracing import check_langsmith_tracing
from insurance_assistant import guardrails
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

_cached_chunks = None
_cached_path = None


def get_document_chunks(file_path: str = config.data_path) -> list:
    """Return cached header-aware document chunks, loading and splitting on first call."""
    global _cached_chunks, _cached_path
    if _cached_chunks is None or _cached_path != file_path:
        logger.info("Loading and preparing chunks for document: %s", file_path)
        documents = load_document(file_path)
        _cached_chunks = split_into_chunks(documents)
        _cached_path = file_path
    return _cached_chunks


def build_vector_store_for_document(file_path: str = config.data_path, force_reindex: bool = False):
    """Load an existing vector store or build and persist a new one from the document."""
    if force_reindex:
        logger.info("Force reindex requested. Deleting existing Qdrant collection if present")
        delete_vector_store_collection()

    if vector_store_exists() and not force_reindex:
        logger.info("Qdrant collection '%s' found. Loading existing vector store from Qdrant Cloud", config.QDRANT_COLLECTION_NAME)
        return load_vector_store()

    logger.info("Qdrant collection '%s' not found. Ingesting from document: %s", config.QDRANT_COLLECTION_NAME, file_path)
    chunks = get_document_chunks(file_path)
    vector_store = build_vector_store(chunks)
    logger.info("Vector store built and indexed in Qdrant Cloud successfully")

    return vector_store


def build_insurance_assistant(file_path: str = config.data_path, force_reindex: bool = False):
    """Initialize and return the complete insurance assistant agent pipeline with hybrid retrieval."""
    logger.info("Initializing insurance assistant pipeline")
    config.check_api_keys()
    check_langsmith_tracing()

    vector_store = build_vector_store_for_document(file_path, force_reindex=force_reindex)
    chunks = get_document_chunks(file_path)
    retriever = get_retriever(vector_store, chunks=chunks)
    search_tool = create_search_tool(retriever)
    llm = get_llm()
    agent = create_insurance_agent(llm, [search_tool])
    logger.info("Insurance assistant pipeline ready")

    return agent


def ask(agent, question: str, chat_history: list = None, return_context: bool = False):
    """Send a question to the insurance assistant agent and return the response (and optionally retrieved context)."""
    logger.info("Asking assistant question: %s", question)

    # Input guardrail check
    is_input_safe, input_reason = guardrails.check_input(question)
    if not is_input_safe:
        logger.warning("Input blocked by guardrails. Reason: %s", input_reason)
        return (guardrails.REFUSAL_MESSAGE, "") if return_context else guardrails.REFUSAL_MESSAGE

    messages = []
    if chat_history:
        for msg in chat_history:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    # Extract retrieved context from tool messages if present
    retrieved_context = ""
    tool_contexts = []
    for msg in response.get("messages", []):
        if getattr(msg, "type", None) == "tool" and msg.content:
            tool_contexts.append(msg.content)
    if tool_contexts:
        retrieved_context = "\n\n".join(tool_contexts)

    # Output guardrail check
    is_output_safe, output_reason = guardrails.check_output(answer)
    if not is_output_safe:
        logger.warning("Output blocked by guardrails. Reason: %s", output_reason)
        return (guardrails.REFUSAL_MESSAGE, retrieved_context) if return_context else guardrails.REFUSAL_MESSAGE

    logger.info("Received assistant response (%d characters)", len(answer))
    logger.info("Received assistant response: %s", answer)
    return (answer, retrieved_context) if return_context else answer
