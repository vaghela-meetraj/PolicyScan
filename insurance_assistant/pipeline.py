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


def build_vector_store_for_document(file_path: str = config.data_path):
    """Load an existing vector store or build and persist a new one from the document."""
    if vector_store_exists():
        print("Vector store index found. Loading existing vector store...")
        return load_vector_store()

    print(f"Loading document from: {file_path}")
    documents = load_document(file_path)

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    chunks = split_into_chunks(documents)

    print(f"Building vector store for {len(chunks)} chunks...")
    vector_store = build_vector_store(chunks)

    print("Saving vector store to disk...")
    save_vector_store(vector_store)
    print("Vector store saved successfully.")

    return vector_store


def build_insurance_assisstant(file_path: str = config.data_path):
    """Initialize and return the complete insurance assistant agent pipeline."""
    print("Checking API keys...")
    config.check_api_keys()

    vector_store = build_vector_store_for_document(file_path)

    print("Creating retriever and search tool...")
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    print("Initializing LLM and agent...")
    llm = get_llm()
    agent = create_insurance_agent(llm, [search_tool])
    print("Insurance assistant pipeline ready.")

    return agent


def ask(agent, question: str) -> str:
    """Send a question to the insurance assistant agent and return the response."""
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
    return response["messages"][-1].content
