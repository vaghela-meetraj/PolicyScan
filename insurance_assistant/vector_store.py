# import os
# from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from insurance_assistant import config
from insurance_assistant.embeddings import get_embedding_model
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)


def build_vector_store(chunks):
    """Build and return a vector store from document chunks."""
    # logger.info("Building FAISS vector store from %d chunks", len(chunks))
    # embeddings_model = get_embedding_model()
    # return FAISS.from_documents(chunks, embeddings_model)
    logger.info("Building Qdrant vector store from %d chunks", len(chunks))
    embeddings_model = get_embedding_model()
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )


def save_vector_store(vector_store, path: str = "") -> None:
    """Save the vector store (Qdrant persists automatically in cloud)."""
    # logger.info("Saving FAISS vector store to directory: %s", path)
    # vector_store.save_local(path)
    pass


def load_vector_store():
    """Load and return an existing Qdrant vector store from cloud."""
    # logger.info("Loading FAISS vector store from directory: %s", path)
    # embeddings_model = get_embedding_model()
    # return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
    logger.info("Loading Qdrant vector store from collection: %s", config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embedding_model()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        collection_name=config.QDRANT_COLLECTION_NAME,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )


def vector_store_exists() -> bool:
    """Check if the vector store exists in Qdrant Cloud."""
    # exists = os.path.exists(os.path.join(path, "index.faiss"))
    # logger.info("Checking vector store existence at '%s': %s", path, exists)
    # return exists
    client = QdrantClient(
        url=config.QDRANT_URL, 
        api_key=config.QDRANT_API_KEY
        )
    exists = client.collection_exists(config.QDRANT_COLLECTION_NAME)
    logger.info("Checking Qdrant collection existence at '%s': %s", config.QDRANT_COLLECTION_NAME, exists)
    return exists


def get_retriever(vector_store, k: int = config.top_k_results):
    """Return a retriever configured to fetch the top-k relevant documents."""
    logger.info("Configuring retriever with top_k=%d", k)
    return vector_store.as_retriever(search_kwargs={"k": k})
