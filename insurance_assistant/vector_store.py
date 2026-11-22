from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from insurance_assistant import config
from insurance_assistant.embeddings import get_embedding_model
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)


def build_vector_store(chunks):
    """Build and return a vector store from document chunks."""
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
    pass


def load_vector_store():
    """Load and return an existing Qdrant vector store from cloud."""
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
    client = QdrantClient(
        url=config.QDRANT_URL, 
        api_key=config.QDRANT_API_KEY,
    )
    exists = client.collection_exists(config.QDRANT_COLLECTION_NAME)
    logger.info("Checking Qdrant collection existence at '%s': %s", config.QDRANT_COLLECTION_NAME, exists)
    return exists


def delete_vector_store_collection() -> bool:
    """Delete the existing collection in Qdrant Cloud if it exists."""
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    if client.collection_exists(config.QDRANT_COLLECTION_NAME):
        logger.info("Deleting existing Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
        client.delete_collection(config.QDRANT_COLLECTION_NAME)
        return True
    return False


def get_hybrid_retriever(
    vector_store,
    chunks: list,
    k: int = None,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
):
    """Return an EnsembleRetriever combining Qdrant dense vector search with BM25 sparse keyword search."""
    if k is None:
        k = config.top_n_candidates if config.ENABLE_RERANKING else config.top_k_results

    logger.info(
        "Configuring Hybrid EnsembleRetriever (dense_weight=%.2f, sparse_weight=%.2f, candidate_k=%d)",
        dense_weight,
        sparse_weight,
        k,
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )


def get_retriever(vector_store, k: int = None, chunks: list = None):
    """Return a retriever configured to fetch the top candidates (pre-rerank).

    If chunks are provided and ENABLE_HYBRID_SEARCH is enabled, returns a hybrid
    EnsembleRetriever (Qdrant Dense + BM25 Sparse). Otherwise returns the dense retriever.
    """
    if k is None:
        k = config.top_n_candidates if config.ENABLE_RERANKING else config.top_k_results

    if chunks and getattr(config, "ENABLE_HYBRID_SEARCH", True):
        return get_hybrid_retriever(vector_store, chunks=chunks, k=k)

    logger.info("Configuring dense retriever with candidate k=%d", k)
    return vector_store.as_retriever(search_kwargs={"k": k})
