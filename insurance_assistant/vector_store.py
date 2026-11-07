import os
from langchain_community.vectorstores import FAISS
from insurance_assistant import config
from insurance_assistant.embeddings import get_embedding_model

def build_vector_store(chunks):
    """Build and return a FAISS vector store from document chunks."""
    embeddings_model = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings_model)

def save_vector_store(vector_store, path : str = config.vector_store_path)-> None:
    """Save the FAISS vector store to the specified local directory."""
    vector_store.save_local(path)

def load_vector_store(path: str = config.vector_store_path):
    """Load and return a FAISS vector store from the specified local directory."""
    embeddings_model = get_embedding_model()
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)

def vector_store_exists(path : str = config.vector_store_path) -> bool:
    """Check if a FAISS vector store index exists at the specified path."""
    return os.path.exists(os.path.join(path,"index.faiss"))

def get_retriever(vector_store, k: int = config.top_k_results):
    """Return a retriever configured to fetch the top-k relevant documents."""
    return vector_store.as_retriever(search_kwargs={"k":k})

