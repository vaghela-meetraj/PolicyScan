from langchain_community.embeddings import JinaEmbeddings
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embedding_model():
    """Returns the embedding model."""
    logger.info("Initializing embedding model: %s", config.embedding_model_name)
    return JinaEmbeddings(session=None, model_name=config.embedding_model_name)
