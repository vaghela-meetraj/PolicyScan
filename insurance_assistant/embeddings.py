from langchain_community.embeddings import JinaEmbeddings
from insurance_assistant import config

def get_embedding_model():
    """ Returns the embedding model """
    return JinaEmbeddings(session=None, model_name=config.embedding_model_name)