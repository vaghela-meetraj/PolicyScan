from langchain_community.document_loaders import TextLoader
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def load_document(file_path : str = config.data_path):
    """ Load a .txt file and return it as a list of langChain Documents """
    logger.info("Loading Documents from : %s", file_path)
    loader = TextLoader(file_path,encoding="utf-8")
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))
    return documents