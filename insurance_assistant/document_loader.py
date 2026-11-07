from langchain_community.document_loaders import TextLoader
from insurance_assistant import config

def load_document(file_path : str = config.data_path):
    """ Load a .txt file and return it as a list of langChain Documents """
    loader = TextLoader(file_path,encoding="utf-8")
    return loader.load()