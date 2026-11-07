from langchain_text_splitters import RecursiveCharacterTextSplitter
from insurance_assistant import config
def split_into_chunks(documents):
    """ Split documents into chunks """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    return text_splitter.split_documents(documents)