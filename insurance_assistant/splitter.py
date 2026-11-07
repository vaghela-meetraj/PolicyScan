from langchain_text_splitters import RecursiveCharacterTextSplitter
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def split_into_chunks(documents):
    """Split documents into chunks."""
    logger.info("Splitting %d document(s) into chunks (size=%d, overlap=%d)", len(documents), config.chunk_size, config.chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Created %d chunks from document(s)", len(chunks))
    return chunks
