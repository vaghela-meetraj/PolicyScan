import os
from dotenv import load_dotenv
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

## ENV

groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

# Tracing

LANGSMITH_TRACING= os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT= os.getenv("LANGSMITH_PROJECT")

## Data Path

data_path = os.path.join("data", "apex_health_insurance.txt")

## Vector Store (FAISS - Legacy local index)
# vector_store_path = os.path.join("data", "faiss_index")

## Cloud Vector DB (Qdrant Cloud)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "Insurance_Assistant")

## Models
# LLM and Embedding Models 

llm_model_name = "openai/gpt-oss-120b"
embedding_model_name = "jina-embeddings-v2-base-en"

# Guardrails Models
guardrails_model_name = "openai/gpt-oss-safeguard-20b"

## Chunks / Text Splitting Configs

chunk_size = 500
chunk_overlap = 75

## Retrieval Result

top_k_results = 5

## System Prompt

system_prompt = """
    You are an assistant for answering questions about insurance policies.
    Respond politely to greetings and questions about who you are.
    Always answer questions strictly using only the provided context. If the requested information is not found in the context, state that you do not know the answer based on the provided documents.
    Do not use any outside knowledge, assumptions, or information not present in the context.
    Keep your response clear, accurate, and concise.
    """

def check_api_keys() -> None:
    """Validate required API keys from environment variables."""
    if not groq_key:
        logger.error("Missing GROQ_API_KEY environment variable")
        raise ValueError("Missing GROQ API KEY. Please add it in .env file or environment variables.")
    if not jina_key:
        logger.error("Missing JINA_API_KEY environment variable")
        raise ValueError("Missing JINA API KEY. Please add it in .env file or environment variables.")
    if not QDRANT_API_KEY:
        logger.error("Missing QDRANT_API_KEY environment variable")
        raise ValueError("Missing QDRANT API KEY. Please add it in .env file or environment variables.")
    if not QDRANT_URL:
        logger.error("Missing QDRANT_URL environment variable")
        raise ValueError("Missing QDRANT URL. Please add it in .env file or environment variables.")
    logger.info("API keys validated successfully")
