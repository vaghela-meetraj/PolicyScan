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

## Vector Store

vector_store_path = os.path.join("data", "faiss_index")

## Models
# LLM and Embedding Models 

llm_model_name = "openai/gpt-oss-120b"
embedding_model_name = "jina-embeddings-v2-base-en"

## Chunks / Text Splitting Configs

chunk_size = 1000
chunk_overlap = 150

## Retrieval Result

top_k_results = 5

## System Prompt

system_prompt = """
    You are an assistant for answering questions about insurance policies, you have mastery in english so whenever you give response your sentence framing and choice of words is perfect.
    Respond politely to greetings and questions about who you are.
    For policy questions, answer using only the provided context; if the information is not in the context, say that you do not know.
    Keep your response clear, accurate, and concise without making assumptions.
    """

def check_api_keys() -> None:
    """Validate required API keys from environment variables."""
    if not groq_key:
        logger.error("Missing GROQ_API_KEY environment variable")
        raise ValueError("Missing GROQ API KEY. Please add it in .env file or environment variables.")
    if not jina_key:
        logger.error("Missing JINA_API_KEY environment variable")
        raise ValueError("Missing JINA API KEY. Please add it in .env file or environment variables.")
    logger.info("API keys validated successfully")
