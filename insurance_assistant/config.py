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

data_path = os.path.join("data", "policy.txt")

## Vector Store (FAISS - Legacy local index)
# vector_store_path = os.path.join("data", "faiss_index")

## Cloud Vector DB (Qdrant Cloud)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "Insurance_Assistant")

## Cloudflare AI Gateway
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_GATEWAY_TOKEN")
CLOUDFLARE_GATEWAY_ID = os.getenv("CLOUDFLARE_GATEWAY_ID")
CLOUDFLARE_GATEWAY_MAIN = os.getenv("CLOUDFLARE_GATEWAY_MAIN", "main-key")
CLOUDFLARE_GATEWAY_BACKUP = os.getenv("CLOUDFLARE_GATEWAY_BACKUP", "backup-key")

## Models
# LLM and Embedding Models 

llm_model_name = "openai/gpt-oss-120b"
llm_backup_model_name = os.getenv("CLOUDFLARE_BACKUP_MODEL", "openai/gpt-oss-20b")
embedding_model_name = "jina-embeddings-v2-base-en"

# Neural Re-ranker Configs
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
reranker_model_name = "jina-reranker-v2-base-multilingual"
top_n_candidates = 15

# Guardrails Configs
ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
guardrails_model_name = "openai/gpt-oss-safeguard-20b"
GUARDRAIL_PACING_DELAY = float(os.getenv("GUARDRAIL_PACING_DELAY", "20.5"))

# LLM Judge & Evaluation Configs
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gemini-3.1-flash-lite")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
EVAL_DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "insurance-policy-qna")
# Pacing delays to strictly protect Groq 8K TPM and Gemini 15 RPM limits
EVAL_PACING_DELAY = float(os.getenv("EVAL_PACING_DELAY", "20.0"))
EVAL_JUDGE_DELAY = float(os.getenv("EVAL_JUDGE_DELAY", "5.0"))

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
    if CLOUDFLARE_ACCOUNT_ID:
        logger.info(
            "Cloudflare AI Gateway enabled (main: %s, backup: %s)",
            CLOUDFLARE_GATEWAY_MAIN,
            CLOUDFLARE_GATEWAY_BACKUP,
        )
    logger.info("API keys validated successfully")
