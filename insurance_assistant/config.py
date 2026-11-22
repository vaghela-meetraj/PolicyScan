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
CLOUDFLARE_SKIP_CACHE = os.getenv("CLOUDFLARE_SKIP_CACHE", "false").lower() == "true"

## Models
# LLM and Embedding Models 

llm_model_name = "openai/gpt-oss-120b"
llm_backup_model_name = os.getenv("CLOUDFLARE_BACKUP_MODEL", "openai/gpt-oss-20b")
embedding_model_name = "jina-embeddings-v2-base-en"

# Neural Re-ranker Configs
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
reranker_model_name = "jina-reranker-v2-base-multilingual"
top_n_candidates = int(os.getenv("TOP_N_CANDIDATES", "12"))

# Hybrid Retrieval Configs (Dense Qdrant + Sparse BM25)
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

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

chunk_size = 900
chunk_overlap = 200

## Retrieval Result

top_k_results = int(os.getenv("TOP_K_RESULTS", "8"))

## System Prompt

system_prompt = """
    You are an expert insurance and policy analysis assistant.
    Your role is to analyze policy documents with legal precision, objective factual accuracy, and strict grounding in the provided context.

    Core Answering Rules:
    1. Grounding: Answer strictly and solely using the provided context. Do not volunteer external examples, unmentioned documents, or unstated assumptions.
    2. Legal Precedence: Exclusions and waiting periods strictly supersede Day Care procedure lists. If a procedure is subject to an exclusion, state that it is not covered.
    3. Contract Precision: When quoting timeframes, waiting periods, percentages, and monetary caps, extract the exact values directly from the retrieved context.
    4. Calculations: When calculating payouts, state the governing policy cap or percentage first, then show step-by-step arithmetic.
    5. Out-of-Scope Refusals: If a fact is absent from the text (e.g. premiums, claim settlement ratios, competitor plans), decline clearly without inventing numbers.
    6. Directness: Answer the user's question directly without adding unrequested administrative boilerplate or extraneous commentary.
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
