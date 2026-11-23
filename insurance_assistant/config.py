import os
from dotenv import load_dotenv
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

## 1. Environment Secrets & External Endpoints (Loaded from .env)

# Core Model APIs
groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

# Qdrant Cloud Vector Database
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Cloudflare AI Gateway
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_GATEWAY_TOKEN")
CLOUDFLARE_GATEWAY_ID = os.getenv("CLOUDFLARE_GATEWAY_ID")

# Observability & Tracing (LangSmith)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "PolicyScan")

# Judge API for Evaluation
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY")


## 2. Application Configuration & Hyperparameters (Defined in Code)

# Data & Collection
data_path = os.path.join("data", "policy.txt")
QDRANT_COLLECTION_NAME = "Insurance_AssistantD2"

# Gateway Aliases & Routing
CLOUDFLARE_GATEWAY_MAIN = "mainapi"
CLOUDFLARE_GATEWAY_BACKUP = "backupapi"
CLOUDFLARE_SKIP_CACHE = False

# Models
llm_model_name = "openai/gpt-oss-120b"
llm_backup_model_name = "openai/gpt-oss-20b"
embedding_model_name = "jina-embeddings-v2-base-en"
guardrails_model_name = "openai/gpt-oss-safeguard-20b"
reranker_model_name = "jina-reranker-v2-base-multilingual"

# Text Splitting & Windowing
chunk_size = 900
chunk_overlap = 200

# Retrieval & Re-ranking Hyperparameters
ENABLE_HYBRID_SEARCH = True
dense_weight = 0.6
sparse_weight = 0.4
ENABLE_RERANKING = True
top_n_candidates = 25
top_k_results = 8

# Guardrails Policy & Pacing
ENABLE_GUARDRAILS = True
GUARDRAIL_PACING_DELAY = 20.5  # seconds between guardrail checks to respect 3 RPM limit

# Evaluation Harness Settings
JUDGE_MODEL_NAME = "gemini-3.1-flash-lite"
JUDGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EVAL_DATASET_NAME = "insurance-policy-qna"
EVAL_PACING_DELAY = 20.0  # seconds between test cases to respect 8K TPM
EVAL_JUDGE_DELAY = 5.0    # seconds between judge calls to respect 15 RPM


## 3. System Prompt & Answering Rules

system_prompt = """
    You are an expert insurance and policy analysis assistant.
    Your role is to analyze policy documents with legal precision, objective factual accuracy, and strict grounding in the provided context.

    Core Answering Rules:
    1. Grounding: Answer strictly and solely using the provided context. Do not volunteer external examples, unmentioned documents, or unstated assumptions.
    2. Hierarchical Precedence: Specific exclusions, waiting periods, and restrictive conditions strictly supersede general coverage inclusions or procedure schedules. If a condition or treatment is subject to an exclusion, state that it is not covered.
    3. Contract Precision: When quoting timeframes, waiting periods, percentages, and monetary caps, extract the exact values directly from the retrieved context.
    4. Calculations: When calculating payouts, state the governing policy cap or percentage first, then show step-by-step arithmetic.
    5. Negative Scope & Refusals: If the requested information is absent from the provided context (such as pricing schedules, external company metrics, comparative product evaluations, or unmentioned plan tiers), decline clearly without fabricating numbers or speculating.
    6. Directness: Answer the user's question directly without adding unrequested administrative boilerplate or extraneous commentary.
    """


## 4. Environment Validation

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
