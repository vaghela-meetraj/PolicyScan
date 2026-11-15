from langchain_groq import ChatGroq
from insurance_assistant import config
from insurance_assistant.gateway import create_gateway_llm
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """Initialize and return the LLM instance with Cloudflare AI Gateway and fallback."""
    if config.CLOUDFLARE_ACCOUNT_ID:
        logger.info(
            "Initializing LLM via Cloudflare AI Gateway (gateway: %s, main: %s [%s], fallback: %s [%s])",
            config.CLOUDFLARE_GATEWAY_ID,
            config.CLOUDFLARE_GATEWAY_MAIN,
            config.llm_model_name,
            config.CLOUDFLARE_GATEWAY_BACKUP,
            config.llm_backup_model_name,
        )
        main_llm = create_gateway_llm(
            alias=config.CLOUDFLARE_GATEWAY_MAIN,
            model_name=config.llm_model_name,
            attempt=1,
        )
        backup_llm = create_gateway_llm(
            alias=config.CLOUDFLARE_GATEWAY_BACKUP,
            model_name=config.llm_backup_model_name,
            attempt=2,
        )

        # RAG fallback: mainapi (120b) -> backupapi (20b)
        return main_llm.with_fallbacks([backup_llm])

    logger.info("Initializing direct Groq LLM with model: %s", config.llm_model_name)
    return ChatGroq(model=config.llm_model_name, api_key=config.groq_key, temperature=0)
