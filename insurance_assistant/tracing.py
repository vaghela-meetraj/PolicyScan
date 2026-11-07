from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def check_langsmith_tracing() -> bool:
    """Check and validate LangSmith tracing configuration."""
    is_enabled = str(config.LANGSMITH_TRACING or "").strip().lower() == "true"

    if not is_enabled:
        logger.info("LangSmith tracing is disabled")
        return False

    if not config.LANGSMITH_API_KEY:
        logger.warning("LangSmith tracing is enabled, but LANGSMITH_API_KEY is missing")
        return False

    project_name = config.LANGSMITH_PROJECT or "default"
    logger.info("LangSmith tracing is enabled for project: %s", project_name)
    return True

