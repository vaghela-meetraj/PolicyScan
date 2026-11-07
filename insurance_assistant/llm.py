from langchain_groq import ChatGroq
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """Initialize and return the Groq chat language model instance."""
    logger.info("Initializing Groq LLM with model: %s", config.llm_model_name)
    return ChatGroq(model=config.llm_model_name, temperature=0)
