from insurance_assistant import config
from langchain_groq import ChatGroq

def get_llm():
    """Initialize and return the Groq chat language model instance."""
    return ChatGroq(model=config.llm_model_name, temperature=0)
