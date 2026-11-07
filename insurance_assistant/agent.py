from langchain.agents import create_agent
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)

def create_insurance_agent(llm, tools):
    """Create and return the insurance assistant agent configured with the LLM, tools, and system prompt."""
    logger.info("Creating insurance assistant agent with %d tool(s)", len(tools))
    return create_agent(model=llm,
                        tools=tools,
                        system_prompt=config.system_prompt)
