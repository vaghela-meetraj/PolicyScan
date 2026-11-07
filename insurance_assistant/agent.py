from langchain.agents import create_agent
from insurance_assistant import config

def create_insurance_agent(llm, tools):
    """Create and return the insurance assistant agent configured with the LLM, tools, and system prompt."""
    return create_agent(model=llm,
                        tools=tools,
                        system_prompt=config.system_prompt)
