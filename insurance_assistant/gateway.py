"""Cloudflare AI Gateway integration with BYOK aliases for Groq LLMs."""

import json
from langchain_openai import ChatOpenAI
from insurance_assistant import config
from insurance_assistant.logger import get_logger

logger = get_logger(__name__)


def get_gateway_url() -> str:
    """Return the Cloudflare AI Gateway base URL for Groq."""
    return f"https://gateway.ai.cloudflare.com/v1/{config.CLOUDFLARE_ACCOUNT_ID}/{config.CLOUDFLARE_GATEWAY_ID}/groq"


def get_gateway_headers(alias: str, attempt: int = 1) -> dict:
    """Return headers for Cloudflare AI Gateway with BYOK alias and tracking metadata."""
    headers = {
        "cf-aig-collect-log-payload": "false",
        "cf-aig-metadata": json.dumps({
            "application": "PolicyScan",
            "key_alias": alias,
            "attempt": attempt,
        }),
        "Authorization": "",
    }
    if config.CLOUDFLARE_API_TOKEN:
        headers["cf-aig-authorization"] = f"Bearer {config.CLOUDFLARE_API_TOKEN}"
    if alias:
        headers["cf-aig-byok-alias"] = alias
    return headers


def create_gateway_llm(alias: str, model_name: str = config.llm_model_name, attempt: int = 1) -> ChatOpenAI:
    """Create and return a ChatOpenAI instance routed through Cloudflare AI Gateway with BYOK alias."""
    base_url = get_gateway_url()
    headers = get_gateway_headers(alias, attempt)
    logger.info(
        "Configuring Gateway LLM (gateway: %s, alias: %s, model: %s)",
        config.CLOUDFLARE_GATEWAY_ID,
        alias,
        model_name,
    )
    return ChatOpenAI(
        model=model_name,
        api_key="placeholder",
        base_url=base_url,
        default_headers=headers,
        temperature=0,
    )
