"""Single place where chat models are created."""

from langchain_openai import ChatOpenAI

from agentops import config


def get_chat_model(
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    temperature: float = config.TEMPERATURE,
) -> ChatOpenAI:
    """
    Build the chat model used by every agent.

    max_retries=0 and a fixed timeout keep latency honest: a retried call
    would otherwise look like one slow call in the monitoring data.
    """
    return ChatOpenAI(
        model=config.MODEL,
        base_url=config.NVIDIA_BASE_URL,
        api_key=config.get_nvidia_api_key(),
        temperature=temperature,
        max_retries=config.LLM_MAX_RETRIES,
        timeout=config.LLM_TIMEOUT_SECONDS,
        # `max_tokens` goes through extra_body on purpose. ChatOpenAI's own
        # max_tokens field is sent as `max_completion_tokens`, but the
        # NVIDIA endpoint is called with plain `max_tokens` (as in v1).
        extra_body={
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )