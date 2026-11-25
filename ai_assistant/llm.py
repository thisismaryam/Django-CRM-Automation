"""LLM providers: Gemini primary -> Gemini fallback -> Groq fallback.

Models are built lazily on first use, so importing the assistant (Django
startup, management commands, tests) never needs API keys or network access.
Model names, keys and the completion budget come from environment variables:

GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL, GEMINI_FALLBACK_MODEL,
GROQ_API_KEY, GROQ_MODEL, AI_ASSISTANT_MAX_TOKENS.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.runnables import Runnable

load_dotenv()

DEFAULT_MAX_TOKENS = 2048


def build_fallback_llm(primary: Runnable, fallbacks: list[Runnable]) -> Runnable:
    """Chains the primary model to a list of fallback models in order."""
    return primary.with_fallbacks(fallbacks)


def _max_tokens() -> int:
    try:
        return int(os.getenv("AI_ASSISTANT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def build_providers(tools: list) -> list[Runnable]:
    """The three tool-bound providers, in fallback order."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    max_tokens = _max_tokens()
    gemini_primary = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        api_key=gemini_key,
        temperature=0,
        max_tokens=max_tokens,
        retries=0,
    ).bind_tools(tools)
    gemini_fallback = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash"),
        api_key=gemini_key,
        temperature=0,
        max_tokens=max_tokens,
        retries=0,
    ).bind_tools(tools)
    groq_fallback = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=max_tokens,
    ).bind_tools(tools, parallel_tool_calls=False)
    return [gemini_primary, gemini_fallback, groq_fallback]


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    from ai_assistant.tools import tools

    primary, *fallbacks = build_providers(tools)
    return build_fallback_llm(primary=primary, fallbacks=fallbacks)
