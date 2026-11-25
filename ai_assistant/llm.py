"""Centralized provider selection and transient failover for the CRM agent.

Provider order and model names are configurable through the environment. A
provider that is rate-limited or unavailable is temporarily skipped so the
next invocation can use a healthy provider without repeatedly hitting it.
"""
import logging
import os
import threading
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda

load_dotenv()

logger = logging.getLogger(__name__)
DEFAULT_MAX_TOKENS = 2048
_provider_backoff_until: dict[str, float] = {}
_backoff_lock = threading.Lock()


def build_fallback_llm(primary: Runnable, fallbacks: list[Runnable]) -> Runnable:
    """Keep the simple LangChain fallback helper available to callers."""
    return primary.with_fallbacks(fallbacks)


def _max_tokens() -> int:
    try:
        return int(os.getenv("AI_ASSISTANT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def _configured_models(tools: list) -> list[tuple[str, Runnable]]:
    """Build tool-bound clients for configured providers, in failover order."""
    models: list[tuple[str, Runnable]] = []
    max_tokens = _max_tokens()

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq

        groq_models = [m.strip() for m in os.getenv(
            "GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",")
            if m.strip()]
        # Both Groq model IDs use one quota and one health circuit.
        for model_name in groq_models:
            model = ChatGroq(
                model=model_name, api_key=groq_key, temperature=0,
                max_tokens=max_tokens,
            ).bind_tools(tools, parallel_tool_calls=False)
            models.append(("groq", model))

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            api_key=gemini_key, temperature=0, max_tokens=max_tokens,
            retries=0,
        ).bind_tools(tools)
        models.append(("gemini", model))

    cf_key = os.getenv("CLOUDFLARE_API_TOKEN")
    cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if cf_key and cf_account:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.getenv("CLOUDFLARE_MODEL", "@cf/openai/gpt-oss-120b"),
            api_key=cf_key,
            base_url=("https://api.cloudflare.com/client/v4/accounts/"
                      f"{cf_account}/ai/v1"),
            temperature=0, max_tokens=max_tokens,
        ).bind_tools(tools, parallel_tool_calls=False)
        models.append(("cloudflare", model))

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.getenv(
                "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0, max_tokens=max_tokens,
        ).bind_tools(tools, parallel_tool_calls=False)
        models.append(("openrouter", model))

    order = [name.strip().lower() for name in os.getenv(
        "LLM_PROVIDER_ORDER", "groq,gemini,cloudflare,openrouter").split(",")]
    position = {name: index for index, name in enumerate(order)}
    models.sort(key=lambda item: position.get(item[0], len(position)))
    return models


def build_providers(tools: list) -> list[Runnable]:
    """Public list of configured providers in failover order."""
    return [model for _name, model in _configured_models(tools)]


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _error_category(exc: Exception) -> str:
    """Return a safe, bounded category without exposing provider error text."""
    status = _status_code(exc)
    message = str(exc).lower()
    if status == 429 or "rate limit" in message or "quota" in message:
        return "rate limit"
    if (status in {401, 403} or any(marker in message for marker in
            ("unauthorized", "unauthenticated", "authentication", "api key",
             "invalid key", "permission denied"))):
        return "authentication"
    error_type = type(exc).__name__.lower()
    if (status is not None and status >= 500) or isinstance(
            exc, (TimeoutError, ConnectionError)) or any(
                marker in error_type for marker in
                ("timeout", "connecterror", "networkerror", "connectionerror")):
        return "unavailable"
    return "other"


def _cooldown_seconds(exc: Exception) -> int:
    status = _status_code(exc)
    message = str(exc).lower()
    if status == 429 or "rate limit" in message or "quota" in message:
        default = 120
        setting = "LLM_RATE_LIMIT_COOLDOWN_SECONDS"
    elif status in {401, 403, 404}:
        default = 900
        setting = "LLM_CONFIG_ERROR_COOLDOWN_SECONDS"
    elif status is None or status >= 500 or isinstance(
            exc, (TimeoutError, ConnectionError)):
        default = 30
        setting = "LLM_UNAVAILABLE_COOLDOWN_SECONDS"
    else:
        return 0
    try:
        return max(1, int(os.getenv(setting, default)))
    except ValueError:
        return default


def _invoke_with_failover(providers, messages, config=None):
    if not providers:
        raise RuntimeError(
            "No LLM provider is configured. Add at least one provider API key.")

    now = time.monotonic()
    with _backoff_lock:
        ready = [(name, model) for name, model in providers
                 if _provider_backoff_until.get(name, 0) <= now]
        open_names = {name for name, _model in providers
                      if _provider_backoff_until.get(name, 0) > now}
    # When every circuit is open, probe the provider closest to recovery.
    if ready:
        attempts = ready
        skipped_names = open_names
    else:
        with _backoff_lock:
            attempts = [min(providers, key=lambda item:
                            _provider_backoff_until.get(item[0], 0))]
        skipped_names = open_names - {attempts[0][0]}
    for skipped_name in sorted(skipped_names):
        logger.info("LLM provider %s temporarily skipped; circuit is open",
                    skipped_name)

    failures = []
    for name, model in attempts:
        logger.info("Attempting LLM provider %s", name)
        try:
            result = (model.invoke(messages, config=config) if config is not None
                      else model.invoke(messages))
        except Exception as exc:
            failures.append(exc)
            category = _error_category(exc)
            cooldown = _cooldown_seconds(exc)
            if cooldown:
                with _backoff_lock:
                    _provider_backoff_until[name] = time.monotonic() + cooldown
            logger.warning("LLM provider %s failed (category: %s); trying next provider",
                           name, category)
            continue
        with _backoff_lock:
            _provider_backoff_until.pop(name, None)
        logger.info("LLM provider %s successfully generated a response", name)
        if failures:
            logger.info("LLM request handled by provider %s after failover", name)
        return result

    raise RuntimeError(
        f"All configured LLM providers failed ({len(failures)} attempted).") from (
            failures[-1] if failures else None)


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """Return a lazy manager so configuration changes take effect on restart."""
    from ai_assistant.tools import tools

    providers = _configured_models(tools)
    return RunnableLambda(lambda messages, config: _invoke_with_failover(
        providers, messages, config))
