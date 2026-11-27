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
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

load_dotenv()

logger = logging.getLogger(__name__)
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = 40
DEFAULT_TOTAL_TIMEOUT_SECONDS = 90
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


def _timeout_seconds() -> int:
    """Per-provider timeout (default 40s, env LLM_TIMEOUT_SECONDS)."""
    try:
        return int(os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _total_timeout_seconds() -> int:
    """Total timeout for all provider attempts (default 90s, env LLM_TOTAL_TIMEOUT_SECONDS)."""
    try:
        return int(os.getenv("LLM_TOTAL_TIMEOUT_SECONDS", DEFAULT_TOTAL_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_TOTAL_TIMEOUT_SECONDS


def _configured_models(tools: list) -> list[tuple[str, Runnable]]:
    """Build tool-bound clients for configured providers, in failover order."""
    models: list[tuple[str, Runnable]] = []
    max_tokens = _max_tokens()

    timeout_s = _timeout_seconds()
    # Groq: qwen/qwen3-27b (primary), openai/gpt-oss-120b (fallback)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq

        groq_models = [m.strip() for m in os.getenv(
            "GROQ_MODELS", "qwen/qwen3-27b,openai/gpt-oss-120b").split(",")
            if m.strip()]
        for model_name in groq_models:
            chat_model = ChatGroq(
                model=model_name, api_key=groq_key, temperature=0,
                max_tokens=max_tokens, timeout=timeout_s, max_retries=0,
            )
            model = chat_model.bind_tools(tools, parallel_tool_calls=False) if tools else chat_model
            models.append(("groq", model))

    # Gemini: gemini-1.5-flash (primary), gemini-2.0-flash (fallback)
    # Note: "gemini-3.8-flash" doesn't exist; using current stable models
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_models = [m.strip() for m in os.getenv(
            "GEMINI_MODELS", "gemini-1.5-flash,gemini-2.0-flash").split(",")
            if m.strip()]
        for model_name in gemini_models:
            chat_model = ChatGoogleGenerativeAI(
                model=model_name, api_key=gemini_key, temperature=0,
                max_tokens=max_tokens, retries=0,
                timeout=timeout_s,
            )
            model = chat_model.bind_tools(tools) if tools else chat_model
            models.append(("gemini", model))

    # Cloudflare Workers AI: verified free models with tool calling support
    # @cf/qwen/qwen3-27b - Strong for agentic tasks
    # @cf/google/gemma-4-26b-a4b-it - Native function calling
    # @cf/zai-org/glm-4.7-flash - Alternative model family
    cf_key = os.getenv("CLOUDFLARE_API_TOKEN")
    cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if cf_key and cf_account:
        from langchain_openai import ChatOpenAI

        cf_models = [m.strip() for m in os.getenv(
            "CLOUDFLARE_MODELS",
            "@cf/qwen/qwen3-27b,@cf/google/gemma-4-26b-a4b-it,@cf/zai-org/glm-4.7-flash"
        ).split(",") if m.strip()]
        for model_name in cf_models:
            chat_model = ChatOpenAI(
                model=model_name, api_key=cf_key,
                base_url=("https://api.cloudflare.com/client/v4/accounts/"
                          f"{cf_account}/ai/v1"),
                temperature=0, max_tokens=max_tokens, timeout=timeout_s, max_retries=0,
            )
            model = chat_model.bind_tools(tools, parallel_tool_calls=False) if tools else chat_model
            models.append(("cloudflare", model))

    # OpenRouter: verified free models with tool calling / agentic capabilities
    # nvidia/nemotron-3-ultra-550b-a55b:free - Agent orchestration & reasoning
    # google/gemma-4-26b-a4b-it:free - Native function calling & structured outputs
    # inclusionai/ling-3.2-flash:free - Agentic capabilities & long context
    # baidu/ernie-5.5-flash-preview:free - Structured instructions
    # Note: Avoided duplicate model families (qwen, gemma) already used above
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI

        or_models = [m.strip() for m in os.getenv(
            "OPENROUTER_MODELS",
            "nvidia/nemotron-3-ultra-550b-a55b:free,"
            "google/gemma-4-26b-a4b-it:free,"
            "inclusionai/ling-3.2-flash:free,"
            "baidu/ernie-5.5-flash-preview:free"
        ).split(",") if m.strip()]
        for model_name in or_models:
            chat_model = ChatOpenAI(
                model=model_name, api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0, max_tokens=max_tokens, timeout=timeout_s, max_retries=0,
            )
            model = chat_model.bind_tools(tools, parallel_tool_calls=False) if tools else chat_model
            models.append(("openrouter", model))

    # Provider failover order (can be overridden via LLM_PROVIDER_ORDER)
    order = [name.strip().lower() for name in os.getenv(
        "LLM_PROVIDER_ORDER", "groq,gemini,cloudflare,openrouter").split(",")]
    position = {name: index for index, name in enumerate(order)}
    models.sort(key=lambda item: position.get(item[0], len(position)))
    return models


def build_providers(tools: list) -> list[Runnable]:
    """Public list of configured providers in failover order."""
    return [model for _name, model in _configured_models(tools)]


_REASONING_BLOCKS = {"thinking", "reasoning", "reasoning_content", "redacted_thinking"}
_KEEP_KWARGS = {"tool_calls", "function_call"}


def _portable_messages(messages):
    """Copies of the conversation without provider-specific extras.

    Reasoning text, thought signatures, token usage and similar metadata from
    one provider (Groq "reasoning_content", Gemini thought signatures) are
    meaningless to another and cause cross-provider warnings when the request
    fails over. They also cost tokens against per-minute limits. Message text
    and tool calls are kept."""
    if not isinstance(messages, (list, tuple)):
        return messages
    cleaned = []
    for message in messages:
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):
                blocks = [b for b in content
                          if not (isinstance(b, dict) and b.get("type") in _REASONING_BLOCKS)]
                content = (blocks[0] if len(blocks) == 1 and isinstance(blocks[0], str)
                           else blocks)
            message = message.model_copy(update={
                "content": content,
                "additional_kwargs": {k: v for k, v in message.additional_kwargs.items()
                                      if k in _KEEP_KWARGS},
                "response_metadata": {}, "usage_metadata": None})
        cleaned.append(message)
    return cleaned


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


_REJECTED_STATUSES = {400, 413, 422}


def _log_provider_error_body(name: str, exc: Exception) -> None:
    """Opt-in: log the provider's own error text for rejected requests.

    Off by default because it can echo CRM data (for example a failed tool
    call). Set AI_ASSISTANT_LOG_PROVIDER_ERRORS=1 in a development session."""
    if os.getenv("AI_ASSISTANT_LOG_PROVIDER_ERRORS") != "1":
        return
    logger.warning("LLM provider %s error detail: %s", name, str(exc)[:1500])


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


def _model_name(model) -> str:
    inner = getattr(model, "bound", model)
    for attr in ("model_name", "model"):
        val = getattr(inner, attr, None)
        if val:
            return str(val)
    return "unknown"


def _extract_token_usage(result) -> dict[str, int]:
    inp, out, tot = 0, 0, 0
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        tot = int(usage.get("total_tokens") or (inp + out))
    else:
        meta = getattr(result, "response_metadata", {})
        if isinstance(meta, dict):
            token_usage = meta.get("token_usage") or meta.get("usage") or {}
            if isinstance(token_usage, dict):
                inp = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
                out = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
                tot = int(token_usage.get("total_tokens") or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def _invoke_with_failover(providers, messages, config=None):
    if not providers:
        raise RuntimeError(
            "No LLM provider is configured. Add at least one provider API key.")

    total_timeout = _total_timeout_seconds()
    start_time = time.monotonic()

    messages = _portable_messages(messages)
    failures = []
    # Models that rejected THIS request outright (HTTP 400/413/422).
    rejected: set[int] = set()
    tried: set[int] = set()

    while True:
        now = time.monotonic()
        if now - start_time >= total_timeout:
            logger.warning("Total LLM timeout exceeded (%ds); giving up", total_timeout)
            from ai_assistant.agent import PROVIDER_ERROR_TEXT
            return AIMessage(content=PROVIDER_ERROR_TEXT)

        # Remaining candidates that haven't been tried for this request
        candidates = [(n, m) for n, m in providers
                      if id(m) not in tried and id(m) not in rejected]
        if not candidates:
            break

        with _backoff_lock:
            ready = [(n, m) for n, m in candidates
                     if _provider_backoff_until.get(n, 0) <= now]

        if ready:
            name, model = ready[0]
        else:
            with _backoff_lock:
                name, model = min(candidates, key=lambda item:
                                  _provider_backoff_until.get(item[0], 0))

        tried.add(id(model))
        model_lbl = _model_name(model)
        logger.info("Attempting LLM provider %s (model: %s)", name, model_lbl)

        try:
            result = (model.invoke(messages, config=config) if config is not None
                      else model.invoke(messages))
        except Exception as exc:
            failures.append(exc)
            category = _error_category(exc)
            if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                category = "unavailable"
            cooldown = _cooldown_seconds(exc)
            if _status_code(exc) in _REJECTED_STATUSES:
                rejected.add(id(model))
            _log_provider_error_body(name, exc)
            if cooldown:
                with _backoff_lock:
                    _provider_backoff_until[name] = time.monotonic() + cooldown
            logger.warning(
                "LLM provider %s (model: %s) failed (category: %s, error type: %s, HTTP status: %s, reason: %s); "
                "trying next provider",
                name, model_lbl, category, type(exc).__name__, _status_code(exc), str(exc)[:250])
            continue

        with _backoff_lock:
            _provider_backoff_until.pop(name, None)
        usage = _extract_token_usage(result)
        logger.info(
            "LLM call handled successfully by provider %s (model: %s). "
            "Tokens consumed: %d input, %d output, %d total.",
            name, model_lbl, usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]
        )
        if failures:
            logger.info("LLM request handled by provider %s (model: %s) after %d failover attempt(s)",
                        name, model_lbl, len(failures))
        return result

    # When all candidates have been attempted
    if rejected and len(rejected) == len(providers):
        logger.warning("Every LLM provider rejected this request; not retrying")
        raise RuntimeError("All configured LLM providers rejected the request.") from (
            failures[-1] if failures else None)

    if failures and all(
        isinstance(f, TimeoutError) or "timed out" in str(f).lower() for f in failures
    ):
        logger.warning("All configured LLM providers timed out; returning provider error")
        from ai_assistant.agent import PROVIDER_ERROR_TEXT
        return AIMessage(content=PROVIDER_ERROR_TEXT)

    raise RuntimeError(
        f"All configured LLM providers failed ({len(failures)} attempted).") from (
            failures[-1] if failures else None)


_llm_cache: dict[tuple[str, ...], Runnable] = {}
_llm_cache_lock = threading.Lock()


def get_llm(tools: list | tuple | None = None) -> Runnable:
    """Return a lazy manager so configuration changes take effect on restart."""
    if tools is None:
        from ai_assistant.tools import tools as default_tools
        tools = default_tools
    tools_list = list(tools)
    key = tuple(getattr(t, "name", str(t)) for t in tools_list)
    with _llm_cache_lock:
        if key not in _llm_cache:
            providers = _configured_models(tools_list)
            _llm_cache[key] = RunnableLambda(
                lambda messages, config=None, p=providers: _invoke_with_failover(
                    p, messages, config))
        return _llm_cache[key]


def clear_llm_cache():
    with _backoff_lock:
        _provider_backoff_until.clear()
    with _llm_cache_lock:
        _llm_cache.clear()


get_llm.cache_clear = clear_llm_cache
