import os
import re

from django.core.management.base import BaseCommand
from langchain_core.messages import HumanMessage

from ai_assistant.llm import _configured_models, _status_code
from ai_assistant.tools import tools

_SECRET_ENV = ("API_KEY", "TOKEN", "SECRET", "ACCOUNT_ID")


def _mask(text):
    for name, value in os.environ.items():
        if value and len(value) > 6 and any(m in name.upper() for m in _SECRET_ENV):
            text = text.replace(value, "***")
    return re.sub(r"(?i)(key|token|bearer)[=: ]+[A-Za-z0-9_\-\.]{12,}", r"\1=***", text)


def _model_name(model):
    inner = getattr(model, "bound", model)
    for attr in ("model_name", "model"):
        value = getattr(inner, attr, None)
        if value:
            return str(value)
    return "?"


class Command(BaseCommand):
    help = "Test each configured LLM provider once and print the real error."

    def handle(self, *args, **options):
        providers = _configured_models(tools)
        self.stdout.write("Provider order: " + os.getenv("LLM_PROVIDER_ORDER", "groq,gemini,cloudflare,openrouter"))
        if not providers:
            self.stdout.write("No provider is configured.")
            return
        for name, model in providers:
            label = f"{name} / {_model_name(model)}"
            try:
                reply = model.invoke([HumanMessage(content="Reply with the single word OK.")])
            except Exception as exc:
                detail = _mask(str(exc)).replace("\n", " ")[:600]
                self.stdout.write(f"FAIL {label}: {type(exc).__name__} (HTTP {_status_code(exc)}) {detail}")
                continue
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            self.stdout.write(f"OK   {label}: {text.strip()[:60]!r}")
