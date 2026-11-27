"""Lightweight local intent router for the Sales Department agent.

Classifies incoming user messages into one of four distinct intents:
- greeting: Standalone greeting or general conversational opening (0 tools).
- schema: Inquiries about CRM structure, models, fields, and relationships (search_database_schema).
- read: Inquiries about live CRM records, pipeline, and sales figures (READ_TOOLS).
- write: Requests to add, create, update, or change CRM data (propose_sales_workflow + relevant read tools).

Routing is performed purely locally using fast heuristic pattern matching without
making any preliminary LLM calls, saving tokens and preserving quota.
"""
import re
from typing import Any

from ai_assistant.conversation import is_continuation_reply

INTENT_GREETING = "greeting"
INTENT_SCHEMA = "schema"
INTENT_READ = "read"
INTENT_WRITE = "write"
INTENT_GENERAL = "general"

_PURE_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|greetings|good\s+(?:morning|afternoon|evening)|"
    r"hi\s+there|hello\s+there|hey\s+there)[!.?\s]*$",
    re.IGNORECASE,
)

_CONVERSATIONAL_HELP_RE = re.compile(
    r"^\s*(?:what\s+can\s+you\s+do|who\s+are\s+you|help(?:\s+me)?|how\s+can\s+you\s+help)[!.?\s]*$",
    re.IGNORECASE,
)

_SCHEMA_KEYWORDS_RE = re.compile(
    r"\b(?:schema|database\s+schema|crm\s+schema|data\s+model|table\s+structure|"
    r"fields?|required\s+fields?|what\s+fields|list\s+fields|which\s+fields|"
    r"relationships?|foreign\s+keys?|how\s+is\s+.*linked|how\s+are\s+.*linked|"
    r"how\s+is\s+.*related|what\s+does\s+.*mean\s+in\s+(?:the\s+)?crm|"
    r"what\s+is\s+a\s+(?:deal|request|lead|payment|contact|company)\??|"
    r"what\s+models\s+exist)\b",
    re.IGNORECASE,
)

_WRITE_ACTION_RE = re.compile(
    r"\b(?:create|add|new|update|change|modify|set|edit|register|save|record|"
    r"log|propose|convert)\b",
    re.IGNORECASE,
)

_WRITE_ENTITIES_RE = re.compile(
    r"\b(?:lead|prospect|company|customer|account|contact|request|inquiry|"
    r"deal|opportunity|payment|invoice|stage|amount|currency|probability|"
    r"next_step|ticket)\b",
    re.IGNORECASE,
)

_WRITE_SPECIFIC_PATTERNS = [
    re.compile(r"\bdeal\s+(?:called|named)\b", re.IGNORECASE),
    re.compile(r"\b(?:set\s+)?amount\s+(?:to|=|is)\b", re.IGNORECASE),
    re.compile(r"\b(?:next\s+)?stage\s+(?:is|=|to)\b", re.IGNORECASE),
    re.compile(r"\bregister\s+(?:the\s+|an\s+)?opportunity\b", re.IGNORECASE),
    re.compile(r"\bcreate_deal\b", re.IGNORECASE),
    re.compile(r"\bcreate\s+a\s+deal\b", re.IGNORECASE),
    re.compile(r"\badd\s+(?:a\s+|this\s+)?(?:lead|contact|company|deal|payment|request|prospect)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:add|create|register|insert)\s+[A-Z][a-zA-Z0-9_\-\s]+", re.IGNORECASE),
]

_READ_ACTION_RE = re.compile(
    r"\b(?:search|find|get|show|list|lookup|view|display|fetch|summary|pipeline|"
    r"status|who\s+is|who\s+are|what\s+is\s+the\s+status|how\s+many|total\s+amount|"
    r"revenue|sales\s+summary|closing\s+this\s+month|tell\s+me\s+about|"
    r"what\s+(?:payments|deals|leads|contacts|companies|requests|inquiries))\b",
    re.IGNORECASE,
)

_FOLLOWUP_PRONOUNS_RE = re.compile(
    r"\b(?:he|she|they|it|his|her|their|that\s+one|the\s+first|the\s+second|"
    r"the\s+last|this\s+contact|that\s+deal|that\s+company|which\s+one)\b",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """True if text is a standalone greeting or general intro question."""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_PURE_GREETING_RE.match(t) or _CONVERSATIONAL_HELP_RE.match(t))


def is_schema_query(text: str) -> bool:
    """True if the text asks about CRM schema, structure, models, or fields."""
    t = text or ""
    # Avoid classifying live data questions as schema queries
    # e.g. "show fields for lead John" vs "what fields does Lead have"
    if _SCHEMA_KEYWORDS_RE.search(t):
        # If it specifically talks about schema/fields/model structure
        return True
    return False


def is_write_query(text: str) -> bool:
    """True if the text asks to create, update, or modify CRM data."""
    t = text or ""
    if "--- UPLOADED DOCUMENT CONTENT" in t:
        return True
    for pat in _WRITE_SPECIFIC_PATTERNS:
        if pat.search(t):
            return True
    if _WRITE_ACTION_RE.search(t) and _WRITE_ENTITIES_RE.search(t):
        return True
    return False


def is_read_query(text: str) -> bool:
    """True if the text asks to search, find, or view live CRM data."""
    t = text or ""
    return bool(_READ_ACTION_RE.search(t))


def classify_intent(
    message: str,
    messages: list | None = None,
    latest_reply: str = "",
    original_request: str = "",
) -> str:
    """Determine the intent of the user message.

    Takes into account conversation context, continuation replies, and explicit requests.
    Returns one of: 'greeting', 'schema', 'read', 'write'.
    """
    clean_msg = (message or "").strip()

    # 1. Check for continuation / follow-up answers to write workflows
    if latest_reply or is_continuation_reply(clean_msg):
        return INTENT_WRITE

    # 2. Check standalone greetings first (only if no previous ongoing task or if fresh)
    if is_greeting(clean_msg):
        return INTENT_GREETING

    # 3. Document uploads default to write workflow processing
    if "--- UPLOADED DOCUMENT CONTENT" in clean_msg:
        return INTENT_WRITE

    # 4. Check for write requests (creating/updating deals, leads, payments, requests, etc.)
    # Write takes precedence over read when both verbs appear (e.g. "find Acme and create a deal")
    if is_write_query(clean_msg):
        return INTENT_WRITE

    # 5. Check for schema/structure questions
    if is_schema_query(clean_msg):
        return INTENT_SCHEMA

    # 6. Check for live data read questions
    if is_read_query(clean_msg):
        return INTENT_READ

    # 7. Context-dependent follow-up checks
    if messages:
        # Check recent messages to see what kind of conversation we are in
        for prev in reversed(messages[-4:]):
            calls = getattr(prev, "tool_calls", None) or []
            if any(c.get("name") == "propose_sales_workflow" for c in calls):
                return INTENT_WRITE
            if any(c.get("name") in {"search_leads", "search_companies", "search_contacts",
                                     "search_requests", "search_deals", "search_payments",
                                     "get_crm_record", "sales_summary"} for c in calls):
                return INTENT_READ
            if any(c.get("name") == "search_database_schema" for c in calls):
                return INTENT_SCHEMA

    # 8. Check original request if this turn is short
    if original_request and original_request != clean_msg:
        if is_write_query(original_request):
            return INTENT_WRITE
        if is_schema_query(original_request):
            return INTENT_SCHEMA
        if is_read_query(original_request):
            return INTENT_READ

    # Default fallback: if any CRM entity is mentioned, read; otherwise general
    if _WRITE_ENTITIES_RE.search(clean_msg):
        return INTENT_READ

    return INTENT_GENERAL
