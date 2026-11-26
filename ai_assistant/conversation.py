"""Short replies that continue the previous request.

When the assistant asks for optional details and the user answers "skip",
that reply means "carry on with the request above"; it is not a new request.
"""
import re

_CONTINUATION_RE = re.compile(
    r"(?:skip(?:\s+it)?|none|no(?:\s+more)?(?:\s+details)?|nope|nothing(?:\s+else)?"
    r"|that(?:'|’)?s\s+all|that\s+is\s+all|just\s+create\s+it|create\s+it"
    r"|go\s+ahead|proceed|continue|ok(?:ay)?|yes|yep|yeah|sure|please\s+do"
    r"|do\s+it|done|n/a)[\s.!]*",
    re.IGNORECASE,
)
MAX_REPLY_CHARS = 40


def is_continuation_reply(text: str) -> bool:
    """True for a bare "skip" / "none" / "that's all" style answer."""
    text = (text or "").strip()
    return 0 < len(text) <= MAX_REPLY_CHARS and bool(_CONTINUATION_RE.fullmatch(text))
