import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ai_assistant.conversation import is_continuation_reply


MAX_HISTORY_TURNS = 3
# Bound multi-turn context to about 1,500-2,000 tokens including JSON/message
# overhead, while retaining up to three complete recent user turns.
MAX_HISTORY_CHARS = 6000
# Tool results in earlier turns are shortened to this many characters (the
# newest turn keeps its full results).
OLD_TOOL_RESULT_CHARS = 600


def trim_conversation_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep recent complete user turns within a bounded serialized size.

    Older turns are compacted (long tool results shortened, provider metadata
    removed) before any turn is dropped. If the newest message is a bare
    reply such as "skip", the turn it answers is the task itself, so that
    turn is never dropped: without it the model would only see "skip"."""
    user_turn_starts = [
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if not user_turn_starts:
        return messages

    user_turn_starts = user_turn_starts[-MAX_HISTORY_TURNS:]
    keep_at_least = 1
    if is_continuation_reply(_text(messages[user_turn_starts[-1]])):
        keep_at_least = min(2, len(user_turn_starts))

    def build(starts):
        history = messages[starts[0]:]
        last_start = starts[-1] - starts[0]
        return _compact(history[:last_start]) + history[last_start:]

    history = build(user_turn_starts)
    while (
        len(user_turn_starts) > keep_at_least
        and _serialized_size(history) > MAX_HISTORY_CHARS
    ):
        user_turn_starts = user_turn_starts[1:]
        history = build(user_turn_starts)

    return history


def _text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else ""


def _compact(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Copies of earlier-turn messages without bulky tool output/metadata."""
    compacted = []
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.content, str) \
                and len(message.content) > OLD_TOOL_RESULT_CHARS:
            message = message.model_copy(update={
                "content": message.content[:OLD_TOOL_RESULT_CHARS] + " ...[shortened]"})
        elif isinstance(message, AIMessage):
            message = message.model_copy(update={
                "additional_kwargs": {}, "response_metadata": {},
                "usage_metadata": None})
        compacted.append(message)
    return compacted


def _serialized_size(messages: list[BaseMessage]) -> int:
    return sum(
        len(json.dumps(message.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
        for message in messages
    )
