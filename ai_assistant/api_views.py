import logging
import uuid

from django.core.exceptions import (
    ObjectDoesNotExist,
    PermissionDenied,
    ValidationError,
)
from django.db import DatabaseError, IntegrityError
from django.db.utils import DataError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from rest_framework.authentication import SessionAuthentication
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.permissions import BasePermission, IsAuthenticated
from rest_framework.response import Response

from ai_assistant.agent import get_app, is_simple_greeting
from ai_assistant.conversation import answers_question, is_continuation_reply
from ai_assistant.documents.extract import extract_text_from_file

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_CHARS = 15000
RECURSION_LIMIT = 40


class IsCrmStaff(BasePermission):
    """The CRM (and so the assistant) is for active staff accounts only,
    matching the admin site's ``has_permission``."""
    message = "You are not permitted to use the CRM assistant."

    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and user.is_active and user.is_staff)


def _request_config(user, public_thread_id: str) -> dict:
    # The thread is namespaced by the authenticated user, so one user can
    # never load or resume another user's conversation or pending approval.
    return {
        "configurable": {
            "thread_id": f"user-{user.pk}:{public_thread_id}",
            "user_id": user.pk,
        },
        "recursion_limit": RECURSION_LIMIT,
    }


def _get_public_thread_id(data: dict) -> str | None:
    thread_id = data.get("thread_id")
    if thread_id is None:
        return str(uuid.uuid4())
    if not isinstance(thread_id, str) or not thread_id or len(thread_id) > 128:
        return None
    return thread_id


def _snapshot_interrupt(snapshot) -> dict | None:
    """Return the pending interrupt payload from the persisted graph state."""
    if not snapshot:
        return None
    for task in snapshot.tasks or ():
        for pending in getattr(task, "interrupts", ()) or ():
            value = getattr(pending, "value", None)
            if value is not None:
                return value
    return None


def _last_ai_text(result: dict) -> str:
    """Return the last AI message text, ignoring tool-call-only messages."""
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content and content.strip():
                return content
    return ""


def _build_response(app, config: dict, thread_id: str, *, decision: str | None = None):
    snapshot = app.get_state(config)
    payload = _snapshot_interrupt(snapshot)
    if payload is not None:
        return Response({
            "status": "pending_approval",
            "thread_id": thread_id,
            "payload": payload,
        })
    state = getattr(snapshot, "values", None) or {}
    text = _last_ai_text(state)
    if not text:
        text = ("Decision processed." if decision else
                "I did not get an answer from the AI service, so nothing was done "
                "or changed. Please send your message again.")
    body = {"status": "success", "thread_id": thread_id, "message": text}
    if decision and state.get("execution"):
        body["execution"] = state["execution"]
    return Response(body)


def _error_response(exc: Exception, action: str):
    if isinstance(exc, PermissionDenied):
        return Response(
            {"status": "error", "message": "You are not permitted to perform this action."},
            status=403)
    if isinstance(exc, (IntegrityError, DataError, ValidationError, ObjectDoesNotExist)):
        return Response({
            "status": "error",
            "message": f"The {action} could not be completed because the CRM data "
                       "is invalid or unavailable.",
        }, status=400)
    if isinstance(exc, DatabaseError):
        return Response({
            "status": "error",
            "message": f"The CRM database could not complete the {action}.",
        }, status=503)
    if isinstance(exc, GraphRecursionError):
        return Response({
            "status": "error",
            "message": "The assistant could not finish this request. Please "
                       "rephrase it or split it into smaller steps.",
        }, status=500)
    logger.exception("AI assistant %s failed", action)
    return Response({
        "status": "error",
        "message": f"The assistant hit an unexpected error during the {action}. "
                   "Nothing was saved unless a success message was shown.",
    }, status=500)


@api_view(["POST"])
@authentication_classes([SessionAuthentication])
@permission_classes([IsAuthenticated, IsCrmStaff])
def chat_view(request):
    data = request.data
    message = data.get("message", "") or ""
    if not isinstance(message, str):
        return Response({"status": "error", "message": "message must be text."}, status=400)
    uploaded_file = request.FILES.get("file")

    # Allow sending just a file, just a message, or both
    if not message.strip() and not uploaded_file:
        return Response(
            {"status": "error", "message": "A message or a file is required."},
            status=400,
        )

    original_request = message.strip() or "Process the uploaded document."
    if uploaded_file:
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            return Response(
                {"status": "error", "message": "File size exceeds 10MB limit."},
                status=400,
            )
        try:
            extracted_text = extract_text_from_file(uploaded_file, uploaded_file.name)
        except ValueError as e:
            return Response({"status": "error", "message": str(e)}, status=400)
        if len(extracted_text) > MAX_DOCUMENT_CHARS:
            extracted_text = (extracted_text[:MAX_DOCUMENT_CHARS]
                              + "\n[document truncated]")
        message = (f"{message}\n\n--- UPLOADED DOCUMENT CONTENT "
                   f"({uploaded_file.name}) ---\n{extracted_text}")

    thread_id = _get_public_thread_id(data)
    if thread_id is None:
        return Response(
            {"status": "error", "message": "Invalid thread_id."},
            status=400,
        )

    app = get_app()
    config = _request_config(request.user, thread_id)
    try:
        # A new message while a plan waits for approval means the user moved
        # on (e.g. to correct it): reject the pending plan first. Rejection
        # never writes CRM data.
        snapshot = app.get_state(config)
        pending = _snapshot_interrupt(snapshot)
        # A bare "skip"/"yes" answers the assistant's last question: it
        # continues the earlier request instead of replacing it. (Not when it
        # was typed instead of answering an approval card.)
        latest_reply = ""
        previous_task = (getattr(snapshot, "values", None) or {}).get("original_request", "")
        if (pending is None and not uploaded_file and previous_task
                and not is_simple_greeting(previous_task)
                and (is_continuation_reply(message)
                     or answers_question(
                         _last_ai_text(getattr(snapshot, "values", None) or {}),
                         message))):
            latest_reply = message.strip()
            original_request = previous_task
        if pending is not None:
            app.invoke(Command(resume={
                "decision": "rejected",
                "operation_id": pending.get("operation_id"),
                "reason": "you sent a new message instead of approving",
            }), config=config)

        app.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "original_request": original_request,
                "latest_reply": latest_reply,
                "operation_id": "",
                "execution": None,
            },
            config=config,
        )
        return _build_response(app, config, thread_id)
    except Exception as exc:
        return _error_response(exc, "request")


@api_view(["POST"])
@authentication_classes([SessionAuthentication])
@permission_classes([IsAuthenticated, IsCrmStaff])
def approve_view(request):
    data = request.data
    thread_id = _get_public_thread_id(data)
    decision = data.get("decision")
    operation_id = data.get("operation_id")
    if thread_id is None or "thread_id" not in data:
        return Response(
            {"status": "error", "message": "A valid thread_id is required."},
            status=400,
        )
    if decision not in ("approved", "rejected"):
        return Response(
            {"status": "error", "message": "decision must be approved or rejected."},
            status=400,
        )
    if not isinstance(operation_id, str) or not operation_id:
        return Response(
            {"status": "error", "message": "operation_id is required."},
            status=400,
        )

    app = get_app()
    config = _request_config(request.user, thread_id)
    try:
        # Threads are namespaced per user, so another user's thread id simply
        # has no pending approval here. (409 rather than 404: LocaleMiddleware
        # turns 404s on these unprefixed URLs into language redirects.)
        snapshot = app.get_state(config)
        pending = _snapshot_interrupt(snapshot)
        if pending is None:
            return Response({
                "status": "error",
                "message": "There is no change waiting for approval in this "
                           "conversation. It may already have been processed.",
            }, status=409)
        if pending.get("operation_id") != operation_id:
            return Response({
                "status": "error",
                "message": "This approval card is out of date; a different plan "
                           "is waiting for a decision.",
                "payload": pending,
            }, status=409)
        app.invoke(Command(resume={"decision": decision, "operation_id": operation_id}),
                   config=config)
        return _build_response(app, config, thread_id, decision=decision)
    except Exception as exc:
        return _error_response(exc, "approval")
