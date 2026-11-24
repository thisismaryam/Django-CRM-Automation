"""LangGraph routing, human approval, resumption and the HTTP API.

The LLM is replaced by a scripted stand-in; everything else (graph, tools,
validation, CRM admin save logic, checkpointing) is real.
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import DatabaseError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rest_framework.test import APIClient

from ai_assistant.agent import (
    PROVIDER_ERROR_TEXT, build_graph, set_app, sqlite_checkpointer,
)
from ai_assistant.models import ActionRequest
from ai_assistant.tests.base import SalesTestCase, ScriptedLLM, tool_call
from ai_assistant.workflow import PLAN_TOOL_NAME
from common.utils.helpers import USER_MODEL
from crm.models import Lead

LEAD_PLAN = {
    "goal": "Add Nina as a lead",
    "operations": [{
        "op_id": "op1", "action": "create", "record_type": "lead",
        "fields": [{"name": "first_name", "value": "Nina"},
                   {"name": "email", "value": "nina@newco.example"}],
    }],
}


class GraphTestCase(SalesTestCase):
    def setUp(self):
        self.graph = build_graph(MemorySaver())
        set_app(self.graph)
        self.addCleanup(set_app, None)

    def script(self, *responses) -> ScriptedLLM:
        llm = ScriptedLLM(*responses)
        patcher = patch("ai_assistant.agent.get_llm", return_value=llm)
        patcher.start()
        self.addCleanup(patcher.stop)
        return llm

    def start(self, user, text, thread="t1", graph=None):
        return (graph or self.graph).invoke(
            {"messages": [HumanMessage(content=text)], "original_request": text},
            config=self.config(user, thread))

    def pending_payload(self, user, thread="t1", graph=None):
        state = (graph or self.graph).get_state(self.config(user, thread))
        for task in state.tasks:
            for item in task.interrupts:
                return item.value
        return None


class RoutingTests(GraphTestCase):
    def test_read_question_routes_to_scoped_read_tool(self):
        llm = self.script(tool_call("search_companies", {"name": "Acme"}),
                          AIMessage(content="Acme Corp is in the CRM (#1)."))
        result = self.start(self.manager, "Is Acme a customer?")
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("Acme Corp", tool_messages[0].content)
        self.assertEqual(result["messages"][-1].content, "Acme Corp is in the CRM (#1).")
        system = llm.prompts[0][0]
        self.assertIsInstance(system, SystemMessage)
        self.assertIn("Is Acme a customer?", system.content)
        self.assertIn(PLAN_TOOL_NAME, system.content)
        self.assertIsNone(self.pending_payload(self.manager))

    def test_read_tool_respects_department_inside_the_graph(self):
        self.script(tool_call("search_companies", {"name": "Hidden"}),
                    AIMessage(content="Not found."))
        result = self.start(self.manager, "Find Hidden Local GmbH")
        tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertNotIn("Hidden Local GmbH", tool_message.content)

    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_structure_question_routes_to_schema_rag(self, search):
        search.return_value = {"matched_models": [], "expanded_models": [], "schemas": {}}
        self.script(tool_call("search_database_schema", {"query": "opportunity fields"}),
                    AIMessage(content="A deal has ..."))
        result = self.start(self.manager, "What fields does an opportunity have?")
        tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertIn("not live records", tool_message.content)
        self.assertIn("## Deal", tool_message.content)

    def test_tool_failure_is_reported_to_the_model_not_raised(self):
        self.script(tool_call("search_leads", {"query": "x"}),
                    AIMessage(content="The lookup failed, sorry."))
        with patch("ai_assistant.access.CrmAccess.queryset", side_effect=DatabaseError("boom")):
            result = self.start(self.manager, "find leads")
        tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertIn("ERROR: the CRM lookup failed", tool_message.content)
        self.assertNotIn("boom", tool_message.content)

    def test_provider_outage_gives_a_clear_message(self):
        self.script(RuntimeError("quota exceeded: key=abc"))
        result = self.start(self.manager, "hello")
        self.assertEqual(result["messages"][-1].content, PROVIDER_ERROR_TEXT)


class ApprovalFlowTests(GraphTestCase):
    def test_plan_pauses_for_approval_then_executes(self):
        llm = self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        self.start(self.manager, "Add Nina, nina@newco.example, as a lead")
        payload = self.pending_payload(self.manager)
        self.assertEqual(payload["action"], "SALES_WORKFLOW")
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())

        result = self.graph.invoke(
            Command(resume={"decision": "approved", "operation_id": payload["operation_id"]}),
            config=self.config(self.manager))
        self.assertTrue(Lead.objects.filter(email="nina@newco.example").exists())
        self.assertIn("Done. All changes were saved", result["messages"][-1].content)
        self.assertEqual(result["execution"]["status"], "success")
        self.assertEqual(len(llm.prompts), 1)   # the report needs no extra LLM call
        # Every tool call got its tool result (valid history for providers).
        calls = {c["id"] for m in result["messages"] if isinstance(m, AIMessage)
                 for c in m.tool_calls}
        answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
        self.assertEqual(calls, answered)

    def test_rejection_writes_nothing(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        self.start(self.manager, "Add Nina")
        payload = self.pending_payload(self.manager)
        result = self.graph.invoke(
            Command(resume={"decision": "rejected", "operation_id": payload["operation_id"]}),
            config=self.config(self.manager))
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.REJECTED)
        self.assertTrue(result["messages"][-1].content.startswith("Cancelled"))

    def test_resume_with_a_different_operation_id_is_not_approval(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        self.start(self.manager, "Add Nina")
        self.graph.invoke(Command(resume={"decision": "approved", "operation_id": "stale"}),
                          config=self.config(self.manager))
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())

    def test_invalid_plan_goes_back_to_the_agent_without_approval(self):
        bad_plan = {"goal": "x", "operations": [{
            "op_id": "op1", "action": "create", "record_type": "lead",
            "fields": [{"name": "first_name", "value": "Nina"}]}]}
        llm = self.script(tool_call(PLAN_TOOL_NAME, bad_plan),
                          AIMessage(content="What is Nina's email address?"))
        result = self.start(self.manager, "Add Nina as a lead")
        self.assertIsNone(self.pending_payload(self.manager))
        feedback = [m for m in llm.prompts[1] if isinstance(m, ToolMessage)][-1]
        self.assertIn("NEEDS_DETAILS", feedback.content)
        self.assertEqual(result["messages"][-1].content, "What is Nina's email address?")

    def test_plan_proposed_together_with_lookups_runs_alone(self):
        message = AIMessage(content="", tool_calls=[
            {"name": "search_leads", "args": {"query": "Nina"}, "id": "a", "type": "tool_call"},
            {"name": PLAN_TOOL_NAME, "args": LEAD_PLAN, "id": "b", "type": "tool_call"},
        ])
        self.script(message)
        result = self.start(self.manager, "Add Nina")
        skipped = next(m for m in result["messages"]
                       if isinstance(m, ToolMessage) and m.tool_call_id == "a")
        self.assertTrue(skipped.content.startswith("NOT_RUN"))
        self.assertIsNotNone(self.pending_payload(self.manager))

    def test_unexpected_validation_error_keeps_conversation_usable(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN),
                    AIMessage(content="Sorry, I could not check that change."))
        with patch("ai_assistant.agent.validate_plan", side_effect=RuntimeError("boom")):
            result = self.start(self.manager, "Add Nina")
        self.assertEqual(result["messages"][-1].content,
                         "Sorry, I could not check that change.")
        calls = {c["id"] for m in result["messages"] if isinstance(m, AIMessage)
                 for c in m.tool_calls}
        answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
        self.assertEqual(calls, answered)

    def test_user_deactivated_while_waiting_closes_the_plan(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        self.start(self.manager, "Add Nina")
        payload = self.pending_payload(self.manager)
        USER_MODEL.objects.filter(pk=self.manager.pk).update(is_active=False)
        result = self.graph.invoke(
            Command(resume={"decision": "approved", "operation_id": payload["operation_id"]}),
            config=self.config(self.manager))
        self.assertEqual(result["execution"]["status"], "failed")
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.FAILED)
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())

    def test_interrupted_workflow_resumes_after_restart(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = Path(directory) / "checkpoints.sqlite"
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))

        saver1 = sqlite_checkpointer(path)
        first_process = build_graph(saver1)
        self.start(self.manager, "Add Nina", graph=first_process)
        saver1.conn.close()                  # "server restart"

        saver2 = sqlite_checkpointer(path)
        self.addCleanup(saver2.conn.close)
        second_process = build_graph(saver2)
        payload = self.pending_payload(self.manager, graph=second_process)
        self.assertEqual(payload["operations"][0]["title"], "Create Lead")
        second_process.invoke(
            Command(resume={"decision": "approved", "operation_id": payload["operation_id"]}),
            config=self.config(self.manager))
        self.assertTrue(Lead.objects.filter(email="nina@newco.example").exists())


class ApiTests(GraphTestCase):
    def client_for(self, user):
        client = APIClient()
        client.force_login(user)
        return client

    def chat(self, user, text, thread="api-1", **extra):
        return self.client_for(user).post(
            "/api/ai/chat/", {"message": text, "thread_id": thread, **extra}, format="multipart")

    def approve(self, user, decision, operation_id, thread="api-1"):
        return self.client_for(user).post(
            "/api/ai/approve/",
            {"thread_id": thread, "decision": decision, "operation_id": operation_id},
            format="json")

    def test_chat_requires_an_authenticated_staff_user(self):
        response = APIClient().post("/api/ai/chat/", {"message": "hi"}, format="json")
        self.assertIn(response.status_code, (401, 403))
        outsider = USER_MODEL.objects.create_user(username="outsider", password="x")
        response = self.chat(outsider, "hi")
        self.assertEqual(response.status_code, 403)

    def test_full_approval_round_trip(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        response = self.chat(self.manager, "Add Nina")
        self.assertEqual(response.status_code, 200, response.content)
        body = response.json()
        self.assertEqual(body["status"], "pending_approval")
        operation_id = body["payload"]["operation_id"]

        stale = self.approve(self.manager, "approved", "not-the-plan")
        self.assertEqual(stale.status_code, 409)
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())

        response = self.approve(self.manager, "approved", operation_id)
        self.assertEqual(response.status_code, 200, response.content)
        body = response.json()
        self.assertEqual(body["execution"]["status"], "success")
        self.assertIn("Done.", body["message"])
        self.assertTrue(Lead.objects.filter(email="nina@newco.example").exists())

        again = self.approve(self.manager, "approved", operation_id)
        self.assertEqual(again.status_code, 409)
        self.assertEqual(Lead.objects.filter(email="nina@newco.example").count(), 1)

    def test_user_cannot_resume_another_users_pending_plan(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN))
        operation_id = self.chat(self.manager, "Add Nina").json()["payload"]["operation_id"]
        response = self.approve(self.operator, "approved", operation_id)
        self.assertEqual(response.status_code, 409)
        self.assertIn("no change waiting", response.json()["message"])
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.PENDING)

    def test_new_message_cancels_a_pending_plan(self):
        self.script(tool_call(PLAN_TOOL_NAME, LEAD_PLAN),
                    AIMessage(content="Sure, what should change?"))
        self.chat(self.manager, "Add Nina")
        response = self.chat(self.manager, "Wait, use her work email instead")
        self.assertEqual(response.json()["message"], "Sure, what should change?")
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.REJECTED)
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())

    def test_document_upload_is_passed_as_data(self):
        llm = self.script(AIMessage(content="I found Jane Doe."))
        upload = SimpleUploadedFile("card.txt", b"Jane Doe\njane@doe.example\nDoe Inc",
                                    content_type="text/plain")
        response = self.chat(self.manager, "Add this prospect", file=upload)
        self.assertEqual(response.status_code, 200, response.content)
        human = [m for m in llm.prompts[0] if isinstance(m, HumanMessage)][-1]
        self.assertIn("--- UPLOADED DOCUMENT CONTENT (card.txt) ---", human.content)
        self.assertIn("jane@doe.example", human.content)

        bad = SimpleUploadedFile("evil.exe", b"MZ", content_type="application/octet-stream")
        response = self.chat(self.manager, "", file=bad)
        self.assertEqual(response.status_code, 400)

    def test_unexpected_errors_do_not_leak_details(self):
        with patch.object(self.graph, "invoke", side_effect=RuntimeError("secret-key-123")):
            response = self.chat(self.manager, "hello")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret-key-123", response.content.decode())

    def test_chat_page_renders_inside_crm_admin(self):
        self.client.force_login(self.manager)
        response = self.client.get("/en/api/ai/chat-ui/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "renderApprovalCard")
        self.assertContains(response, "Approve all")

    def test_approve_requires_operation_id(self):
        response = self.approve(self.manager, "approved", "")
        self.assertEqual(response.status_code, 400)
