"""Regression tests for two real conversations that failed.

Conversation 1: "Create a new deal called ERP Integration for the existing
company Byteboom. Link it to the existing contact Sadia ... USD 420000, 98%,
Global Sales, stage completed. Do not create any new stages."  -> the
assistant answered "I can only help with sales CRM tasks."

Conversation 2: "create the CRM Implementation Request and Deal for Sadia,
with USD 650,000 and 75% probability" -> the assistant asked for optional
details and said to type "skip"; "skip" produced the generic welcome message.

The LLM itself is replaced by a stand-in that only knows what it is shown in
its prompt (it cannot see anything the code removed from the context). Graph,
checkpointing, tools, validation, approval and the CRM save logic are real.
"""
import json
import re
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from rest_framework.test import APIClient

from ai_assistant.agent import (
    SYSTEM_PROMPT, build_graph, build_system_prompt, set_app,
    PROVIDER_ERROR_TEXT,
)
from ai_assistant.context import (
    MAX_HISTORY_CHARS, trim_conversation_history,
)
from ai_assistant.conversation import is_continuation_reply
from ai_assistant.models import ActionRequest
from ai_assistant.tests.base import (
    GLOBAL_SALES, SalesTestCase, ScriptedLLM, tool_call,
)
from ai_assistant.workflow import PLAN_TOOL_NAME
from crm.models import Company, Contact, Deal, Lead, Payment, Request, Stage

# REFUSAL is still used for non-CRM requests
REFUSAL = "I can only help with sales CRM tasks."

CONVERSATION_1 = ("Create a new deal called ERP Integration for the existing company "
                  "Byteboom. Link it to the existing contact Sadia. Set the deal amount "
                  "to 420000 USD, probability to 98%, and assign the department to "
                  "Global Sales. the stage is \"completed\" Do not create any new stages.")
CONVERSATION_2 = ("create the CRM Implementation Request and Deal for Sadia, with "
                  "USD 650,000 and 75% probability")


class VisibleContextLLM:
    """A sensible assistant that only knows what is in the prompt it receives.

    * the exact first conversation: plans the deal (or answers the validation
      error it is shown);
    * the second: looks up Sadia and asks for optional details; on "skip" it
      continues the task if the task is visible in its prompt, otherwise it
      returns the provider error message (since greetings now go to LLM).
    Like real providers, its tool-call messages carry bulky metadata.
    """

    def __init__(self, stage="completed"):
        self.stage = stage
        self.prompts = []

    @staticmethod
    def _with_metadata(message: AIMessage) -> AIMessage:
        message.additional_kwargs = {"reasoning_content": "r" * 3500}
        message.response_metadata = {"token_usage": {"x": "y" * 3500}}
        return message

    def invoke(self, messages):
        self.prompts.append(messages)
        system = messages[0].content
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        last = humans[-1]
        # tool results produced since the newest user message
        since = messages[max(i for i, m in enumerate(messages)
                             if isinstance(m, HumanMessage)) + 1:]
        results = [m for m in since if isinstance(m, ToolMessage)]

        if "ERP Integration" in last:
            if results and results[-1].name == PLAN_TOOL_NAME:
                return AIMessage(content=f"Could not plan it: {results[-1].content}")
            if not results:
                return self._with_metadata(tool_call("search_contacts", {"name": "Sadia"}))
            contact_id = self._contact_id(results[-1].content)
            return self._with_metadata(tool_call(PLAN_TOOL_NAME, {
                "goal": "Create the ERP Integration deal for Sadia",
                "operations": [
                    {"op_id": "op1", "action": "create", "record_type": "request",
                     "create_deal": True, "fields": [
                         {"name": "request_for", "value": "ERP Integration"},
                         {"name": "contact", "value": str(contact_id)},
                         {"name": "department", "value": "Global Sales"}]},
                    {"op_id": "op2", "action": "update", "record_type": "deal",
                     "record_id": "$op1.deal", "fields": [
                         {"name": "amount", "value": "420000"},
                         {"name": "currency", "value": "USD"},
                         {"name": "probability", "value": "98"},
                         {"name": "stage", "value": self.stage}]}],
            }, "plan-1"))

        task_visible = "Sadia" in system.split("SCOPE")[0] or any(
            "Sadia" in h for h in humans[:-1])
        if last.strip().lower() == "skip":
            if not task_visible:
                return AIMessage(content=PROVIDER_ERROR_TEXT)   # greetings now go to LLM
            contact_id = self._contact_id(*[m.content for m in messages
                                            if isinstance(m, ToolMessage)
                                            and m.name == "search_contacts"])
            if contact_id is None:
                return self._with_metadata(tool_call("search_contacts", {"name": "Sadia"}))
            return self._with_metadata(tool_call(PLAN_TOOL_NAME, {
                "goal": "Register the CRM Implementation opportunity for Sadia",
                "operations": [
                    {"op_id": "op1", "action": "create", "record_type": "request",
                     "create_deal": True, "fields": [
                         {"name": "request_for", "value": "CRM Implementation"},
                         {"name": "contact", "value": str(contact_id)}]},
                    {"op_id": "op2", "action": "update", "record_type": "deal",
                     "record_id": "$op1.deal", "fields": [
                         {"name": "amount", "value": "650000"},
                         {"name": "currency", "value": "USD"},
                         {"name": "probability", "value": "75"}]}],
            }, "plan-2"))

        if "Sadia" in last:
            if not results:
                return self._with_metadata(tool_call("search_contacts", {"name": "Sadia"}, "c1"))
            if len(results) == 1:
                return self._with_metadata(tool_call("search_requests", {"query": "Sadia"}, "c2"))
            return AIMessage(content=(
                "We have a contact Sadia at Byteboom. We have no requests for Sadia. "
                "More details can be added, or type \"skip\" to create the request and deal."))
        # Unknown message (e.g., greeting) -> provider error since no LLM available
        return AIMessage(content=PROVIDER_ERROR_TEXT)

    @staticmethod
    def _contact_id(*contents):
        for content in contents:
            found = re.search(r'"contacts": \[\{"id": (\d+)', content)
            if found:
                return int(found.group(1))
        return None


class ConversationBase(SalesTestCase):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.byteboom = Company.objects.create(
            full_name="Byteboom", email="hello@byteboom.example", phone="+92 21 111 0000",
            website="https://byteboom.example", city_name="Karachi", address="Clifton",
            country=cls.country, owner=cls.manager, department_id=GLOBAL_SALES)
        cls.sadia = Contact.objects.create(
            first_name="Sadia", last_name="Khan", email="sadia@byteboom.example",
            company=cls.byteboom, owner=cls.manager, department_id=GLOBAL_SALES)
        # Test data standing in for a stage that already exists in the CRM.
        cls.completed = Stage.objects.create(
            name="completed", department_id=GLOBAL_SALES, index_number=12)

    def setUp(self):
        self.graph = build_graph(MemorySaver())
        set_app(self.graph)
        self.addCleanup(set_app, None)
        self.client = APIClient()
        self.client.force_login(self.manager)

    def use_llm(self, llm):
        from ai_assistant.llm import get_llm
        get_llm.cache_clear()
        patcher = patch("ai_assistant.agent.get_llm", return_value=llm)
        patcher.start()
        self.addCleanup(patcher.stop)
        return llm

    def chat(self, text, thread="conv"):
        response = self.client.post("/api/ai/chat/", {"message": text, "thread_id": thread},
                                    format="multipart")
        self.assertEqual(response.status_code, 200, response.content)
        return response.json()

    def approve(self, operation_id, thread="conv"):
        response = self.client.post("/api/ai/approve/", {
            "thread_id": thread, "decision": "approved", "operation_id": operation_id},
            format="json")
        self.assertEqual(response.status_code, 200, response.content)
        return response.json()

    def crm_counts(self):
        return (Request.objects.count(), Deal.objects.count(), Stage.objects.count(),
                Company.objects.count(), Contact.objects.count())


# ---------------------------------------------------------------------------
# Conversation 1
# ---------------------------------------------------------------------------

class ErpIntegrationDealConversation(ConversationBase):
    def test_deal_request_reaches_an_approval_card_and_creates_no_stage(self):
        llm = self.use_llm(VisibleContextLLM())
        before = self.crm_counts()
        body = self.chat(CONVERSATION_1)
        self.assertEqual(body["status"], "pending_approval", body)
        self.assertNotIn(REFUSAL, json.dumps(body))
        self.assertEqual(self.crm_counts(), before, "nothing may be written before approval")

        card = body["payload"]
        request_op, deal_op = card["operations"]
        self.assertEqual(request_op["fields"]["Request for"], "ERP Integration")
        self.assertIn("Sadia", request_op["related"]["Contact"])
        self.assertIn(f"#{self.sadia.pk}", request_op["related"]["Contact"])
        self.assertEqual(request_op["inferred"]["Department"], "Global sales (from your department)")
        self.assertEqual(deal_op["fields"]["Amount"], "420000")
        self.assertEqual(deal_op["fields"]["Probability (%)"], "98")
        self.assertEqual(deal_op["fields"]["Stage"], "completed")
        self.assertEqual(deal_op["warnings"], [])      # no bogus "already set" note

        result = self.approve(card["operation_id"])
        self.assertEqual(result["execution"]["status"], "success", result)
        deal = Deal.objects.get(name="ERP Integration")
        self.assertEqual((float(deal.amount), deal.probability), (420000.0, 98))
        self.assertEqual(deal.stage_id, self.completed.pk)
        self.assertEqual(deal.department_id, GLOBAL_SALES)
        self.assertEqual(deal.contact_id, self.sadia.pk)
        self.assertEqual(deal.company_id, self.byteboom.pk)
        self.assertEqual(Stage.objects.count(), before[2])       # no stage created

    def test_unknown_stage_is_reported_with_real_options_never_invented(self):
        Stage.objects.filter(pk=self.completed.pk).delete()
        self.use_llm(VisibleContextLLM())
        stages = Stage.objects.count()
        body = self.chat(CONVERSATION_1)
        self.assertEqual(body["status"], "success", body)            # no card
        self.assertIn("'completed' is not available to you", body["message"])
        self.assertIn("closed (successful)", body["message"])        # real options listed
        self.assertNotIn(REFUSAL, body["message"])
        self.assertEqual(Stage.objects.count(), stages)
        self.assertFalse(Deal.objects.filter(name="ERP Integration").exists())
        self.assertFalse(Request.objects.filter(request_for="ERP Integration").exists())

    def test_naming_another_department_is_refused_not_applied(self):
        from ai_assistant.access import CrmAccess
        from ai_assistant.tests.test_workflow import op, plan
        from ai_assistant.workflow import validate_plan
        outcome = validate_plan(CrmAccess(self.manager), plan(op(
            "op1", "create", "request", request_for="X", create_deal=True,
            contact=self.sadia.pk, department="Local sales")))
        self.assertFalse(outcome.ok)
        self.assertIn("cannot be set", outcome.message)
        self.assertIn("Global sales", outcome.message)

    def test_prompt_marks_deal_requests_as_in_scope(self):
        prompt = build_system_prompt(CONVERSATION_1).content
        self.assertIn("creating a deal", prompt)
        self.assertIn("never refuse these", prompt)
        self.assertIn("Create a deal called X", prompt)
        self.assertIn("Never create stages", prompt)
        # the one-sentence refusal is still available for non-CRM requests
        self.assertIn("clearly unrelated to the CRM", SYSTEM_PROMPT)
        self.assertIn("you can only help with sales CRM tasks", SYSTEM_PROMPT)


class CompleteNewDealConversation(ConversationBase):
    def setUp(self):
        super().setUp()
        Stage.objects.create(
            name="commercial offer", department_id=GLOBAL_SALES, index_number=99)

    def nestle_plan(self):
        return {
            "goal": "Onboard Nestle Pakistan and create its deal and payment",
            "operations": [
                {"op_id": "op1", "action": "create", "record_type": "company",
                 "fields": [
                     {"name": "full_name", "value": "Nestle Pakistan"},
                     {"name": "email", "value": "info@nestle.example.com"},
                     {"name": "website", "value": "https://nestle.example.com"},
                     {"name": "country", "value": self.country.name},
                 ]},
                {"op_id": "op2", "action": "create", "record_type": "lead",
                 "fields": [
                     {"name": "first_name", "value": "Robert"},
                     {"name": "email", "value": "robert@example.com"},
                     {"name": "phone", "value": "03111222333"},
                     {"name": "company", "value": "$op1"},
                 ]},
                {"op_id": "op3", "action": "create", "record_type": "contact",
                 "fields": [
                     {"name": "first_name", "value": "Hamza"},
                     {"name": "email", "value": "hamza@example.com"},
                     {"name": "title", "value": "Marketing Director"},
                     {"name": "company", "value": "$op1"},
                 ]},
                {"op_id": "op4", "action": "create", "record_type": "request",
                 "create_deal": True,
                 "fields": [
                     {"name": "request_for", "value": "Nestle Marketing Automation"},
                     {"name": "contact", "value": "$op3"},
                 ]},
                {"op_id": "op5", "action": "update", "record_type": "deal",
                 "record_id": "$op4.deal",
                 "fields": [
                     {"name": "amount", "value": "7500"},
                     {"name": "currency", "value": "USD"},
                     {"name": "probability", "value": "85"},
                     {"name": "stage", "value": "commercial offer"},
                 ]},
                {"op_id": "op6", "action": "create", "record_type": "payment",
                 "fields": [
                     {"name": "deal", "value": "$op4.deal"},
                     {"name": "amount", "value": "20"},
                     {"name": "currency", "value": "USD"},
                 ]},
            ],
        }

    def submit_nestle_plan(self, thread):
        llm = ScriptedLLM(tool_call(PLAN_TOOL_NAME, self.nestle_plan()))
        self.use_llm(llm)
        response = self.chat(
            "Onboard Nestle Pakistan and create the Nestle Marketing Automation "
            "deal for Hamza, amount 7500 USD, probability 85%, commercial offer "
            "stage, with a 20 USD payment. Skip optional details.",
            thread=thread)
        if len(llm.prompts) > 1:
            feedback = [message.content for message in llm.prompts[-1]
                        if isinstance(message, ToolMessage)]
            self.fail(feedback[-1] if feedback else "The plan was unexpectedly retried.")
        return response

    def crm_counts(self):
        return tuple(model.objects.count() for model in (
            Company, Lead, Contact, Request, Deal, Payment, Stage))

    def test_complete_customer_deal_and_payment_share_one_approval(self):
        before = self.crm_counts()
        body = self.submit_nestle_plan("nestle-complete")

        self.assertEqual(body["status"], "pending_approval", body)
        self.assertEqual(self.crm_counts(), before)
        operations = body["payload"]["operations"]
        self.assertEqual(
            [(op["action"], op["record_type"]) for op in operations],
            [("create", "company"), ("create", "lead"), ("create", "contact"),
             ("create", "request"), ("update", "deal"), ("create", "payment")])
        self.assertEqual(operations[4]["fields"]["Amount"], "7500")
        self.assertEqual(operations[4]["fields"]["Currency"], "USD")
        self.assertEqual(operations[4]["fields"]["Probability (%)"], "85")
        self.assertEqual(operations[4]["fields"]["Stage"], "commercial offer")
        self.assertEqual(operations[4]["related"]["Deal"],
                         "the new deal created by op4")
        self.assertEqual(operations[5]["fields"]["Amount"], "20")
        self.assertEqual(operations[5]["fields"]["Currency"], "USD")
        self.assertEqual(operations[5]["related"]["Deal"],
                         "the new deal created by op4")

        result = self.approve(body["payload"]["operation_id"], thread="nestle-complete")
        self.assertEqual(result["execution"]["status"], "success", result)

        company = Company.objects.get(full_name="Nestle Pakistan")
        self.assertEqual(company.country, self.country)
        lead = Lead.objects.get(email="robert@example.com")
        contact = Contact.objects.get(email="hamza@example.com")
        request = Request.objects.get(request_for="Nestle Marketing Automation")
        deal = Deal.objects.get(request=request)
        payment = Payment.objects.get(deal=deal)
        self.assertEqual(lead.company, company)
        self.assertEqual(contact.company, company)
        self.assertEqual(request.contact, contact)
        self.assertEqual(request.company, company)
        self.assertEqual((deal.contact, deal.company), (contact, company))
        self.assertEqual(
            (deal.amount, deal.currency.name, deal.probability, deal.stage.name),
            (7500, "USD", 85, "commercial offer"))
        self.assertEqual((payment.deal, payment.amount, payment.currency.name),
                         (deal, 20, "USD"))

    def test_rejecting_complete_customer_deal_plan_writes_no_crm_records(self):
        before = self.crm_counts()
        body = self.submit_nestle_plan("nestle-reject")
        self.assertEqual(body["status"], "pending_approval", body)
        self.assertEqual(self.crm_counts(), before)

        response = self.client.post("/api/ai/approve/", {
            "thread_id": "nestle-reject",
            "decision": "rejected",
            "operation_id": body["payload"]["operation_id"],
        }, format="json")
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(self.crm_counts(), before)
        self.assertEqual(
            ActionRequest.objects.get(
                operation_id=body["payload"]["operation_id"]).status,
            ActionRequest.REJECTED)


# ---------------------------------------------------------------------------
# Conversation 2
# ---------------------------------------------------------------------------

class SkipConversation(ConversationBase):
    def run_first_two_turns(self):
        body = self.chat("Create a new deal called ERP Integration for the existing company "
                         "Byteboom. Link it to the existing contact Sadia ... Do not create "
                         "any new stages.", thread="skip")
        # (turn 1 is conversation 1; approve nothing: the next message rejects it)
        self.assertEqual(body["status"], "pending_approval")
        body = self.chat(CONVERSATION_2, thread="skip")
        self.assertEqual(body["status"], "success", body)
        self.assertIn('type "skip"', body["message"])
        return body

    def test_skip_continues_the_task_and_shows_an_approval_card(self):
        llm = self.use_llm(VisibleContextLLM())
        self.run_first_two_turns()
        before = self.crm_counts()
        body = self.chat("skip", thread="skip")

        # skip must not restart the chat (should show approval card)
        self.assertEqual(body["status"], "pending_approval", body)
        self.assertEqual(self.crm_counts(), before, "typing skip must not write to the CRM")
        self.assertFalse(Deal.objects.filter(name="CRM Implementation").exists())

        pending = ActionRequest.objects.get(operation_id=body["payload"]["operation_id"])
        self.assertEqual(pending.status, ActionRequest.PENDING)

        card = body["payload"]
        self.assertEqual(card["operations"][0]["fields"]["Request for"], "CRM Implementation")
        self.assertEqual(card["operations"][1]["fields"]["Amount"], "650000")
        self.assertEqual(card["operations"][1]["fields"]["Probability (%)"], "75")
        self.assertIn(f"#{self.sadia.pk}", card["operations"][0]["related"]["Contact"])

        # what the model was shown on "skip": the task, and the continuation note
        shown = llm.prompts[-1] if llm.prompts else []
        system = [p for p in llm.prompts if p][-1][0].content
        self.assertIn(CONVERSATION_2, system)
        self.assertIn('"skip"', system)

        # only the user's approval writes - all or nothing
        result = self.approve(card["operation_id"], thread="skip")
        self.assertEqual(result["execution"]["status"], "success", result)
        deal = Deal.objects.get(name="CRM Implementation")
        self.assertEqual((float(deal.amount), deal.probability), (650000.0, 75))
        self.assertEqual(deal.contact_id, self.sadia.pk)
        pending.refresh_from_db()
        self.assertEqual(pending.status, ActionRequest.EXECUTED)

    def test_rejecting_the_card_after_skip_writes_nothing(self):
        self.use_llm(VisibleContextLLM())
        self.run_first_two_turns()
        before = self.crm_counts()
        body = self.chat("skip", thread="skip")
        response = self.client.post("/api/ai/approve/", {
            "thread_id": "skip", "decision": "rejected",
            "operation_id": body["payload"]["operation_id"]}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.crm_counts(), before)

    def test_the_pending_task_survives_in_the_checkpoint(self):
        self.use_llm(VisibleContextLLM())
        self.run_first_two_turns()
        self.chat("skip", thread="skip")
        state = self.graph.get_state(
            {"configurable": {"thread_id": f"user-{self.manager.pk}:skip"}}).values
        self.assertEqual(state["original_request"], CONVERSATION_2)
        self.assertEqual(state["latest_reply"], "skip")

    def test_skip_with_nothing_to_continue_is_a_normal_message(self):
        self.use_llm(VisibleContextLLM())
        body = self.chat("skip", thread="fresh")
        # Greetings/short messages now go to LLM; with test LLM returns PROVIDER_ERROR_TEXT
        self.assertEqual(body["message"], PROVIDER_ERROR_TEXT)
        state = self.graph.get_state(
            {"configurable": {"thread_id": f"user-{self.manager.pk}:fresh"}}).values
        self.assertEqual(state["original_request"], "skip")
        self.assertEqual(state["latest_reply"], "")

    def test_a_new_request_replaces_the_old_task(self):
        self.use_llm(VisibleContextLLM())
        self.run_first_two_turns()
        self.chat("What is the weather?", thread="skip")
        state = self.graph.get_state(
            {"configurable": {"thread_id": f"user-{self.manager.pk}:skip"}}).values
        self.assertEqual(state["original_request"], "What is the weather?")

    def test_a_reply_typed_over_an_approval_card_does_not_continue_it(self):
        self.use_llm(VisibleContextLLM())
        body = self.chat(CONVERSATION_1, thread="card")
        self.assertEqual(body["status"], "pending_approval")
        before = self.crm_counts()
        self.chat("yes", thread="card")      # typed instead of using the card
        self.assertEqual(self.crm_counts(), before)
        self.assertEqual(
            ActionRequest.objects.get(operation_id=body["payload"]["operation_id"]).status,
            ActionRequest.REJECTED)

    def test_uploads_are_never_treated_as_continuations(self):
        self.use_llm(VisibleContextLLM())
        self.run_first_two_turns()
        response = self.client.post("/api/ai/chat/", {
            "message": "skip", "thread_id": "skip",
            "file": SimpleUploadedFile("a.txt", b"Acme wants things")}, format="multipart")
        self.assertEqual(response.status_code, 200)
        state = self.graph.get_state(
            {"configurable": {"thread_id": f"user-{self.manager.pk}:skip"}}).values
        self.assertEqual(state["latest_reply"], "")


class ContinuationReplyTests(SalesTestCase):
    def test_recognises_short_answers_only(self):
        for text in ("skip", "Skip", " skip. ", "none", "No", "that's all", "that’s all",
                     "just create it", "go ahead", "OK", "yes", "Nothing else"):
            self.assertTrue(is_continuation_reply(text), text)
        for text in ("", "   ", "skip the part about Byteboom", "no, use Acme instead",
                     "create a lead for Nina", "x" * 41, "yes please create the deal now"):
            self.assertFalse(is_continuation_reply(text), text)


class HistoryTrimmingTests(SalesTestCase):
    def history(self, final):
        big = {"reasoning_content": "r" * 4000}
        return [
            HumanMessage(content="Create a deal for Sadia, USD 650,000"),
            AIMessage(content="", tool_calls=[{"name": "search_contacts", "args": {"query": "Sadia"},
                                               "id": "c1", "type": "tool_call"}],
                      additional_kwargs=big),
            ToolMessage(content='{"contacts": [{"id": 7}]}' + " pad" * 2000, tool_call_id="c1",
                        name="search_contacts"),
            AIMessage(content='We have Sadia. Type "skip" to create it.', additional_kwargs=big),
            HumanMessage(content=final),
        ]

    def test_skip_keeps_the_turn_it_answers(self):
        trimmed = trim_conversation_history(self.history("skip"))
        self.assertEqual(trimmed[0].content, "Create a deal for Sadia, USD 650,000")
        self.assertEqual(trimmed[-1].content, "skip")
        # tool-call / result pairs stay intact, but bulky parts are shortened
        self.assertEqual([type(m) for m in trimmed],
                         [HumanMessage, AIMessage, ToolMessage, AIMessage, HumanMessage])
        self.assertIn('"id": 7', trimmed[2].content)
        self.assertLess(len(trimmed[2].content), 700)
        self.assertEqual(trimmed[1].additional_kwargs, {})

    def test_skip_keeps_the_task_even_when_the_old_turn_is_huge(self):
        messages = self.history("skip")
        messages[0] = HumanMessage(content="Create a deal for Sadia " + "pad " * MAX_HISTORY_CHARS)
        trimmed = trim_conversation_history(messages)
        self.assertIn("Sadia", trimmed[0].content)

    def test_other_messages_still_drop_oversized_old_turns(self):
        trimmed = trim_conversation_history(self.history("What deals are open?")
                                            [:4] + [HumanMessage(content="old " * 100)] * 0
                                            + [HumanMessage(content="What deals are open?")])
        self.assertLessEqual(len(trimmed), 5)

    def test_state_itself_is_not_modified(self):
        messages = self.history("skip")
        original_size = len(messages[2].content)
        trim_conversation_history(messages)
        self.assertEqual(len(messages[2].content), original_size)
        self.assertTrue(messages[1].additional_kwargs)
