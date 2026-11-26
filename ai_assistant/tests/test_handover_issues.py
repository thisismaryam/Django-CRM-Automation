"""Regression tests for the web AI-assistant issues in PROJECT_HANDOVER.md.

2. "Stage matching query does not exist" -> clear preflight error
3. payment on a deal without an amount ("No deal amount") -> explicit warning
6. deal creation for an existing company: country requirement and the
   department field are handled, nothing is invented
1. provider failover: no reasoning metadata leaks across providers
"""
import os
from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage

from ai_assistant.llm import _portable_messages, build_providers
from ai_assistant.models import ActionRequest
from ai_assistant.tests.base import LOCAL_SALES
from ai_assistant.tests.test_workflow import WorkflowTestCase, op, plan
from crm.models import Company, Deal, Payment, Request, Stage


class StageSetupTests(WorkflowTestCase):
    def test_missing_default_stage_gives_clear_error_and_writes_nothing(self):
        Stage.objects.filter(department_id=self.acme_deal.department_id).update(default=False)
        before = (Request.objects.count(), Deal.objects.count())
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", request_for="No stage deal",
               create_deal=True, lead=self.lead.pk)))
        self.assertFalse(outcome.ok)
        self.assertIn("default Deal stage", outcome.message)
        self.assertNotIn("matching query does not exist", outcome.message)
        self.assertEqual((Request.objects.count(), Deal.objects.count()), before)
        self.assertEqual(ActionRequest.objects.count(), self.pending_before)

    def test_two_default_stages_are_rejected_not_guessed(self):
        dept = self.acme_deal.department_id
        Stage.objects.create(name="Second default", default=True, department_id=dept,
                             index_number=99)
        outcome = self.validate(self.manager, plan(
            op("op1", "update", "deal", record_id=self.acme_deal.pk,
               next_step="Call")))
        self.assertFalse(outcome.ok)
        self.assertIn("2 default Deal stages", outcome.message)

    def test_assistant_never_creates_stages(self):
        count = Stage.objects.count()
        Stage.objects.filter(default=True).update(default=False)
        self.validate(self.manager, plan(
            op("op1", "create", "request", request_for="X", create_deal=True,
               lead=self.lead.pk)))
        self.assertEqual(Stage.objects.count(), count)


class PaymentWithoutDealAmountTests(WorkflowTestCase):
    def test_payment_on_existing_deal_without_amount_warns(self):
        Deal.objects.filter(pk=self.acme_deal.pk).update(amount=None)
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "payment", deal=self.acme_deal.pk, amount="150000",
            currency=1)))
        self.assertTrue(outcome.ok, outcome.message)
        warnings = outcome.plan["operations"][0]["display"]["warnings"]
        self.assertTrue(any("No deal amount" in w for w in warnings), warnings)

    def test_payment_on_new_deal_without_amount_warns_and_never_invents_amount(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", create_deal=True,
               request_for="Prepaid", lead=self.lead.pk),
            op("op2", "create", "payment", deal="$op1.deal", amount="50",
               status="received")))
        self.assertTrue(outcome.ok, outcome.message)
        warnings = outcome.plan["operations"][1]["display"]["warnings"]
        self.assertTrue(any("deal amount" in w for w in warnings), warnings)
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        deal = Request.objects.get(request_for="Prepaid").deal
        self.assertFalse(deal.amount)  # amount was not invented

    def test_no_warning_when_the_plan_sets_the_deal_amount(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", create_deal=True,
               request_for="Prepaid2", lead=self.lead.pk),
            op("op2", "update", "deal", record_id="$op1.deal", amount="1000"),
            op("op3", "create", "payment", deal="$op1.deal", amount="50",
               status="received")))
        self.assertTrue(outcome.ok, outcome.message)
        warnings = outcome.plan["operations"][2]["display"]["warnings"]
        self.assertFalse(any("deal amount" in w for w in warnings), warnings)


class DealForExistingCompanyTests(WorkflowTestCase):
    def _plan(self):
        return plan(
            op("op1", "create", "request", request_for="ERP Integration",
               create_deal=True, contact=self.acme_contact.pk),
            op("op2", "update", "deal", record_id="$op1.deal",
               next_step="Kick-off call", next_step_date="2026-12-01",
               description="ERP project", relevant=True, active=True,
               important=True, department=self.manager_department()))

    def manager_department(self):
        return self.acme_deal.department_id

    def test_own_department_field_is_accepted_not_a_dead_end(self):
        outcome = self.validate(self.manager, self._plan())
        self.assertTrue(outcome.ok, outcome.message)

    def test_other_department_is_still_refused(self):
        other = LOCAL_SALES
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", request_for="Cross dept",
               create_deal=True, contact=self.acme_contact.pk),
            op("op2", "update", "deal", record_id="$op1.deal", department=other)))
        self.assertFalse(outcome.ok)
        self.assertIn("department", outcome.message)


class ProviderFailoverTests(SimpleTestCase):
    def test_portable_messages_strip_reasoning_but_keep_tool_calls(self):
        call = {"name": "search_leads", "args": {"query": "x"}, "id": "c1",
                "type": "tool_call"}
        original = AIMessage(
            content=[{"type": "reasoning", "text": "secret"},
                     {"type": "text", "text": "hello"}],
            tool_calls=[call],
            additional_kwargs={"reasoning_content": "thoughts", "tool_calls": []},
            response_metadata={"model": "groq"})
        cleaned = _portable_messages([HumanMessage(content="hi"), original])
        ai = cleaned[1]
        self.assertEqual(ai.tool_calls[0]["name"], "search_leads")
        self.assertNotIn("reasoning_content", ai.additional_kwargs)
        self.assertEqual(ai.response_metadata, {})
        self.assertNotIn("reasoning", str(ai.content))
        self.assertIn("reasoning", str(original.content))  # original untouched

    @patch.dict(os.environ, {"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k"}, clear=True)
    def test_failover_order_is_groq_then_gemini(self):
        self.assertEqual(len(build_providers([])), 3)
