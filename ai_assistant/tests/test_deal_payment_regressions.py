"""Regressions for deal/payment requests: name-based links, ambiguity, the
department-less superuser chain and the per-message LLM call cap."""
from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_assistant import agent
from ai_assistant.access import CrmAccess
from ai_assistant.tests.base import SalesTestCase
from ai_assistant.workflow import validate_plan


def _fields(**kw):
    return [{"name": k, "value": str(v)} for k, v in kw.items()]


class NameResolutionTests(SalesTestCase):
    def _payment_plan(self, deal_value):
        return {"goal": "payment", "operations": [{
            "op_id": "op1", "action": "create", "record_type": "payment",
            "fields": _fields(deal=deal_value, amount=250)}]}

    def test_payment_resolves_a_unique_deal_name(self):
        outcome = validate_plan(CrmAccess(self.manager),
                                self._payment_plan("Acme widgets"))
        self.assertTrue(outcome.ok, outcome.message)

    def test_payment_resolves_a_deal_ticket(self):
        outcome = validate_plan(CrmAccess(self.manager),
                                self._payment_plan("T-ACME-1"))
        self.assertTrue(outcome.ok, outcome.message)

    def test_unknown_deal_name_is_reported_not_guessed(self):
        outcome = validate_plan(CrmAccess(self.manager),
                                self._payment_plan("No such deal"))
        self.assertFalse(outcome.ok)
        self.assertIn("no deal matching", outcome.message.lower())

    def test_ambiguous_name_lists_ids_and_asks(self):
        outcome = validate_plan(CrmAccess(self.manager),
                                self._payment_plan("T-"))
        self.assertFalse(outcome.ok)
        self.assertIn("matches several", outcome.message)
        self.assertIn(f"#{self.acme_deal.pk}", outcome.message)

    def test_name_lookup_stays_inside_department_scope(self):
        # A Local-sales deal must not be resolvable by a Global-sales user.
        outcome = validate_plan(CrmAccess(self.manager),
                                self._payment_plan("Hidden local deal"))
        self.assertFalse(outcome.ok)
        self.assertIn("accessible to you", outcome.message)


class SuperuserWithoutDepartmentTests(SalesTestCase):
    def test_new_counterparty_chain_fails_with_actionable_message(self):
        admin = self.admin
        admin.is_superuser = True
        admin.save()
        plan = {"goal": "deal", "operations": [
            {"op_id": "op1", "action": "create", "record_type": "lead",
             "fields": _fields(first_name="Sara", email="sara@xyz.example",
                               company_name="XYZ Ltd")},
            {"op_id": "op2", "action": "create", "record_type": "request",
             "create_deal": True,
             "fields": _fields(request_for="CRM setup", lead="$op1")}]}
        access = CrmAccess(admin)
        if access.home_department_id:
            self.skipTest("fixture admin already has a department")
        outcome = validate_plan(access, plan)
        self.assertFalse(outcome.ok)
        self.assertIn("Do NOT retry", outcome.message)


class LoopCapTests(SalesTestCase):
    def test_counts_only_llm_replies_since_last_user_message(self):
        messages = [HumanMessage(content="old"), AIMessage(content="a"),
                    AIMessage(content="b"), HumanMessage(content="new"),
                    AIMessage(content="c"),
                    ToolMessage(content="x", tool_call_id="1", name="t"),
                    AIMessage(content="d")]
        self.assertEqual(agent._llm_calls_this_turn(messages), 2)

    def test_agent_stops_without_calling_the_llm_at_the_cap(self):
        messages = [HumanMessage(content="add a payment")] + [
            AIMessage(content=str(i)) for i in range(agent.MAX_LLM_CALLS_PER_TURN)]

        def boom():
            raise AssertionError("LLM must not be called once the cap is hit")

        from unittest import mock
        with mock.patch.object(agent, "get_llm", boom):
            result = agent.agent_node(
                {"messages": messages, "original_request": "add a payment"})
        self.assertEqual(result["messages"][0].content, agent.LOOP_LIMIT_TEXT)


class SuperuserStageTests(SalesTestCase):
    """A superuser has no department of their own, so the CRM offers no stage
    choices until the deal exists. A deal planned in the same plan must use the
    department resolved for its request."""

    def _plan(self):
        return {"goal": "deal", "operations": [
            {"op_id": "op1", "action": "create", "record_type": "request",
             "create_deal": True,
             "fields": _fields(request_for="Acme automation",
                               contact=self.acme_contact.pk)},
            {"op_id": "op2", "action": "update", "record_type": "deal",
             "record_id": "$op1.deal",
             "fields": _fields(name="Acme automation", amount=7500,
                               currency="USD", probability=85,
                               stage="commercial offer")},
            {"op_id": "op3", "action": "create", "record_type": "payment",
             "fields": _fields(deal="$op1.deal", amount=20, currency="USD")},
        ]}

    def test_unassigned_superuser_can_set_stage_by_name_on_planned_deal(self):
        access = CrmAccess(self.admin)
        self.assertFalse(access.home_department_id)
        outcome = validate_plan(access, self._plan())
        self.assertNotIn("not available to you", outcome.message or "")
        self.assertTrue(outcome.ok, outcome.message)

    def test_department_user_still_works(self):
        outcome = validate_plan(CrmAccess(self.manager), self._plan())
        self.assertTrue(outcome.ok, outcome.message)


class RejectedRequestIsNotRetriedTests(SimpleTestCase):
    """A deterministic HTTP 400/413 must not be re-sent in a loop."""

    def setUp(self):
        super().setUp()
        from ai_assistant.llm import clear_llm_cache
        clear_llm_cache()

    def test_each_model_is_tried_once_then_the_request_fails_fast(self):
        from langchain_core.runnables import RunnableLambda
        from ai_assistant.llm import _invoke_with_failover

        class Rejected(Exception):
            status_code = 400

        calls = []

        def make(name):
            def reject(messages, config=None):
                calls.append(name)
                raise Rejected("bad request")
            return (name, RunnableLambda(reject))

        providers = [make("groq"), make("cloudflare"), make("openrouter")]
        with self.assertRaises(RuntimeError):
            _invoke_with_failover(providers, [HumanMessage(content="hi")])
        self.assertEqual(calls, ["groq", "cloudflare", "openrouter"])

    def test_a_later_model_can_still_succeed_after_a_rejection(self):
        from langchain_core.runnables import RunnableLambda
        from ai_assistant.llm import _invoke_with_failover

        class Rejected(Exception):
            status_code = 400

        def reject(messages, config=None):
            raise Rejected("bad request")

        providers = [("groq", RunnableLambda(reject)),
                     ("gemini", RunnableLambda(lambda m, config=None: AIMessage(content="ok")))]
        result = _invoke_with_failover(providers, [HumanMessage(content="hi")])
        self.assertEqual(result.content, "ok")


class ToolSchemaAcceptsNumbersTests(SimpleTestCase):
    """Providers validate the model's tool call against the advertised JSON
    schema and return HTTP 400 on a mismatch. A multi-operation document makes
    models send amount/probability/record_id as numbers."""

    def _args(self):
        return {"goal": "deal", "operations": [
            {"op_id": "op1", "action": "create", "record_type": "request",
             "create_deal": True,
             "fields": [{"name": "request_for", "value": "Cloud upgrade"}]},
            {"op_id": "op2", "action": "update", "record_type": "deal",
             "record_id": "$op1.deal",
             "fields": [{"name": "amount", "value": 12500},
                        {"name": "probability", "value": 90},
                        {"name": "currency", "value": "USD"}]},
            {"op_id": "op3", "action": "update", "record_type": "company",
             "record_id": 12, "fields": [{"name": "active", "value": True}]},
        ]}

    def test_advertised_schema_accepts_numeric_values(self):
        import jsonschema
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from ai_assistant.tools import propose_sales_workflow
        schema = convert_to_openai_tool(propose_sales_workflow)["function"]["parameters"]
        jsonschema.validate(self._args(), schema)

    def test_numbers_are_turned_into_text(self):
        from ai_assistant.workflow import ProposeWorkflowInput
        parsed = ProposeWorkflowInput(**self._args())
        fields = {f.name: f.value for f in parsed.operations[1].fields}
        self.assertEqual(fields["amount"], "12500")
        self.assertEqual(fields["probability"], "90")
        self.assertEqual(parsed.operations[2].record_id, "12")
