"""Unit tests for the local intent router, dynamic prompt assembly,
dynamic tool binding, and token usage logging.
"""
import logging
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_assistant.agent import (
    HEADER_SECTION,
    PLAN_TOOL_NAME,
    READ_SECTION,
    SCHEMA_SECTION,
    SYSTEM_PROMPT,
    build_system_prompt,
    get_tools_for_intent,
)
from ai_assistant.llm import _extract_token_usage, _model_name
from ai_assistant.read_tools import READ_TOOLS
from ai_assistant.router import (
    INTENT_GREETING,
    INTENT_READ,
    INTENT_SCHEMA,
    INTENT_WRITE,
    classify_intent,
    is_greeting,
    is_read_query,
    is_schema_query,
    is_write_query,
)
from ai_assistant.tools import propose_sales_workflow, search_database_schema, tools


class IntentRouterTests(SimpleTestCase):
    def test_greeting_intent_classification(self):
        greetings = [
            "hello", "hi", "hey!", "Good morning.", "good evening", "howdy",
            "hello there", "what can you do", "help me",
        ]
        for g in greetings:
            self.assertTrue(is_greeting(g), f"Expected {g!r} to be recognized as greeting")
            self.assertEqual(classify_intent(g), INTENT_GREETING)

    def test_schema_intent_classification(self):
        schema_queries = [
            "what fields does a deal have",
            "show me the schema for contacts",
            "how is a Contact linked to a Company",
            "what is the CRM data model for leads",
            "what is a request in the crm?",
            "what models exist in the database?",
            "required fields for company",
        ]
        for q in schema_queries:
            self.assertTrue(is_schema_query(q), f"Expected {q!r} to be recognized as schema query")
            self.assertEqual(classify_intent(q), INTENT_SCHEMA)

    def test_read_intent_classification(self):
        read_queries = [
            "search leads named Alice",
            "find company Acme Corp",
            "show me deals in the pipeline",
            "who is Leo Lead?",
            "sales summary for this quarter",
            "what payments are recorded for deal #12?",
            "list contacts for Acme",
        ]
        for q in read_queries:
            self.assertTrue(is_read_query(q), f"Expected {q!r} to be recognized as read query")
            self.assertEqual(classify_intent(q), INTENT_READ)

    def test_write_intent_classification(self):
        write_queries = [
            "Create a deal called ERP for Acme Corp",
            "add a new lead named Bob with email bob@example.com",
            "Register the opportunity for NovaTech",
            "update deal #5 next step to Call tomorrow",
            "set amount to 5000 USD and stage to closed",
            "Add Nina to the CRM",
            "create a payment of $500 for Acme widgets",
        ]
        for q in write_queries:
            self.assertTrue(is_write_query(q), f"Expected {q!r} to be recognized as write query")
            self.assertEqual(classify_intent(q), INTENT_WRITE)

    def test_continuation_and_document_routing(self):
        # A continuation reply ("skip", "that's all", "yes") routes to write
        self.assertEqual(classify_intent("skip"), INTENT_WRITE)
        self.assertEqual(classify_intent("that's all"), INTENT_WRITE)
        self.assertEqual(classify_intent("", latest_reply="skip"), INTENT_WRITE)

        # Uploaded document routes to write workflow
        doc_msg = "Please process this:\n\n--- UPLOADED DOCUMENT CONTENT (cv.pdf) ---\nJohn Doe..."
        self.assertEqual(classify_intent(doc_msg), INTENT_WRITE)

    def test_write_precedes_read_when_combined(self):
        # "Find Acme and create a deal for them" contains both find and create
        combined = "Find Acme Corp and create a deal for 10000 USD"
        self.assertEqual(classify_intent(combined), INTENT_WRITE)


class DynamicPromptAndToolBindingTests(SimpleTestCase):
    def test_greeting_prompt_and_tools(self):
        prompt = build_system_prompt("hello", intent=INTENT_GREETING).content
        bound_tools = get_tools_for_intent(INTENT_GREETING)

        # Greeting intent binds 0 tools
        self.assertEqual(bound_tools, [])

        # Prompt contains scope & capabilities, but excludes full workflow process
        self.assertIn("GREETINGS AND CAPABILITIES", prompt)
        self.assertNotIn("CRM SALES PROCESS", prompt)
        self.assertNotIn("propose_sales_workflow", prompt)

        # Token footprint is significantly smaller than full prompt
        self.assertLess(len(prompt), len(SYSTEM_PROMPT) // 2)

    def test_schema_prompt_and_tools(self):
        prompt = build_system_prompt("what fields does a deal have", intent=INTENT_SCHEMA).content
        bound_tools = get_tools_for_intent(INTENT_SCHEMA)

        # Schema intent binds ONLY search_database_schema
        self.assertEqual(bound_tools, [search_database_schema])

        # Prompt focuses on schema inquiries without write workflow rules
        self.assertIn("CRM SCHEMA INQUIRIES", prompt)
        self.assertNotIn("CRM SALES PROCESS", prompt)
        self.assertNotIn("COMPLETENESS CHECK", prompt)

    def test_read_prompt_and_tools(self):
        prompt = build_system_prompt("find Acme", intent=INTENT_READ).content
        bound_tools = get_tools_for_intent(INTENT_READ)

        # Read intent binds read tools only; NO propose_sales_workflow and NO schema tool
        self.assertEqual(
            sorted(t.name for t in bound_tools),
            sorted(t.name for t in READ_TOOLS),
        )
        self.assertNotIn(propose_sales_workflow.name, [t.name for t in bound_tools])
        self.assertNotIn(search_database_schema.name, [t.name for t in bound_tools])

        # Prompt includes direct result formatting instruction
        self.assertIn("READING CRM DATA", prompt)
        self.assertIn("FORMAT RESULTS DIRECTLY", prompt)
        self.assertNotIn("CRM SALES PROCESS", prompt)

    def test_write_prompt_and_tools(self):
        prompt = build_system_prompt("Create a deal for Acme", intent=INTENT_WRITE).content
        bound_tools = get_tools_for_intent(INTENT_WRITE)

        # Write intent binds propose_sales_workflow and read tools (excluding schema)
        self.assertIn(propose_sales_workflow, bound_tools)
        self.assertNotIn(search_database_schema, bound_tools)

        # Prompt includes workflow rules and instructs avoiding preliminary searches
        self.assertIn("Avoid unnecessary preliminary searches", prompt)
        self.assertIn("CRM SALES PROCESS", prompt)
        self.assertIn(PLAN_TOOL_NAME, prompt)


class UsageLoggingTests(SimpleTestCase):
    def test_token_usage_extraction_from_usage_metadata(self):
        msg = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
        )
        usage = _extract_token_usage(msg)
        self.assertEqual(usage["input_tokens"], 120)
        self.assertEqual(usage["output_tokens"], 45)
        self.assertEqual(usage["total_tokens"], 165)

    def test_token_usage_extraction_from_response_metadata(self):
        msg = AIMessage(
            content="ok",
            response_metadata={"token_usage": {"prompt_tokens": 300, "completion_tokens": 50}},
        )
        usage = _extract_token_usage(msg)
        self.assertEqual(usage["input_tokens"], 300)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 350)

    def test_model_name_extraction(self):
        mock_model = MagicMock()
        mock_model.bound.model_name = "test-model-1"
        self.assertEqual(_model_name(mock_model), "test-model-1")

        mock_model_2 = MagicMock()
        del mock_model_2.bound.model_name
        mock_model_2.bound.model = "test-model-2"
        self.assertEqual(_model_name(mock_model_2), "test-model-2")
