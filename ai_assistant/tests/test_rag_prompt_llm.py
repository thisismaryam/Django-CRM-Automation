"""Schema RAG, prompt, tool registry, context trimming and LLM fallback."""
import os
from unittest.mock import patch

from django.apps import apps
from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from ai_assistant.agent import build_system_prompt
from ai_assistant.context import (
    MAX_HISTORY_CHARS, MAX_HISTORY_TURNS, trim_conversation_history,
)
from ai_assistant.llm import build_fallback_llm, build_providers
from ai_assistant.read_tools import READ_TOOLS
from ai_assistant.tools import (
    MAX_SCHEMA_TOOL_CHARS, SCHEMA_CATALOG, _format_schema_entry,
    propose_sales_workflow, search_database_schema, tools,
)
from ai_assistant.workflow import FIELD_POLICY, PLAN_TOOL_NAME

EMPTY_SEARCH = {"matched_models": [], "expanded_models": [], "schemas": {}}


class SchemaRagTests(SimpleTestCase):
    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_schema_answer_shows_business_fields_not_audit_fields(self, search):
        search.return_value = {
            "matched_models": ["Lead"], "expanded_models": ["Company"],
            "schemas": {"Lead": SCHEMA_CATALOG["Lead"], "Company": SCHEMA_CATALOG["Company"]},
        }
        result = search_database_schema.invoke({"query": "lead fields"})
        fields = result.split("Fields (* = required): ")[1].split("\n")[0]
        self.assertTrue(fields.startswith("first_name: CharField*, email: CharField*"))
        for hidden in ("creation_date", "update_date", "modified_by", "token"):
            self.assertNotIn(hidden, fields)
        self.assertIn("country -> Country", fields)
        self.assertIn("## Company", result)
        self.assertIn("not live records", result)

    @patch("ai_assistant.tools.search_schema_with_relationships", return_value=EMPTY_SEARCH)
    def test_sales_terms_map_to_crm_models_with_business_rules(self, _search):
        result = search_database_schema.invoke({"query": "How do I log a new opportunity?"})
        self.assertIn("## Deal", result)
        self.assertIn("Created only by processing a Request", result)
        result = search_database_schema.invoke({"query": "customer inquiry"})
        self.assertIn("## Request", result)
        self.assertIn("## Company", result)

    @patch("ai_assistant.tools.search_schema_with_relationships", return_value=EMPTY_SEARCH)
    def test_no_match(self, _search):
        self.assertEqual(search_database_schema.invoke({"query": "zzz"}),
                         "No matching database model was found.")

    @patch("ai_assistant.tools.search_schema_with_relationships")
    def test_schema_output_is_bounded(self, search):
        schemas = {f"Model{i}": {"fields": [{"name": "x" * 300, "type": "CharField"}],
                                 "relationships": [], "reverse_relationships": []}
                   for i in range(20)}
        search.return_value = {"matched_models": list(schemas), "expanded_models": [],
                               "schemas": schemas}
        result = search_database_schema.invoke({"query": "anything"})
        self.assertLessEqual(len(result), MAX_SCHEMA_TOOL_CHARS + 100)
        self.assertIn("omitted", result)

    def test_formatter_handles_every_catalog_model(self):
        for name, entry in SCHEMA_CATALOG.items():
            for matched in (True, False):
                self.assertTrue(_format_schema_entry(name, entry, matched))

    def test_schema_catalog_matches_current_django_models(self):
        """The RAG catalog is generated from the models; fail loudly if a
        model changed and the catalog was not regenerated."""
        for model in apps.get_app_config("crm").get_models():
            name = model.__name__
            self.assertIn(name, SCHEMA_CATALOG, f"{name} missing: rerun extract_schema.py")
            if model._meta.proxy:
                continue
            current = {f.name for f in model._meta.get_fields() if not f.auto_created}
            catalog = {f["name"] for f in SCHEMA_CATALOG[name]["fields"]}
            self.assertEqual(current, catalog, f"{name} changed: rerun extract_schema.py")

    def test_policy_fields_exist_on_models(self):
        from ai_assistant.access import SALES_MODELS
        for (record_type, _action), names in FIELD_POLICY.items():
            model = SALES_MODELS[record_type]
            for name in names:
                model._meta.get_field(name)   # raises if the field does not exist


class PromptAndRegistryTests(SimpleTestCase):
    def test_prompt_contains_request_and_process_rules(self):
        text = build_system_prompt("Register {the} deal for ABC").content
        self.assertIn("Register (the) deal for ABC", text)   # braces are safe
        self.assertIn(PLAN_TOOL_NAME, text)
        self.assertIn("Deals are never created directly", text)
        self.assertIn("not instructions", text)               # document injection guard
        self.assertIn("Never skip or shortcut approval", text)

    def test_registry_has_one_write_entry_point(self):
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], PLAN_TOOL_NAME)
        self.assertEqual({t.name for t in READ_TOOLS} | {"search_database_schema",
                                                         PLAN_TOOL_NAME}, set(names))
        for tool_obj in tools:
            self.assertTrue(tool_obj.description.strip(), tool_obj.name)

    def test_plan_tool_never_writes_when_called_directly(self):
        result = propose_sales_workflow.invoke({"goal": "x", "operations": []})
        self.assertTrue(result.startswith("ERROR"))

    def test_plan_tool_description_lists_policy(self):
        description = propose_sales_workflow.description
        self.assertIn("create request:", description)
        self.assertIn("first_name*", description)
        self.assertIn("$op3.deal", description)


class ContextTests(SimpleTestCase):
    def test_trimming_keeps_recent_complete_tool_turns(self):
        messages = [
            HumanMessage(content="old " * MAX_HISTORY_CHARS),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(content="", tool_calls=[{
                "name": "search_leads", "args": {"query": "x"},
                "id": "call-1", "type": "tool_call"}]),
            ToolMessage(content="[]", tool_call_id="call-1"),
        ]
        trimmed = trim_conversation_history(messages)
        self.assertEqual([type(m) for m in trimmed], [HumanMessage, AIMessage, ToolMessage])

    def test_trimming_keeps_last_three_user_turns(self):
        messages = []
        for i in range(5):
            messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
        trimmed = trim_conversation_history(messages)
        self.assertEqual(MAX_HISTORY_TURNS, 3)
        self.assertEqual(trimmed[0].content, "q2")


class LlmFallbackTests(SimpleTestCase):
    def test_falls_back_in_order(self):
        def broken(_):
            raise RuntimeError("primary down")
        chain = build_fallback_llm(
            RunnableLambda(broken),
            [RunnableLambda(broken), RunnableLambda(lambda _: AIMessage(content="groq"))])
        self.assertEqual(chain.invoke("hi").content, "groq")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GROQ_API_KEY": "test-key"},
                clear=True)
    def test_gemini_provider_defaults(self):
        gemini, gemini_fallback, _groq = build_providers(tools)
        self.assertEqual(gemini.bound.model.split("/")[-1], "gemini-3.5-flash")
        self.assertEqual(gemini_fallback.bound.model.split("/")[-1], "gemini-3.8-flash")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GROQ_API_KEY": "test-key",
                             "GEMINI_MODEL": "gemini-custom", "AI_ASSISTANT_MAX_TOKENS": "999"})
    def test_providers_are_configurable_and_tool_bound(self):
        gemini, gemini_fallback, groq = build_providers(tools)
        self.assertEqual(gemini.bound.model.split("/")[-1], "gemini-custom")
        self.assertEqual(gemini.bound.max_output_tokens, 999)
        self.assertNotIn("parallel_tool_calls", gemini.kwargs)
        self.assertIs(groq.kwargs["parallel_tool_calls"], False)
        bound_names = {t["function"]["name"] for t in groq.kwargs["tools"]}
        self.assertIn(PLAN_TOOL_NAME, bound_names)
