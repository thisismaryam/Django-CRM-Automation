"""Tool registry for the Sales agent.

* Schema RAG: ``search_database_schema`` (CRM *structure*, never live data).
* Read tools: ``ai_assistant.read_tools`` (live, permission-scoped records).
* Write planning: ``propose_sales_workflow``. The graph routes this call to
  the plan-review / approval / execution nodes; it never writes by itself.
"""
import json
from pathlib import Path

from langchain_core.tools import tool

from ai_assistant.rag.search import search_schema_with_relationships
from ai_assistant.read_tools import READ_TOOLS
from ai_assistant.workflow import (
    PLAN_TOOL_NAME,
    ProposeWorkflowInput,
    describe_policy,
)

SCHEMA_CATALOG_PATH = Path(__file__).resolve().parent / \
    "data" / "schema_catalog.json"

# Bounded-output constants for the schema tool
MAX_SCHEMA_FIELDS_PER_MODEL = 45
MAX_SCHEMA_RELATIONSHIPS_PER_MODEL = 10
MAX_SCHEMA_TOOL_CHARS = 4500

# System/audit fields the assistant never fills in; hidden from schema answers.
HIDDEN_SCHEMA_FIELDS = {
    "creation_date", "update_date", "modified_by", "token", "avatar",
}

# Sales vocabulary -> CRM model. Used to make sure a question phrased in
# business terms ("opportunity", "inquiry") retrieves the right model even
# when the embedding search ranks it lower.
SALES_TERMS = {
    "opportunit": "Deal",
    "pipeline": "Deal",
    "deal": "Deal",
    "prospect": "Lead",
    "lead": "Lead",
    "inquir": "Request",
    "enquir": "Request",
    "request": "Request",
    "customer": "Company",
    "account": "Company",
    "organisation": "Company",
    "organization": "Company",
    "company": "Company",
    "contact person": "Contact",
    "contact": "Contact",
    "payment": "Payment",
    "invoice": "Payment",
    "revenue": "Payment",
}

# Business process rules that are not visible in the model schema; taken
# from the CRM's admin code (requestadmin, dealadmin, leadadmin, ...).
SALES_PROCESS_NOTES = {
    "Deal": "Created only by processing a Request (Create deal). Deals cannot be added directly.",
    "Request": "Customer inquiry. Links a Contact (and its Company) or a Lead, not both. Processing it with Create deal creates the Deal.",
    "Lead": "Prospect not yet converted. Converting a Lead creates a Company and Contact and moves its deals/requests to them.",
    "Contact": "A person who always belongs to a Company; owner and department follow the company.",
    "Payment": "Belongs to a Deal; status received/guaranteed/high/low probability.",
    "Company": "Customer organisation; name + country must be unique.",
}

with SCHEMA_CATALOG_PATH.open(encoding="utf-8") as schema_file:
    SCHEMA_CATALOG = json.load(schema_file)


def _format_field(field: dict, relationships: dict) -> str:
    """One compact field description, e.g. `email*`, `country -> Country`,
    `company (Company of contact) -> Company`, `tags -> [Tag]`."""
    text = field["name"]
    if field.get("verbose_name"):
        text += f" ({field['verbose_name']})"
    if field["name"] in relationships:
        rel = relationships[field["name"]]
        target = rel["related_model"]
        text += f" -> [{target}]" if rel["type"] == "ManyToMany" else f" -> {target}"
    else:
        text += f": {field['type']}"
    if field.get("required_for_input"):
        text += "*"
    return text


def _format_schema_entry(name: str, entry: dict, is_matched: bool) -> str:
    """Render one model entry compactly.

    Matched models list their business fields (required first, audit fields
    hidden, foreign keys shown with their target model). Expanded models
    only list relationships, to save context."""
    verbose = entry.get("verbose_name", name)
    if verbose and verbose.lower() != name.lower():
        lines = [f"## {name} ({verbose})"]
    else:
        lines = [f"## {name}"]
    if name in SALES_PROCESS_NOTES:
        lines.append(f"Business rule: {SALES_PROCESS_NOTES[name]}")

    forward = entry.get("relationships", [])
    reverse = entry.get("reverse_relationships", [])[
        :MAX_SCHEMA_RELATIONSHIPS_PER_MODEL
    ]

    if is_matched:
        relationships = {r["field"]: r for r in forward}
        visible = [
            f for f in entry.get("fields", [])
            if f["name"] not in HIDDEN_SCHEMA_FIELDS
        ]
        # Required fields first; sorted() is stable so catalog order is kept.
        visible = sorted(
            visible, key=lambda f: not f.get("required_for_input"))
        shown = visible[:MAX_SCHEMA_FIELDS_PER_MODEL]
        if shown:
            field_lines = ", ".join(
                _format_field(f, relationships) for f in shown
            )
            remaining = len(visible) - len(shown)
            if remaining > 0:
                field_lines += f" [+{remaining} more]"
            lines.append(f"Fields (* = required): {field_lines}")
    else:
        forward = [
            r for r in forward if r["field"] not in HIDDEN_SCHEMA_FIELDS
        ][:MAX_SCHEMA_RELATIONSHIPS_PER_MODEL]
        if forward:
            rel_lines = "; ".join(
                f"{r['field']} -> {r['related_model']} ({r['type']})"
                for r in forward
            )
            lines.append(f"Relationships: {rel_lines}")

    if reverse:
        rev_lines = "; ".join(
            f"{r['related_model']} via {r['field']}"
            for r in reverse
        )
        lines.append(f"Referenced by: {rev_lines}")

    return "\n".join(lines)


def _term_models(query: str) -> list[str]:
    text = query.lower()
    found = []
    for term, model in SALES_TERMS.items():
        if term in text and model not in found and model in SCHEMA_CATALOG:
            found.append(model)
    return found


@tool
def search_database_schema(query: str) -> str:
    """Look up the CRM data model (models, fields, required fields,
    relationships and Sales business rules). This describes STRUCTURE only;
    it never returns actual CRM records. Use it for questions such as "what
    fields does a deal have" or "how is a contact linked to a company".
    """
    result = search_schema_with_relationships(query, top_k=2)
    term_models = _term_models(query)
    if not result["schemas"] and not term_models:
        return "No matching database model was found."

    matched_list = list(dict.fromkeys(term_models[:2] + list(result["matched_models"])))
    matched = set(matched_list)
    schemas = dict(result["schemas"])
    for name in term_models:
        schemas.setdefault(name, SCHEMA_CATALOG[name])
    ordered = matched_list + [
        n for n in result["expanded_models"] if n not in matched
    ]

    blocks = ["(Schema information: describes CRM structure, not live records.)"]
    total_chars = len(blocks[0])

    for name in ordered:
        entry = schemas.get(name)
        if not entry:
            continue

        block = _format_schema_entry(
            name, entry, is_matched=(name in matched))

        if total_chars + len(block) + 2 > MAX_SCHEMA_TOOL_CHARS:
            blocks.append(
                "[additional related models omitted to fit context limit]")
            break

        blocks.append(block)
        total_chars += len(block) + 2

    return "\n\n".join(blocks)


@tool(PLAN_TOOL_NAME, args_schema=ProposeWorkflowInput)
def propose_sales_workflow(goal: str, operations: list) -> str:
    """placeholder - replaced below"""
    # The LangGraph workflow intercepts this tool call (plan_review node).
    # If it is ever invoked directly, it must not write anything.
    return "ERROR: workflow proposals are handled by the approval workflow."


propose_sales_workflow.description = (
    "Propose ALL CRM changes needed for the user's goal as ONE plan. The "
    "user sees the whole plan on an approval card and approves or rejects "
    "it; nothing is saved before approval and the approved plan is saved in "
    "one transaction (all or nothing). Search first and reuse existing "
    "records by id; never invent ids.\n"
    "Each operation: op_id (op1, op2, ...), action create|update, "
    "record_type, record_id (updates), fields [{name, value}], create_deal "
    "(requests only), note.\n"
    "Link fields (company, contact, lead, deal) take an existing id or a "
    "reference to an earlier create operation: $op1. A request with "
    "create_deal=true creates its Deal; refer to that deal as $op3.deal "
    "(e.g. update deal record_id=$op3.deal to set amount/next step). For a "
    "new Deal, put its name in Request.request_for, set create_deal=true, "
    "then use a separate Deal update at record_id=$<request op>.deal for every "
    "supplied Deal field (amount, currency, probability, stage, etc.). If a "
    "payment is requested, create it with deal=$<same request op>.deal. Include the Request, "
    "Deal update and Payment in this same plan; never put Deal fields on the "
    "Request or create a Deal directly.\n"
    "Rules: deals are never created directly; a contact needs a company; a "
    "request links a contact OR a lead (not both, and not a lead with a "
    "company); to create a deal the request must link a contact or lead; "
    "payments belong to a deal. Choice fields (country, lead_source, stage, "
    "currency, products, industry, type, closing_reason) accept names. "
    "Dates are YYYY-MM-DD.\n"
    "Settable fields (* = required on create):\n" + describe_policy()
)


ALL_READ_TOOLS = [search_database_schema, *READ_TOOLS]
tools = [*ALL_READ_TOOLS, propose_sales_workflow]
