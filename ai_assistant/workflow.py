"""Sales workflow plans: structure, validation, approval payload, execution.

A *plan* is an ordered list of operations (create/update of Leads,
Companies, Contacts, Requests, Deals and Payments) proposed by the agent for
one business goal. The life cycle is:

1. ``validate_plan``   - runs before the user sees anything. Checks
   permissions, record visibility, references between operations, CRM
   business rules and the CRM's own admin form validation. Works out values
   the CRM will fill in automatically and looks for duplicates. Writes
   nothing to CRM tables; it records the pending plan in ``ActionRequest``.
2. ``approval_payload`` - what the approval card shows.
3. ``execute_plan``    - runs only after approval. Re-checks the user,
   permissions and every referenced record, then saves all operations in a
   single database transaction through the CRM admin's ``save_model`` /
   ``save_related`` (so CRM side effects such as Request -> Deal creation
   happen exactly as in the CRM UI). Any failure rolls back everything. The
   ``ActionRequest`` row is claimed inside the same transaction, so a plan
   can never be executed twice.

Deals are never created directly: in this CRM ``DealAdmin.has_add_permission``
is False and no group has ``crm.add_deal``. A Deal is created by processing a
Request (the "Create deal" button), which the plan expresses as a request
operation with ``create_deal=true``.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from django.core.exceptions import FieldDoesNotExist, PermissionDenied
from django.db import DatabaseError, transaction
from django.db.models import BooleanField, Q
from django.http import QueryDict
from django.utils import timezone
from django.utils.html import strip_tags
from pydantic import BaseModel, Field, field_validator

from ai_assistant.access import SALES_MODELS, CrmAccess, record_url
from ai_assistant.models import ActionRequest
from common.models import Department
from common.utils.helpers import get_delta_date, get_department_id
from crm.models.others import ClosingReason
from crm.models import (
    Company, Contact, Currency, Deal, Lead, Payment, Product,
    Request, Stage,
)
from crm.settings import FIRST_STEP

logger = logging.getLogger(__name__)

DEFAULT_APPROVAL_TTL_MINUTES = 60


def approval_ttl() -> timedelta:
    """How long a plan may wait for approval (AI_ASSISTANT_APPROVAL_TTL_MINUTES,
    default 60). After that it can no longer be executed."""
    try:
        minutes = int(os.environ.get("AI_ASSISTANT_APPROVAL_TTL_MINUTES",
                                     DEFAULT_APPROVAL_TTL_MINUTES))
    except ValueError:
        minutes = DEFAULT_APPROVAL_TTL_MINUTES
    if minutes <= 0:
        minutes = DEFAULT_APPROVAL_TTL_MINUTES
    return timedelta(minutes=minutes)


PLAN_TOOL_NAME = "propose_sales_workflow"
MAX_OPERATIONS = 10
_OP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
_REF_RE = re.compile(r"^\$(?P<op>[A-Za-z0-9_-]{1,20})(?:\.(?P<attr>deal))?$")


# ---------------------------------------------------------------------------
# Tool input contract
# ---------------------------------------------------------------------------

class FieldValue(BaseModel):
    name: str = Field(description="CRM field name, e.g. first_name")
    value: str = Field(description="Value as text. Record links: an existing id "
                       "(e.g. 12) or a reference to an earlier operation "
                       "($op1, or $op2.deal for the deal created by request op2).")

    @field_validator("value", mode="before")
    @classmethod
    def _as_text(cls, value):
        # Models sometimes send numbers or booleans; the CRM forms take text.
        return "" if value is None else str(value)


class OperationSpec(BaseModel):
    op_id: str = Field(description="Short unique id, e.g. op1")
    action: Literal["create", "update"]
    record_type: Literal["lead", "company", "contact", "request", "deal", "payment"]
    record_id: str = Field(default="", description="update only: id of the existing "
                           "record, or $opN.deal for a deal created in this plan")
    fields: list[FieldValue] = Field(default_factory=list)
    create_deal: bool = Field(default=False, description="request only: process the "
                              "request and create its Deal (the CRM's Create deal step)")
    note: str = Field(default="", description="Why this operation is needed")

    @field_validator("record_id", "op_id", mode="before")
    @classmethod
    def _id_as_text(cls, value):
        return "" if value is None else str(value)


class ProposeWorkflowInput(BaseModel):
    goal: str = Field(description="The user's business goal in one sentence")
    operations: list[OperationSpec]


# ---------------------------------------------------------------------------
# Field policy: which fields the agent may set, per record type and action.
# Restricting fields keeps ownership/department/system fields under the
# CRM's own control (set_owner, save_model, ...).
# ---------------------------------------------------------------------------

_PERSON = [
    "first_name", "middle_name", "last_name", "title", "email",
    "secondary_email", "phone", "mobile", "other_phone", "address",
    "city_name", "country", "region", "district", "description",
    "lead_source", "birth_date", "sex",
]
_COMPANY = [
    "full_name", "alternative_names", "email", "phone", "website", "address",
    "city_name", "country", "region", "district", "registration_number",
    "type", "industry", "lead_source", "description",
]
_REQUEST = [
    "request_for", "first_name", "middle_name", "last_name", "email", "phone",
    "website", "company_name", "country", "city_name", "description",
    "remark", "lead_source", "receipt_date", "products", "lead", "contact",
    "company",
]
_DEAL = [
    "name", "next_step", "next_step_date", "description", "stage", "amount",
    "currency", "probability", "closing_reason", "relevant", "active",
    "important",
]
_PAYMENT = [
    "amount", "currency", "payment_date", "status", "contract_number",
    "invoice_number", "order_number",
]

FIELD_POLICY: dict[tuple[str, str], list[str]] = {
    ("lead", "create"): _PERSON + [
        "company_name", "company_email", "company_phone", "company_address",
        "website", "type", "company", "contact"],
    ("lead", "update"): _PERSON + [
        "company_name", "company_email", "company_phone", "company_address",
        "website", "type", "company", "contact", "disqualified"],
    ("company", "create"): _COMPANY,
    ("company", "update"): _COMPANY + ["disqualified"],
    ("contact", "create"): _PERSON + ["company"],
    ("contact", "update"): _PERSON + ["disqualified"],
    ("request", "create"): _REQUEST,
    ("request", "update"): _REQUEST,
    ("deal", "update"): _DEAL,
    ("payment", "create"): ["deal"] + _PAYMENT,
    ("payment", "update"): _PAYMENT,
}

REQUIRED_ON_CREATE = {
    "lead": ["first_name", "email"],
    "company": ["full_name", "email"],
    "contact": ["first_name", "email", "company"],
    "request": ["request_for"],
    "payment": ["deal", "amount"],
}

# Foreign keys that point to Sales records (existing id or plan reference).
RECORD_LINKS: dict[tuple[str, str], Any] = {
    ("lead", "company"): Company,
    ("lead", "contact"): Contact,
    ("contact", "company"): Company,
    ("request", "lead"): Lead,
    ("request", "contact"): Contact,
    ("request", "company"): Company,
    ("payment", "deal"): Deal,
}

# Choice-like foreign keys / many-to-many fields resolvable by name.
LOOKUP_FIELDS = {
    "country", "lead_source", "type", "industry", "products", "stage",
    "currency", "closing_reason",
}


def describe_policy() -> str:
    lines = []
    for (record_type, action), names in FIELD_POLICY.items():
        required = REQUIRED_ON_CREATE.get(record_type, []) if action == "create" else []
        shown = [f"{n}*" if n in required else n for n in names]
        lines.append(f"- {action} {record_type}: {', '.join(shown)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

class PlanError(Exception):
    pass


@dataclass
class PlannedOp:
    op_id: str
    action: str
    record_type: str
    note: str = ""
    record_id: int | None = None        # existing record (update)
    target_ref: str | None = None       # $opN.deal (update of a planned deal)
    create_deal: bool = False
    provided: dict[str, Any] = field(default_factory=dict)   # normalised user values
    data: dict[str, Any] = field(default_factory=dict)       # effective form data
    links: dict[str, Any] = field(default_factory=dict)      # field -> int id or "$opN"
    errors: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inferred: dict[str, str] = field(default_factory=dict)
    related: dict[str, str] = field(default_factory=dict)
    shown_fields: dict[str, str] = field(default_factory=dict)
    changes: dict[str, dict[str, str]] = field(default_factory=dict)
    existing_label: str = ""
    cleaned: dict[str, Any] = field(default_factory=dict)   # validated form values
    # update only: current raw values of the fields the plan touches, taken
    # when the approval card was generated. Execution refuses to overwrite a
    # record whose values no longer match.
    original: dict[str, Any] | None = None

    @property
    def model(self):
        return SALES_MODELS[self.record_type]

    @property
    def form_fields(self) -> list[str]:
        return list(dict.fromkeys([*self.data, *self.links]))

    def to_dict(self) -> dict:
        return {
            "op_id": self.op_id, "action": self.action,
            "record_type": self.record_type, "note": self.note,
            "record_id": self.record_id, "target_ref": self.target_ref,
            "create_deal": self.create_deal, "data": self.data,
            "links": self.links,
            "original": self.original,
            "display": {
                "title": self.title(), "fields": self.shown_fields,
                "changes": self.changes, "related": self.related,
                "inferred": self.inferred, "warnings": self.warnings,
                "existing_record": self.existing_label or None,
            },
        }

    def title(self) -> str:
        name = self.model._meta.verbose_name.title()
        if self.action == "create":
            text = f"Create {name}"
            if self.create_deal:
                text += " and its Deal"
            return text
        if self.record_type == "request" and self.create_deal:
            return f"Process Request #{self.record_id} into a Deal"
        target = f"#{self.record_id}" if self.record_id else f"({self.target_ref})"
        return f"Update {name} {target}"


@dataclass
class PlanValidation:
    ok: bool
    plan: dict | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_GUARD_FIELDS = {
    "request": ["deal", "pending", "duplicate", "case"],
    "deal": ["stage"],
}


def _snapshot(instance, names) -> dict[str, Any]:
    """Raw, comparable current values of ``names`` on a saved record."""
    snap: dict[str, Any] = {}
    for name in dict.fromkeys(names):
        try:
            field_obj = instance._meta.get_field(name)
        except FieldDoesNotExist:
            continue
        if field_obj.many_to_many:
            snap[name] = sorted(getattr(instance, name).values_list("pk", flat=True))
        elif field_obj.is_relation:
            snap[name] = getattr(instance, f"{field_obj.name}_id")
        else:
            value = getattr(instance, name, None)
            snap[name] = None if value is None else str(value)
    return snap


def _changed_since(instance, original: dict[str, Any]) -> list[str]:
    """Human-readable list of fields that differ from the approved snapshot."""
    now = _snapshot(instance, list(original))
    return [f"{name}: was {original[name]!r}, now {now.get(name)!r}"
            for name in original if now.get(name) != original[name]]


def _verbose(model, name: str) -> str:
    try:
        return str(model._meta.get_field(name).verbose_name).capitalize()
    except FieldDoesNotExist:
        return name.replace("_", " ").capitalize()


def _display(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if hasattr(value, "all") and callable(value.all):
        return ", ".join(str(v) for v in value.all())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_display(v) for v in value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _form_errors(form) -> list[str]:
    errors = []
    for name, messages in form.errors.items():
        label = "" if name == "__all__" else f"{name}: "
        for message in messages:
            errors.append(label + strip_tags(str(message)).strip())
    return errors


def _to_bool(value: str) -> str:
    text = str(value).strip().lower()
    if text in ("true", "yes", "y", "1", "on"):
        return "true"
    if text in ("false", "no", "n", "0", "off", ""):
        return "false"
    raise PlanError(f"expected yes/no, got {value!r}")


def _to_number(value: str) -> str:
    text = re.sub(r"[\s$€£%]", "", str(value))
    if "," in text:
        # Only strip commas that are clearly thousands separators
        # (25,000 or 1,250,000.50). "1,5" or "1.234,56" are ambiguous.
        if not re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):
            raise PlanError(f"ambiguous number {value!r}; write it like 1234.56")
        text = text.replace(",", "")
    try:
        Decimal(text)
    except InvalidOperation:
        raise PlanError(f"expected a number, got {value!r}")
    return text


def _to_choice(model, name: str, value: str) -> str:
    field_obj = model._meta.get_field(name)
    text = str(value).strip().lower()
    for code, label in field_obj.flatchoices:
        if text in (str(code).lower(), str(label).lower()):
            return str(code)
    options = ", ".join(str(label) for _, label in field_obj.flatchoices)
    raise PlanError(f"must be one of: {options}")


def _normalise(model, name: str, value: str) -> Any:
    field_obj = model._meta.get_field(name)
    if field_obj.many_to_many:
        return [part.strip() for part in str(value).split(",") if part.strip()]
    if isinstance(field_obj, BooleanField):
        return _to_bool(value)
    if field_obj.choices and not field_obj.is_relation:
        return _to_choice(model, name, value)
    if field_obj.get_internal_type() in ("DecimalField", "PositiveSmallIntegerField"):
        return _to_number(value)
    return str(value).strip()


def _plan_default_currency(access: CrmAccess):
    """The currency the CRM assigns to a new deal (requestadmin._get_or_create_deal)."""
    if access.department_id:
        department = Department.objects.filter(pk=access.department_id).first()
        return department.default_currency if department else None
    return Currency.objects.filter(is_state_currency=True).first()


def _emails(value: str) -> list[str]:
    return [e.strip() for e in str(value or "").split(",") if e.strip()]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class _PlanValidator:
    def __init__(self, access: CrmAccess, goal: str, specs: list[OperationSpec]):
        self.access = access
        self.goal = goal.strip()
        self.specs = specs
        self.ops: dict[str, PlannedOp] = {}
        self.plan_warnings: list[str] = []
        # op_id -> field names copied from a linked company (lead creation)
        self.inherited: dict[str, set[str]] = {}

    # -- entry point ---------------------------------------------------

    def run(self) -> PlanValidation:
        if not self.specs:
            return PlanValidation(False, message="ERROR: the plan has no operations.")
        if len(self.specs) > MAX_OPERATIONS:
            return PlanValidation(False, message=(
                f"ERROR: a plan may contain at most {MAX_OPERATIONS} operations."))
        for spec in self.specs:
            op = PlannedOp(
                op_id=spec.op_id.strip(), action=spec.action,
                record_type=spec.record_type, note=spec.note.strip(),
                create_deal=bool(spec.create_deal),
            )
            if not _OP_ID_RE.match(op.op_id) or op.op_id in self.ops:
                return PlanValidation(False, message=(
                    f"ERROR: op_id {spec.op_id!r} must be unique and contain only "
                    "letters, digits, '-' or '_'."))
            self.ops[op.op_id] = op
            try:
                self._validate_op(op, spec)
            except PermissionDenied as exc:
                op.errors.append(f"permission denied: {exc}")
            except PlanError as exc:
                op.errors.append(str(exc))
        self._cross_plan_checks()
        return self._outcome()

    def _outcome(self) -> PlanValidation:
        problems = []
        needs_details = False
        for op in self.ops.values():
            label = f"{op.op_id} ({op.action} {op.record_type})"
            if op.missing:
                needs_details = True
                problems.append(f"- {label}: missing required information: "
                                + ", ".join(op.missing))
            for error in op.errors:
                problems.append(f"- {label}: {error}")
        if problems:
            status = "NEEDS_DETAILS" if needs_details and all(
                not op.errors for op in self.ops.values()) else "VALIDATION_FAILED"
            return PlanValidation(False, message=(
                f"{status}: the plan was NOT shown to the user and nothing was "
                "changed. Ask the user for missing information or fix the plan, "
                "then propose it again.\n" + "\n".join(problems)))
        plan = {
            "operation_id": str(uuid.uuid4()),
            "goal": self.goal,
            "created_at": timezone.now().isoformat(),
            "expires_at": (timezone.now() + approval_ttl()).isoformat(),
            "user_id": self.access.user.pk,
            "operations": [op.to_dict() for op in self.ops.values()],
            "warnings": self.plan_warnings,
        }
        return PlanValidation(True, plan=plan)

    # -- per operation -------------------------------------------------

    def _validate_op(self, op: PlannedOp, spec: OperationSpec) -> None:
        if op.record_type == "deal" and op.action == "create":
            raise PlanError(
                "Deals cannot be created directly in this CRM. Create or "
                "update a Request with create_deal=true; the CRM then creates "
                "the Deal from the Request. Use $<request op>.deal to update "
                "that new deal's amount, next step, etc.")
        if (op.record_type, op.action) not in FIELD_POLICY:
            raise PlanError(f"{op.action} {op.record_type} is not supported.")
        if op.create_deal and op.record_type != "request":
            raise PlanError("create_deal is only valid for request operations.")

        model = op.model
        instance = self._target_instance(op, spec)

        # Field names and values.
        allowed = FIELD_POLICY[(op.record_type, op.action)]
        for item in spec.fields:
            name = item.name.strip().lower()
            if name == "department" and name not in allowed:
                # Departments are never set by the agent: the CRM assigns the
                # user's own. Naming that same department is harmless (the
                # user simply said what will happen anyway); naming another
                # one is refused with an explanation, never applied.
                self._own_department_only(item.value)
                continue
            if name not in allowed:
                raise PlanError(
                    f"field {item.name!r} cannot be set on {op.action} "
                    f"{op.record_type}. Allowed: {', '.join(allowed)}")
            value = item.value
            if (op.record_type, name) in RECORD_LINKS:
                op.links[name] = self._resolve_link(op, name, value)
                continue
            if str(value).strip() == "" and op.action == "create":
                continue
            try:
                op.provided[name] = _normalise(model, name, value)
            except PlanError as exc:
                op.errors.append(f"{name}: {exc}")

        if op.action == "create":
            op.missing = [
                name for name in REQUIRED_ON_CREATE.get(op.record_type, [])
                if name not in op.provided and name not in op.links
            ]
        elif not op.provided and not op.links and not op.create_deal:
            raise PlanError("an update must change at least one field.")

        op.data = dict(op.provided)
        # Capture current values now: ModelForm validation copies the new
        # values onto the instance.
        before = {name: _display(getattr(instance, name, ""))
                  for name in op.provided} if instance.pk else {}
        if op.action == "update" and instance.pk:
            op.original = _snapshot(instance, [
                *op.provided, *op.links, *_GUARD_FIELDS.get(op.record_type, [])])
        self._business_rules(op, instance)
        if op.missing or op.errors:
            return
        self._check_with_crm_form(op, instance)
        if op.errors:
            return
        self._duplicates(op, instance)
        self._describe(op, instance, before)

    def _own_department_only(self, value: str) -> None:
        access = self.access
        own = Department.objects.filter(pk=access.home_department_id).first() \
            if access.home_department_id else None
        text = str(value).strip()
        if own is not None and (text.lower() == own.name.lower()
                                or text == str(own.pk)):
            return
        raise PlanError(
            f"department {text!r} cannot be set: new records and deals always "
            + (f"belong to your own department ({own.name}); do not send a "
               "department field." if own else
               "get their department from the linked records or your account; "
               "do not send a department field."))

    def _target_instance(self, op: PlannedOp, spec: OperationSpec):
        model = op.model
        access = self.access
        if op.action == "create":
            if not access.can_add(model):
                raise PermissionDenied(
                    f"you are not allowed to add {model._meta.verbose_name_plural}")
            if spec.record_id.strip():
                raise PlanError("record_id is only used for updates.")
            return model()
        ref = spec.record_id.strip()
        match = _REF_RE.match(ref)
        if match:
            target = self.ops.get(match.group("op"))
            if (op.record_type != "deal" or match.group("attr") != "deal"
                    or target is None or target is op
                    or not (target.record_type == "request" and target.create_deal)):
                raise PlanError(
                    "record_id references must be $opN.deal pointing to an "
                    "earlier request operation with create_deal=true.")
            if not access.user.has_perm("crm.change_deal"):
                raise PermissionDenied("you are not allowed to change deals")
            op.target_ref = ref
            department_id = access.home_department_id
            owner = access.user
            return Deal(
                department_id=department_id, owner=owner,
                stage=Stage.objects.filter(department_id=department_id,
                                           default=True).first(),
            )
        if not ref.isdigit():
            raise PlanError("an update needs record_id (the existing record id).")
        obj = access.get_visible(model, ref)
        if obj is None:
            raise PlanError(
                f"{model._meta.verbose_name} #{ref} does not exist or is not "
                "accessible to you.")
        if not access.can_change(obj):
            raise PermissionDenied(
                f"you are not allowed to change {model._meta.verbose_name} #{ref}")
        op.record_id = obj.pk
        op.existing_label = f"#{obj.pk} {obj}"
        return obj

    def _resolve_link(self, op: PlannedOp, name: str, value: str):
        target_model = RECORD_LINKS[(op.record_type, name)]
        text = str(value).strip()
        if text in ("", "0", "none", "null"):
            return None
        match = _REF_RE.match(text)
        if match:
            target = self.ops.get(match.group("op"))
            if target is None or target is op:
                raise PlanError(f"{name}: {text} must refer to an earlier operation.")
            if match.group("attr") == "deal":
                # $opN.deal is a Deal: only valid where a Deal is expected.
                # (Planned references skip the visibility check, so a type
                # mismatch would link an arbitrary record with that id.)
                ok = (target_model is Deal and target.record_type == "request"
                      and target.create_deal)
            else:
                ok = target.action == "create" and target.model is target_model
            if not ok:
                raise PlanError(
                    f"{name}: {text} does not produce a "
                    f"{target_model._meta.verbose_name}.")
            return text
        if not text.isdigit():
            raise PlanError(
                f"{name}: use the existing {target_model._meta.verbose_name} id "
                "from a search, or a $opN reference.")
        obj = self.access.get_visible(target_model, text)
        if obj is None:
            raise PlanError(
                f"{name}: {target_model._meta.verbose_name} #{text} does not "
                "exist or is not accessible to you.")
        return obj.pk

    # -- business rules and inferred values ---------------------------

    def _linked(self, op: PlannedOp, name: str):
        """(existing object or None, planned op or None) for a link field."""
        value = op.links.get(name)
        if value is None:
            return None, None
        if isinstance(value, int):
            model = RECORD_LINKS[(op.record_type, name)]
            return model.objects.get(pk=value), None
        match = _REF_RE.match(value)
        return None, self.ops[match.group("op")]

    def _business_rules(self, op: PlannedOp, instance) -> None:
        access = self.access
        op.inferred["Owner"] = access.user_label() if op.action == "create" else ""
        if op.action == "create" and op.record_type != "contact":
            op.inferred["Department"] = access.department_label()
        if not op.inferred["Owner"]:
            del op.inferred["Owner"]

        if op.record_type == "lead":
            self._lead_rules(op, instance)
        elif op.record_type == "contact" and op.action == "create":
            company, planned = self._linked(op, "company")
            if company is not None or planned is not None:
                source = company.full_name if company else f"new company ({planned.op_id})"
                op.inferred["Owner and department"] = f"same as the company: {source}"
        elif op.record_type == "request":
            self._request_rules(op, instance)
        elif op.record_type == "deal":
            self._deal_rules(op, instance)
        elif op.record_type == "payment":
            self._payment_rules(op, instance)

    def _lead_rules(self, op: PlannedOp, instance) -> None:
        company, planned = self._linked(op, "company")
        if company is None and planned is None:
            if op.data.get("company_name") and op.action == "create":
                op.warnings.append(
                    f'"{op.data["company_name"]}" is not linked to a CRM company; '
                    "the lead will store the company name only.")
            return
        if company is not None:
            source = {
                "company_name": company.full_name,
                "country": company.country_id and str(company.country_id),
                "city_name": company.city_name,
                "company_phone": company.phone,
                "website": company.website,
                "company_address": company.address,
            }
            origin = f"company #{company.pk} {company.full_name}"
        else:
            values = planned.data
            source = {
                "company_name": values.get("full_name"),
                "country": values.get("country"),
                "city_name": values.get("city_name"),
                "company_phone": values.get("phone"),
                "website": values.get("website"),
                "company_address": values.get("address"),
            }
            origin = f"new company from {planned.op_id}"
        for name, value in source.items():
            # Never overwrite what an existing lead already has.
            current = getattr(instance, f"{name}_id" if name == "country" else name, None)
            if value and name not in op.data and not current:
                op.data[name] = str(value)
                self.inherited.setdefault(op.op_id, set()).add(name)
                op.inferred[_verbose(Lead, name)] = f"{value} (from {origin})" \
                    if name != "country" else f"from {origin}"

    def _request_rules(self, op: PlannedOp, instance) -> None:
        access = self.access
        links = {name: op.links.get(name, getattr(instance, f"{name}_id", None))
                 for name in ("lead", "contact", "company")}
        if links["contact"] and links["lead"]:
            op.errors.append("a request links either a contact or a lead, not both "
                             "(CRM rule).")
        if links["company"] and links["lead"]:
            op.errors.append("a request links either a company or a lead, not both "
                             "(CRM rule).")
        contact, contact_op = self._linked(op, "contact")
        lead, lead_op = self._linked(op, "lead")
        company, company_op = self._linked(op, "company")
        if op.action == "update" and "contact" not in op.links:
            contact = instance.contact
        if op.action == "update" and "lead" not in op.links:
            lead = instance.lead
        if op.action == "update" and "company" not in op.links:
            company = instance.company

        # Contact and company must match (RequestForm.clean).
        if contact is not None and company is not None and contact.company_id != company.pk:
            op.errors.append("the contact does not belong to the linked company.")
        if contact_op is not None and (company is not None or company_op is not None):
            planned_company = contact_op.links.get("company")
            wanted = company.pk if company is not None else f"${company_op.op_id}"
            if planned_company != wanted:
                op.errors.append("the new contact does not belong to the linked company.")

        # Person details are copied from the linked contact/lead by the CRM.
        person = contact or lead
        person_op = contact_op or lead_op
        if person is not None:
            op.inferred["Person details"] = (
                f"copied from {person._meta.verbose_name} #{person.pk} {person}")
        elif person_op is not None:
            for name in ("first_name", "last_name", "email", "phone"):
                if name not in op.data and person_op.data.get(name):
                    op.data[name] = person_op.data[name]
            op.inferred["Person details"] = f"copied from {person_op.op_id}"
        if op.action == "create" and not op.data.get("first_name") \
                and person is None and person_op is None:
            op.missing.append("first_name (or link an existing contact or lead)")

        if contact is not None and company is None and company_op is None:
            op.inferred["Company"] = f"#{contact.company_id} {contact.company} (the contact's company)"

        if op.action == "create" and access.user.groups.filter(name="managers").exists():
            op.inferred["Subsequent request"] = "Yes (the CRM marks requests added by managers)"

        if not op.create_deal:
            return
        # --- Request -> Deal (RequestAdmin "_create-deal") -------------
        if not access.user.has_perm("crm.change_deal"):
            # The CRM shows "Create deal" only to users with change_deal
            # (RequestAdmin.change_view -> has_change_deal_perm).
            raise PermissionDenied("you are not allowed to create deals from requests")

        # Managers process requests in their own department. A superuser has
        # no implicit department, so mirror the CRM's _get_or_create_deal rule:
        # use the linked counterparty's assigned owner and department. Never
        # add an admin to a department just to make this workflow work.
        deal_owner = access.user
        deal_department_id = access.home_department_id
        department_source = "your department"
        if access.user.is_superuser and not deal_department_id:
            if op.action == "update" and instance.owner_id:
                deal_owner = instance.owner
                deal_department_id = instance.department_id or get_department_id(deal_owner)
                department_source = f"Request #{instance.pk}"
            elif person is not None:
                deal_owner = person.owner
                deal_department_id = person.department_id or (
                    get_department_id(deal_owner) if deal_owner else None)
                department_source = f"linked {person._meta.verbose_name}"
            elif person_op is not None:
                deal_owner = access.user
                deal_department_id = person_op.data.get("department") or access.home_department_id
                department_source = f"planned {person_op.record_type}"

        if not deal_department_id:
            op.errors.append(
                "a sales department could not be inferred for this deal. Link a "
                "counterparty assigned to a sales department or process the request "
                "as a member of one.")
            return
        if not deal_owner:
            op.errors.append(
                "an owner could not be inferred for this deal. Assign the linked "
                "counterparty to a CRM user first.")
            return

        op.data["owner"] = str(deal_owner.pk)
        op.data["department"] = str(deal_department_id)
        op.inferred["Owner"] = (
            f"{deal_owner.get_full_name() or deal_owner.username} "
            f"(from {department_source})")
        department = Department.objects.filter(pk=deal_department_id).first()
        if department is None:
            op.errors.append("the inferred sales department no longer exists.")
            return
        op.inferred["Department"] = f"{department} (from {department_source})"

        if op.action == "update":
            if instance.deal_id:
                op.errors.append(f"request #{instance.pk} already has deal #{instance.deal_id}.")
            if instance.duplicate or instance.case:
                op.errors.append("duplicate requests and cases do not create deals (CRM rule).")
        if person is None and person_op is None:
            op.errors.append(
                "to create a deal, link the request to a contact or a lead "
                "(existing id or one created earlier in this plan). Search first; "
                "create a lead or contact if the person is new.")
        if department and department.works_globally and not op.data.get("country"):
            country = instance.country_id if op.action == "update" else None
            country_source = "kept from the existing Request" if country else None
            if not country:
                country = self._counterparty_country(
                    contact, lead, company, contact_op, lead_op, company_op)
                if country:
                    country_source = "taken from the linked counterparty"
            if country:
                op.data["country"] = str(country)
                op.inferred["Country"] = country_source
            else:
                op.errors.append(
                    f"{department} works globally, so the deal needs a country, and the "
                    "linked contact, lead or company has none. Ask the user for the "
                    "country (it is only used when they name one), or add it to that "
                    "record first.")
        setup_problem = _stage_setup_error(deal_department_id)
        if setup_problem:
            op.errors.append(setup_problem)
        if op.action == "update":
            for name in ("country", "products"):
                if name not in op.data:
                    value = getattr(instance, f"{name}_id", None) if name == "country" \
                        else [p.pk for p in instance.products.all()]
                    if value:
                        op.data[name] = value if isinstance(value, list) else str(value)
        stage = Stage.objects.filter(department_id=deal_department_id,
                                     default=True).first()
        currency = _plan_default_currency(access)
        op.inferred["New deal"] = (
            f"name = request subject; stage = {stage or 'department default'}; "
            f"currency = {currency or 'state currency'}; next step = "
            f"\"{FIRST_STEP}\" by {get_delta_date(1).isoformat()}; "
            f"owner = {deal_owner.get_full_name() or deal_owner.username}")
        if Product.objects.filter(department_id=deal_department_id).exists():
            op.data.setdefault("products", [])
        op.data.setdefault("country", "")

    def _counterparty_country(self, contact, lead, company,
                              contact_op, lead_op, company_op):
        """Country of the request's counterparty, existing or planned."""
        if contact is not None and contact.company_id and contact.company.country_id:
            return contact.company.country_id
        if lead is not None and lead.country_id:
            return lead.country_id
        if company is not None and company.country_id:
            return company.country_id
        if contact_op is not None:
            link = contact_op.links.get("company")
            if isinstance(link, int):
                country_id = Company.objects.get(pk=link).country_id
                if country_id:
                    return country_id
            if isinstance(link, str):
                country_id = self.ops[link[1:]].data.get("country")
                if country_id:
                    return country_id
        if lead_op is not None:
            country_id = lead_op.data.get("country")
            if country_id:
                return country_id
        if company_op is not None:
            country_id = company_op.data.get("country")
            if country_id:
                return country_id
        return None

    def _deal_rules(self, op: PlannedOp, instance) -> None:
        if self.access.user.groups.filter(name="chiefs").exists() and not self.access.user.is_superuser:
            op.errors.append("chiefs can view deals but their key fields are read-only.")
        if not op.target_ref:
            # A deal planned in this same plan is checked by its request op.
            setup_problem = _stage_setup_error(
                instance.department_id, closing="closing_reason" in op.data)
            if setup_problem:
                op.errors.append(setup_problem)
        if instance.stage_id and "stage" not in op.data:
            default_stage = Stage.objects.filter(
                department_id=instance.department_id, default=True).first()
            second = Stage.objects.filter(
                department_id=instance.department_id, second_default=True).first()
            if default_stage and second and instance.stage_id == default_stage.pk:
                op.warnings.append(
                    f'Saving a deal in the default stage moves it to "{second}" '
                    "(CRM rule for edited deals).")
        if "closing_reason" in op.data:
            op.warnings.append(
                "Closing a deal as unsuccessful deletes its payments that were not "
                "received (CRM rule).")

    def _payment_rules(self, op: PlannedOp, instance) -> None:
        access = self.access
        deal, deal_op = (self._linked(op, "deal") if op.action == "create"
                         else (instance.deal, None))
        if deal is not None:
            probe = Payment(deal=deal)
            if not access.admin_for(Payment).has_change_permission(
                    access.build_request(), probe):
                op.errors.append(
                    f"you are not allowed to manage payments of deal #{deal.pk} "
                    "(managers may only manage payments of their own deals).")
            currency = deal.currency
        else:
            currency = _plan_default_currency(access)
        if op.action == "create":
            if deal is not None and not deal.amount:
                op.warnings.append(
                    f"deal #{deal.pk} has no amount, so the payment share will show "
                    "'No deal amount'. Set the deal amount (ask the user) so the "
                    "share can be calculated.")
            elif deal_op is not None and not any(
                    other.target_ref == f"${deal_op.op_id}.deal"
                    and _positive(other.data.get("amount"))
                    for other in self.ops.values()):
                op.warnings.append(
                    "the new deal has no amount, so the payment's share of the deal "
                    "would show 'No deal amount'. Ask the user for the deal amount "
                    "and add it to the deal (update deal with record_id "
                    f"${deal_op.op_id}.deal); never guess it.")
        if op.action == "create" and "currency" not in op.data:
            if currency is None:
                op.missing.append("currency")
            else:
                op.data["currency"] = str(currency.pk)
                op.inferred["Currency"] = f"{currency} (the deal's currency)"
        if op.action == "create" and "payment_date" not in op.data:
            op.inferred["Payment date"] = f"{timezone.localdate().isoformat()} (today)"
        if op.action == "create" and "status" not in op.data:
            op.inferred["Status"] = "received (CRM default)"

    # -- CRM admin form validation -------------------------------------

    def _check_with_crm_form(self, op: PlannedOp, instance) -> None:
        access = self.access
        model = op.model
        existing = instance if instance.pk else None
        request = access.build_request(post=_post_for(op, access), obj=existing)
        # Links to records planned earlier in this plan cannot be checked by
        # the form until those records exist; the same form re-validates them
        # at execution time. Links to existing records are checked now.
        fields = [n for n in op.form_fields if not isinstance(op.links.get(n), str)]
        form_data = {
            **op.data,
            **{n: "" if v is None else str(v)
               for n, v in op.links.items() if not isinstance(v, str)},
        }
        try:
            form_class = access.admin_for(model).get_form(request, existing, fields=fields)
        except (Currency.DoesNotExist, Department.DoesNotExist):
            op.errors.append("your department has no default currency configured; "
                             "ask an administrator.")
            return
        read_only = [n for n in fields if n not in form_class.base_fields]
        if read_only:
            op.errors.append("these fields are read-only for you: " + ", ".join(read_only))
            return
        self._resolve_lookups(op, form_class)
        if op.errors:
            return
        form_data.update({n: op.data[n] for n in LOOKUP_FIELDS & set(op.data)})
        form = form_class(data=_query_dict(form_data, fields), instance=instance)
        valid = form.is_valid()
        inherited = self.inherited.get(op.op_id, set())
        if not valid and inherited:
            dropped = [name for name in form.errors if name in inherited]
            for name in dropped:
                op.data.pop(name, None)
                op.inferred.pop(_verbose(Lead, name), None)
                op.warnings.append(
                    f"The company's {_verbose(Lead, name).lower()} is not valid for a "
                    "lead, so it was not copied.")
                form_data.pop(name, None)
            if dropped:
                fields = [n for n in fields if n not in dropped]
                form_class = access.admin_for(model).get_form(request, existing, fields=fields)
                form = form_class(data=_query_dict(form_data, fields), instance=instance)
                valid = form.is_valid()
        if not valid:
            op.errors.extend(_form_errors(form))
            return
        op.cleaned = form.cleaned_data

    def _resolve_lookups(self, op: PlannedOp, form_class) -> None:
        for name in LOOKUP_FIELDS & set(op.data):
            value = op.data[name]
            form_field = form_class.base_fields.get(name)
            queryset = getattr(form_field, "queryset", None)
            if queryset is None:
                continue
            values = value if isinstance(value, list) else [value]
            resolved = []
            for item in values:
                item = str(item).strip()
                if item == "" or item.isdigit():
                    resolved.append(item)
                    continue
                match = queryset.filter(name__iexact=item).first()
                if match is None:
                    options = ", ".join(str(o) for o in queryset[:12])
                    op.errors.append(
                        f"{name}: {item!r} is not available to you."
                        + (f" Options: {options}" if options else
                           " No options are configured for your department."))
                    continue
                resolved.append(str(match.pk))
            op.data[name] = resolved if isinstance(value, list) else (resolved[0] if resolved else "")

    # -- duplicates ------------------------------------------------------

    def _duplicates(self, op: PlannedOp, instance) -> None:
        if op.action != "create":
            return
        errors, warnings = _duplicate_findings(
            self.access, op.record_type, op.data, op.links)
        op.errors.extend(errors)
        op.warnings.extend(warnings)

    # -- approval card content ------------------------------------------

    def _describe(self, op: PlannedOp, instance, before: dict) -> None:
        model = op.model
        cleaned = op.cleaned
        for name in op.provided:
            label = _verbose(model, name)
            new = _display(cleaned.get(name, op.provided[name]))
            if op.action == "update" and instance.pk:
                old = before.get(name, "")
                if old != new:
                    op.changes[label] = {"from": old or "(empty)", "to": new or "(empty)"}
            elif new:
                op.shown_fields[label] = new
        for name, value in op.links.items():
            label = _verbose(model, name)
            if value is None:
                op.related[label] = "(cleared)"
            elif isinstance(value, int):
                target = RECORD_LINKS[(op.record_type, name)].objects.get(pk=value)
                op.related[label] = f"#{target.pk} {target} (existing)"
            else:
                match = _REF_RE.match(value)
                source = self.ops[match.group("op")]
                if match.group("attr") == "deal":
                    op.related[label] = f"the new deal created by {source.op_id}"
                else:
                    op.related[label] = f"the new {source.record_type} from {source.op_id}"
        if op.target_ref:
            source = self.ops[_REF_RE.match(op.target_ref).group("op")]
            op.related["Deal"] = f"the new deal created by {source.op_id}"
        if (op.action == "update" and not op.changes and not op.links
                and not op.create_deal and not op.target_ref):
            op.warnings.append("These values are already set; nothing will change.")

    def _cross_plan_checks(self) -> None:
        seen: dict[str, str] = {}
        for op in self.ops.values():
            if op.action != "create" or op.record_type not in ("lead", "contact"):
                continue
            for email in _emails(op.provided.get("email")):
                key = f"{op.record_type}:{email.lower()}"
                if key in seen:
                    op.errors.append(f"{email} is used by {seen[key]} in this plan too.")
                seen[key] = op.op_id
        used = set()
        for op in self.ops.values():
            for value in [*op.links.values(), op.target_ref]:
                if isinstance(value, str):
                    used.add(_REF_RE.match(value).group("op"))
        for op in self.ops.values():
            if op.record_type == "company" and op.action == "create" and op.op_id not in used:
                self.plan_warnings.append(
                    f"{op.op_id} creates a company that no other operation links to.")


def _positive(value) -> bool:
    try:
        return Decimal(str(value)) > 0
    except (InvalidOperation, TypeError, ValueError):
        return False


def _stage_setup_error(department_id, *, closing: bool = False) -> str:
    """Why the CRM could not pick the deal's stage, or "".

    The CRM looks the default stage up with ``Stage.objects.get`` when it saves
    a deal, which fails with "Stage matching query does not exist" if the
    department has none (or several). Checking first gives a clear message
    instead of a crash. The assistant never creates stages."""
    if not department_id:
        return ("the deal has no department, so the CRM cannot choose its default "
                "stage.")
    name = Department.objects.filter(pk=department_id).values_list("name", flat=True).first()
    name = name or f"department #{department_id}"
    defaults = Stage.objects.filter(default=True, department_id=department_id).count()
    if defaults != 1:
        return (f"{name} has {defaults} default Deal stages; the CRM needs exactly one, "
                "so the deal cannot be saved. An "
                "administrator must set up the stages for this department (the "
                "assistant never creates stages).")
    if closing:
        wins = Stage.objects.filter(success_stage=True, department_id=department_id).count()
        reasons = ClosingReason.objects.filter(
            success_reason=True, department_id=department_id).count()
        if wins != 1 or reasons != 1:
            return (f"{name} needs exactly one success stage and one success closing "
                    "reason to close a deal as won; an administrator must configure them.")
    return ""


def _email_match_q(email: str) -> Q:
    """Exact, case-insensitive match of one whole address in ``email`` or
    ``secondary_email`` (either may hold a comma-separated list). A substring
    such as ``eo@x.com`` never matches ``leo@x.com``."""
    pattern = r"(^|,)\s*" + re.escape(email.strip()) + r"\s*(,|$)"
    return Q(email__iregex=pattern) | Q(secondary_email__iregex=pattern)


def _duplicate_findings(access: CrmAccess, record_type: str, data: dict,
                        links: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a record about to be created. Used when the plan
    is validated and again right before it executes."""
    errors: list[str] = []
    warnings: list[str] = []
    if record_type in ("lead", "contact"):
        own_model = Lead if record_type == "lead" else Contact
        other_model = Contact if own_model is Lead else Lead
        for email in _emails(data.get("email")):
            same = _email_match_q(email)
            if access.can_view(own_model):
                dup = access.queryset(own_model).filter(same).first()
                if dup is not None:
                    errors.append(
                        f"duplicate: {own_model._meta.verbose_name} #{dup.pk} "
                        f"({dup.full_name}) already uses {email}. Update or link "
                        "that record instead of creating a new one.")
            if access.can_view(other_model):
                other = access.queryset(other_model).filter(same).first()
                if other is not None:
                    warnings.append(
                        f"{other_model._meta.verbose_name} #{other.pk} "
                        f"({other.full_name}) already uses {email}.")
    elif record_type == "company":
        name = data.get("full_name", "")
        candidates = access.queryset(Company).filter(
            Q(full_name__iexact=name) | Q(alternative_names__icontains=name))
        for company in candidates:
            aliases = {a.strip().lower() for a in company.alternative_names.split(",")}
            if company.full_name.lower() == name.lower() or name.lower() in aliases:
                errors.append(
                    f"duplicate: company #{company.pk} ({company.full_name}) "
                    "already exists. Link it instead of creating a new company.")
                break
    elif record_type == "request" and data.get("request_for"):
        same = access.queryset(Request).filter(
            pending=True, request_for__iexact=data["request_for"])
        if data.get("email"):
            same = same.filter(email__iexact=data["email"])
        dup = same.first()
        if dup is not None:
            warnings.append(
                f"pending request #{dup.pk} has the same subject"
                + (" and email" if data.get("email") else "")
                + "; this may be a duplicate.")
    elif record_type == "payment":
        deal_id = links.get("deal")
        if isinstance(deal_id, int):
            dup = access.queryset(Payment).filter(
                deal_id=deal_id, amount=Decimal(data["amount"]),
                payment_date=data.get("payment_date") or timezone.localdate(),
            ).first()
            if dup is not None:
                warnings.append(
                    f"payment #{dup.pk} of the same amount and date already "
                    "exists on this deal.")
    return errors, warnings


def _post_for(op: PlannedOp | dict, access: CrmAccess) -> dict:
    create_deal = op.create_deal if isinstance(op, PlannedOp) else op["create_deal"]
    if not create_deal:
        return {}
    data = op.data if isinstance(op, PlannedOp) else op["data"]
    department_id = data.get("department") or access.home_department_id
    return {"_create-deal": "1", "department": str(department_id or "")}


def _query_dict(data: dict, fields: list[str]) -> QueryDict:
    query = QueryDict(mutable=True)
    for name in fields:
        value = data.get(name, "")
        if isinstance(value, list):
            query.setlist(name, [str(v) for v in value])
        elif value is None:
            query[name] = ""
        else:
            query[name] = str(value)
    return query


def validate_plan(access: CrmAccess, args: dict, *, thread_id: str = "") -> PlanValidation:
    """Validate the agent's proposal. Nothing is written to CRM tables."""
    try:
        proposal = ProposeWorkflowInput.model_validate(args)
    except Exception as exc:  # pydantic.ValidationError
        return PlanValidation(False, message=f"ERROR: the proposal is malformed: {exc}")
    try:
        access.require_scope()
    except PermissionDenied as exc:
        return PlanValidation(False, message=f"PERMISSION_DENIED: {exc}")
    outcome = _PlanValidator(access, proposal.goal, proposal.operations).run()
    if outcome.ok:
        ActionRequest.objects.create(
            user=access.user,
            operation_id=outcome.plan["operation_id"],
            thread_id=thread_id[:200],
            action_type="SALES_WORKFLOW",
            payload=outcome.plan,
        )
    return outcome


def approval_payload(plan: dict) -> dict:
    """The data rendered on the approval card."""
    operations = [
        {"op_id": op["op_id"], "action": op["action"],
         "record_type": op["record_type"], "note": op["note"], **op["display"]}
        for op in plan["operations"]
    ]
    return {
        "action": "SALES_WORKFLOW",
        "operation_id": plan["operation_id"],
        "expires_at": plan.get("expires_at"),
        "goal": plan["goal"],
        "record_type": ", ".join(op["display"]["title"] for op in plan["operations"]),
        "operations": operations,
        "warnings": plan.get("warnings", []),
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class OperationFailed(Exception):
    def __init__(self, op_id: str, message: str):
        super().__init__(message)
        self.op_id = op_id
        self.message = message


def execute_plan(access: CrmAccess, plan: dict) -> dict:
    """Run an approved plan atomically. Returns a JSON-serialisable report."""
    operation_id = plan["operation_id"]
    base = {"operation_id": operation_id, "goal": plan.get("goal", "")}
    record = ActionRequest.objects.filter(
        operation_id=operation_id, user=access.user).first()
    if record is None:
        return {**base, "status": "failed", "results": [],
                "error": "This approval does not belong to your account."}
    if record.status != ActionRequest.PENDING:
        return _already_processed(record, base)
    if is_expired(record):
        report = {**base, "status": "expired", "results": _rolled_back(plan, ""),
                  "error": "This approval expired before it was approved."}
        ActionRequest.objects.filter(
            pk=record.pk, status=ActionRequest.PENDING).update(
            status=ActionRequest.FAILED, result=report, decided_at=timezone.now())
        return report
    try:
        access.require_scope()
    except PermissionDenied as exc:
        return mark_failed(access.user.pk, plan, f"Permission denied: {exc}")

    results: list[dict] = []
    try:
        with transaction.atomic():
            claimed = ActionRequest.objects.filter(
                pk=record.pk, status=ActionRequest.PENDING,
            ).update(status=ActionRequest.APPROVED, decided_at=timezone.now())
            if claimed != 1:
                raise _AlreadyClaimed()
            produced: dict[str, Any] = {}
            for op in plan["operations"]:
                results.append(_execute_operation(access, op, produced))
            report = {**base, "status": "success", "results": results}
            # Recorded in the same transaction as the CRM writes, so a crash
            # can never leave a committed plan looking unprocessed.
            ActionRequest.objects.filter(pk=record.pk).update(
                status=ActionRequest.EXECUTED, result=report)
        return report
    except _AlreadyClaimed:
        # Another approval of the same plan (double click, second tab) got
        # there first. Report that; never overwrite its outcome.
        record.refresh_from_db()
        return _already_processed(record, base)
    except OperationFailed as exc:
        report = {**base, "status": "failed", "failed_op": exc.op_id,
                  "error": exc.message, "results": _rolled_back(plan, exc.op_id)}
    except PermissionDenied as exc:
        report = {**base, "status": "failed", "error": f"Permission denied: {exc}",
                  "results": _rolled_back(plan, "")}
    except DatabaseError as exc:
        report = {**base, "status": "failed",
                  "error": f"The CRM database rejected the changes ({type(exc).__name__}).",
                  "results": _rolled_back(plan, "")}
    except Exception as exc:  # CRM business code raised; nothing was committed
        logger.exception("Workflow %s failed", operation_id)
        # The operation after the last successful one is the one that failed.
        operations = plan["operations"]
        failed_op = operations[len(results)]["op_id"] if len(results) < len(operations) else ""
        report = {**base, "status": "failed", "failed_op": failed_op,
                  "error": f"The CRM could not complete {failed_op or 'the workflow'} "
                           f"({type(exc).__name__}: {strip_tags(str(exc))[:300]}).",
                  "results": _rolled_back(plan, failed_op)}

    # The transaction rolled back, so the row is PENDING again (claimed by
    # this run). Only that state may be turned into FAILED.
    ActionRequest.objects.filter(pk=record.pk, status=ActionRequest.PENDING).update(
        status=ActionRequest.FAILED, result=report, decided_at=timezone.now())
    return report


class _AlreadyClaimed(Exception):
    pass


def is_expired(record: ActionRequest) -> bool:
    """Pending approvals older than the TTL can no longer be executed."""
    return timezone.now() - record.created_at > approval_ttl()


def expire_pending(*, dry_run: bool = False) -> int:
    """Close PENDING plans older than the TTL (status FAILED, result.status
    "expired"). Returns how many were (or would be) closed. Never touches CRM
    records."""
    cutoff = timezone.now() - approval_ttl()
    stale = ActionRequest.objects.filter(
        status=ActionRequest.PENDING, created_at__lt=cutoff)
    count = stale.count()
    if not dry_run and count:
        stale.update(
            status=ActionRequest.FAILED, decided_at=timezone.now(),
            result={"status": "expired",
                    "error": "This approval expired before it was approved."})
    return count


def _already_processed(record: ActionRequest, base: dict) -> dict:
    previous = record.result or {}
    return {**previous, **base, "status": "already_processed",
            "previous_status": record.status}


def mark_failed(user_id, plan: dict, error: str) -> dict:
    """Close a waiting plan that could not run at all (e.g. the user was
    deactivated while it waited for approval)."""
    report = {"operation_id": plan["operation_id"], "goal": plan.get("goal", ""),
              "status": "failed", "error": error, "results": _rolled_back(plan, "")}
    ActionRequest.objects.filter(
        operation_id=plan["operation_id"], user_id=user_id,
        status=ActionRequest.PENDING,
    ).update(status=ActionRequest.FAILED, result=report, decided_at=timezone.now())
    return report


def mark_rejected(user_id, plan: dict, reason: str = "") -> None:
    ActionRequest.objects.filter(
        operation_id=plan["operation_id"], user_id=user_id,
        status=ActionRequest.PENDING,
    ).update(status=ActionRequest.REJECTED, decided_at=timezone.now(),
             result={"status": "rejected", "reason": reason})


def format_report(result: dict) -> str:
    """Plain, exact summary of a plan outcome for the chat."""
    status = result.get("status")
    if status == "rejected":
        text = "Cancelled. Nothing was changed in the CRM."
        if result.get("reason"):
            text += f" ({result['reason']})"
        return text
    if status == "expired":
        return ("This approval expired, so nothing was changed in the CRM. "
                "Please ask again and approve the new plan.")
    if status == "already_processed":
        return ("This plan was already processed earlier "
                f"({str(result.get('previous_status', '')).lower()}); nothing was run again.")
    lines = []
    if status == "success":
        lines.append("Done. All changes were saved:")
        for item in result.get("results", []):
            link = f"[{item['label']}]({item['url']})" if item.get("url") else item["label"]
            lines.append(f"- {item['action'].capitalize()} {item['record_type']} "
                         f"#{item['record_id']}: {link}")
            if item.get("deal"):
                deal = item["deal"]
                deal_link = f"[{deal['label']}]({deal['url']})" if deal.get("url") else deal["label"]
                lines.append(f"- Created Deal #{deal['record_id']} from this request: {deal_link}")
        return "\n".join(lines)
    lines.append("Nothing was saved: the workflow failed and every change in it was "
                 "rolled back.")
    if result.get("error"):
        where = f" in {result['failed_op']}" if result.get("failed_op") else ""
        lines.append(f"Problem{where}: {result['error']}")
    return "\n".join(lines)


def _rolled_back(plan: dict, failed_op: str) -> list[dict]:
    return [
        {"op_id": op["op_id"], "title": op["display"]["title"],
         "status": "failed" if op["op_id"] == failed_op else "not saved"}
        for op in plan["operations"]
    ]


def _execute_operation(access: CrmAccess, op: dict, produced: dict) -> dict:
    op_id = op["op_id"]
    record_type = op["record_type"]
    model = SALES_MODELS[record_type]
    is_update = op["action"] == "update"

    # 1. The record to save, re-checked against current permissions.
    if not is_update:
        if not access.can_add(model):
            raise OperationFailed(op_id, f"You can no longer add "
                                         f"{model._meta.verbose_name_plural}.")
        instance = model()
    else:
        if op["target_ref"]:
            instance = produced.get(op["target_ref"][1:])
            if instance is None:
                raise OperationFailed(op_id, "The referenced deal was not created.")
        else:
            instance = access.get_visible(model, op["record_id"], lock=True)
            if instance is None:
                raise OperationFailed(op_id, f"{model._meta.verbose_name} "
                                             f"#{op['record_id']} is no longer accessible.")
        if not access.can_change(instance):
            raise OperationFailed(op_id, f"You can no longer change this "
                                         f"{model._meta.verbose_name}.")
        if not op["target_ref"]:
            # Never overwrite data that changed after the approval card was
            # generated: the record must still match what the user approved.
            original = op.get("original")
            if original is None:
                raise OperationFailed(op_id, (
                    "This plan has no saved snapshot of the record, so it cannot "
                    "be checked for changes. Nothing was saved; please ask again."))
            changed = _changed_since(instance, original)
            if changed:
                raise OperationFailed(op_id, (
                    f"{model._meta.verbose_name.capitalize()} #{instance.pk} was "
                    "changed after this approval card was generated ("
                    + "; ".join(changed) + "). Nothing was saved; please ask "
                    "again so you can review the current values."))
    # Business-rule permissions that were checked when the plan was proposed.
    if (op["create_deal"] or op["target_ref"]) \
            and not access.user.has_perm("crm.change_deal"):
        raise OperationFailed(op_id, "You can no longer create or change deals.")
    if not is_update and record_type in ("lead", "contact", "company", "request", "payment"):
        errors, _warnings = _duplicate_findings(
            access, record_type, op["data"],
            {k: v for k, v in op["links"].items() if isinstance(v, int)})
        if errors:
            raise OperationFailed(op_id, "; ".join(errors))

    # 2. Form data with links resolved to real, visible records.
    data = dict(op["data"])
    for name, value in op["links"].items():
        target_model = RECORD_LINKS[(record_type, name)]
        if value is None:
            data[name] = ""
            continue
        if isinstance(value, str):
            target = produced.get(value[1:])
            if target is None:
                raise OperationFailed(op_id, f"{value} was not created.")
        else:
            target = access.get_visible(target_model, value)
            if target is None:
                raise OperationFailed(op_id, f"{target_model._meta.verbose_name} "
                                             f"#{value} is no longer accessible.")
        data[name] = str(target.pk)
    if record_type == "payment" and not is_update:
        deal = Deal.objects.get(pk=data["deal"])
        if not access.admin_for(Payment).has_change_permission(
                access.build_request(), Payment(deal=deal)):
            raise OperationFailed(op_id, "You are not allowed to add payments to this deal.")
    fields = list(dict.fromkeys([*op["data"], *op["links"]]))

    # 3. The CRM admin form validates; the CRM admin saves.
    model_admin = access.admin_for(model)
    existing = instance if instance.pk else None
    request = access.build_request(post=_post_for(op, access), obj=existing)
    form_class = model_admin.get_form(request, existing, fields=fields)
    missing = [n for n in fields if n not in form_class.base_fields]
    if missing:
        raise OperationFailed(op_id, "Read-only fields for you: " + ", ".join(missing))
    form = form_class(data=_query_dict(data, fields), instance=instance)
    if not form.is_valid():
        raise OperationFailed(op_id, "; ".join(_form_errors(form)))
    obj = form.save(commit=False)
    model_admin.save_model(request, obj, form, is_update)
    model_admin.save_related(request, form, [], is_update)
    _log_change(model_admin, request, obj, form, is_update)
    produced[op_id] = obj

    result = {
        "op_id": op_id, "title": op["display"]["title"], "status": "success",
        "action": "updated" if is_update else "created",
        "record_type": model._meta.verbose_name.title(),
        "record_id": obj.pk, "label": str(obj), "url": record_url(obj),
        "notes": [strip_tags(m) for m in request._messages.messages],
    }
    if op["create_deal"]:
        obj.refresh_from_db()
        if obj.deal_id is None:
            notes = "; ".join(result["notes"]) or "no deal was produced"
            raise OperationFailed(op_id, f"The CRM did not create the deal ({notes}).")
        produced[f"{op_id}.deal"] = obj.deal
        result["deal"] = {"record_id": obj.deal_id, "label": str(obj.deal),
                          "url": record_url(obj.deal)}
    return result


def _log_change(model_admin, request, obj, form, is_update: bool) -> None:
    """Record the change in the CRM's admin history (LogEntry), as the admin
    change form does, so AI-made changes appear in each record's history."""
    try:
        message = model_admin.construct_change_message(
            request, form, [], add=not is_update)
        if is_update:
            model_admin.log_change(request, obj, message)
        else:
            model_admin.log_addition(request, obj, message)
    except Exception:  # history logging must never break a saved workflow
        logger.exception("Could not write admin history for %r", obj)
