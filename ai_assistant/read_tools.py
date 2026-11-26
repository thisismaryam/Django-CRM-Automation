"""Read-only CRM tools for the Sales agent.

Every query starts from ``CrmAccess.queryset(model)``, i.e. the CRM admin's
own ``get_queryset`` for the authenticated user after a ``view`` permission
check, so department scoping applies to reads exactly as it does in the CRM
pages. Related records are only included when they are also visible.
"""
from __future__ import annotations

import functools
import json
from datetime import timedelta
from decimal import Decimal
from typing import Any, Callable

from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q, Sum
from django.utils import timezone
from django.utils.dateparse import parse_date
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ai_assistant.access import SALES_MODELS, CrmAccess, record_url
from common.models import Department
from crm.models import (
    ClientType, ClosingReason, Company, Contact, Country, Currency, Deal,
    Industry, Lead, LeadSource, Payment, Product, Request, Stage,
)

DEFAULT_LIMIT = 10
MAX_LIMIT = 25
RELATED_LIMIT = 10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _limit(value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


def crm_read_tool(func: Callable) -> Callable:
    """Turn permission problems into a message the model can relay, instead
    of crashing the conversation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionDenied as exc:
            return f"PERMISSION_DENIED: {exc}"
    return wrapper


def _access(config: RunnableConfig) -> CrmAccess:
    access = CrmAccess.from_config(config)
    # Deny by default: a non-admin user with no department would otherwise
    # get the CRM admin's unfiltered (all departments) queryset and options.
    access.require_scope()
    return access


def _ref(obj, label: str | None = None) -> dict | None:
    if obj is None:
        return None
    return {"id": obj.pk, "name": label if label is not None else str(obj)}


def _visible_ref(access: CrmAccess, obj) -> dict | None:
    """Reference a related record only if the user may see it."""
    if obj is None:
        return None
    model = type(obj)
    if not access.can_view(model):
        return {"id": obj.pk, "name": "(not visible to you)"}
    if not access.queryset(model).filter(pk=obj.pk).exists():
        return {"id": obj.pk, "name": "(in another department)"}
    return _ref(obj, _label(obj))


def _label(obj) -> str:
    if isinstance(obj, Company):
        return obj.full_name
    if isinstance(obj, (Contact, Lead)):
        return " ".join(filter(None, (obj.first_name, obj.middle_name, obj.last_name)))
    if isinstance(obj, Request):
        return obj.request_for
    if isinstance(obj, Deal):
        return obj.name
    if isinstance(obj, Payment):
        return f"{obj.amount} {obj.currency or ''} ({obj.get_status_display()})"
    return str(obj)


def _user(user) -> str | None:
    if user is None:
        return None
    return user.get_full_name() or user.username


def _person_q(query: str, *, email_fields=("email", "secondary_email")) -> Q:
    """Every word must match a name part or email/phone."""
    q = Q()
    for word in query.split():
        word_q = (
            Q(first_name__icontains=word)
            | Q(middle_name__icontains=word)
            | Q(last_name__icontains=word)
            | Q(phone__icontains=word)
        )
        for field in email_fields:
            word_q |= Q(**{f"{field}__icontains": word})
        q &= word_q
    return q


def _money(value) -> str | None:
    if value is None:
        return None
    return str(Decimal(value).quantize(Decimal("0.01")))


# ---------------------------------------------------------------------------
# serialisers (compact dictionaries, never model objects)
# ---------------------------------------------------------------------------

def lead_dict(access: CrmAccess, lead: Lead, detailed: bool = False) -> dict:
    data = {
        "id": lead.pk,
        "name": _label(lead),
        "email": lead.email,
        "phone": lead.phone or None,
        "company_name": lead.company_name or None,
        "company": _visible_ref(access, lead.company) if lead.company_id else None,
        "country": str(lead.country) if lead.country_id else None,
        "owner": _user(lead.owner),
        "disqualified": lead.disqualified,
        "created": lead.creation_date.date().isoformat(),
        "url": record_url(lead),
    }
    if detailed:
        data.update({
            "title": lead.title or None,
            "mobile": lead.mobile or None,
            "city": lead.city_name or None,
            "website": lead.website or None,
            "company_email": lead.company_email or None,
            "company_phone": lead.company_phone or None,
            "lead_source": str(lead.lead_source) if lead.lead_source_id else None,
            "contact": _visible_ref(access, lead.contact) if lead.contact_id else None,
            "description": lead.description or None,
            "department": str(lead.department) if lead.department_id else None,
        })
    return data


def company_dict(access: CrmAccess, company: Company, detailed: bool = False) -> dict:
    data = {
        "id": company.pk,
        "name": company.full_name,
        "alternative_names": company.alternative_names or None,
        "email": company.email or None,
        "phone": company.phone or None,
        "website": company.website or None,
        "country": str(company.country) if company.country_id else None,
        "city": company.city_name or (str(company.city) if company.city_id else None),
        "owner": _user(company.owner),
        "url": record_url(company),
    }
    if detailed:
        data.update({
            "type": str(company.type) if company.type_id else None,
            "industries": [str(i) for i in company.industry.all()],
            "address": company.address or None,
            "registration_number": company.registration_number or None,
            "lead_source": str(company.lead_source) if company.lead_source_id else None,
            "description": company.description or None,
            "disqualified": company.disqualified,
            "department": str(company.department) if company.department_id else None,
        })
    return data


def contact_dict(access: CrmAccess, contact: Contact, detailed: bool = False) -> dict:
    data = {
        "id": contact.pk,
        "name": _label(contact),
        "title": contact.title or None,
        "email": contact.email,
        "phone": contact.phone or None,
        "company": _visible_ref(access, contact.company),
        "owner": _user(contact.owner),
        "url": record_url(contact),
    }
    if detailed:
        data.update({
            "mobile": contact.mobile or None,
            "secondary_email": contact.secondary_email or None,
            "country": str(contact.country) if contact.country_id else None,
            "city": contact.city_name or None,
            "address": contact.address or None,
            "description": contact.description or None,
            "disqualified": contact.disqualified,
        })
    return data


def request_dict(access: CrmAccess, req: Request, detailed: bool = False) -> dict:
    data = {
        "id": req.pk,
        "subject": req.request_for,
        "ticket": req.ticket,
        "status": "pending" if req.pending else "processed",
        "receipt_date": req.receipt_date.isoformat() if req.receipt_date else None,
        "person": " ".join(filter(None, (req.first_name, req.last_name))) or None,
        "email": req.email or None,
        "company_name": req.company_name or None,
        "lead": _visible_ref(access, req.lead) if req.lead_id else None,
        "contact": _visible_ref(access, req.contact) if req.contact_id else None,
        "company": _visible_ref(access, req.company) if req.company_id else None,
        "deal": _visible_ref(access, req.deal) if req.deal_id else None,
        "owner": _user(req.owner),
        "url": record_url(req),
    }
    if detailed:
        data.update({
            "phone": req.phone or None,
            "country": str(req.country) if req.country_id else None,
            "products": [p.name for p in req.products.all()],
            "description": req.description or None,
            "remark": req.remark or None,
            "duplicate": req.duplicate,
            "case": req.case,
            "lead_source": str(req.lead_source) if req.lead_source_id else None,
        })
    return data


def deal_dict(access: CrmAccess, deal: Deal, detailed: bool = False) -> dict:
    data = {
        "id": deal.pk,
        "name": deal.name,
        "stage": str(deal.stage) if deal.stage_id else None,
        "amount": _money(deal.amount),
        "currency": str(deal.currency) if deal.currency_id else None,
        "probability": deal.probability,
        "next_step": deal.next_step,
        "next_step_date": deal.next_step_date.isoformat() if deal.next_step_date else None,
        "active": deal.active,
        "company": _visible_ref(access, deal.company) if deal.company_id else None,
        "contact": _visible_ref(access, deal.contact) if deal.contact_id else None,
        "lead": _visible_ref(access, deal.lead) if deal.lead_id else None,
        "owner": _user(deal.owner),
        "url": record_url(deal),
    }
    if detailed:
        data.update({
            "ticket": deal.ticket,
            "request": _visible_ref(access, deal.request) if deal.request_id else None,
            "relevant": deal.relevant,
            "important": deal.important,
            "closing_reason": str(deal.closing_reason) if deal.closing_reason_id else None,
            "closing_date": deal.closing_date.isoformat() if deal.closing_date else None,
            "description": deal.description or None,
            "workflow": (deal.workflow or "")[:600] or None,
            "co_owner": _user(deal.co_owner),
            "created": deal.creation_date.date().isoformat(),
            "department": str(deal.department) if deal.department_id else None,
        })
    return data


def payment_dict(access: CrmAccess, payment: Payment, detailed: bool = False) -> dict:
    return {
        "id": payment.pk,
        "deal": _visible_ref(access, payment.deal),
        "amount": _money(payment.amount),
        "currency": str(payment.currency) if payment.currency_id else None,
        "status": payment.get_status_display(),
        "payment_date": payment.payment_date.isoformat(),
        "invoice_number": payment.invoice_number or None,
        "contract_number": payment.contract_number or None,
        "order_number": payment.order_number or None,
        "url": record_url(payment),
    }


SERIALISERS = {
    "lead": lead_dict,
    "company": company_dict,
    "contact": contact_dict,
    "request": request_dict,
    "deal": deal_dict,
    "payment": payment_dict,
}


def describe_record(access: CrmAccess, obj, detailed: bool = False) -> dict:
    for key, model in SALES_MODELS.items():
        if isinstance(obj, model):
            return SERIALISERS[key](access, obj, detailed)
    return {"id": obj.pk, "name": str(obj)}


def _result(kind: str, rows: list[dict], total: int) -> str:
    if not rows:
        return f"No {kind} found that you can access with these criteria."
    payload = {"count": len(rows), "total_matches": total, kind: rows}
    if total > len(rows):
        payload["note"] = "More records match; narrow the search to see them."
    return _dump(payload)


# ---------------------------------------------------------------------------
# search tools
# ---------------------------------------------------------------------------

@tool
@crm_read_tool
def search_leads(
    query: str = "",
    company_name: str = "",
    include_disqualified: bool = False,
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Leads (prospects not yet converted to a Company/Contact) that
    you are allowed to see. `query` matches name, email or phone words;
    `company_name` matches the lead's company text or linked company.
    Returns ids to use in workflow proposals."""
    access = _access(config)
    qs = access.queryset(Lead).select_related("company", "country", "owner")
    if query.strip():
        qs = qs.filter(_person_q(query))
    if company_name.strip():
        name = company_name.strip()
        qs = qs.filter(Q(company_name__icontains=name)
                       | Q(company__full_name__icontains=name))
    if not include_disqualified:
        qs = qs.filter(disqualified=False)
    qs = qs.order_by("-creation_date")
    rows = list(qs[:_limit(limit)])
    return _result("leads", [lead_dict(access, x) for x in rows], qs.count())


@tool
@crm_read_tool
def search_companies(
    name: str = "",
    email: str = "",
    country: str = "",
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Companies you are allowed to see by partial name (also matches
    alternative names), email/domain or country. Use it to check whether a
    company already exists before proposing to create one."""
    access = _access(config)
    qs = access.queryset(Company).select_related("country", "city", "owner")
    if name.strip():
        qs = qs.filter(Q(full_name__icontains=name.strip())
                       | Q(alternative_names__icontains=name.strip()))
    if email.strip():
        qs = qs.filter(email__icontains=email.strip())
    if country.strip():
        qs = qs.filter(country__name__icontains=country.strip())
    qs = qs.order_by("full_name")
    rows = list(qs[:_limit(limit)])
    return _result("companies", [company_dict(access, x) for x in rows], qs.count())


@tool
@crm_read_tool
def search_contacts(
    name: str = "",
    email: str = "",
    company_id: int = 0,
    company_name: str = "",
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Contacts (people at a Company) you are allowed to see by name,
    email, company id or partial company name."""
    access = _access(config)
    qs = access.queryset(Contact).select_related("company", "owner")
    if name.strip():
        qs = qs.filter(_person_q(name))
    if email.strip():
        qs = qs.filter(Q(email__icontains=email.strip())
                       | Q(secondary_email__icontains=email.strip()))
    if company_id:
        qs = qs.filter(company_id=company_id)
    if company_name.strip():
        qs = qs.filter(company__full_name__icontains=company_name.strip())
    qs = qs.order_by("first_name", "last_name")
    rows = list(qs[:_limit(limit)])
    return _result("contacts", [contact_dict(access, x) for x in rows], qs.count())


@tool
@crm_read_tool
def search_requests(
    query: str = "",
    status: str = "any",
    company_id: int = 0,
    contact_id: int = 0,
    lead_id: int = 0,
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Requests (customer inquiries; deals are created from them).
    `query` matches subject, person, email, company name or ticket.
    `status` is "pending" (not yet processed), "processed" or "any"."""
    access = _access(config)
    qs = access.queryset(Request).select_related(
        "lead", "contact", "company", "deal", "owner")
    for word in query.split():
        qs = qs.filter(
            Q(request_for__icontains=word) | Q(first_name__icontains=word)
            | Q(last_name__icontains=word) | Q(email__icontains=word)
            | Q(company_name__icontains=word) | Q(ticket__iexact=word)
        )
    if status == "pending":
        qs = qs.filter(pending=True)
    elif status == "processed":
        qs = qs.filter(pending=False)
    if company_id:
        qs = qs.filter(company_id=company_id)
    if contact_id:
        qs = qs.filter(contact_id=contact_id)
    if lead_id:
        qs = qs.filter(lead_id=lead_id)
    qs = qs.order_by("-creation_date")
    rows = list(qs[:_limit(limit)])
    return _result("requests", [request_dict(access, x) for x in rows], qs.count())


@tool
@crm_read_tool
def search_deals(
    query: str = "",
    status: str = "active",
    stage: str = "",
    company_id: int = 0,
    contact_id: int = 0,
    lead_id: int = 0,
    mine_only: bool = False,
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Deals (sales opportunities) you are allowed to see.
    `query` matches deal name, ticket or counterparty name. `status` is
    "active", "closed" or "any". `mine_only` limits to deals you own or
    co-own."""
    access = _access(config)
    qs = access.queryset(Deal).select_related(
        "stage", "currency", "company", "contact", "lead", "owner")
    for word in query.split():
        qs = qs.filter(
            Q(name__icontains=word) | Q(ticket__iexact=word)
            | Q(company__full_name__icontains=word)
            | Q(contact__last_name__icontains=word)
            | Q(lead__last_name__icontains=word)
            | Q(lead__company_name__icontains=word)
        )
    if status == "active":
        qs = qs.filter(active=True)
    elif status == "closed":
        qs = qs.filter(active=False)
    if stage.strip():
        qs = qs.filter(stage__name__icontains=stage.strip())
    if company_id:
        qs = qs.filter(company_id=company_id)
    if contact_id:
        qs = qs.filter(contact_id=contact_id)
    if lead_id:
        qs = qs.filter(lead_id=lead_id)
    if mine_only:
        qs = qs.filter(Q(owner=access.user) | Q(co_owner=access.user))
    qs = qs.order_by("next_step_date")
    rows = list(qs[:_limit(limit)])
    return _result("deals", [deal_dict(access, x) for x in rows], qs.count())


@tool
@crm_read_tool
def search_payments(
    deal_id: int = 0,
    status: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = DEFAULT_LIMIT,
    *,
    config: RunnableConfig,
) -> str:
    """Search Payments you are allowed to see. `status` is one of received,
    guaranteed, high probability, low probability (or empty for all).
    Dates are YYYY-MM-DD. Also returns totals per currency."""
    access = _access(config)
    qs = access.queryset(Payment).select_related("deal", "currency")
    if deal_id:
        qs = qs.filter(deal_id=deal_id)
    if status.strip():
        code = _payment_status_code(status)
        if code is None:
            return ("ERROR: unknown payment status. Use received, guaranteed, "
                    "high probability or low probability.")
        qs = qs.filter(status=code)
    for value, lookup in ((date_from, "payment_date__gte"), (date_to, "payment_date__lte")):
        if value.strip():
            parsed = parse_date(value.strip())
            if parsed is None:
                return f"ERROR: invalid date {value!r}; use YYYY-MM-DD."
            qs = qs.filter(**{lookup: parsed})
    totals = _totals(qs)
    qs = qs.order_by("-payment_date")
    rows = list(qs[:_limit(limit)])
    if not rows:
        return "No payments found that you can access with these criteria."
    return _dump({
        "count": len(rows), "total_matches": qs.count(),
        "totals_by_currency": totals,
        "payments": [payment_dict(access, x) for x in rows],
    })


def _payment_status_code(value: str) -> str | None:
    value = value.strip().lower()
    for code, label in Payment.STATUS_CHOICES:
        if value in (code, str(label).lower()):
            return code
    return None


# ---------------------------------------------------------------------------
# record details with related records
# ---------------------------------------------------------------------------

def _related(access: CrmAccess, model, serialiser, **filters) -> list[dict] | str:
    if not access.can_view(model):
        return "not visible to you"
    qs = access.queryset(model).filter(**filters).order_by("-pk")
    return [serialiser(access, x) for x in qs[:RELATED_LIMIT]]


@tool
@crm_read_tool
def get_crm_record(
    record_type: str,
    record_id: int,
    *,
    config: RunnableConfig,
) -> str:
    """Get full details of one CRM record plus its related records.
    `record_type` is lead, company, contact, request, deal or payment.
    Related records you cannot access are hidden."""
    access = _access(config)
    key = record_type.strip().lower()
    model = SALES_MODELS.get(key)
    if model is None:
        return ("ERROR: record_type must be one of "
                + ", ".join(SALES_MODELS) + ".")
    obj = access.get_visible(model, record_id)
    if obj is None:
        return (f"No {key} with id {record_id} exists that you can access.")
    data = describe_record(access, obj, detailed=True)
    related: dict[str, Any] = {}
    if key == "company":
        related["contacts"] = _related(access, Contact, contact_dict, company=obj)
        related["deals"] = _related(access, Deal, deal_dict, company=obj)
        related["requests"] = _related(access, Request, request_dict, company=obj)
        related["leads"] = _related(access, Lead, lead_dict, company=obj)
    elif key == "contact":
        related["deals"] = _related(access, Deal, deal_dict, contact=obj)
        related["requests"] = _related(access, Request, request_dict, contact=obj)
    elif key == "lead":
        related["deals"] = _related(access, Deal, deal_dict, lead=obj)
        related["requests"] = _related(access, Request, request_dict, lead=obj)
    elif key == "request":
        pass  # lead/contact/company/deal are already part of the record
    elif key == "deal":
        related["payments"] = _related(access, Payment, payment_dict, deal=obj)
        related["requests"] = _related(access, Request, request_dict, deal=obj)
    data["related"] = related
    return _dump({key: data})


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _totals(qs) -> dict:
    # order_by() clears the admin's default ordering, which would otherwise
    # be added to GROUP BY and split the totals per record.
    return {
        str(row["currency__name"] or "no currency"): _money(row["total"])
        for row in qs.order_by().values("currency__name").annotate(total=Sum("amount"))
    }


@tool
@crm_read_tool
def sales_summary(days: int = 30, mine_only: bool = False, *, config: RunnableConfig) -> str:
    """A Sales report built from live CRM records you can access: open
    pipeline by stage, overdue next steps, deals won, new leads and requests,
    and payments received/expected over the last `days` days."""
    access = _access(config)
    days = max(1, min(int(days or 30), 366))
    today = timezone.localdate()
    since = today - timedelta(days=days)
    report: dict[str, Any] = {
        "period": f"{since.isoformat()} to {today.isoformat()}",
        "scope": access.department_label() + (" (your records only)" if mine_only else ""),
    }

    def own(qs, *fields):
        if not mine_only:
            return qs
        q = Q()
        for field in fields:
            q |= Q(**{field: access.user})
        return qs.filter(q)

    if access.can_view(Deal):
        deals = own(access.queryset(Deal), "owner", "co_owner")
        active = deals.filter(active=True)
        report["open_deals"] = {
            "count": active.count(),
            "amount_by_currency": _totals(active),
            "by_stage": [
                {"stage": row["stage__name"] or "no stage", "count": row["n"],
                 "amount": _money(row["total"])}
                for row in active.order_by().values("stage__name", "stage__index_number")
                .annotate(n=Count("id"), total=Sum("amount"))
                .order_by("stage__index_number")
            ],
        }
        overdue = active.filter(next_step_date__lt=today).order_by("next_step_date")
        report["overdue_next_steps"] = {
            "count": overdue.count(),
            "examples": [deal_dict(access, d) for d in overdue[:5]],
        }
        won = deals.filter(win_closing_date__date__gte=since)
        report["deals_won"] = {"count": won.count(), "amount_by_currency": _totals(won)}
        report["deals_created"] = deals.filter(creation_date__date__gte=since).count()
    else:
        report["deals"] = "not visible to you"

    if access.can_view(Lead):
        report["new_leads"] = own(access.queryset(Lead), "owner").filter(
            creation_date__date__gte=since).count()
    if access.can_view(Request):
        requests = own(access.queryset(Request), "owner", "co_owner")
        report["requests"] = {
            "received": requests.filter(creation_date__date__gte=since).count(),
            "pending_now": requests.filter(pending=True).count(),
        }
    if access.can_view(Payment):
        payments = access.queryset(Payment)
        if mine_only:
            payments = payments.filter(Q(deal__owner=access.user) | Q(deal__co_owner=access.user))
        report["payments"] = {
            "received_in_period": _totals(payments.filter(
                status=Payment.RECEIVED, payment_date__gte=since, payment_date__lte=today)),
            "expected_upcoming": _totals(payments.exclude(
                status=Payment.RECEIVED).filter(payment_date__gte=today)),
        }
    else:
        report["payments"] = "not visible to you"
    return _dump(report)


# ---------------------------------------------------------------------------
# reference values
# ---------------------------------------------------------------------------

_OPTION_MODELS = {
    "stages": Stage,
    "lead_sources": LeadSource,
    "closing_reasons": ClosingReason,
    "client_types": ClientType,
    "industries": Industry,
    "products": Product,
}

# Some reference models (notably Stage) have no standalone view permission in
# the CRM's group policy. Their values are nevertheless available in a model
# form to users who can operate on the consuming record type. Product is a
# full CRM record type and therefore keeps its own view-permission check.
_OPTION_CONSUMERS = {
    "stages": (Deal,),
    "lead_sources": (Lead, Company, Contact, Request),
    "closing_reasons": (Deal,),
    "client_types": (Company, Lead),
    "industries": (Company, Lead),
}


@tool
@crm_read_tool
def list_crm_options(kind: str, query: str = "", *, config: RunnableConfig) -> str:
    """List valid values for CRM choice fields in your department: stages,
    lead_sources, closing_reasons, client_types, industries, products,
    currencies, payment_statuses or countries (use `query` to filter
    countries by name)."""
    access = _access(config)
    kind = kind.strip().lower()
    department_id = access.department_id or access.home_department_id
    if kind == "currencies":
        return _dump([{"id": c.pk, "name": c.name} for c in Currency.objects.order_by("name")])
    if kind == "payment_statuses":
        return _dump([str(label) for _, label in Payment.STATUS_CHOICES])
    if kind == "countries":
        qs = Country.objects.order_by("name")
        if query.strip():
            qs = qs.filter(name__icontains=query.strip())
        return _dump([{"id": c.pk, "name": c.name} for c in qs[:30]])
    model = _OPTION_MODELS.get(kind)
    if model is None:
        return ("ERROR: kind must be one of "
                + ", ".join([*_OPTION_MODELS, "currencies", "payment_statuses", "countries"]) + ".")
    consumers = _OPTION_CONSUMERS.get(kind, ())
    try:
        model_view_allowed = access.can_view(model)
    except PermissionDenied:
        # Some CRM reference models are intentionally not registered as
        # standalone admin pages; their choices follow consumer permissions.
        model_view_allowed = False

    def can_use_consumer(consumer):
        try:
            return access.can_view(consumer) or access.can_add(consumer)
        except PermissionDenied:
            return False

    if not model_view_allowed and not any(map(can_use_consumer, consumers)):
        raise PermissionDenied(
            f"You do not have permission to view {model._meta.verbose_name_plural}.")
    qs = model.objects.all()
    if department_id:
        qs = qs.filter(department_id=department_id)
    if query.strip():
        qs = qs.filter(name__icontains=query.strip())
    rows = [{"id": x.pk, "name": str(x)} for x in qs.order_by("name")[:50]]
    return _dump(rows) if rows else f"No {kind} are configured for your department."


@tool
@crm_read_tool
def list_departments(*, config: RunnableConfig) -> str:
    """List the departments the authenticated user belongs to."""
    access = _access(config)
    departments = Department.objects.filter(user=access.user).distinct().order_by("name")
    return _dump([{"id": d.pk, "name": d.name} for d in departments])


READ_TOOLS = [
    search_leads,
    search_companies,
    search_contacts,
    search_requests,
    search_deals,
    search_payments,
    get_crm_record,
    sales_summary,
    list_crm_options,
    list_departments,
]
