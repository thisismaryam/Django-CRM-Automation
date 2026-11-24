"""Permission and department scoping for the AI assistant.

The assistant never re-implements CRM access rules. It builds a server-side
request for the authenticated user, applies the same role/department
attributes that ``common.utils.usermiddleware.UserMiddleware`` sets on every
CRM request, and then asks the CRM's own registered ModelAdmin classes:

* ``get_queryset``          -> which records the user may see (department scope)
* ``has_view_permission``   -> may the user read this record type at all
* ``has_add_permission``    -> may the user create this record type
* ``has_change_permission`` -> may the user change this record (record-level
  rules such as ``clarify_permission`` and the Payment/Deal owner rules)
* ``get_form`` / ``save_model`` / ``save_related`` -> validation and the
  business side effects of saving

The user is always loaded from the backend request context (the ``user_id``
the API view puts in the LangGraph config), never from LLM output.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, QueryDict
from django.urls import NoReverseMatch, reverse

from common.models import Department
from common.utils.helpers import get_department_id
from common.utils.usermiddleware import set_user_department, set_user_groups
from crm.models import Company, Contact, Deal, Lead, Payment, Request
from crm.site.crmadminsite import crm_site

# Record types the Sales agent works with, keyed by the name used in tools.
SALES_MODELS: dict[str, Any] = {
    "lead": Lead,
    "company": Company,
    "contact": Contact,
    "request": Request,
    "deal": Deal,
    "payment": Payment,
}


class MessageCollector:
    """Stands in for django.contrib.messages storage on the server-side
    request, so CRM admin code that calls ``messages.success(...)`` works and
    its notices can be reported back to the user."""

    def __init__(self):
        self.messages: list[str] = []

    def add(self, level, message, extra_tags=""):
        self.messages.append(str(message))

    def __iter__(self):
        return iter(self.messages)


def get_active_staff_user(user_id: Any):
    """Load the authenticated user; inactive or non-staff users get no
    access, matching the CRM admin site's own ``has_permission`` check."""
    if user_id is None:
        raise PermissionDenied("An authenticated user is required.")
    user = get_user_model().objects.filter(pk=user_id).first()
    if user is None or not user.is_active or not user.is_staff:
        raise PermissionDenied("This account cannot use the CRM.")
    return user


class CrmAccess:
    """The authenticated user's view of the CRM, enforced by the CRM admin."""

    def __init__(self, user):
        if not user.is_active or not user.is_staff:
            raise PermissionDenied("This account cannot use the CRM.")
        self.user = user
        # build_request() applies UserMiddleware's role flags and active
        # department to the user object (user.department_id, is_manager, ...).
        self.build_request()
        # Department used for scoping: None means "all departments" (superusers
        # and chiefs), exactly as on the CRM pages.
        self.department_id = getattr(user, "department_id", None)
        # The department new records are assigned to (CRM set_owner logic).
        self.home_department_id = get_department_id(user)

    # -- construction -------------------------------------------------

    @classmethod
    def from_config(cls, config: dict | None) -> "CrmAccess":
        user_id = ((config or {}).get("configurable") or {}).get("user_id")
        return cls(get_active_staff_user(user_id))

    def build_request(self, post: dict | None = None, obj=None) -> HttpRequest:
        """A request object for calling CRM admin methods on the user's
        behalf. ``post`` carries admin button names such as ``_create-deal``.
        """
        request = HttpRequest()
        request.method = "POST"
        request.user = self.user
        request.session = {}
        request.META["SERVER_NAME"] = "localhost"
        request.META["SERVER_PORT"] = "80"
        request.POST = QueryDict(mutable=True)
        for key, value in (post or {}).items():
            request.POST[key] = value
        # CrmModelAdmin.set_queryset reads the object id from the URL match.
        request.resolver_match = SimpleNamespace(
            kwargs={"object_id": obj.pk} if obj is not None and obj.pk else {}
        )
        request._messages = MessageCollector()
        groups = self.user.groups.all()
        set_user_groups(request, groups)
        set_user_department(request, groups)
        return request

    # -- admin delegation ---------------------------------------------

    @staticmethod
    def admin_for(model):
        model_admin = crm_site._registry.get(model)
        if model_admin is None:
            raise PermissionDenied(
                f"{model._meta.verbose_name} is not available in the CRM.")
        return model_admin

    def can_view(self, model) -> bool:
        return self.admin_for(model).has_view_permission(self.build_request())

    def can_add(self, model) -> bool:
        return self.admin_for(model).has_add_permission(self.build_request())

    def can_change(self, obj) -> bool:
        model_admin = self.admin_for(type(obj))
        return model_admin.has_change_permission(
            self.build_request(obj=obj), obj)

    def require_view(self, model) -> None:
        if not self.can_view(model):
            raise PermissionDenied(
                f"You do not have permission to view "
                f"{model._meta.verbose_name_plural}.")

    def queryset(self, model):
        """Records of ``model`` the user is allowed to see."""
        self.require_view(model)
        return self.admin_for(model).get_queryset(self.build_request())

    def get_visible(self, model, pk) -> Any | None:
        """The record with ``pk`` if it exists *and* the user may see it."""
        try:
            pk = int(pk)
        except (TypeError, ValueError):
            return None
        return self.queryset(model).filter(pk=pk).first()

    # -- descriptions -------------------------------------------------

    def department_label(self) -> str:
        department_id = self.department_id or self.home_department_id
        if not department_id:
            return "no department (all departments visible)" \
                if self.user.is_superuser else "no department"
        department = Department.objects.filter(pk=department_id).first()
        return department.name if department else "unknown department"

    def user_label(self) -> str:
        return self.user.get_full_name() or self.user.username


def record_url(obj) -> str:
    """Link to the record's page in the CRM UI."""
    meta = obj._meta
    for name in (f"site:{meta.app_label}_{meta.model_name}_change",
                 f"admin:{meta.app_label}_{meta.model_name}_change"):
        try:
            return reverse(name, args=(obj.pk,))
        except NoReverseMatch:
            continue
    return ""
