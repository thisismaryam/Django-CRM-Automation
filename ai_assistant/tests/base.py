"""Shared fixtures for the Sales agent tests.

Uses the CRM's own test fixtures (groups with their real permissions,
departments, users, stages, currencies), so permission tests exercise the
same rules as the CRM UI.

Run:  python manage.py test ai_assistant
"""
import json
from datetime import timedelta

from django.contrib.auth.models import Group
from django.test import TestCase
from django.utils import timezone
from langchain_core.messages import AIMessage

from common.utils.helpers import USER_MODEL
from crm.models import Company, Contact, Country, Deal, Lead, Request, Stage
from tests.base_test_classes import BaseTestCase

GLOBAL_SALES = 9
LOCAL_SALES = 10


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedLLM:
    """Stands in for the provider chain: returns queued responses in order
    and records the prompts it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.responses:
            return AIMessage(content="(no more scripted responses)")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SalesTestCase(TestCase):
    fixtures = BaseTestCase.fixtures

    @classmethod
    def setUpTestData(cls):
        get = USER_MODEL.objects.get
        cls.manager = get(username="Andrew.Manager.Global")
        cls.manager2 = get(username="Darian.Manager.Co-worker.Head.Global")
        cls.operator = get(username="Valeria.Operator.Global")
        cls.chief = get(username="Garry.Chief")
        cls.coworker = get(username="Masha.Co-worker.Bookkeeping")
        cls.admin = get(username="Adam.Admin")

        cls.local_manager = USER_MODEL.objects.create_user(
            username="Lucy.Manager.Local", password="x", is_staff=True)
        cls.local_manager.groups.add(
            Group.objects.get(name="managers"), Group.objects.get(pk=LOCAL_SALES))

        cls.country = Country.objects.order_by("pk").first()
        today = timezone.localdate()

        cls.acme = Company.objects.create(
            full_name="Acme Corp", email="info@acme.example", phone="+1 555 0100",
            website="https://acme.example", city_name="Springfield",
            address="1 Acme Way", country=cls.country,
            owner=cls.manager, department_id=GLOBAL_SALES)
        cls.acme_contact = Contact.objects.create(
            first_name="Alice", last_name="Anders", email="alice@acme.example",
            company=cls.acme, owner=cls.manager, department_id=GLOBAL_SALES)
        cls.lead = Lead.objects.create(
            first_name="Leo", last_name="Lead", email="leo@prospect.example",
            company_name="Prospect Ltd", country=cls.country,
            owner=cls.manager, department_id=GLOBAL_SALES)

        cls.local_company = Company.objects.create(
            full_name="Hidden Local GmbH", email="hello@hidden.example",
            owner=cls.local_manager, department_id=LOCAL_SALES)
        cls.local_contact = Contact.objects.create(
            first_name="Hans", last_name="Hidden", email="hans@hidden.example",
            company=cls.local_company, owner=cls.local_manager,
            department_id=LOCAL_SALES)

        stage = Stage.objects.get(department_id=GLOBAL_SALES, default=True)
        cls.acme_request = Request.objects.create(
            request_for="Acme wants 10 widgets", first_name="Alice",
            email="alice@acme.example", contact=cls.acme_contact,
            company=cls.acme, owner=cls.manager, department_id=GLOBAL_SALES,
            pending=False)
        cls.acme_deal = Deal.objects.create(
            name="Acme widgets", next_step="Call Alice",
            next_step_date=today + timedelta(days=3), stage=stage,
            amount=1000, currency_id=1, contact=cls.acme_contact,
            company=cls.acme, request=cls.acme_request, ticket="T-ACME-1",
            owner=cls.manager, department_id=GLOBAL_SALES)
        cls.acme_request.deal = cls.acme_deal
        cls.acme_request.save(update_fields=["deal"])
        cls.other_deal = Deal.objects.create(
            name="Darian deal", next_step="Send offer",
            next_step_date=today - timedelta(days=2), stage=stage,
            amount=500, currency_id=1, company=cls.acme,
            ticket="T-DARIAN-1", owner=cls.manager2,
            department_id=GLOBAL_SALES)
        cls.local_deal = Deal.objects.create(
            name="Hidden local deal", next_step="Wait",
            next_step_date=today + timedelta(days=1),
            stage=Stage.objects.get(department_id=LOCAL_SALES, default=True),
            amount=999, currency_id=1, company=cls.local_company,
            ticket="T-LOCAL-1", owner=cls.local_manager,
            department_id=LOCAL_SALES)

    @staticmethod
    def config(user, thread: str = "t1") -> dict:
        return {"configurable": {"user_id": user.pk,
                                 "thread_id": f"user-{user.pk}:{thread}"}}

    def run_tool(self, tool, user, **args):
        return tool.invoke(args, config=self.config(user))

    def run_json(self, tool, user, **args):
        return json.loads(self.run_tool(tool, user, **args))
