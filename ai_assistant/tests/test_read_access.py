"""CRM read tools: department scoping and permission enforcement."""
import json
from django.core.exceptions import PermissionDenied
from django.contrib.auth.models import Group, Permission
from unittest.mock import patch
from django.test import TestCase

from ai_assistant.access import CrmAccess
from ai_assistant.read_tools import (
    get_crm_record, list_crm_options, list_departments, sales_summary,
    search_companies, search_contacts, search_deals, search_leads,
    search_payments, search_requests,
)
from ai_assistant.tools import search_database_schema
from ai_assistant.tests.base import GLOBAL_SALES, SalesTestCase
from common.utils.helpers import USER_MODEL
from crm.models import Company, Deal, Payment, Product


class CrmAccessTests(SalesTestCase):
    def test_manager_sees_only_own_department(self):
        access = CrmAccess(self.manager)
        names = set(access.queryset(Company).values_list("full_name", flat=True))
        self.assertIn("Acme Corp", names)
        self.assertNotIn("Hidden Local GmbH", names)
        self.assertIsNone(access.get_visible(Company, self.local_company.pk))

    def test_chief_without_department_sees_all_departments(self):
        access = CrmAccess(self.chief)
        self.assertTrue(access.queryset(Deal).filter(pk=self.local_deal.pk).exists())
        self.assertFalse(access.can_add(Company))   # chiefs only view

    def test_authentication_alone_grants_no_read_access(self):
        access = CrmAccess(self.coworker)   # staff, but no CRM view permissions
        with self.assertRaises(PermissionDenied):
            access.queryset(Company)

    def test_non_staff_and_inactive_users_are_refused(self):
        user = USER_MODEL.objects.create_user(username="nostaff", password="x")
        with self.assertRaises(PermissionDenied):
            CrmAccess(user)
        self.manager.is_active = False
        with self.assertRaises(PermissionDenied):
            CrmAccess(self.manager)

    def test_user_id_comes_only_from_config(self):
        with self.assertRaises(PermissionDenied):
            CrmAccess.from_config({"configurable": {}})

    def test_record_level_change_permission_follows_crm_rules(self):
        manager = CrmAccess(self.manager)
        self.assertTrue(manager.can_change(self.acme_deal))      # owner
        self.assertFalse(manager.can_change(self.other_deal))    # another manager's
        operator = CrmAccess(self.operator)
        self.assertTrue(operator.can_change(self.other_deal))    # operator, same dept
        self.assertFalse(CrmAccess(self.local_manager).can_change(self.acme_deal))

    def test_deals_cannot_be_added_directly(self):
        self.assertFalse(CrmAccess(self.manager).can_add(Deal))
        self.assertFalse(CrmAccess(self.admin).can_add(Deal))


class EmptyDatabaseReadTests(TestCase):
    def test_absent_acme_returns_not_found_from_database_tool(self):
        Group.objects.get_or_create(name="co-workers")
        admin = USER_MODEL.objects.create_superuser(
            username="empty_database_admin", email="admin@example.test", password="test")
        result = search_companies.invoke(
            {"name": "Acme Corp"},
            config={"configurable": {"user_id": admin.pk}})
        self.assertEqual(result, "No companies found that you can access with these criteria.")
        self.assertNotIn("Acme Corp", result)


class ReadToolTests(SalesTestCase):
    def test_search_companies_is_department_scoped(self):
        data = self.run_json(search_companies, self.manager, name="")
        names = {c["name"] for c in data["companies"]}
        self.assertIn("Acme Corp", names)
        self.assertNotIn("Hidden Local GmbH", names)
        result = self.run_tool(search_companies, self.manager, name="Hidden")
        self.assertIn("No companies found", result)

    def test_local_manager_cannot_see_global_records(self):
        result = self.run_tool(search_contacts, self.local_manager, name="Alice")
        self.assertIn("No contacts found", result)
        data = self.run_json(search_contacts, self.local_manager, name="Hans")
        self.assertEqual(data["contacts"][0]["name"], "Hans Hidden")

    def test_get_record_of_other_department_is_hidden(self):
        result = self.run_tool(get_crm_record, self.manager,
                               record_type="company", record_id=self.local_company.pk)
        self.assertIn("No company", result)
        self.assertNotIn("Hidden Local", result)

    def test_get_record_includes_visible_related_records(self):
        data = self.run_json(get_crm_record, self.manager,
                             record_type="company", record_id=self.acme.pk)
        company = data["company"]
        self.assertEqual(company["name"], "Acme Corp")
        self.assertEqual([c["name"] for c in company["related"]["contacts"]], ["Alice Anders"])
        self.assertEqual({d["name"] for d in company["related"]["deals"]},
                         {"Acme widgets", "Darian deal"})

    def test_reads_require_view_permission(self):
        for tool in (search_leads, search_companies, search_deals, search_requests):
            result = self.run_tool(tool, self.coworker)
            self.assertTrue(result.startswith("PERMISSION_DENIED"), tool.name)

    def test_search_leads_and_requests(self):
        data = self.run_json(search_leads, self.manager, query="leo")
        self.assertEqual(data["leads"][0]["email"], "leo@prospect.example")
        data = self.run_json(search_requests, self.manager, query="widgets")
        self.assertEqual(data["requests"][0]["deal"]["name"], "Acme widgets")
        self.assertEqual(data["requests"][0]["status"], "processed")

    def test_search_deals_filters(self):
        data = self.run_json(search_deals, self.manager, mine_only=True)
        self.assertEqual({d["name"] for d in data["deals"]}, {"Acme widgets"})
        data = self.run_json(search_deals, self.manager)
        self.assertEqual({d["name"] for d in data["deals"]}, {"Acme widgets", "Darian deal"})
        data = self.run_json(search_deals, self.chief)
        self.assertIn("Hidden local deal", {d["name"] for d in data["deals"]})

    def test_payments_are_scoped_through_their_deal(self):
        Payment.objects.create(deal=self.acme_deal, amount=100, currency_id=1)
        Payment.objects.create(deal=self.local_deal, amount=999, currency_id=1)
        data = self.run_json(search_payments, self.manager)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["totals_by_currency"], {"USD": "100.00"})
        self.assertTrue(self.run_tool(search_payments, self.operator)
                        .startswith("PERMISSION_DENIED"))   # operators: no payments

    def test_sales_summary_uses_scoped_live_data(self):
        data = self.run_json(sales_summary, self.manager, days=30)
        self.assertEqual(data["open_deals"]["count"], 2)
        self.assertEqual(data["open_deals"]["amount_by_currency"], {"USD": "1500.00"})
        self.assertEqual(data["overdue_next_steps"]["count"], 1)
        mine = self.run_json(sales_summary, self.manager, days=30, mine_only=True)
        self.assertEqual(mine["open_deals"]["count"], 1)

    def test_options_are_department_specific(self):
        raw = self.run_tool(list_crm_options, self.manager, kind="stages")
        self.assertFalse(raw.startswith("PERMISSION_DENIED"), raw)
        data = json.loads(raw)
        self.assertTrue(data)
        from crm.models import Stage
        self.assertEqual(
            {row["id"] for row in data},
            set(Stage.objects.filter(department_id=GLOBAL_SALES).values_list("pk", flat=True)))

    def test_list_departments(self):
        data = self.run_json(list_departments, self.manager)
        self.assertEqual([d["name"] for d in data], ["Global sales"])

    def test_ordinary_user_can_use_schema_and_permitted_read_tools(self):
        ordinary = USER_MODEL.objects.create_user(
            username="temporary_ordinary", password="test", is_staff=True)
        ordinary.groups.add(Group.objects.get(pk=GLOBAL_SALES))
        config = self.config(ordinary)

        # Schema RAG has no CRM-record permissions or write capability.
        with patch("ai_assistant.tools.search_schema_with_relationships",
                   return_value={"matched_models": [], "expanded_models": [],
                                 "schemas": {}}):
            schema = search_database_schema.invoke({"query": "lead fields"})
        self.assertIn("## Lead", schema)
        self.assertIn("not live records", schema)

        self.assertTrue(search_companies.invoke({}, config=config).startswith(
            "PERMISSION_DENIED"))
        Product.objects.create(name="Restricted test product", department_id=GLOBAL_SALES)
        self.assertNotIn("Restricted test product", list_crm_options.invoke(
            {"kind": "products"}, config=config))

        # Explicit view permissions enable read-only RAG, still department scoped.
        view_codenames = [f"view_{model}" for model in (
            "lead", "company", "contact", "request", "deal", "payment")]
        ordinary.user_permissions.set(Permission.objects.filter(
            content_type__app_label="crm", codename__in=view_codenames))
        ordinary = USER_MODEL.objects.get(pk=ordinary.pk)
        config = self.config(ordinary)
        product_result = list_crm_options.invoke({"kind": "products"}, config=config)
        self.assertTrue(product_result.startswith("PERMISSION_DENIED"))
        ordinary.user_permissions.add(Permission.objects.get(
            content_type__app_label="crm", codename="view_product"))
        ordinary = USER_MODEL.objects.get(pk=ordinary.pk)
        config = self.config(ordinary)
        product_result = list_crm_options.invoke({"kind": "products"}, config=config)
        self.assertIn("Restricted test product", product_result)
        company_result = search_companies.invoke({}, config=config)
        self.assertIn("Acme Corp", company_result)
        self.assertNotIn("Hidden Local GmbH", company_result)
        lead_result = search_leads.invoke({}, config=config)
        self.assertIn("Leo Lead", lead_result)
        self.assertNotIn("Hidden", lead_result)

        for model in ("lead", "company", "contact", "request", "deal", "payment"):
            self.assertFalse(ordinary.has_perm(f"crm.add_{model}"), model)
            self.assertFalse(ordinary.has_perm(f"crm.change_{model}"), model)
            self.assertFalse(ordinary.has_perm(f"crm.delete_{model}"), model)
