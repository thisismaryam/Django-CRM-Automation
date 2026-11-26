"""Regression tests for the audit fixes:

1. users without a department are denied (no all-department leak)
2. superusers can do every CRM operation without a department group
3. duplicate emails are matched exactly (case-insensitive), not by substring
4. approved plans never overwrite data changed after the card was generated;
   pending approvals expire
"""
import os
from datetime import timedelta
from io import StringIO
from unittest.mock import patch

from django.contrib.auth.models import Group
from django.core.exceptions import PermissionDenied
from django.core.management import call_command
from django.utils import timezone

from ai_assistant.access import CrmAccess
from ai_assistant.models import ActionRequest
from ai_assistant.read_tools import (
    list_crm_options, list_departments, search_companies, search_deals,
)
from ai_assistant.tests.base import GLOBAL_SALES, LOCAL_SALES, SalesTestCase
from ai_assistant.tests.test_workflow import op, plan
from ai_assistant.workflow import (
    approval_payload, approval_ttl, execute_plan, format_report, validate_plan,
)
from common.utils.helpers import USER_MODEL
from crm.models import Company, Deal, Lead, Request

U = USER_MODEL.objects


def make_user(username, groups=(), **flags):
    user = U.create_user(username=username, password="x",
                         **({"is_staff": True} | flags))
    for group in groups:
        user.groups.add(Group.objects.get(pk=group) if isinstance(group, int)
                        else Group.objects.get(name=group))
    return U.get(pk=user.pk)


class AuditBase(SalesTestCase):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.nodept_manager = make_user("nodept_mgr", ["managers"])
        cls.superuser = make_user("root_no_dept", is_superuser=True)

    def validate(self, user, args):
        return validate_plan(CrmAccess(U.get(pk=user.pk)), args, thread_id="user-x:t")

    def run_plan(self, user, args):
        outcome = self.validate(user, args)
        self.assertTrue(outcome.ok, outcome.message)
        return execute_plan(CrmAccess(U.get(pk=user.pk)), outcome.plan), outcome


# ---------------------------------------------------------------------------
# 1. No department => no access (deny by default)
# ---------------------------------------------------------------------------

class NoDepartmentManagerTests(AuditBase):
    def test_manager_without_department_is_denied_reads(self):
        access = CrmAccess(self.nodept_manager)
        self.assertTrue(access.scope_error())
        with self.assertRaises(PermissionDenied):
            access.queryset(Company)
        self.assertFalse(access.can_view(Company))
        for tool, args in [(search_companies, {}), (search_deals, {}),
                           (list_crm_options, {"kind": "stages"}),
                           (list_departments, {})]:
            result = self.run_tool(tool, self.nodept_manager, **args)
            self.assertTrue(result.startswith("PERMISSION_DENIED"), (tool.name, result))
            self.assertNotIn("Acme", result)
            self.assertNotIn("Hidden Local", result)

    def test_manager_without_department_cannot_write(self):
        access = CrmAccess(self.nodept_manager)
        self.assertFalse(access.can_add(Lead))
        before = Lead.objects.count()
        outcome = self.validate(self.nodept_manager, plan(op(
            "op1", "create", "lead", first_name="No", email="nodept@z.example")))
        self.assertFalse(outcome.ok)
        self.assertIn("PERMISSION", outcome.message.upper())
        outcome = self.validate(self.nodept_manager, plan(op(
            "op1", "update", "company", record_id=self.local_company.pk, phone="1")))
        self.assertFalse(outcome.ok)
        self.assertEqual(Lead.objects.count(), before)

    def test_plan_created_before_losing_department_is_not_executed(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Late", email="late@z.example")))
        self.assertTrue(outcome.ok, outcome.message)
        self.manager.groups.remove(Group.objects.get(pk=GLOBAL_SALES))
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(Lead.objects.filter(email="late@z.example").exists())

    def test_superuser_chief_and_accountant_keep_their_access(self):
        self.assertEqual(CrmAccess(self.superuser).scope_error(), "")
        self.assertEqual(CrmAccess(self.chief).scope_error(), "")
        accountant = make_user("acct", ["accountants"])
        self.assertEqual(CrmAccess(accountant).scope_error(), "")
        # superuser and chief still see every department
        for user in (self.superuser, self.chief):
            names = set(CrmAccess(U.get(pk=user.pk)).queryset(Company)
                        .values_list("full_name", flat=True))
            self.assertLessEqual({"Acme Corp", "Hidden Local GmbH"}, names)

    def test_superoperator_without_department_group_is_denied(self):
        Group.objects.get_or_create(name="superoperators")
        user = make_user("superop", ["superoperators"])
        self.assertTrue(CrmAccess(user).scope_error())

    def test_department_users_are_unchanged(self):
        names = set(CrmAccess(self.manager).queryset(Company)
                    .values_list("full_name", flat=True))
        self.assertIn("Acme Corp", names)
        self.assertNotIn("Hidden Local GmbH", names)
        names = set(CrmAccess(self.local_manager).queryset(Company)
                    .values_list("full_name", flat=True))
        self.assertEqual(names, {"Hidden Local GmbH"})


# ---------------------------------------------------------------------------
# 2. Superuser works without a department group (department comes from the
#    linked counterparty, as in the CRM) - and the scope check keeps allowing it
# ---------------------------------------------------------------------------

class SuperuserWithoutDepartmentTests(AuditBase):
    def test_superuser_has_no_department_group(self):
        self.assertIsNone(CrmAccess(self.superuser).home_department_id)
        self.assertEqual(CrmAccess(self.superuser).scope_error(), "")

    def test_request_and_deal_without_joining_a_department(self):
        report, _ = self.run_plan(self.superuser, plan(
            op("op1", "create", "request", request_for="From admin",
               create_deal=True, contact=self.acme_contact.pk),
            op("op2", "update", "deal", record_id="$op1.deal", amount="5000")))
        self.assertEqual(report["status"], "success", report)
        deal = Deal.objects.get(name="From admin")
        self.assertEqual(deal.department_id, GLOBAL_SALES)
        self.assertEqual(float(deal.amount), 5000.0)
        # still in no department group
        self.assertIsNone(CrmAccess(U.get(pk=self.superuser.pk)).home_department_id)

    def test_superuser_updates_any_department(self):
        report, _ = self.run_plan(self.superuser, plan(op(
            "op1", "update", "company", record_id=self.local_company.pk,
            phone="+1 202 555 0000")))
        self.assertEqual(report["status"], "success", report)

    def test_manager_without_department_still_cannot_make_deals(self):
        outcome = self.validate(self.nodept_manager, plan(op(
            "op1", "create", "request", request_for="Nope", create_deal=True,
            contact=self.acme_contact.pk)))
        self.assertFalse(outcome.ok)
        self.assertFalse(Deal.objects.filter(name="Nope").exists())


# ---------------------------------------------------------------------------
# 3. Exact, case-insensitive duplicate emails
# ---------------------------------------------------------------------------

class DuplicateEmailTests(AuditBase):
    def lead_plan(self, email):
        return plan(op("op1", "create", "lead", first_name="Dee", email=email))

    def test_substring_of_existing_email_is_not_a_duplicate(self):
        # leo@prospect.example exists; eo@... / o@... are different people.
        for email in ("eo@prospect.example", "xleo@prospect.example",
                      "leo@prospect.example.org"):
            outcome = self.validate(self.manager, self.lead_plan(email))
            self.assertTrue(outcome.ok, (email, outcome.message))

    def test_exact_match_is_a_duplicate_regardless_of_case(self):
        for email in ("leo@prospect.example", "LEO@Prospect.Example"):
            outcome = self.validate(self.manager, self.lead_plan(email))
            self.assertFalse(outcome.ok, email)
            self.assertIn("duplicate", outcome.message)

    def test_address_inside_a_comma_separated_list_is_matched(self):
        Lead.objects.create(
            first_name="Multi", email="m1@x.example, M2@x.example",
            owner=self.manager, department_id=GLOBAL_SALES)
        self.assertFalse(self.validate(self.manager, self.lead_plan("m2@x.example")).ok)
        self.assertFalse(self.validate(self.manager, self.lead_plan("m1@X.example")).ok)
        self.assertTrue(self.validate(self.manager, self.lead_plan("2@x.example")).ok)

    def test_secondary_email_is_matched_exactly(self):
        Lead.objects.create(
            first_name="Sec", email="primary@x.example",
            secondary_email="Second@x.example",
            owner=self.manager, department_id=GLOBAL_SALES)
        self.assertFalse(self.validate(self.manager, self.lead_plan("second@x.example")).ok)
        self.assertTrue(self.validate(self.manager, self.lead_plan("econd@x.example")).ok)

    def test_email_with_regex_characters_is_literal(self):
        # An unescaped "." would make l.o@... match the existing leo@...
        outcome = self.validate(self.manager, self.lead_plan("l.o@prospect.example"))
        self.assertTrue(outcome.ok, outcome.message)

    def test_contact_dedup_and_other_duplicate_rules_still_work(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "contact", first_name="Al", email="ALICE@acme.example",
            company=self.acme.pk)))
        self.assertFalse(outcome.ok)
        self.assertIn("duplicate", outcome.message)
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "company", full_name="acme corp", email="n@acme2.example")))
        self.assertFalse(outcome.ok)
        self.assertIn("duplicate", outcome.message)

    def test_duplicate_created_after_approval_blocks_execution(self):
        outcome = self.validate(self.manager, self.lead_plan("race@z.example"))
        self.assertTrue(outcome.ok, outcome.message)
        Lead.objects.create(first_name="Fast", email="race@z.example",
                            owner=self.manager, department_id=GLOBAL_SALES)
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.assertIn("duplicate", report["error"])
        self.assertEqual(Lead.objects.filter(email="race@z.example").count(), 1)


# ---------------------------------------------------------------------------
# 4. Stale approvals and expiry
# ---------------------------------------------------------------------------

class StaleApprovalTests(AuditBase):
    def company_phone_plan(self, phone="+1 111 0000"):
        return plan(op("op1", "update", "company", record_id=self.acme.pk, phone=phone))

    def test_unchanged_record_is_updated(self):
        report, _ = self.run_plan(self.manager, self.company_phone_plan())
        self.assertEqual(report["status"], "success", report)
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.phone, "+1 111 0000")

    def test_edit_after_approval_card_is_not_overwritten(self):
        outcome = self.validate(self.manager, self.company_phone_plan())
        self.assertTrue(outcome.ok, outcome.message)
        Company.objects.filter(pk=self.acme.pk).update(phone="+9 CHANGED BY A COLLEAGUE")
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.assertIn("changed after this approval card", report["error"])
        self.assertIn("phone", report["error"])
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.phone, "+9 CHANGED BY A COLLEAGUE")
        record = ActionRequest.objects.get(operation_id=outcome.plan["operation_id"])
        self.assertEqual(record.status, ActionRequest.FAILED)
        self.assertIn("Nothing was saved", format_report(report))

    def test_stale_check_is_all_or_nothing(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "lead", first_name="Pair", email="pair@z.example"),
            op("op2", "update", "company", record_id=self.acme.pk, phone="+1 222 0000")))
        self.assertTrue(outcome.ok, outcome.message)
        Company.objects.filter(pk=self.acme.pk).update(phone="+9 NEW")
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(Lead.objects.filter(email="pair@z.example").exists())

    def test_unrelated_field_edits_do_not_block(self):
        outcome = self.validate(self.manager, self.company_phone_plan("+1 333 0000"))
        Company.objects.filter(pk=self.acme.pk).update(city_name="Elsewhere")
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "success", report)
        self.acme.refresh_from_db()
        self.assertEqual((self.acme.phone, self.acme.city_name),
                         ("+1 333 0000", "Elsewhere"))

    def test_deal_stage_change_after_card_blocks_update(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "deal", record_id=self.acme_deal.pk, amount="2000")))
        self.assertTrue(outcome.ok, outcome.message)
        other_stage = Deal._meta.get_field("stage").related_model.objects.filter(
            department_id=GLOBAL_SALES).exclude(pk=self.acme_deal.stage_id).first()
        Deal.objects.filter(pk=self.acme_deal.pk).update(stage=other_stage)
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.acme_deal.refresh_from_db()
        self.assertEqual(float(self.acme_deal.amount), 1000.0)

    def test_plan_without_snapshot_fails_safe(self):
        outcome = self.validate(self.manager, self.company_phone_plan())
        for operation in outcome.plan["operations"]:
            operation.pop("original")
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.acme.refresh_from_db()
        self.assertNotEqual(self.acme.phone, "+1 111 0000")

    def test_permission_removed_after_approval_blocks_execution(self):
        outcome = self.validate(self.manager, self.company_phone_plan())
        self.manager.groups.remove(Group.objects.get(name="managers"))
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.acme.refresh_from_db()
        self.assertNotEqual(self.acme.phone, "+1 111 0000")

    def test_record_moved_out_of_scope_blocks_execution(self):
        outcome = self.validate(self.manager, self.company_phone_plan())
        Company.objects.filter(pk=self.acme.pk).update(department_id=LOCAL_SALES)
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "failed")


class ApprovalExpiryTests(AuditBase):
    def pending_lead_plan(self, email="exp@z.example"):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Exp", email=email)))
        self.assertTrue(outcome.ok, outcome.message)
        return outcome

    def age(self, outcome, minutes):
        ActionRequest.objects.filter(
            operation_id=outcome.plan["operation_id"]
        ).update(created_at=timezone.now() - timedelta(minutes=minutes))

    def test_card_shows_expiry(self):
        outcome = self.pending_lead_plan()
        self.assertIn("expires_at", approval_payload(outcome.plan))

    def test_fresh_plan_executes(self):
        outcome = self.pending_lead_plan()
        self.age(outcome, 59)
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "success", report)

    def test_expired_plan_is_refused_and_closed(self):
        outcome = self.pending_lead_plan()
        self.age(outcome, 61)
        report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "expired")
        self.assertIn("expired", format_report(report))
        self.assertFalse(Lead.objects.filter(email="exp@z.example").exists())
        record = ActionRequest.objects.get(operation_id=outcome.plan["operation_id"])
        self.assertEqual(record.status, ActionRequest.FAILED)
        self.assertEqual(record.result["status"], "expired")

    def test_ttl_is_configurable(self):
        outcome = self.pending_lead_plan("ttl@z.example")
        self.age(outcome, 11)
        with patch.dict(os.environ, {"AI_ASSISTANT_APPROVAL_TTL_MINUTES": "10"}):
            report = execute_plan(CrmAccess(U.get(pk=self.manager.pk)), outcome.plan)
        self.assertEqual(report["status"], "expired")

    def test_invalid_ttl_falls_back_to_default(self):
        outcome = self.pending_lead_plan("ttl2@z.example")
        self.age(outcome, 30)
        for bad in ("abc", "0", "-5"):
            with patch.dict(os.environ, {"AI_ASSISTANT_APPROVAL_TTL_MINUTES": bad}):
                self.assertEqual(approval_ttl(), timedelta(minutes=60))

    def test_expire_command_closes_only_old_pending_rows(self):
        old = self.pending_lead_plan("old@z.example")
        fresh = self.pending_lead_plan("fresh@z.example")
        self.age(old, 500)
        leads_before = Lead.objects.count()
        out = StringIO()
        call_command("expire_ai_approvals", "--dry-run", stdout=out)
        self.assertIn("Would expire 1", out.getvalue())
        self.assertEqual(ActionRequest.objects.get(
            operation_id=old.plan["operation_id"]).status, ActionRequest.PENDING)
        call_command("expire_ai_approvals", stdout=out)
        self.assertEqual(ActionRequest.objects.get(
            operation_id=old.plan["operation_id"]).status, ActionRequest.FAILED)
        self.assertEqual(ActionRequest.objects.get(
            operation_id=fresh.plan["operation_id"]).status, ActionRequest.PENDING)
        self.assertEqual(Lead.objects.count(), leads_before)   # CRM data untouched
