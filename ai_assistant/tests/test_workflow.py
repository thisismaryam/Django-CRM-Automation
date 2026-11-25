"""Workflow plans: validation, approval payload, atomic execution."""
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4
from unittest.mock import patch

from django.contrib.admin.models import LogEntry
from django.contrib.auth.models import Group
from django.utils import timezone

from ai_assistant.access import CrmAccess, SALES_MODELS
from ai_assistant.models import ActionRequest
from ai_assistant.tests.base import GLOBAL_SALES, SalesTestCase
from ai_assistant.workflow import (
    approval_payload, execute_plan, format_report, validate_plan,
)
from common.utils.helpers import USER_MODEL
from crm.models import (
    ClosingReason, Company, Contact, Deal, Lead, LeadSource, Payment, Request, Stage,
)


def op(op_id, action, record_type, record_id="", create_deal=False, **fields):
    return {
        "op_id": op_id, "action": action, "record_type": record_type,
        "record_id": str(record_id), "create_deal": create_deal,
        "fields": [{"name": k, "value": v} for k, v in fields.items()],
    }


def plan(*ops, goal="test goal"):
    return {"goal": goal, "operations": list(ops)}


class WorkflowTestCase(SalesTestCase):
    def validate(self, user, args):
        self.pending_before = ActionRequest.objects.count()
        return validate_plan(CrmAccess(user), args, thread_id="user-x:t")

    def execute(self, user, outcome):
        self.assertTrue(outcome.ok, outcome.message)
        fresh = USER_MODEL.objects.get(pk=user.pk)
        return execute_plan(CrmAccess(fresh), outcome.plan)

    def assertRejected(self, outcome, *fragments):
        self.assertFalse(outcome.ok)
        for fragment in fragments:
            self.assertIn(fragment, outcome.message)
        # An invalid plan is never stored for approval.
        self.assertEqual(ActionRequest.objects.count(), self.pending_before)


class LeadWorkflowTests(WorkflowTestCase):
    def test_create_lead_needs_approval_then_saves_with_owner_and_department(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Nina", last_name="New",
            email="nina@newco.example", company_name="NewCo")))
        self.assertTrue(outcome.ok, outcome.message)

        # Nothing written to CRM tables before approval.
        self.assertFalse(Lead.objects.filter(email="nina@newco.example").exists())
        pending = ActionRequest.objects.get()
        self.assertEqual(pending.status, ActionRequest.PENDING)
        self.assertEqual(pending.operation_id, outcome.plan["operation_id"])

        card = approval_payload(outcome.plan)
        self.assertEqual(card["action"], "SALES_WORKFLOW")
        first = card["operations"][0]
        self.assertEqual(first["title"], "Create Lead")
        self.assertEqual(first["fields"]["First name"], "Nina")
        self.assertIn("Owner", first["inferred"])
        self.assertEqual(first["inferred"]["Department"], "Global sales")
        self.assertTrue(any("not linked to a CRM company" in w for w in first["warnings"]))

        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        lead = Lead.objects.get(email="nina@newco.example")
        self.assertEqual(lead.owner, self.manager)
        self.assertEqual(lead.department_id, GLOBAL_SALES)
        self.assertEqual(report["results"][0]["record_id"], lead.pk)
        self.assertTrue(report["results"][0]["url"].endswith(f"/crm/lead/{lead.pk}/change/"))
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.EXECUTED)
        self.assertTrue(LogEntry.objects.filter(object_id=str(lead.pk)).exists())
        self.assertIn("Done. All changes were saved", format_report(report))

    def test_missing_required_information_asks_for_details(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Nina")))
        self.assertRejected(outcome, "NEEDS_DETAILS", "email")

    def test_duplicate_lead_email_is_blocked_and_contact_email_warned(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Leo", email="leo@prospect.example")))
        self.assertRejected(outcome, "VALIDATION_FAILED", "duplicate", f"#{self.lead.pk}")

        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Al", email="alice@acme.example")))
        self.assertTrue(outcome.ok, outcome.message)
        warnings = outcome.plan["operations"][0]["display"]["warnings"]
        self.assertTrue(any(f"#{self.acme_contact.pk}" in w for w in warnings))

    def test_same_email_twice_in_one_plan_is_blocked(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "lead", first_name="A", email="twin@x.example"),
            op("op2", "create", "lead", first_name="B", email="twin@x.example")))
        self.assertRejected(outcome, "twin@x.example is used by op1")

    def test_lead_linked_to_company_inherits_company_details(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Kim", email="kim@acme.example",
            company=self.acme.pk)))
        self.assertTrue(outcome.ok, outcome.message)
        display = outcome.plan["operations"][0]["display"]
        self.assertIn(f"#{self.acme.pk} Acme Corp (existing)", display["related"].values())
        self.assertIn("Company phone", display["inferred"])
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        lead = Lead.objects.get(email="kim@acme.example")
        self.assertEqual(lead.company, self.acme)
        self.assertEqual(lead.company_name, "Acme Corp")
        self.assertEqual(lead.company_phone, "+1 555 0100")
        self.assertEqual(lead.website, "https://acme.example")
        self.assertEqual(lead.country, self.country)

    def test_linking_existing_lead_to_company_keeps_its_own_values(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "lead", record_id=self.lead.pk, company=self.acme.pk)))
        self.assertTrue(outcome.ok, outcome.message)
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        self.lead.refresh_from_db()
        self.assertEqual(self.lead.company, self.acme)
        self.assertEqual(self.lead.company_name, "Prospect Ltd")      # kept
        self.assertEqual(self.lead.company_phone, "+1 555 0100")      # was empty

    def test_choice_fields_resolve_by_name_within_department(self):
        source = LeadSource.objects.filter(department_id=GLOBAL_SALES).first()
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Sam", email="sam@x.example",
            lead_source=source.name)))
        self.assertTrue(outcome.ok, outcome.message)
        self.execute(self.manager, outcome)
        self.assertEqual(Lead.objects.get(email="sam@x.example").lead_source, source)

        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Sam", email="sam2@x.example",
            lead_source="Carrier pigeon")))
        self.assertRejected(outcome, "lead_source", "not available to you")

    def test_unknown_field_is_rejected(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="X", email="x@x.example", owner="1")))
        self.assertRejected(outcome, "cannot be set", "Allowed:")


class UnassignedAdminWorkflowTests(WorkflowTestCase):
    def setUp(self):
        self.maryam = USER_MODEL.objects.create_superuser(
            username="Maryam", email="maryam@example.test", password="test")
        self.assertFalse(self.maryam.groups.filter(pk=GLOBAL_SALES).exists())

    def test_admin_has_every_supported_permission_except_direct_deal_create(self):
        access = CrmAccess(self.maryam)
        for model in SALES_MODELS.values():
            self.assertTrue(access.can_view(model), model.__name__)
            self.assertEqual(access.can_add(model), model is not Deal, model.__name__)
        payment = Payment.objects.create(deal=self.acme_deal, amount=5, currency_id=1)
        for record in (self.lead, self.acme, self.acme_contact,
                       self.acme_request, self.acme_deal, payment):
            self.assertTrue(access.can_change(record), record._meta.label)

    def test_admin_can_create_request_without_a_department_membership(self):
        outcome = self.validate(self.maryam, plan(op(
            "op1", "create", "request", request_for="Admin unassigned request",
            first_name="Maryam")))
        self.assertTrue(outcome.ok, outcome.message)
        self.assertFalse(Request.objects.filter(
            request_for="Admin unassigned request").exists())

        report = self.execute(self.maryam, outcome)
        self.assertEqual(report["status"], "success", report)
        request = Request.objects.get(request_for="Admin unassigned request")
        self.assertEqual(request.owner, self.maryam)
        self.assertIsNone(request.department_id)
        self.assertFalse(self.maryam.groups.filter(pk=GLOBAL_SALES).exists())

    def test_admin_request_to_deal_infers_counterparty_department_and_owner(self):
        invalid = self.validate(self.maryam, plan(op(
            "bad", "create", "request", request_for="Invalid mixed links",
            first_name="Maryam", lead=self.lead.pk, contact=self.acme_contact.pk)))
        self.assertRejected(invalid, "either a contact or a lead, not both")

        outcome = self.validate(self.maryam, plan(op(
            "op1", "create", "request", create_deal=True,
            request_for="Admin approved opportunity", lead=self.lead.pk)))
        self.assertTrue(outcome.ok, outcome.message)
        self.assertFalse(Request.objects.filter(
            request_for="Admin approved opportunity").exists())
        self.assertFalse(Deal.objects.filter(
            name="Admin approved opportunity").exists())

        inferred = outcome.plan["operations"][0]["display"]["inferred"]
        self.assertEqual(inferred["Department"], "Global sales (from linked Lead)")
        self.assertEqual(
            inferred["Owner"],
            f"{self.manager.get_full_name() or self.manager.username} (from linked Lead)")

        first = self.execute(self.maryam, outcome)
        self.assertEqual(first["status"], "success", first)
        request = Request.objects.get(request_for="Admin approved opportunity")
        deal = Deal.objects.get(request=request)
        self.assertEqual(request.lead, self.lead)
        self.assertEqual(request.owner, self.manager)
        self.assertEqual(request.department_id, GLOBAL_SALES)
        self.assertEqual(request.deal, deal)
        self.assertEqual(deal.request, request)
        self.assertEqual(deal.lead, self.lead)
        self.assertEqual(deal.owner, self.manager)
        self.assertEqual(deal.department_id, GLOBAL_SALES)

        second = self.execute(self.maryam, outcome)
        self.assertEqual(second["status"], "already_processed")
        self.assertEqual(Request.objects.filter(
            request_for="Admin approved opportunity").count(), 1)
        self.assertEqual(Deal.objects.filter(request=request).count(), 1)


class PermissionTests(WorkflowTestCase):
    def test_user_without_add_permission_cannot_plan_creation(self):
        outcome = self.validate(self.chief, plan(op(
            "op1", "create", "company", full_name="Chief Co", email="c@c.example")))
        self.assertRejected(outcome, "permission denied")

    def test_records_of_other_departments_cannot_be_linked(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "contact", first_name="Eve", email="eve@x.example",
            company=self.local_company.pk)))
        self.assertRejected(outcome, "not accessible to you")
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "company", record_id=self.local_company.pk, phone="123")))
        self.assertRejected(outcome, "not accessible to you")

    def test_manager_cannot_update_another_managers_deal(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "deal", record_id=self.other_deal.pk, amount="1")))
        self.assertRejected(outcome, "permission denied")

    def test_permissions_are_rechecked_after_approval(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Late", email="late@x.example")))
        self.assertTrue(outcome.ok)
        self.manager.groups.remove(Group.objects.get(name="managers"))
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(Lead.objects.filter(email="late@x.example").exists())
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.FAILED)

    def test_deals_cannot_be_created_directly(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "deal", name="Direct deal")))
        self.assertRejected(outcome, "Deals cannot be created directly", "create_deal=true")

    def test_ordinary_read_only_user_cannot_write_even_with_approval(self):
        ordinary = USER_MODEL.objects.create_user(
            username="temporary_ordinary_workflow", password="test", is_staff=True)
        ordinary.groups.add(Group.objects.get(pk=GLOBAL_SALES))
        from django.contrib.auth.models import Permission
        view_codenames = [f"view_{model}" for model in (
            "lead", "company", "contact", "request", "deal", "payment")]
        ordinary.user_permissions.set(Permission.objects.filter(
            content_type__app_label="crm", codename__in=view_codenames))

        for record_type, fields in (
                ("lead", {"first_name": "Denied", "email": "denied@example.test"}),
                ("company", {"full_name": "Denied Co", "email": "denied@example.test"}),
                ("contact", {"first_name": "Denied", "email": "denied@example.test",
                              "company": self.acme.pk}),
                ("request", {"request_for": "Denied request", "first_name": "Denied"}),
                ("payment", {"deal": self.acme_deal.pk, "amount": "10"})):
            rejected = self.validate(ordinary, plan(op(
                "op1", "create", record_type, **fields)))
            self.assertRejected(rejected, "permission denied")

        deal_request = self.validate(ordinary, plan(op(
            "op1", "create", "request", create_deal=True,
            request_for="Denied deal", first_name="Denied", lead=self.lead.pk)))
        self.assertRejected(deal_request, "permission denied")

        payment = Payment.objects.create(deal=self.acme_deal, amount=1, currency_id=1)
        for record_type, record_id, fields in (
                ("lead", self.lead.pk, {"first_name": "Changed"}),
                ("company", self.acme.pk, {"phone": "123"}),
                ("contact", self.acme_contact.pk, {"first_name": "Changed"}),
                ("request", self.acme_request.pk, {"request_for": "Changed"}),
                ("deal", self.acme_deal.pk, {"amount": "12"}),
                ("payment", payment.pk, {"amount": "12"})):
            rejected = self.validate(ordinary, plan(op(
                "op1", "update", record_type, record_id=record_id, **fields)))
            self.assertRejected(rejected, "permission denied")

        delete = self.validate(ordinary, {
            "goal": "delete a lead", "operations": [{
                "op_id": "op1", "action": "delete", "record_type": "lead",
                "fields": [],
            }],
        })
        self.assertRejected(delete, "malformed")

        # Simulate a forged/previously approved action row: execution must
        # recheck add permission independently of the approval state.
        operation_id = str(uuid4())
        forged_plan = {
            "operation_id": operation_id, "goal": "forged approval",
            "operations": [{
                "op_id": "op1", "action": "create", "record_type": "lead",
                "record_id": None, "target_ref": None, "create_deal": False,
                "data": {"first_name": "Forged", "email": "forged@example.test"},
                "links": {}, "display": {"title": "Create Lead"},
            }],
        }
        ActionRequest.objects.create(
            user=ordinary, operation_id=operation_id, action_type="SALES_WORKFLOW",
            payload=forged_plan, status=ActionRequest.PENDING)
        result = execute_plan(CrmAccess(ordinary), forged_plan)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(Lead.objects.filter(email="forged@example.test").exists())

        operation_id = str(uuid4())
        forged_update = {
            "operation_id": operation_id, "goal": "forged update approval",
            "operations": [{
                "op_id": "op1", "action": "update", "record_type": "company",
                "record_id": self.acme.pk, "target_ref": None, "create_deal": False,
                "data": {"phone": "999-9999"}, "links": {},
                "display": {"title": f"Update Company #{self.acme.pk}"},
            }],
        }
        ActionRequest.objects.create(
            user=ordinary, operation_id=operation_id, action_type="SALES_WORKFLOW",
            payload=forged_update, status=ActionRequest.PENDING)
        result = execute_plan(CrmAccess(ordinary), forged_update)
        self.assertEqual(result["status"], "failed")
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.phone, "+1 555 0100")


class MultiRecordWorkflowTests(WorkflowTestCase):
    def prospect_plan(self):
        step_date = (timezone.localdate() + timedelta(days=7)).isoformat()
        return plan(
            op("op1", "create", "company", full_name="ABC Technologies",
               email="info@abc.example", country=self.country.name),
            op("op2", "create", "contact", first_name="Bob", last_name="Builder",
               email="bob@abc.example", company="$op1"),
            op("op3", "create", "request", create_deal=True,
               request_for="ABC Technologies CRM rollout", contact="$op2"),
            op("op4", "update", "deal", record_id="$op3.deal", amount="25,000",
               next_step="Send proposal", next_step_date=step_date, probability="40"),
            goal="Add the ABC Technologies prospect and register their opportunity",
        )

    def test_prospect_to_opportunity_workflow(self):
        outcome = self.validate(self.manager, self.prospect_plan())
        self.assertTrue(outcome.ok, outcome.message)
        card = approval_payload(outcome.plan)
        titles = [o["title"] for o in card["operations"]]
        self.assertEqual(titles, ["Create Company", "Create Contact Person",
                                  "Create Request and its Deal", "Update Deal ($op3.deal)"])
        request_op = card["operations"][2]
        self.assertIn("New deal", request_op["inferred"])
        self.assertEqual(request_op["related"]["Contact"], "the new contact from op2")
        self.assertTrue(any("moves it to" in w for w in card["operations"][3]["warnings"]))
        self.assertFalse(Company.objects.filter(full_name="ABC Technologies").exists())

        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)

        company = Company.objects.get(full_name="ABC Technologies")
        contact = Contact.objects.get(email="bob@abc.example")
        request = Request.objects.get(request_for="ABC Technologies CRM rollout")
        deal = request.deal
        self.assertEqual(contact.company, company)
        self.assertEqual((company.owner, company.department_id), (self.manager, GLOBAL_SALES))
        self.assertEqual((contact.owner, contact.department_id), (self.manager, GLOBAL_SALES))
        self.assertFalse(request.pending)
        self.assertEqual(request.company, company)
        self.assertEqual(request.first_name, "Bob")
        self.assertEqual(deal.request, request)
        self.assertEqual((deal.contact, deal.company), (contact, company))
        self.assertEqual(deal.amount, Decimal("25000"))
        self.assertEqual(deal.probability, 40)
        self.assertEqual(deal.next_step, "Send proposal")
        self.assertIn("Send proposal", deal.workflow)
        self.assertEqual(deal.owner, self.manager)
        self.assertEqual(deal.currency_id, 1)          # department default currency
        self.assertEqual(deal.stage, Stage.objects.get(
            department_id=GLOBAL_SALES, second_default=True))
        self.assertEqual(report["results"][2]["deal"]["record_id"], deal.pk)

    def test_failure_rolls_back_every_operation(self):
        outcome = self.validate(self.manager, self.prospect_plan())
        contact_admin = CrmAccess.admin_for(Contact)
        with patch.object(type(contact_admin), "save_model",
                          side_effect=RuntimeError("disk full")):
            report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_op"], "op2")
        self.assertEqual([r["status"] for r in report["results"]],
                         ["not saved", "failed", "not saved", "not saved"])
        self.assertFalse(Company.objects.filter(full_name="ABC Technologies").exists())
        self.assertFalse(Request.objects.filter(request_for__startswith="ABC").exists())
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.FAILED)
        self.assertIn("rolled back", format_report(report))

    def test_record_deleted_while_waiting_fails_safely(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "contact", first_name="Zed", email="zed@acme.example",
               company=self.acme.pk)))
        Company.objects.filter(pk=self.acme.pk).delete()
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "failed")
        self.assertIn("no longer accessible", report["error"])

    def test_plan_cannot_be_executed_twice(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Once", email="once@x.example")))
        first = self.execute(self.manager, outcome)
        second = self.execute(self.manager, outcome)
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "already_processed")
        self.assertEqual(Lead.objects.filter(email="once@x.example").count(), 1)

    def test_other_user_cannot_execute_someone_elses_plan(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Mine", email="mine@x.example")))
        report = execute_plan(CrmAccess(self.operator), outcome.plan)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(Lead.objects.filter(email="mine@x.example").exists())

    def test_request_rules_are_enforced(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "request", request_for="Both", contact=self.acme_contact.pk,
            lead=self.lead.pk)))
        self.assertRejected(outcome, "contact or a lead, not both")

        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "request", create_deal=True, request_for="No person",
            first_name="Nobody", country=self.country.name)))
        self.assertRejected(outcome, "link the request to a contact or a lead")

        other = Company.objects.create(full_name="Other Co", email="o@o.example",
                                       owner=self.manager, department_id=GLOBAL_SALES)
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "request", request_for="Mismatch",
            contact=self.acme_contact.pk, company=other.pk)))
        self.assertRejected(outcome, "does not belong to the linked company")

    def test_existing_pending_request_is_processed_into_a_deal(self):
        pending = Request.objects.create(
            request_for="Leo needs a quote", first_name="Leo", email="leo@prospect.example",
            lead=self.lead, owner=self.manager, department_id=GLOBAL_SALES)
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "request", record_id=pending.pk, create_deal=True)))
        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.plan["operations"][0]["display"]["title"],
                         f"Process Request #{pending.pk} into a Deal")
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        pending.refresh_from_db()
        self.assertFalse(pending.pending)
        self.assertEqual(pending.deal.lead, self.lead)
        self.assertEqual(pending.deal.name, "Leo needs a quote")

        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "request", record_id=pending.pk, create_deal=True)))
        self.assertRejected(outcome, "already has deal")

    def test_update_shows_before_and_after_values(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "company", record_id=self.acme.pk, phone="+1 555 0199")))
        self.assertTrue(outcome.ok, outcome.message)
        display = outcome.plan["operations"][0]["display"]
        self.assertEqual(display["changes"]["Phone"],
                         {"from": "+1 555 0100", "to": "+1 555 0199"})
        self.assertEqual(display["existing_record"], f"#{self.acme.pk} Acme Corp")
        self.execute(self.manager, outcome)
        self.acme.refresh_from_db()
        self.assertEqual(self.acme.phone, "+1 555 0199")


class ReviewRegressionTests(WorkflowTestCase):
    """Issues found in the independent code review."""

    def test_deal_reference_cannot_be_used_as_another_record_type(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", create_deal=True, request_for="R",
               contact=self.acme_contact.pk),
            op("op2", "create", "contact", first_name="Mal", email="mal@x.example",
               company="$op1.deal")))
        self.assertRejected(outcome, "company: $op1.deal does not produce a Company")

    def test_second_approval_of_running_or_finished_plan_never_marks_it_failed(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Dbl", email="dbl@x.example")))
        # Another approval of the same plan claimed it first.
        ActionRequest.objects.update(status=ActionRequest.APPROVED)
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "already_processed")
        self.assertEqual(ActionRequest.objects.get().status, ActionRequest.APPROVED)
        self.assertFalse(Lead.objects.filter(email="dbl@x.example").exists())

    def test_success_is_recorded_with_the_writes(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "lead", first_name="Rec", email="rec@x.example")))
        self.execute(self.manager, outcome)
        row = ActionRequest.objects.get()
        self.assertEqual(row.status, ActionRequest.EXECUTED)
        self.assertEqual(row.result["status"], "success")

    def test_updating_request_contact_must_match_its_existing_company(self):
        other = Company.objects.create(full_name="Other Co", email="o@o.example",
                                       owner=self.manager, department_id=GLOBAL_SALES)
        stranger = Contact.objects.create(first_name="Sue", email="sue@o.example",
                                          company=other, owner=self.manager,
                                          department_id=GLOBAL_SALES)
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "request", record_id=self.acme_request.pk,
            contact=stranger.pk)))
        self.assertRejected(outcome, "does not belong to the linked company")

    def test_creating_deals_requires_change_deal_permission(self):
        from django.contrib.auth.models import Permission
        clerk = USER_MODEL.objects.create_user(username="clerk", password="x", is_staff=True)
        intake = Group.objects.create(name="request intake")
        intake.permissions.set(Permission.objects.filter(
            content_type__app_label="crm",
            codename__in=["add_request", "view_request", "view_lead"]))
        clerk.groups.add(intake, Group.objects.get(pk=GLOBAL_SALES))
        outcome = self.validate(clerk, plan(op(
            "op1", "create", "request", create_deal=True, request_for="Quote",
            lead=self.lead.pk)))
        self.assertRejected(outcome, "not allowed to create deals")
        outcome = self.validate(clerk, plan(op(
            "op1", "create", "request", request_for="Quote", lead=self.lead.pk)))
        self.assertTrue(outcome.ok, outcome.message)   # plain intake is fine

    def test_ambiguous_number_formats_are_rejected(self):
        for value, accepted in (("25,000", True), ("1,250,000.50", True),
                                ("1,5", False), ("1.234,56", False)):
            outcome = self.validate(self.manager, plan(op(
                "op1", "update", "deal", record_id=self.acme_deal.pk, amount=value)))
            self.assertEqual(outcome.ok, accepted, (value, outcome.message))

    def test_unsuccessful_deal_closure_side_effect_is_disclosed_before_approval(self):
        payment = Payment.objects.create(
            deal=self.acme_deal, amount=25, currency_id=1,
            status=Payment.GUARANTEED)
        reason = ClosingReason.objects.create(
            name="Audit unsuccessful reason", department_id=GLOBAL_SALES,
            index_number=99, success_reason=False)
        outcome = self.validate(self.manager, plan(op(
            "op1", "update", "deal", record_id=self.acme_deal.pk,
            closing_reason=reason.name)))
        self.assertTrue(outcome.ok, outcome.message)
        payload = approval_payload(outcome.plan)
        self.assertTrue(any(
            "deletes its payments that were not received" in warning
            for warning in payload["operations"][0]["warnings"]))
        self.assertTrue(Payment.objects.filter(pk=payment.pk).exists())


class PaymentWorkflowTests(WorkflowTestCase):
    def test_manager_adds_payment_to_own_deal(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "payment", deal=self.acme_deal.pk, amount="300",
            status="guaranteed", invoice_number="INV-1")))
        self.assertTrue(outcome.ok, outcome.message)
        display = outcome.plan["operations"][0]["display"]
        self.assertIn("Currency", display["inferred"])
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        payment = Payment.objects.get(invoice_number="INV-1")
        self.assertEqual((payment.deal, payment.amount, payment.status, payment.currency_id),
                         (self.acme_deal, Decimal("300"), Payment.GUARANTEED, 1))

    def test_manager_cannot_add_payment_to_another_managers_deal(self):
        outcome = self.validate(self.manager, plan(op(
            "op1", "create", "payment", deal=self.other_deal.pk, amount="10")))
        self.assertRejected(outcome, "not allowed to manage payments")

    def test_operator_has_no_payment_permission(self):
        outcome = self.validate(self.operator, plan(op(
            "op1", "create", "payment", deal=self.acme_deal.pk, amount="10")))
        self.assertRejected(outcome, "permission denied")

    def test_payment_on_deal_created_in_same_plan(self):
        outcome = self.validate(self.manager, plan(
            op("op1", "create", "request", create_deal=True,
               request_for="Prepaid order", lead=self.lead.pk),
            op("op2", "create", "payment", deal="$op1.deal", amount="50",
               status="received")))
        self.assertTrue(outcome.ok, outcome.message)
        report = self.execute(self.manager, outcome)
        self.assertEqual(report["status"], "success", report)
        deal = Request.objects.get(request_for="Prepaid order").deal
        self.assertEqual(Payment.objects.get(deal=deal).amount, Decimal("50"))
