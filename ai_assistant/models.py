from django.conf import settings
from django.db import models


class ActionRequest(models.Model):
    """A CRM change plan proposed by the AI assistant.

    One row per proposed workflow. It is created when a plan passes
    validation (before approval) and moves to APPROVED inside the same
    database transaction that writes the CRM records, so a plan can never be
    executed twice. EXECUTED/FAILED rows keep the outcome for auditing.
    """
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    EXECUTED = 'EXECUTED'
    FAILED = 'FAILED'
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (APPROVED, 'Approved'),
        (REJECTED, 'Rejected'),
        (EXECUTED, 'Executed'),
        (FAILED, 'Failed'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    action_type = models.CharField(max_length=50)  # e.g. 'SALES_WORKFLOW'
    payload = models.JSONField()  # the validated plan shown for approval
    status = models.CharField(
        max_length=10, choices=STATUS_CHOICES, default=PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    operation_id = models.CharField(
        max_length=36, unique=True, null=True, blank=True)
    thread_id = models.CharField(max_length=200, blank=True, default='')
    result = models.JSONField(null=True, blank=True)
    decided_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.action_type} - {self.status}"
