from django.contrib import admin

# ActionRequest (the audit trail of AI-proposed plans) is intentionally not
# registered: the CRM pins its admin main menu (tests/test_base.py) and the
# assistant should not change existing CRM pages.
