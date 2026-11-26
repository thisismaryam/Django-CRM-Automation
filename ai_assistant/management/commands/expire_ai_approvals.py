from django.core.management.base import BaseCommand

from ai_assistant.workflow import approval_ttl, expire_pending


class Command(BaseCommand):
    help = ("Close AI assistant approvals that have waited longer than "
            "AI_ASSISTANT_APPROVAL_TTL_MINUTES (default 60). Only ActionRequest "
            "rows are updated; CRM records are never touched.")

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true",
                            help="Only report how many approvals would be closed.")

    def handle(self, *args, **options):
        count = expire_pending(dry_run=options["dry_run"])
        verb = "Would expire" if options["dry_run"] else "Expired"
        self.stdout.write(f"{verb} {count} pending approval(s) older than {approval_ttl()}.")
