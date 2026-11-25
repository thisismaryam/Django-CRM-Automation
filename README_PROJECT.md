# Project README: Django-CRM and the Sales AI Assistant

This guide explains how the Django-CRM codebase is organized, how to run it,
and how the Sales Department AI assistant added to this checkout works. It is
intended for developers and operators who are new to the project.

For the upstream CRM product overview, see the [main README](README.md).
For full CRM installation and business configuration instructions, see the
[installation and configuration guide](docs/installation_and_configuration_guide.md)
and the [user guide](docs/django-crm_user_guide.md).

## What this project does

Django-CRM is a self-hosted customer relationship management system built with
Django. Its primary interface is Django Admin, extended with CRM-specific
screens and workflows. The project includes customer and sales records, task
management, analytics, email and mailing tools, shared chat, reminders, and
other business utilities.

This checkout also includes a **Sales Department AI assistant**. Sales staff
describe what they need in plain language. The assistant:

- answers questions about live CRM data (leads, companies, contacts,
  requests, deals, payments, pipeline summaries), limited to what the user is
  allowed to see;
- answers questions about the CRM's structure using a schema search index;
- plans multi-record work, such as "add this prospect from ABC Technologies
  and register the opportunity", as one workflow. It shows the whole plan on
  an approval card and, only after approval, saves it in one database
  transaction.

The assistant works through the CRM's own permission checks, admin forms and
save logic. It does not replace the admin interface or bypass CRM business
rules.

## Main components

| Component | Purpose |
| --- | --- |
| `crm/` | Core CRM records and business workflows, including leads, companies, contacts, deals, requests, products, and payments. |
| `tasks/` | Tasks, projects, memos, and related collaboration features. |
| `analytics/` | Reporting and analytics views. |
| `massmail/` | Email accounts, message handling, and campaign/mailing features. |
| `chat/` | Shared chat functionality used by CRM and task objects. |
| `common/` | Shared models, views, utilities, templates, static assets, and Django Admin customizations. |
| `help/`, `quality/`, `settings/`, `voip/` | Help content, quality-related features, site settings, and VoIP integrations. |
| `ai_assistant/` | Sales AI agent: LangGraph workflow, permission-scoped CRM tools, workflow planning and execution, schema search, document extraction, chat UI. |
| `webcrm/` | Django project settings, root URL configuration, and application entry points. |
| `templates/`, `static/`, `media/` | Shared templates, source static assets, and user-uploaded/runtime media. |
| `tests/` | Tests for the CRM applications and workflows. |
| `docs/` | Installation, user, feature, and developer documentation. |

### Django request flow

1. `manage.py` loads `webcrm.settings` and starts Django management commands.
2. `webcrm/urls.py` routes requests to the CRM applications, Django Admin, and
   the AI assistant.
3. Django applications provide models, views, admin classes, templates, and
   migrations for their features.
4. Django ORM reads and writes the configured database. The default local
   configuration uses SQLite.

The CRM and admin URLs use configurable prefixes from `webcrm/settings.py`
(`SECRET_CRM_PREFIX` and `SECRET_ADMIN_PREFIX`) and Django's language URL
prefixes. For example, with the checked-in development settings, the admin
login is under `/en/456-admin/`. Do not assume these example prefixes will
remain the same if settings are changed.

## Sales AI assistant

### CRM business rules it follows

These rules come from the CRM's own code and are enforced, not just described
in the prompt:

- **Deals are created only from Requests.** No group has `crm.add_deal` and
  `DealAdmin.has_add_permission` is `False`. The assistant creates a Deal the
  way the CRM UI does: by processing a Request (the "Create deal" button,
  `RequestAdmin.save_related` with `_create-deal`).
- A Request links a Contact **or** a Lead, never both, and a Lead is never
  combined with a Company; the contact must belong to the linked company.
- A Contact always belongs to a Company and takes its owner and department
  from it.
- Payments belong to a Deal. Managers may only manage payments of deals they
  own or co-own.
- Departments that work globally require a country before a deal is created;
  departments with products require products.

### Access and department scope

The chat page and API require an active staff account. Passing that staff-only
gate does not grant access to CRM records: read tools use the CRM admin's
`has_view_permission` and `get_queryset`, while workflow creates and updates
use `has_add_permission` / `has_change_permission`, CRM forms, and save logic.
Updates and linked records are checked again when an approved plan executes.

Users belong to departments through Django group membership. A Department is
also a Group; role groups such as `managers` grant permissions, while a
department group such as `Global sales` identifies a user's department.
Department groups do not themselves grant CRM model permissions; assign a
separate role group or explicit permissions as appropriate.
`co-workers` is added by the user-creation signal and must already exist when
creating users. Assign ordinary CRM users to a department group as well as an
appropriate permission-bearing role. The CRM admin queryset applies a
department filter when a department is selected; an ordinary user with CRM
permissions but no department group may otherwise receive an unfiltered
queryset. Superusers, chiefs, accountants, and superoperators have special
department-selection behavior in the CRM.

Schema search uses the checked-in model catalog and FAISS index. It returns
model structure and business rules, not live records, and is not department
filtered. Live record searches are permission- and queryset-scoped. Some
reference choices returned by `list_crm_options` (countries, currencies, and
payment statuses) are global; department-specific choices are filtered when
the user has a department.

### Components and files

| Path | Responsibility |
| --- | --- |
| `ai_assistant/agent.py` | LangGraph graph: `agent` (LLM reasoning/routing) → `read_tools` or `plan_review` → `approval` (interrupt) → `execute` → `report`. System prompt and SQLite checkpointer. |
| `ai_assistant/access.py` | Permission and department scoping. Builds a server-side request for the authenticated user and delegates to the CRM's registered ModelAdmins (`get_queryset`, `has_view/add/change_permission`, forms, `save_model`/`save_related`). |
| `ai_assistant/read_tools.py` | Read tools: `search_leads`, `search_companies`, `search_contacts`, `search_requests`, `search_deals`, `search_payments`, `get_crm_record`, `sales_summary`, `list_crm_options`, `list_departments`. |
| `ai_assistant/workflow.py` | Workflow plan contract, validation (permissions, references, business rules, CRM admin form validation, duplicates, inferred values), approval payload, atomic and idempotent execution, result report. |
| `ai_assistant/tools.py` | Tool registry, schema RAG tool (`search_database_schema`) with Sales terminology and process notes, and the single write entry point `propose_sales_workflow`. |
| `ai_assistant/llm.py` | Gemini primary → Gemini fallback → Groq, built lazily, configured from environment variables. |
| `ai_assistant/context.py` | Keeps recent complete user turns within the history limits. |
| `ai_assistant/api_views.py` | Chat endpoint (text and document upload) and approval endpoint. |
| `ai_assistant/models.py` | `ActionRequest`: one row per proposed plan (pending → approved/rejected → executed/failed) for auditing and duplicate-execution protection. |
| `ai_assistant/documents/extract.py` | Text extraction from PDF, DOCX and TXT uploads. |
| `ai_assistant/rag/` | Schema extraction from the Django models, embeddings and FAISS search. |
| `ai_assistant/templates/ai_assistant/chat.html` | Chat page inside the CRM admin layout, approval cards and result display. |
| `ai_assistant/tests/` | Test suite (see below). |

### How a request flows

1. The chat page posts the message (and optionally a PDF/DOCX/TXT up to
   10 MB) to `POST /api/ai/chat/`. Only active staff users may call it.
   Document text (capped at 15,000 characters) is appended to the message and
   treated as data, not instructions.
2. The `agent` node calls the LLM with the tools bound. It routes:
   - structure questions to `search_database_schema` (structure only);
   - data questions to the read tools, which always query through the CRM
     admin's `get_queryset` after a `view` permission check, so department
     scoping matches the CRM pages;
   - change requests to `propose_sales_workflow`, one call with every
     operation the goal needs.
3. `plan_review` validates the plan: permissions (`add`/`change` and
   record-level rules), visibility of every referenced record, references
   between operations (`$op1`, `$op3.deal`), the business rules above, the
   CRM's own admin form validation, and duplicates (in the user's scope and
   inside the plan). It also works out values the CRM will fill in. An invalid
   plan goes back to the agent with the problems listed (`NEEDS_DETAILS` or
   `VALIDATION_FAILED`). No CRM table is written.
4. A valid plan gets a unique `operation_id`, is stored in graph state and in
   `ActionRequest`, and the graph pauses with `interrupt()`. The chat shows
   one approval card: every operation with its values, before → after
   changes, related records (existing ids or "new record from op1"), values
   filled automatically, and warnings.
5. `POST /api/ai/approve/` with `thread_id`, `decision` and `operation_id`
   resumes the graph. The operation id must match the plan that is waiting.
   Threads are namespaced by the authenticated user, so nobody can resume
   another user's plan. If the user sends a new chat message instead, the
   waiting plan is rejected first.
6. `execute` reloads the user, re-checks permissions and every referenced
   record, and runs all operations in one database transaction through the
   CRM admin forms and `save_model`/`save_related`. Any failure rolls back
   the whole plan. The `ActionRequest` row is claimed inside the same
   transaction, so a plan cannot run twice. Each saved record is also written
   to the CRM's admin history.
7. `report` returns an exact summary of what was saved (with links) or why
   nothing was saved. No LLM call is needed for it.

### Supported operations

| Record | Create | Update |
| --- | --- | --- |
| Lead | yes (inherits company details when linked to a company) | yes |
| Company | yes | yes |
| Contact | yes (needs a company) | yes |
| Request | yes, optionally processed into a Deal | yes, including processing an existing pending request into a Deal |
| Deal | no, created from a Request | yes (amount, stage, next step, probability, closing, etc.) |
| Payment | yes | yes |

Deletes and lead conversion stay in the regular CRM interface.

### Conversation context and persistence

LangGraph's SQLite saver persists graph state, including paused approvals, in
`checkpoints.sqlite` next to `manage.py` (override with
`AI_ASSISTANT_CHECKPOINT_DB`). A paused plan survives server restarts. Each
model call keeps at most the three most recent user turns, dropping older
turns when serialized history exceeds 10,000 characters. Treat the checkpoint
file as application data: protect it and include it in backups and retention
planning.

To print the exact messages sent to the LLM, set `AI_ASSISTANT_LOG_PROMPTS=1`
before starting Django. Prompt traces contain CRM and user data; use this only
in a controlled development environment.

### Refreshing the schema index

The schema catalog and FAISS index are generated from the CRM models. A test
fails if a model changes and the catalog was not regenerated. Rebuild both
from the repository root:

```powershell
python ai_assistant\rag\extract_schema.py
python ai_assistant\rag\build_index.py
```

The index builder uses the `BAAI/bge-small-en-v1.5` embedding model and
downloads it the first time it runs.

## Set up a local development environment

The upstream installation guide contains the full CRM setup instructions.
For a typical local SQLite development setup:

1. Install Python 3.12 or newer. The pinned Django 6.0 dependency requires
   this floor even though `setup.cfg` currently advertises an older
   `python_requires` value.
2. Create and activate an environment, for example with conda:

   ```powershell
   conda create -n django-crm python=3.12
   conda activate django-crm
   ```

3. Install the CRM and assistant dependencies:

   ```powershell
   python -m pip install -r requirements.txt -r requirements-ai.txt
   ```

4. To use the AI assistant, create a `.env` file in the project root (it is
   ignored by Git):

   ```
   GEMINI_API_KEY=...
   # Alternatively, use GOOGLE_API_KEY instead of GEMINI_API_KEY.
   GROQ_API_KEY=...
   # optional
   GEMINI_MODEL=gemini-3.5-flash
   GEMINI_FALLBACK_MODEL=gemini-3.8-flash
   GROQ_MODEL=openai/gpt-oss-20b
   AI_ASSISTANT_MAX_TOKENS=2048
   ```

   The default model order is Gemini 3.5 Flash, Gemini 3.8 Flash, then Groq.
   `GEMINI_MODEL` and `GEMINI_FALLBACK_MODEL` can override the Gemini defaults;
   `GROQ_MODEL` can override the Groq model. Set both provider keys to enable
   the full fallback chain. The current code constructs Gemini before Groq, so
   a Groq key alone is not sufficient. Django can start without LLM keys, but
   AI requests cannot. Restart Django after changing `.env`; the variables are
   loaded when the assistant module is imported.

5. For a new, empty local database, apply migrations and load the starter data:

   ```powershell
   python manage.py migrate
   python manage.py setupdata
   ```

   `setupdata` also runs migrations, loads the project's fixtures, and creates
   three demo users: `IamSUPER`, `IamGlobalSALES`, and `IamLocalSALES`. It prints
   generated passwords to the console. The sales accounts are assigned the
   `managers`, `co-workers`, and matching department groups.

   **Do not treat `setupdata` as a repeatable or non-destructive command.** Its
   fixtures use fixed primary keys and can replace rows with matching keys
   (including the local Site record); it also creates users with fixed
   usernames, so a second run can fail on duplicate users. Use it only for a
   fresh database, or review and back up a populated database before loading
   fixtures. It is not needed just to run Django.

   `setupdata` already creates a superuser. Run `python manage.py createsuperuser`
   only if you want an additional account, using a distinct username. If you
   initialize users without `setupdata`, ensure the `co-workers` group exists
   first; the user-creation signal expects it.

6. Ensure `media/locks` exists (the background email importer uses it), then
   start the development server:

   ```powershell
   if (-not (Test-Path media\locks)) {
       New-Item -ItemType Directory -Path media\locks -Force | Out-Null
   }
   python manage.py runserver
   ```

The default settings use SQLite, `DEBUG = True`, and local host names. They
are for local development, not public deployment. Configure secrets,
`ALLOWED_HOSTS`, database, email, HTTPS, static files, and other security
settings before deploying.

### Open the CRM and assistant

With the checked-in settings and English locale:

| Page or endpoint | URL |
| --- | --- |
| Django Admin login | `http://127.0.0.1:8000/en/456-admin/` |
| CRM | `http://127.0.0.1:8000/en/123/` |
| AI assistant page | `http://127.0.0.1:8000/en/api/ai/chat-ui/` |
| AI chat API | `POST http://127.0.0.1:8000/api/ai/chat/` |
| AI approval API | `POST http://127.0.0.1:8000/api/ai/approve/` |

The chat page and both API endpoints require an authenticated, active staff
session. The APIs use Django REST Framework session authentication, so unsafe
requests must include a valid CSRF token. Review production-specific
authentication, rate limiting, and operational requirements before exposing
the application publicly.

## Tests and checks

Run Django's configuration checks and the assistant tests from the repository
root:

```powershell
python manage.py check
python manage.py test ai_assistant
```

The assistant tests use the CRM's real group permissions and departments and
replace only the LLM with a scripted stand-in. They cover:

- reads and department scoping;
- permission enforcement;
- every create/update operation;
- duplicates and missing information;
- relationship validation;
- approval, rejection and stale approvals;
- resuming after a restart;
- multi-record transactions and rollback;
- duplicate-execution protection;
- cross-user isolation;
- document uploads;
- graph routing;
- RAG catalog freshness;
- provider fallback.

The project also has a broader suite under `tests/`; run it with
`python manage.py test tests` when making wider changes.

## Further reading

- [Main project README](README.md) — product overview and feature summary.
- [Installation and configuration](docs/installation_and_configuration_guide.md) —
  CRM setup, email, database, users, and integrations.
- [User guide](docs/django-crm_user_guide.md) — using the CRM.
- [System overview](docs/crm_system_overview.md) — CRM concepts and workflows.
- [Changelog](CHANGELOG.md) — upstream project changes.
