# Project README: Django-CRM and the Sales AI Assistant

This guide explains how the Django-CRM codebase is organized, how to run it, and how the Sales Department AI assistant works. It is intended for developers and operators who are new to the project.

For the upstream CRM product overview, see the [main README](README.md).
For full CRM installation and business configuration instructions, see the [installation and configuration guide](docs/installation_and_configuration_guide.md) and the [user guide](docs/django-crm_user_guide.md).

## What this project does

Django-CRM is a self-hosted customer relationship management system built with Django. Its primary interface is Django Admin, extended with CRM-specific screens and workflows. The project includes customer and sales records, task management, analytics, email and mailing tools, shared chat, reminders, and other business utilities.

This checkout also includes a **Sales Department AI assistant**. Sales staff describe what they need in plain language. The assistant:

- answers questions about live CRM data (leads, companies, contacts, requests, deals, payments, pipeline summaries), limited to what the user is allowed to see;
- answers questions about the CRM's structure using a schema search index;
- plans multi-record work, such as "add this prospect from ABC Technologies and register the opportunity", as one workflow. It shows the whole plan on an approval card and, only after approval, saves it in one database transaction.

The assistant works through the CRM's own permission checks, admin forms and save logic. It does not replace the admin interface or bypass CRM business rules.

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
2. `webcrm/urls.py` routes requests to the CRM applications, Django Admin, and the AI assistant.
3. Django applications provide models, views, admin classes, templates, and migrations for their features.
4. Django ORM reads and writes the configured database. The default local configuration uses SQLite.

The CRM and admin URLs use configurable prefixes from `webcrm/settings.py` (`SECRET_CRM_PREFIX` and `SECRET_ADMIN_PREFIX`) and Django's language URL prefixes. For example, with the checked-in development settings, the admin login is under `/en/456-admin/`. Do not assume these example prefixes will remain the same if settings are changed.

## Sales AI assistant

### CRM business rules it follows

These rules come from the CRM's own code and are enforced, not just described in the prompt:

- **Deals are created only from Requests.** No group has `crm.add_deal` and `DealAdmin.has_add_permission` is `False`. The assistant creates a Deal the way the CRM UI does: by processing a Request (the "Create deal" button, `RequestAdmin.save_related` with `_create-deal`).
- A Request links a Contact **or** a Lead, never both, and a Lead is never combined with a Company; the contact must belong to the linked company.
- A Contact always belongs to a Company and takes its owner and department from it.
- Payments belong to a Deal. Managers may only manage payments of deals they own or co-own.
- Departments that work globally require a country before a deal is created; departments with products require products.

### Access and department scope

The chat page and API require an active staff account. Passing that staff-only gate does not grant access to CRM records: read tools use the CRM admin's `has_view_permission` and `get_queryset`, while workflow creates and updates use `has_add_permission` / `has_change_permission`, CRM forms, and save logic. Updates and linked records are checked again when an approved plan executes.

An ordinary user may use schema RAG without record permissions. Live record reads require the applicable CRM `view` permission and remain subject to the CRM admin queryset. Creating or changing records requires the corresponding CRM add/change permission and validation. An approval card is not a permission grant: the account and each record are checked again when an approved plan is executed. Delete is not an assistant operation.

Users belong to departments through Django group membership. A Department is also a Group; role groups such as `managers` grant permissions, while a department group such as `Global sales` identifies a user's department. Department groups do not themselves grant CRM model permissions; assign a separate role group or explicit permissions as appropriate. `co-workers` is added by the user-creation signal and must already exist when creating users. Assign ordinary CRM users to a department group as well as an appropriate permission-bearing role. The CRM admin queryset applies a department filter when a department is selected; an ordinary user with CRM permissions but no department group may otherwise receive an unfiltered queryset. Superusers, chiefs, accountants, and superoperators have special department-selection behavior in the CRM.

Schema search uses the checked-in model catalog and FAISS index. It returns model structure and business rules, not live records, and is not department filtered. Live record searches are permission- and queryset-scoped. Some reference choices returned by `list_crm_options` (countries, currencies, and payment statuses) are global; department-specific choices are filtered when the user has a department.

The assistant treats live CRM questions as database questions: it must use an authorized read tool and report an empty result as not found. Schema answers come from the schema catalog and must not be presented as live-record results. Standalone greetings are answered locally with a short capability description and do not call an LLM provider. The greeting contains no example company or sample CRM record. Do not use illustrative company names, IDs, emails, amounts, or relationships as though they were real CRM data.

### Components and files

| Path | Responsibility |
| --- | --- |
| `ai_assistant/agent.py` | LangGraph graph: `agent` (LLM reasoning/routing) → `read_tools` or `plan_review` → `approval` (interrupt) → `execute` → `report`. Includes deterministic standalone greeting handling, system prompt, provider-failure message, and SQLite checkpointer. |
| `ai_assistant/access.py` | Permission and department scoping. Builds a server-side request for the authenticated user and delegates to the CRM's registered ModelAdmins (`get_queryset`, `has_view/add/change_permission`, forms, `save_model`/`save_related`). |
| `ai_assistant/read_tools.py` | Read tools: `search_leads`, `search_companies`, `search_contacts`, `search_requests`, `search_deals`, `search_payments`, `get_crm_record`, `sales_summary`, `list_crm_options`, `list_departments`. |
| `ai_assistant/workflow.py` | Workflow plan contract, validation (permissions, references, business rules, CRM admin form validation, duplicates, inferred values), approval payload, atomic and idempotent execution, result report. |
| `ai_assistant/tools.py` | Tool registry, schema RAG tool (`search_database_schema`) with Sales terminology and process notes, and the single write entry point `propose_sales_workflow`. |
| `ai_assistant/llm.py` | Centralized provider selection with failover: Groq (primary) → Gemini → Cloudflare Workers AI → OpenRouter. Built lazily, configured from environment variables. Handles provider errors, rate limits, and timeouts. |
| `ai_assistant/router.py` | Lightweight local intent router. Classifies incoming messages into `greeting`, `schema`, `read`, or `write` intents using fast heuristic pattern matching (no LLM call needed). |
| `ai_assistant/context.py` | Keeps recent complete user turns within the history limits. |
| `ai_assistant/api_views.py` | Chat endpoint (text and document upload) and approval endpoint. |
| `ai_assistant/models.py` | `ActionRequest`: one row per proposed plan (pending → approved/rejected → executed/failed) for auditing and duplicate-execution protection. |
| `ai_assistant/documents/extract.py` | Text extraction from PDF, DOCX and TXT uploads. |
| `ai_assistant/rag/` | Schema extraction from the Django models, embeddings and FAISS search. |
| `ai_assistant/templates/ai_assistant/chat.html` | Chat page inside the CRM admin layout, approval cards and result display. |
| `ai_assistant/tests/` | Test suite (see below). |

### How a request flows

1. The chat page posts the message (and optionally a PDF/DOCX/TXT up to 10 MB) to `POST /api/ai/chat/`. Only active staff users may call it. Document text (capped at 15,000 characters) is appended to the message and treated as data, not instructions.
2. The local intent router (`router.py`) classifies the request into one of four intents:
   - **greeting** — standalone hello, "what can you do", "help"
   - **schema** — questions about CRM structure, models, fields, relationships
   - **read** — questions about live CRM records, pipeline, sales figures
   - **write** — requests to create, update, or modify CRM data
   This classification is fast, uses no LLM tokens, and determines which tools and prompt sections are bound.
3. The `agent` node calls the LLM with the tools bound for that intent. It routes:
   - structure questions to `search_database_schema` (structure only);
   - data questions to the read tools, which always query through the CRM admin's `get_queryset` after a `view` permission check, so department scoping matches the CRM pages;
   - change requests to `propose_sales_workflow`, one call with every operation the goal needs.
4. `plan_review` validates the plan: permissions (`add`/`change` and record-level rules), visibility of every referenced record, references between operations (`$op1`, `$op3.deal`), the business rules above, the CRM's own admin form validation, and duplicates (in the user's scope and inside the plan). It also works out values the CRM will fill in. An invalid plan goes back to the agent with the problems listed (`NEEDS_DETAILS` or `VALIDATION_FAILED`). No CRM table is written.
5. A valid plan gets a unique `operation_id`, is stored in graph state and in `ActionRequest`, and the graph pauses with `interrupt()`. The chat shows one approval card: every operation with its values, before → after changes, related records (existing ids or "new record from op1"), values filled automatically, and warnings.
6. `POST /api/ai/approve/` with `thread_id`, `decision` and `operation_id` resumes the graph. The operation id must match the plan that is waiting. Threads are namespaced by the authenticated user, so nobody can resume another user's plan. If the user sends a new chat message instead, the waiting plan is rejected first.
7. `execute` reloads the user, re-checks permissions and every referenced record, and runs all operations in one database transaction through the CRM admin forms and `save_model`/`save_related`. Any failure rolls back the whole plan. The `ActionRequest` row is claimed inside the same transaction, so a plan cannot run twice. Each saved record is also written to the CRM's admin history.
8. `report` returns an exact summary of what was saved (with links) or why nothing was saved. No LLM call is needed for it.

### Dynamic prompt and tool binding

Based on the detected intent, the assistant uses a minimal prompt and tool set:

| Intent | Tools bound | Prompt sections |
| --- | --- | --- |
| `greeting` | None | Header, Greeting, Style |
| `schema` | `search_database_schema` only | Header, Schema, Style |
| `read` | All read tools (search\*, get_crm_record, sales_summary, list_crm_options, list_departments) | Header, Read, Style |
| `write` | `propose_sales_workflow` + read tools | Full system prompt with sales process |

This keeps token usage low, avoids unnecessary tool calls, and gives the LLM exactly what it needs for each task type.

### LLM provider selection and failover

`ai_assistant/llm.py` manages multiple LLM providers with automatic failover:

**Provider order (configurable via `LLM_PROVIDER_ORDER`):**
1. **Groq** — `qwen/qwen3-27b` (primary), `openai/gpt-oss-120b` (fallback)
2. **Gemini** — `gemini-1.5-flash` (primary), `gemini-2.0-flash` (fallback)
3. **Cloudflare Workers AI** — `@cf/qwen/qwen3-27b`, `@cf/google/gemma-4-26b-a4b-it`, `@cf/zai-org/glm-4.7-flash`
4. **OpenRouter** — `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-26b-a4b-it:free`, `inclusionai/ling-3.2-flash:free`, `baidu/ernie-5.5-flash-preview:free`

**Failover behavior:**
- Providers are tried in order; a provider that returns rate-limit, auth, or availability errors is temporarily skipped (cooldown periods: 2 min for rate limits, 15 min for auth errors, 30 sec for unavailable).
- Total timeout across all providers: 90 seconds (configurable via `LLM_TOTAL_TIMEOUT_SECONDS`).
- Per-provider timeout: 40 seconds (configurable via `LLM_TIMEOUT_SECONDS`).
- If a provider returns HTTP 400/413/422 (bad request), it is not retried for that request.
- Token usage is logged per successful call (input, output, total tokens).
- Model calls are cached per tool-set; configuration changes take effect on server restart.

**Environment variables for providers:**
```bash
GROQ_API_KEY=...
GEMINI_API_KEY=...          # or GOOGLE_API_KEY
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_ACCOUNT_ID=...
OPENROUTER_API_KEY=...
LLM_PROVIDER_ORDER=groq,gemini,cloudflare,openrouter
GROQ_MODELS=qwen/qwen3-27b,openai/gpt-oss-120b
GEMINI_MODELS=gemini-1.5-flash,gemini-2.0-flash
CLOUDFLARE_MODELS=@cf/qwen/qwen3-27b,@cf/google/gemma-4-26b-a4b-it,@cf/zai-org/glm-4.7-flash
OPENROUTER_MODELS=nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-26b-a4b-it:free,inclusionai/ling-3.2-flash:free,baidu/ernie-5.5-flash-preview:free
AI_ASSISTANT_MAX_TOKENS=2048
```

### Assistant record-operation coverage

| CRM record type | Read | Create | Update | Delete |
| --- | --- | --- | --- | --- |
| Lead | Yes, with CRM view permission and queryset scope | Yes | Yes, subject to record-level CRM rules | No assistant operation |
| Company | Yes, with CRM view permission and queryset scope | Yes | Yes, subject to record-level CRM rules | No assistant operation |
| Contact | Yes, with CRM view permission and queryset scope | Yes; requires a Company | Yes, subject to record-level CRM rules | No assistant operation |
| Request | Yes, with CRM view permission and queryset scope | Yes; may create its Deal through the CRM workflow | Yes; may process an eligible pending Request into a Deal | No assistant operation |
| Deal | Yes, with CRM view permission and queryset scope | Not directly; created from a Request | Yes, subject to CRM rules | No assistant operation |
| Payment | Yes, with CRM view permission and queryset scope | Yes, subject to CRM rules | Yes, subject to CRM rules | No assistant operation |

The assistant also exposes schema search, department and reference-option lookups, and sales summaries. These are not additional CRUD operations on CRM records. Delete operations and Lead conversion remain in the regular CRM UI; the assistant does not expose a delete tool or delete workflow.

### Admin Request-to-Deal workflow

A superuser does not need to be added to a sales department to create a Request. If the admin creates a Deal from a Request without a home department, the workflow resolves its owner and department from the linked Contact or Lead (or from the existing Request when processing it). The inferred values are shown for human approval, and the CRM's own validation and save hooks still run. If a department or owner cannot be resolved for the Deal, validation rejects the plan without saving it. Both sides of the Request/Deal link are created through the CRM's Request processing behavior. A plan has a unique operation ID and can execute only once; replaying an already processed plan does not create a duplicate.

This rule is specific to Request-to-Deal processing. It does not grant a superuser an implicit department for unrelated new records; ordinary CRM owner and department assignment rules continue to apply.

### Conversation context and persistence

LangGraph's SQLite saver persists graph state, including paused approvals, in `checkpoints.sqlite` next to `manage.py` (override with `AI_ASSISTANT_CHECKPOINT_DB`). A paused plan survives server restarts. Each model call keeps at most the three most recent user turns, dropping older turns when serialized history exceeds 10,000 characters. Treat the checkpoint file as application data: protect it and include it in backups and retention planning.

To print the exact messages sent to the LLM, set `AI_ASSISTANT_LOG_PROMPTS=1` before starting Django. Prompt traces contain CRM and user data; use this only in a controlled development environment.

### Refreshing the schema index

The schema catalog and FAISS index are generated from the CRM models. A test fails if a model changes and the catalog was not regenerated. Rebuild both from the repository root:

```powershell
python ai_assistant\rag\extract_schema.py
python ai_assistant\rag\build_index.py
```

The index builder uses the `BAAI/bge-small-en-v1.5` embedding model and downloads it the first time it runs.

## Set up a local development environment

The upstream installation guide contains the full CRM setup instructions. For a typical local SQLite development setup:

1. Install Python 3.12 or newer. The pinned Django 6.0 dependency requires this floor even though `setup.cfg` currently advertises an older `python_requires` value.
2. Create and activate an environment, for example with conda:

   ```powershell
   conda create -n django-crm python=3.12
   conda activate django-crm
   ```

3. Install the CRM and assistant dependencies:

   ```powershell
   python -m pip install -r requirements.txt -r requirements-ai.txt
   ```

4. To use the AI assistant, create a `.env` file in the project root (it is ignored by Git):

   ```
   # Add the keys for the providers you want enabled; missing providers are skipped.
   GROQ_API_KEY=...
   GEMINI_API_KEY=...
   CLOUDFLARE_API_TOKEN=...
   CLOUDFLARE_ACCOUNT_ID=...
   OPENROUTER_API_KEY=...
   # optional model/order overrides
   LLM_PROVIDER_ORDER=groq,gemini,cloudflare,openrouter
   GROQ_MODELS=qwen/qwen3-27b,openai/gpt-oss-120b
   GEMINI_MODELS=gemini-1.5-flash,gemini-2.0-flash
   CLOUDFLARE_MODELS=@cf/qwen/qwen3-27b,@cf/google/gemma-4-26b-a4b-it,@cf/zai-org/glm-4.7-flash
   OPENROUTER_MODELS=nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-26b-a4b-it:free,inclusionai/ling-3.2-flash:free,baidu/ernie-5.5-flash-preview:free
   AI_ASSISTANT_MAX_TOKENS=2048
   ```

   The provider manager tries the configured providers in `LLM_PROVIDER_ORDER`. It briefly skips providers that return rate-limit or availability errors. Groq's two GPT-OSS models share one API key and rate-limit circuit. The Cloudflare API token needs Workers AI access, and its account ID is required to build the endpoint URL. OpenRouter uses the listed tool-capable Llama model by default. Install the AI requirements after changing them. Restart Django after editing `.env`; settings are read when the LLM manager is initialized. Standalone greetings do not call a provider.

5. For a new, empty local database, apply migrations and load the starter data:

   ```powershell
   python manage.py migrate
   python manage.py setupdata
   ```

   `setupdata` also runs migrations, loads the project's fixtures, and creates three demo users: `IamSUPER`, `IamGlobalSALES`, and `IamLocalSALES`. It prints generated passwords to the console. The sales accounts are assigned the `managers`, `co-workers`, and matching department groups.

   **Do not treat `setupdata` as a repeatable or non-destructive command.** Its fixtures use fixed primary keys and can replace rows with matching keys (including the local Site record); it also creates users with fixed usernames, so a second run can fail on duplicate users. Use it only for a fresh database, or review and back up a populated database before loading fixtures. It is not needed just to run Django.

   `setupdata` already creates a superuser. Run `python manage.py createsuperuser` only if you want an additional account, using a distinct username. If you initialize users without `setupdata`, ensure the `co-workers` group exists first; the user-creation signal expects it.

6. Ensure `media/locks` exists (the background email importer uses it), then start the development server:

   ```powershell
   if (-not (Test-Path media\locks)) {
       New-Item -ItemType Directory -Path media\locks -Force | Out-Null
   }
   python manage.py runserver
   ```

The default settings use SQLite, `DEBUG = True`, and local host names. They are for local development, not public deployment. Configure secrets, `ALLOWED_HOSTS`, database, email, HTTPS, static files, and other security settings before deploying.

### Open the CRM and assistant

With the checked-in settings and English locale:

| Page or endpoint | URL |
| --- | --- |
| Django Admin login | `http://127.0.0.1:8000/en/456-admin/` |
| CRM | `http://127.0.0.1:8000/en/123/` |
| AI assistant page | `http://127.0.0.1:8000/en/api/ai/chat-ui/` |
| AI chat API | `POST http://127.0.0.1:8000/api/ai/chat/` |
| AI approval API | `POST http://127.0.0.1:8000/api/ai/approve/` |

The chat page and both API endpoints require an authenticated, active staff session. The APIs use Django REST Framework session authentication, so unsafe requests must include a valid CSRF token. Review production-specific authentication, rate limiting, and operational requirements before exposing the application publicly.

## Tests and checks

Run Django's configuration checks and the assistant tests from the repository root:

```powershell
python manage.py check
python manage.py test ai_assistant
```

The assistant tests use the CRM's real group permissions and departments and replace only the LLM with a scripted stand-in. They cover:

- deterministic greetings that do not call an LLM provider;
- schema RAG without CRM record permissions and read tools with explicit view permissions;
- reads and department scoping;
- permission enforcement;
- every create/update operation;
- duplicates and missing information;
- relationship validation;
- approval, rejection and stale approvals;
- resuming after a restart;
- multi-record transactions and rollback;
- duplicate-execution protection;
- superuser Request creation without sales-department membership, including Request-to-Deal department/owner inference and one-time execution;
- ordinary read-only users being unable to write, including forged or approved plans;
- cross-user isolation;
- document uploads;
- graph routing;
- RAG catalog freshness;
- provider fallback;
- **local intent router** (`router.py`) classification for all four intent types;
- **dynamic prompt and tool binding** per intent;
- **token usage logging** from provider responses;
- **name-based deal resolution** for payments (by name or ticket);
- **ambiguous name handling** (lists matches, asks user);
- **department-scoped name lookups** (Global sales cannot see Local sales deals);
- **superuser without department** chain fails with actionable message;
- **LLM call cap** counts only replies since last user message;
- **HTTP 400/413/422 rejections** are not retried in a loop;
- **tool schema accepts numeric values** (amount, probability, record_id, booleans).

The project also has a broader suite under `tests/`; run it with `python manage.py test tests` when making wider changes.

## Further reading

- [Main project README](README.md) — product overview and feature summary.
- [Installation and configuration](docs/installation_and_configuration_guide.md) — CRM setup, email, database, users, and integrations.
- [User guide](docs/django-crm_user_guide.md) — using the CRM.
- [System overview](docs/crm_system_overview.md) — CRM concepts and workflows.
- [Changelog](CHANGELOG.md) — upstream project changes.