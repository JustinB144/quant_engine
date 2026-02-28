# Feature Spec: Automated Documentation System (Source-Derived + Drift Checks)

> **Status:** Draft
> **Author:** Codex (GPT-5)
> **Date:** 2026-02-26
> **Estimated effort:** 60 hours across 12 tasks

---

## Why

The repository already distinguishes between narrative docs and source-derived docs in `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md`, but the refresh process is still manual and ad hoc. As the system grows (FastAPI routers, React frontend, scripts, tests, specs, and package READMEs), documentation drift is inevitable unless generation and validation are automated.

Current risk:
- Source-derived docs become stale after normal code changes.
- Narrative docs silently miss new endpoints/routes/CLI flags.
- Documentation refreshes are expensive, inconsistent, and depend on one-off scripts or manual audits.
- There is no repo-enforced check (`pre-commit`/CI) that fails when generated docs drift from source.

This spec defines a complete auto-doc system that treats the **current working tree (including uncommitted changes)** as source of truth and makes documentation refresh reproducible, testable, and enforceable.

## What

Implement an automated documentation system that:

1. **Discovers and classifies docs** (narrative vs source-derived vs historical) from a maintained registry.
2. **Builds deterministic source inventories** from the current working tree using AST/text parsing (no runtime imports with side effects).
3. **Generates source-derived docs** (reference docs + package READMEs) reproducibly from source.
4. **Runs in `write` mode and `check` mode**:
   - `write`: rewrite generated docs
   - `check`: fail if generated output differs from committed files
5. **Produces a machine-readable report/manifest** (targets, hashes, timings, diffs).
6. **Integrates with local workflow** (CLI + optional pre-commit hook).
7. **Integrates with GitHub CI** (drift-check workflow) if/when `.github/` is present/added.
8. **Adds narrative drift checks** (heuristic assertions for active docs) without auto-rewriting human-written docs.
9. **Is fully tested** (unit + integration/golden tests) and safe to run in a dirty working tree.

Done means a maintainer can run one command to refresh docs and another command (same tool, `--check`) to verify docs are in sync with the current source tree.

## Constraints

### Must-haves
- Must treat the **current working tree** as source of truth (include uncommitted changes).
- Must not import arbitrary project modules for introspection unless explicitly sandboxed/isolated; prefer AST/text parsing to avoid side effects.
- Must generate deterministic output (stable ordering, stable formatting, no wall-clock timestamps in generated bodies unless intentionally isolated).
- Must support target-scoped generation (`--targets`), full generation, and check-only mode.
- Must preserve the narrative/source-derived/historical distinction documented in `docs/maintenance/DOCS_MAINTENANCE.md`.
- Must skip large/generated/cache paths (`data/cache/`, `.venv/`, `node_modules/`, `__pycache__/`, etc.).
- Must fail loudly with actionable errors (missing file, parse error, unsupported doc target).
- Must include automated tests for parsers and renderers.

### Must-nots
- Must NOT rewrite narrative docs automatically.
- Must NOT scan hidden nested repos or ignored directories recursively without an explicit allowlist.
- Must NOT depend on network access.
- Must NOT rely on nondeterministic file ordering from filesystem traversal.
- Must NOT silently overwrite files when generator output is empty/invalid.
- Must NOT assume GitHub Actions exists today; CI integration must be optional/additive.

### Out of scope
- Auto-writing historical reports/specs (`docs/reports/*`, legacy audits, `.docx` files).
- Semantic prose generation for architecture/guides (human-authored narrative remains manual).
- LLM-based doc generation in CI.
- Publishing docs to external sites (MkDocs/Sphinx/Docusaurus).

## Current State

This section is source-verified against the current repository working tree on 2026-02-26.

### Verified facts (current repo)

- `docs/maintenance/DOCS_MAINTENANCE.md` defines three categories:
  - current narrative docs
  - source-derived reference docs (should be regenerated)
  - historical/planning docs
- Source-derived docs currently exist, but there is **no committed auto-doc generator** for the full set.
- There is **no `.github/` directory** in the repo today (no CI workflow configured).
- There is **no `.pre-commit-config.yaml`** in the repo today.
- There is **no `Makefile`/`justfile`/`tox.ini`/`noxfile.py`** in the repo today.
- `pyproject.toml` includes relevant dev dependencies (`pytest`, `ruff`, `pre-commit`) but no docs-generation command.
- There is an existing source-derived generator pattern in `/Users/justinblaise/Documents/quant_engine/scripts/generate_types.py` (script-based deterministic codegen from source metadata).
- `.gitignore` already excludes paths relevant to scanning (`data/cache/`, `.venv/`, `node_modules/`, `__pycache__/`, `*.parquet`, `*.meta.json`, `*.db`, etc.).

### Existing docs intended for source-derived regeneration

Per `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md`, examples include:
- `/Users/justinblaise/Documents/quant_engine/docs/reference/REPO_COMPONENT_MATRIX.md`
- `/Users/justinblaise/Documents/quant_engine/docs/reference/SOURCE_API_REFERENCE.md`
- `/Users/justinblaise/Documents/quant_engine/docs/reference/CONFIG_REFERENCE.md`
- `/Users/justinblaise/Documents/quant_engine/docs/reference/FRONTEND_UI_REFERENCE.md`
- `/Users/justinblaise/Documents/quant_engine/docs/reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md`
- `/Users/justinblaise/Documents/quant_engine/docs/reference/TEST_SPEC_MAP.md`
- Package READMEs in `/Users/justinblaise/Documents/quant_engine/{autopilot,backtest,data,features,kalshi,models,regime,risk,tests}/README.md`

### Key files (implementation context)

| File | Role | Notes |
|------|------|-------|
| `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md` | Canonical docs policy | Defines categories and refresh process |
| `/Users/justinblaise/Documents/quant_engine/docs/reference/*.md` | Source-derived outputs | Should become generator targets |
| `/Users/justinblaise/Documents/quant_engine/scripts/generate_types.py` | Existing codegen pattern | Script-driven, deterministic, no external services |
| `/Users/justinblaise/Documents/quant_engine/pyproject.toml` | Tooling/deps | Has `pytest`, `ruff`, `pre-commit`; no docs command yet |
| `/Users/justinblaise/Documents/quant_engine/.gitignore` | Scan exclusions baseline | Important for path exclusion policy |
| `/Users/justinblaise/Documents/quant_engine/api/routers/__init__.py` | Mounted router list | Canonical for API router inventory |
| `/Users/justinblaise/Documents/quant_engine/frontend/src/App.tsx` | Frontend route map | Canonical route inventory |
| `/Users/justinblaise/Documents/quant_engine/frontend/src/api/endpoints.ts` | Frontend API endpoint constants | Canonical FE endpoint inventory |

### Existing patterns to follow

- Use `Path` + repo-root-relative traversal.
- Prefer AST parsing for Python inventories (modules/classes/functions/routes).
- Use deterministic table ordering and stable formatting in generated markdown.
- Scripts should be invokable via `python scripts/<name>.py`.
- Keep generated docs readable and explicitly labeled as source-derived.

### Configuration / Environment (for the auto-doc system)

Current repo has no auto-doc config yet. This spec introduces:
- a docs registry file (source-of-truth for target classification and generators),
- optional CLI entrypoint in `pyproject.toml`,
- optional `pre-commit` hook config,
- optional GitHub workflow.

## Architecture (Proposed Auto-Doc System)

### Components

1. **Registry**
   - Declares doc targets, categories, generator names, and output paths.
   - Prevents hard-coded target sprawl inside generator code.

2. **Scanner Layer**
   - Parses Python, frontend TS/TSX, config files, and tests into structured inventories.
   - Does not import runtime modules (AST/text parsing only).

3. **Renderer Layer**
   - Converts inventories into markdown docs (one renderer per target type).
   - Produces deterministic output.

4. **Runner / CLI**
   - `write` and `check` modes.
   - Supports target selection and reports.

5. **Drift Checker**
   - Compares generated content to on-disk files.
   - Emits summary + machine-readable JSON.

6. **Narrative Guardrails**
   - Checks active narrative docs for required references/invariants (heuristic assertions).
   - Flags drift without auto-editing prose.

7. **Workflow Integration**
   - Local (`pre-commit`) and CI (`.github/workflows/...`) wrappers.

### Proposed filesystem layout

| Path | Purpose |
|------|---------|
| `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py` | Main CLI entrypoint for docs generation/check |
| `/Users/justinblaise/Documents/quant_engine/docs/automation/doc_registry.yaml` | Target registry + categories + renderers |
| `/Users/justinblaise/Documents/quant_engine/docs/automation/README.md` | Operator/maintainer usage for auto-doc system |
| `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py` | Scanner unit tests |
| `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_renderers.py` | Renderer unit tests |
| `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_integration.py` | End-to-end `--check`/`--write` tests |
| `/Users/justinblaise/Documents/quant_engine/.pre-commit-config.yaml` (optional new) | Local docs drift check hook |
| `/Users/justinblaise/Documents/quant_engine/.github/workflows/docs-drift.yml` (optional new) | CI docs drift check |

## Tasks

Break implementation into small, discrete units. Each task should touch ≤3 files, take ≤30 minutes, and be independently committable.

### T1: Define Doc Target Registry (Single Source of Truth)

**What:** Create a machine-readable registry of documentation targets, categories, and generators.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/docs/automation/doc_registry.yaml` — new file
- `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md` — add reference to registry

**Implementation notes:**
- Registry must include fields:
  - `id`
  - `category` (`narrative`, `source_derived`, `historical`)
  - `path`
  - `generator` (for source-derived only)
  - `enabled`
  - `check_mode` (`generate`, `assert_only`, `skip`)
- Seed with all currently known source-derived docs and package READMEs.
- Include narrative docs with `assert_only` for drift checks (no rewriting).
- Include historical dirs with `skip`.
- Registry should be easy to diff and manually maintain.

**Verify:**
```bash
python - <<'PY'
from pathlib import Path
import yaml
p = Path('docs/automation/doc_registry.yaml')
cfg = yaml.safe_load(p.read_text())
assert 'targets' in cfg and isinstance(cfg['targets'], list)
assert any(t['category'] == 'source_derived' for t in cfg['targets'])
print('PASS: doc registry loads with source-derived targets')
PY
```

---

### T2: Build Deterministic Repository Discovery + Exclusion Policy

**What:** Implement path discovery/exclusion logic that matches repo realities and ignores cache/build artifacts.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py` — add discovery/exclusion core
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py` — add exclusion tests

**Implementation notes:**
- Exclude at minimum:
  - `.git/`, `.venv/`, `node_modules/`, `__pycache__/`, `.pytest_cache/`
  - `data/cache/`
  - `*.parquet`, `*.meta.json`, `*.db`, `*.db-journal`, `*.duckdb`, `*.duckdb.wal`
  - generated frontend build outputs if present
- Sort all traversals deterministically.
- Do not trust `git ls-files` alone because tool must work in dirty working tree and include untracked files when relevant.
- Prefer allowlist-based scanning per target (e.g., `api/`, `frontend/src/`, `tests/`) over whole-repo recursive scans.

**Verify:**
```bash
pytest -q tests/test_docs_autogen_scanners.py -k exclusion
```

---

### T3: Python AST Scanner Layer (Modules, Classes, Functions, CLI Scripts)

**What:** Add AST-based Python scanners used by multiple doc renderers.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py` — AST scanner functions/classes
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py` — parser unit tests

**Implementation notes:**
- Extract (without imports):
  - module docstring
  - LOC (line count)
  - top-level classes
  - top-level functions
  - public method names
- Add specialized scanners for:
  - root `run_*.py` scripts (argparse option count, module docs)
  - package README inventories (`autopilot`, `backtest`, `data`, etc.)
- Parse failures should report file + exception; fail the run unless `--continue-on-parse-error` is explicitly provided.

**Verify:**
```bash
pytest -q tests/test_docs_autogen_scanners.py -k python_ast
```

---

### T4: FastAPI Router + Endpoint Scanner

**What:** Parse mounted routers and endpoint inventories from source.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py`

**Implementation notes:**
- Canonical mounted router list source: `/Users/justinblaise/Documents/quant_engine/api/routers/__init__.py`.
- Parse router modules and then each `APIRouter(...)` definition and route decorators.
- Normalize path concatenation (`prefix` + decorator path).
- Preserve handlers and router file paths.
- Handle mixed absolute-path routers (e.g., system health endpoints mounted without router prefix).
- Output must be reusable by:
  - `SOURCE_API_REFERENCE`
  - `REPO_COMPONENT_MATRIX`
  - optional narrative drift assertions

**Verify:**
```bash
pytest -q tests/test_docs_autogen_scanners.py -k fastapi
python scripts/generate_docs.py --targets source_api_reference --check
```

---

### T5: Frontend Scanner Layer (Routes, Sidebar, Endpoints, Hooks)

**What:** Parse frontend route map, nav items, endpoint constants, and hook exports.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py`

**Implementation notes:**
- Canonical frontend sources:
  - `/Users/justinblaise/Documents/quant_engine/frontend/src/App.tsx`
  - `/Users/justinblaise/Documents/quant_engine/frontend/src/components/layout/Sidebar.tsx`
  - `/Users/justinblaise/Documents/quant_engine/frontend/src/api/endpoints.ts`
  - `/Users/justinblaise/Documents/quant_engine/frontend/src/api/queries/*.ts`
  - `/Users/justinblaise/Documents/quant_engine/frontend/src/api/mutations/*.ts`
- Parsing can be regex/text-based initially but must be covered by tests.
- Must handle current `Sidebar.tsx` `NAV_ITEMS` shape (`label`, `path`, `icon`) and fail clearly on format drift.
- Record counts/LOC for frontend inventories with deterministic file ordering.

**Verify:**
```bash
pytest -q tests/test_docs_autogen_scanners.py -k frontend
python scripts/generate_docs.py --targets frontend_ui_reference --check
```

---

### T6: Config + Test-Suite Scanners

**What:** Add scanners for `config.py` sections/constants and test inventories/spec maps.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_scanners.py`

**Implementation notes:**
- `config.py` scanner:
  - parse top-level constants + source expressions via AST
  - parse section headers from comment separators
  - parse inline `# STATUS:` annotations into `status`/`notes`
- Test scanner:
  - inventory `tests/` and `kalshi/tests/`
  - class names, test method counts, top-level `test_*`
  - scope summary for `TEST_SPEC_MAP`
- Must not execute any tests/modules for introspection.

**Verify:**
```bash
pytest -q tests/test_docs_autogen_scanners.py -k "config or test_map"
```

---

### T7: Markdown Renderer Layer for Source-Derived Docs

**What:** Implement renderers for all source-derived targets currently maintained in the repo.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_renderers.py`

**Implementation notes:**
- Renderers to implement (at minimum):
  - `repo_component_matrix`
  - `source_api_reference`
  - `config_reference`
  - `frontend_ui_reference`
  - `test_spec_map`
  - package README renderers (`autopilot`, `backtest`, `data`, `features`, `kalshi`, `models`, `regime`, `risk`, `tests`)
- Every generated doc header must clearly state:
  - source-derived
  - generated from current working tree
- Must produce stable markdown tables and escaped content (`|`, backticks, long expressions).
- Renderers must be pure functions (`inventory -> str`) for easy testing.

**Verify:**
```bash
pytest -q tests/test_docs_autogen_renderers.py
python scripts/generate_docs.py --targets all_source_derived --check
```

---

### T8: CLI Runner (`write` / `check` / `targets`) + Atomic File Writes

**What:** Build the main CLI workflow and atomic write behavior.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_integration.py`

**Implementation notes:**
- CLI flags:
  - `--write` (default false)
  - `--check`
  - `--targets <id,...>`
  - `--category source_derived`
  - `--report-json <path>`
  - `--verbose`
  - `--include-untracked` (default true)
- `--check` returns non-zero when any generated target differs.
- `--write` writes atomically (`.tmp` + rename), only if content changed.
- Reject invalid combinations (`--write` + `--check` can be allowed if semantics are explicit; otherwise disallow).
- Print concise summary (generated/changed/skipped/failed).

**Verify:**
```bash
python scripts/generate_docs.py --category source_derived --check
pytest -q tests/test_docs_autogen_integration.py -k cli
```

---

### T9: Drift Manifest + Machine-Readable Reporting

**What:** Add a JSON report/manifest for humans and CI.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_integration.py`

**Implementation notes:**
- Report schema (JSON):
  - `run_mode` (`check`/`write`)
  - `repo_root`
  - `targets_total`
  - `targets_processed`
  - `changed`
  - `unchanged`
  - `failed`
  - `duration_ms`
  - `targets[]` with per-target:
    - `id`, `path`, `category`, `status`, `changed`, `error`, `sha256_before`, `sha256_after`
- Optional persisted manifest path:
  - `/Users/justinblaise/Documents/quant_engine/docs/automation/.docgen_manifest.json`
- Manifest should be ignored by default if generated on local runs (or stored only when explicitly requested).

**Verify:**
```bash
python scripts/generate_docs.py --check --report-json /tmp/docgen_report.json || true
python - <<'PY'
import json
r=json.load(open('/tmp/docgen_report.json'))
assert 'targets' in r and 'run_mode' in r
print('PASS: report schema present')
PY
```

---

### T10: Narrative Drift Assertions (No Auto-Rewrite)

**What:** Implement assertions for active narrative docs to catch missing references after code changes.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_renderers.py`
- `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md` (document policy)

**Implementation notes:**
- This is **not generation**. It is validation-only.
- Examples of assertions:
  - `docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` must mention mounted router count or point to `SOURCE_API_REFERENCE`.
  - `docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` must include all root `run_*.py` scripts discovered from source.
  - `docs/README.md` must reference source-derived docs listed in registry.
- Assertion failures should tell maintainers what to update manually.
- Keep checks heuristic and maintainable (avoid brittle exact text matching; prefer required tokens/links/path mentions).

**Verify:**
```bash
python scripts/generate_docs.py --targets narrative_assertions --check
pytest -q tests/test_docs_autogen_renderers.py -k narrative
```

---

### T11: Local Workflow Integration (`pyproject` script + `pre-commit`)

**What:** Add a standard local command and optional pre-commit enforcement.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/pyproject.toml`
- `/Users/justinblaise/Documents/quant_engine/.pre-commit-config.yaml` (new)
- `/Users/justinblaise/Documents/quant_engine/docs/automation/README.md` (new)

**Implementation notes:**
- Add a project script entrypoint (example):
  - `qe-docs = "scripts.generate_docs:main"` if refactored into importable module
  - or document `python scripts/generate_docs.py` if script-only
- Add pre-commit hooks:
  - `docs-autogen-check` on relevant paths (`api/`, `frontend/src/`, `config.py`, `run_*.py`, `tests/`, `docs/`)
- Hook should default to `--check` (fast fail), not `--write`.
- Document how to run locally:
  - refresh docs
  - check drift
  - install hooks
  - bypass/override when needed

**Verify:**
```bash
python scripts/generate_docs.py --check
pre-commit run --all-files docs-autogen-check
```

---

### T12: Optional GitHub CI Workflow + Full Integration Tests + Rollout Guardrails

**What:** Add CI drift checking and finalize test/rollout plan.

**Files:**
- `/Users/justinblaise/Documents/quant_engine/.github/workflows/docs-drift.yml` (new, optional)
- `/Users/justinblaise/Documents/quant_engine/tests/test_docs_autogen_integration.py`
- `/Users/justinblaise/Documents/quant_engine/docs/maintenance/DOCS_MAINTENANCE.md`

**Implementation notes:**
- CI workflow should:
  - set up Python
  - install minimal deps
  - run `python scripts/generate_docs.py --category source_derived --check`
  - optionally run narrative assertions
  - upload JSON drift report artifact on failure
- Since `.github/` does not exist today, this task is additive and must not break local usage if CI is omitted.
- Add rollout strategy:
  - phase 1: manual local usage (`--write`)
  - phase 2: pre-commit `--check`
  - phase 3: CI `--check`
- Add emergency bypass docs (temporarily disable hook/workflow while generator bug is fixed).

**Verify:**
```bash
pytest -q tests/test_docs_autogen_integration.py
python scripts/generate_docs.py --category source_derived --check
```

---

## Validation

### Acceptance criteria

1. Running `python scripts/generate_docs.py --category source_derived --write` rewrites all registered source-derived docs deterministically with no manual edits required.
2. Running `python scripts/generate_docs.py --category source_derived --check` returns:
   - exit code `0` when docs are current
   - non-zero when any generated doc is stale
3. Generator output uses the current working tree (including uncommitted source files/edits).
4. Narrative docs are not auto-rewritten, but drift assertions catch missing critical references.
5. All scanner/renderers are covered by tests and do not import runtime modules with side effects.
6. Path exclusions prevent scanning large cache/build data and hidden artifacts.
7. The tool produces a machine-readable report suitable for CI parsing.
8. Local workflow is documented and repeatable (refresh + check + hook install).

### Verification steps

```bash
# 1) Full source-derived refresh
python scripts/generate_docs.py --category source_derived --write

# 2) Immediate idempotence check (must pass, no changes)
python scripts/generate_docs.py --category source_derived --check

# 3) Run doc-generator tests
pytest -q tests/test_docs_autogen_scanners.py tests/test_docs_autogen_renderers.py tests/test_docs_autogen_integration.py

# 4) (Optional) Narrative assertions
python scripts/generate_docs.py --targets narrative_assertions --check
```

### Rollback plan

- If generator output is incorrect:
  - revert generated docs and generator changes together (`git revert` or selective file revert)
  - temporarily disable `pre-commit` hook / CI docs-check workflow
  - keep registry and tests if stable; disable only broken target(s) via registry `enabled: false`
- If a specific renderer is broken:
  - mark target disabled in registry
  - continue running other targets
  - file follow-up spec/issue for renderer fix

---

## Notes

- This spec intentionally separates **generation** (source-derived docs) from **assertion** (narrative docs) to avoid accidental prose corruption.
- The first implementation can keep everything in `/Users/justinblaise/Documents/quant_engine/scripts/generate_docs.py`; a later refactor can split scanners/renderers into a package if complexity grows.
- If performance becomes an issue, add incremental caching keyed by file hash, but only after deterministic correctness is established.
- `docs/reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md` may require a dedicated parser for DDL extraction from `/Users/justinblaise/Documents/quant_engine/kalshi/storage.py`; include it in the registry with a dedicated renderer from the start or mark it `enabled: false` until implemented.
