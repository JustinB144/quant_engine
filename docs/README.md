# Quant Engine Documentation Index

## Purpose

This folder is the active documentation hub for the current `quant_engine` system (FastAPI backend + React/Vite frontend).

The docs are split into:
- current runtime/architecture/operator docs,
- source-derived reference docs (regenerated from source),
- package-level READMEs,
- and historical reports/specs (kept for context, not current runtime truth).

## Recommended Reading Paths

### Operator / Owner
1. `guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`
2. `guides/WEB_APP_QUICK_START.md`
3. `architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
4. `operations/CLI_AND_WORKFLOW_RUNBOOK.md`
5. `reference/FRONTEND_UI_REFERENCE.md` (UI details)

### Coding Agent / Maintainer
1. `architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
2. `reference/REPO_COMPONENT_MATRIX.md`
3. `reference/SOURCE_API_REFERENCE.md`
4. `reference/CONFIG_REFERENCE.md`
5. `reference/FRONTEND_UI_REFERENCE.md` (if web UI work)
6. Relevant package `README.md`

## Active Documentation Map

### Guides
- `guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`: High-level operator-oriented system guide (current stack)
- `guides/WEB_APP_QUICK_START.md`: FastAPI + React dev/prod startup guide
- `guides/FRONTEND_FOUNDATION_SUMMARY.md`: Source-verified frontend foundation summary
- `guides/UI_IMPROVEMENT_GUIDE.md`: Current React/FastAPI UI improvement backlog (source-verified context)

### Architecture
- `architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`: Runtime architecture and end-to-end flows
- `architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`: Cross-module contracts/invariants and known contract drift

### Operations
- `operations/CLI_AND_WORKFLOW_RUNBOOK.md`: Root scripts, web app startup, and workflow sequences

### Source-Derived Reference
- `reference/REPO_COMPONENT_MATRIX.md`: File/component matrix for Python + frontend source
- `reference/SOURCE_API_REFERENCE.md`: Python module/class/function/router inventory
- `reference/CONFIG_REFERENCE.md`: `config.py`, `api/config.py`, `config_structured.py` reference
- `reference/FRONTEND_UI_REFERENCE.md`: React routes/pages/hooks/endpoints/components reference
- `reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md`: `kalshi/storage.py` schema DDL reference
- `reference/TEST_SPEC_MAP.md`: Test suite behavioral map

### Glossary / Maintenance
- `glossary/QUANT_ENGINE_GLOSSARY.md`: Terms used across the equity, API, and Kalshi subsystems
- `maintenance/DOCS_MAINTENANCE.md`: What is generated vs narrative and how to keep docs current

## Historical / Planning Docs (Not Runtime Truth)

These directories/files may intentionally reference removed stacks (for example Dash) or superseded migration states:
- `docs/reports/` (audits, migration reports, historical investigations)
- `docs/specs/` (implementation specs / design history)
- `docs/plans/` (roadmaps/plans)
- root-level `*AUDIT*`, `*REPORT*`, `*IMPROVEMENT*` markdown files

Use source-derived references and current architecture docs first when behavior conflicts with a historical report.

## Current Source Coverage Snapshot

### Python Packages

| Package | Modules | Classes | Top-level Functions | LOC |
|---|---:|---:|---:|---:|
| `(root)` | 13 | 18 | 23 | 3,352 |
| `api` | 59 | 65 | 125 | 9,267 |
| `autopilot` | 8 | 12 | 0 | 3,685 |
| `backtest` | 11 | 29 | 26 | 5,652 |
| `data` | 12 | 17 | 80 | 7,099 |
| `features` | 10 | 5 | 48 | 4,026 |
| `indicators` | 7 | 97 | 4 | 4,550 |
| `kalshi` | 16 | 26 | 66 | 5,224 |
| `models` | 16 | 34 | 9 | 5,971 |
| `regime` | 13 | 17 | 6 | 4,243 |
| `risk` | 16 | 22 | 21 | 5,845 |
| `tests` | 69 | 286 | 164 | 20,539 |
| `utils` | 2 | 3 | 1 | 446 |

### Frontend (`frontend/src`) 

- Files: 135
- LOC: 8,862
- Pages: 51
- Components: 39
- API query hooks: 12
- API mutation hooks: 6

## Removed Obsolete Active Docs

The previous Dash-specific active docs (`DASH_*` guides and `DASH_UI_REFERENCE.md`) have been removed from the active set because the source tree no longer contains `dash_ui/` or `run_dash.py`.
