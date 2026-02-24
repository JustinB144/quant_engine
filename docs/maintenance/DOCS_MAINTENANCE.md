# Documentation Maintenance Guide

## Purpose

This repo contains a mix of narrative docs and source-derived reference docs. This guide explains what is current runtime documentation vs historical context, and how to keep the active docs aligned with source.

## Documentation Categories

### 1. Current Narrative Docs (human-written, source-verified)

These explain architecture, workflows, operator usage, and invariants for the current FastAPI + React system.

Examples:
- `../architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../operations/CLI_AND_WORKFLOW_RUNBOOK.md`
- `../guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`
- `../guides/WEB_APP_QUICK_START.md`

### 2. Source-Derived Reference Docs (regenerate when source changes)

These are generated from the current filesystem / Python AST / source DDL and should be treated as lookup references, not design intent.

Examples:
- `../reference/REPO_COMPONENT_MATRIX.md`
- `../reference/SOURCE_API_REFERENCE.md`
- `../reference/CONFIG_REFERENCE.md`
- `../reference/FRONTEND_UI_REFERENCE.md`
- `../reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md`
- `../reference/TEST_SPEC_MAP.md`
- package `README.md` files in `autopilot/`, `backtest/`, `data/`, `features/`, `kalshi/`, `models/`, `regime/`, `risk/`, `tests/`

### 3. Historical / Planning Docs (context only)

These may intentionally refer to removed stacks, migration states, or superseded plans.

Examples:
- `../reports/*`
- `../specs/*`
- `../plans/*`
- root-level audit/report markdown files

Do not treat them as current runtime truth when they conflict with source.

## What Must Trigger Doc Updates

### Architecture / workflow changes
- router additions/removals (`api/routers/*`, `api/routers/__init__.py`)
- new background jobs or job payload changes (`api/jobs/*`, `api/schemas/compute.py`)
- frontend route/page changes (`frontend/src/App.tsx`, `frontend/src/components/layout/Sidebar.tsx`)
- CLI entrypoint/flag changes (`run_*.py`)

### Contract changes
- `ApiResponse` envelope or `ResponseMeta` changes (`api/schemas/envelope.py`)
- job status/payload changes (`api/jobs/models.py`)
- config constants/status changes (`config.py`, `api/config.py`, `config_structured.py`)
- result artifact schema changes under `results/` writers/readers
- Kalshi storage DDL changes (`kalshi/storage.py`)

### Core behavior changes
- identity handling (`permno`/ticker)
- leakage controls / split logic
- regime labels/semantics/gating
- promotion gate thresholds or required checks
- execution cost/risk realism assumptions

## Refresh Process (Recommended)

1. Inspect source first (entrypoints, routers, schemas, affected packages).
2. Update narrative docs that describe behavior or architecture.
3. Regenerate source-derived references (component matrix, source API reference, config reference, test map, frontend UI reference, schema reference, package READMEs).
4. Run greps to catch obsolete active references (for example removed modules/stacks).
5. If a doc is historical and intentionally stale, keep it but label/classify it as historical instead of pretending it is current.

## Active UI Canonical Note

The active UI stack is the React/Vite frontend in `frontend/`, backed by FastAPI (`api/`). Dash-era active docs have been removed from the current documentation set.
