# Documentation Maintenance Guide

## Purpose

This folder now contains both curated narrative docs and source-derived reference docs.
This document explains what to treat as canonical and how to keep docs accurate when the code changes.

## Documentation Types

### 1. Curated Narrative Docs (human-written / architecture-oriented)
These explain intent, workflows, and design decisions.
Examples:
- `../guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`
- `../architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`

### 2. Source-Derived Reference Docs (generated from current source tree)
These maximize coverage and lookupability.
Examples:
- `../reference/SOURCE_API_REFERENCE.md`
- `../reference/CONFIG_REFERENCE.md`
- `../reference/DASH_UI_REFERENCE.md`
- `../reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md`
- `../reference/TEST_SPEC_MAP.md`
- `../reference/REPO_COMPONENT_MATRIX.md`

## Update Rules (Recommended)

When changing code:
- Update narrative docs if the system intent, workflow, or contracts changed.
- Regenerate/update source-derived references if modules, callbacks, config constants, or schemas changed.
- Prefer additive docs updates over silent behavior changes.

## High-Risk Changes That Require Doc Updates

- identity handling (`permno`/ticker behavior)
- no-leakage joins / split logic
- regime labels or semantics
- backtest execution/cost assumptions
- promotion gate criteria
- Kalshi storage schema or distribution feature outputs
- Dash page routes/IDs/callbacks
- config constant additions/removals/renames

## What To Read Before Editing (for humans or LLMs)

1. `../reports/QUANT_ENGINE_LLM_CONTEXT_SPEC.md`
2. `../architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
3. `../reference/SOURCE_API_REFERENCE.md` (scoping)
4. Relevant package README(s) in the package you are modifying

## Canonical UI Note

The active UI is `dash_ui` launched by `run_dash.py`.
The legacy `ui` stack has been removed.
