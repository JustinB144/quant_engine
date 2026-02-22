# Quant Engine Documentation Index

## Purpose

This folder is the consolidated documentation hub for `quant_engine`.
It includes:
- human-operator guides,
- architecture and workflow explanations,
- LLM-oriented constraints/context,
- source-derived reference documentation,
- and package-level READMEs for local navigation.

## Recommended Reading Paths

### Human Operator / Owner
1. `guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`
2. `guides/DASH_QUICK_START.md`
3. `architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
4. `operations/CLI_AND_WORKFLOW_RUNBOOK.md`
5. `reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`

### LLM / Coding Agent
1. `reports/QUANT_ENGINE_LLM_CONTEXT_SPEC.md`
2. `architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
3. `reference/DASH_UI_REFERENCE.md` (if UI task)
4. `reference/CONFIG_REFERENCE.md`
5. `reference/SOURCE_API_REFERENCE.md`
6. `reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`

## Documentation Map

### Guides
- `guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md`: Human-oriented system explanation and UI/workflow guide
- `guides/DASH_QUICK_START.md`: Dash UI quick start
- `guides/DASH_FOUNDATION_SUMMARY.md`: Dash UI architecture/foundation summary
- `guides/UI_IMPROVEMENT_GUIDE.md`: UI improvement ideas and guidance

### Architecture
- `architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`: End-to-end architecture, subsystem interactions, data/control flow
- `architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`: Non-negotiable cross-module contracts and safety rules

### Operations
- `operations/CLI_AND_WORKFLOW_RUNBOOK.md`: Operational scripts, when to run them, typical workflows, outputs

### Reference (Source-Derived)
- `reference/SOURCE_API_REFERENCE.md`: Exhaustive module/class/function inventory for current source tree
- `reference/CONFIG_REFERENCE.md`: Configuration constants grouped by config sections
- `reference/DASH_UI_REFERENCE.md`: Dash pages, routes, IDs, callbacks, shared components
- `reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md`: Kalshi event-time storage schema/tables/indexes
- `reference/TEST_SPEC_MAP.md`: Tests as behavioral specification map
- `reference/REPO_COMPONENT_MATRIX.md`: Package-level component matrix and module index

### Glossary / Maintenance
- `glossary/QUANT_ENGINE_GLOSSARY.md`: System terminology and acronyms
- `maintenance/DOCS_MAINTENANCE.md`: How this documentation set is organized and refreshed

### Reports / Historical Analysis
- `reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`: Deep source-based system intent/component audit (master deep-dive)
- `reports/QUANT_ENGINE_LLM_CONTEXT_SPEC.md`: LLM context contract
- `reports/*.md`: Additional audit/integration/data refresh reports

## Current Source Coverage Snapshot

| Package | Modules | Classes | Top-level Functions | LOC |
|---|---:|---:|---:|---:|
| `(root)` | 12 | 0 | 23 | 2,561 |
| `autopilot` | 6 | 9 | 0 | 1,770 |
| `backtest` | 6 | 15 | 16 | 3,372 |
| `dash_ui` | 28 | 3 | 128 | 9,880 |
| `data` | 10 | 13 | 55 | 5,164 |
| `features` | 9 | 3 | 43 | 3,047 |
| `indicators` | 2 | 92 | 2 | 2,692 |
| `kalshi` | 25 | 34 | 66 | 5,947 |
| `models` | 13 | 26 | 7 | 4,397 |
| `regime` | 4 | 5 | 5 | 1,019 |
| `risk` | 11 | 14 | 14 | 2,742 |
| `tests` | 20 | 29 | 9 | 2,572 |
| `utils` | 2 | 3 | 1 | 437 |

## Notes

- The active UI stack is `dash_ui/` (Dash). Legacy `ui/` has been removed.
- The deep audit report remains the most thorough narrative source; the reference docs here maximize discoverability and lookup speed.
