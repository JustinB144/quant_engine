# Feature Spec: [Title]

> **Status:** Draft | In Review | Approved | In Progress | Complete
> **Author:** [name]
> **Date:** [YYYY-MM-DD]
> **Estimated effort:** [X hours across Y tasks]

---

## Why

_1-2 sentences describing the problem being solved. What's broken, missing, or suboptimal? Why does this matter now?_

## What

_Concrete, verifiable deliverable. What does "done" look like? Be specific enough that someone could check a box._

## Constraints

### Must-haves
- _Non-negotiable requirements — the feature doesn't ship without these_

### Must-nots
- _Explicit boundaries — what this feature does NOT do, to prevent scope creep_

### Out of scope
- _Related work that will be addressed separately in future specs_

## Current State

_Describe the existing code and patterns the implementation must follow. Include file paths, function signatures, data structures, and architectural conventions. The agent implementing this needs enough context to write code that looks like it belongs in the codebase._

### Key files
| File | Role | Notes |
|------|------|-------|
| `path/to/file.py` | _What it does_ | _Anything the implementor needs to know_ |

### Existing patterns to follow
- _Naming conventions, error handling style, logging patterns, test structure, etc._

### Configuration
- _Relevant config values, environment variables, feature flags_

## Tasks

_Break implementation into small, discrete units. Each task should touch ≤3 files, take ≤30 minutes, and be independently committable. Start each task in a fresh context session._

### T1: [Short descriptive title]

**What:** _Specific deliverable for this task_

**Files:**
- `path/to/file1.py` — _what changes_
- `path/to/file2.py` — _what changes_

**Implementation notes:**
- _Key decisions, algorithms, edge cases to handle_

**Verify:**
```bash
# Command(s) that prove this task is complete
python -m pytest tests/test_something.py -v
```

---

### T2: [Short descriptive title]

**What:** _Specific deliverable_

**Files:**
- `path/to/file.py` — _what changes_

**Implementation notes:**
- _Details_

**Verify:**
```bash
# Verification command
```

---

_...repeat for each task..._

## Validation

_End-to-end verification that the entire feature works correctly after all tasks are complete._

### Acceptance criteria
1. _Specific, testable condition_
2. _Another condition_
3. _Performance/regression requirement_

### Verification steps
```bash
# Full test suite or integration check
```

### Rollback plan
- _How to revert if something goes wrong (feature flag, git revert, etc.)_

---

## Notes

_Anything that doesn't fit above: design alternatives considered, references, open questions for future iteration._
