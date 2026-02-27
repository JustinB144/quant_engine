#!/usr/bin/env python3
"""
JOB 3: Hotspot Scoring Script
Computes module-level and file-level risk scores for every module and
cross-module file in the quant_engine codebase.

Inputs:
  - docs/audit/MODULE_INVENTORY.yaml (Job 1)
  - docs/audit/DEPENDENCY_EDGES.json (Job 2)
  - git log for change frequency
  - grep for cyclomatic complexity proxy

Outputs:
  - Printed scoring data for HOTSPOT_LIST.md construction
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Module definitions ---
CORE_MODULES = [
    "config", "data", "features", "indicators", "regime", "models",
    "backtest", "risk", "evaluation", "validation", "autopilot",
    "kalshi", "api", "utils",
]

# Shared artifact writers (from spec)
ARTIFACT_WRITERS = {
    "models/trainer.py",
    "backtest/engine.py",
    "models/predictor.py",
    "autopilot/registry.py",
    "autopilot/paper_trader.py",
    "autopilot/engine.py",
    "data/local_cache.py",
    "api/jobs/store.py",
    "kalshi/storage.py",
    "config.py",  # config_data/universe.yaml reader, consumed by many
}

# Contract files (export types used in cross-module signatures)
CONTRACT_FILES = {
    "config.py",
    "features/pipeline.py",
    "regime/shock_vector.py",
    "regime/uncertainty_gate.py",
    "regime/detector.py",
    "validation/preconditions.py",
    "models/predictor.py",
    "models/trainer.py",
    "models/versioning.py",
    "models/governance.py",
    "backtest/engine.py",
    "backtest/validation.py",
    "backtest/execution.py",
    "data/loader.py",
    "data/local_cache.py",
    "data/survivorship.py",
    "indicators/indicators.py",
    "risk/position_sizer.py",
    "risk/portfolio_risk.py",
    "risk/drawdown.py",
    "risk/stop_loss.py",
    "autopilot/engine.py",
    "autopilot/paper_trader.py",
    "autopilot/promotion_gate.py",
    "kalshi/storage.py",
    "api/services/health_service.py",
    "models/retrain_trigger.py",
    "models/cross_sectional.py",
}


def load_edges():
    """Load dependency edges from Job 2 output."""
    edges_path = ROOT / "docs" / "audit" / "DEPENDENCY_EDGES.json"
    with open(edges_path) as f:
        data = json.load(f)
    return data["edges"]


def score_range(value, thresholds):
    """Score a value on 0-3 scale using thresholds [low, mid, high]."""
    if value >= thresholds[2]:
        return 3
    elif value >= thresholds[1]:
        return 2
    elif value >= thresholds[0]:
        return 1
    return 0


def get_git_change_frequency(module_path, days=60):
    """Count git commits for a module path in last N days."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"--since={days}.days", "--", module_path],
            capture_output=True, text=True, cwd=ROOT,
        )
        return len([line for line in result.stdout.strip().split("\n") if line])
    except Exception:
        return 0


def get_complexity_proxy(filepath):
    """Count if/elif/except/try: statements as complexity proxy."""
    full_path = ROOT / filepath
    if not full_path.exists():
        return 0
    try:
        result = subprocess.run(
            ["grep", "-c", "-E", r"if |elif |except |try:", str(full_path)],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        return 0


def get_file_lines(filepath):
    """Get line count for a file."""
    full_path = ROOT / filepath
    if not full_path.exists():
        return 0
    try:
        result = subprocess.run(
            ["wc", "-l", str(full_path)],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip().split()[0])
    except Exception:
        return 0


def compute_module_scores(edges):
    """Compute module-level scores (Step 1)."""
    cross_edges = [e for e in edges if e["cross_module"]]

    # Fan-in: count distinct source modules per target module
    fan_in_modules = {m: set() for m in CORE_MODULES}
    fan_in_edges = {m: 0 for m in CORE_MODULES}
    for e in cross_edges:
        tgt = e["target_module"]
        src = e["source_module"]
        if tgt in fan_in_modules:
            fan_in_modules[tgt].add(src)
            fan_in_edges[tgt] += 1

    # Fan-out: count distinct target modules per source module
    fan_out_modules = {m: set() for m in CORE_MODULES}
    fan_out_edges = {m: 0 for m in CORE_MODULES}
    for e in cross_edges:
        src = e["source_module"]
        tgt = e["target_module"]
        if src in fan_out_modules:
            fan_out_modules[src].add(tgt)
            fan_out_edges[src] += 1

    # Git change frequency
    change_freq = {}
    for mod in CORE_MODULES:
        if mod == "config":
            change_freq[mod] = get_git_change_frequency("config.py") + \
                               get_git_change_frequency("config_structured.py")
        else:
            change_freq[mod] = get_git_change_frequency(f"{mod}/")

    # Largest file per module (lines)
    largest_file = {}
    for mod in CORE_MODULES:
        if mod == "config":
            largest_file[mod] = get_file_lines("config.py")
        else:
            mod_dir = ROOT / mod
            if mod_dir.exists():
                max_lines = 0
                for py_file in mod_dir.rglob("*.py"):
                    if "__pycache__" not in str(py_file):
                        lines = get_file_lines(str(py_file.relative_to(ROOT)))
                        max_lines = max(max_lines, lines)
                largest_file[mod] = max_lines
            else:
                largest_file[mod] = 0

    # Contract surface: count distinct external consumer modules
    contract_consumers = {m: set() for m in CORE_MODULES}
    for e in cross_edges:
        tgt = e["target_module"]
        src = e["source_module"]
        if tgt in contract_consumers:
            contract_consumers[tgt].add(src)

    # Test coverage gaps (from spec's verified data)
    test_gaps = {
        "config": 0, "data": 0, "features": 2, "indicators": 3,
        "regime": 0, "models": 2, "backtest": 1, "risk": 0,
        "evaluation": 1, "validation": 2, "autopilot": 1, "kalshi": 0,
        "api": 3, "utils": 1,
    }

    scores = {}
    for mod in CORE_MODULES:
        fi = score_range(len(fan_in_modules[mod]), [1, 3, 6])
        fo = score_range(len(fan_out_modules[mod]), [1, 3, 6])
        cs = score_range(len(contract_consumers[mod]), [1, 3, 6])
        # Override contract surface for special cases
        if mod == "config":
            cs = 3  # 200+ constants, 14 consumers
        elif mod == "indicators":
            cs = 2  # 90+ classes but only 1 direct consumer
        cf = score_range(change_freq[mod], [3, 11, 21])
        cx = score_range(largest_file[mod], [200, 500, 1000])
        tg = test_gaps.get(mod, 0)
        total = fi + fo + cs + cf + cx + tg

        scores[mod] = {
            "fan_in": fi,
            "fan_in_raw": len(fan_in_modules[mod]),
            "fan_in_edges": fan_in_edges[mod],
            "fan_out": fo,
            "fan_out_raw": len(fan_out_modules[mod]),
            "fan_out_edges": fan_out_edges[mod],
            "contract": cs,
            "contract_raw": len(contract_consumers[mod]),
            "changes": cf,
            "changes_raw": change_freq[mod],
            "complexity": cx,
            "complexity_raw": largest_file[mod],
            "test_gaps": tg,
            "total": total,
        }

    return scores


def compute_file_scores(edges):
    """Compute file-level scores (Step 2)."""
    cross_edges = [e for e in edges if e["cross_module"]]

    # Cross-module fan-in per target file
    file_fan_in = {}
    file_fan_in_sources = {}
    for e in cross_edges:
        tf = e["target_file"]
        if tf not in file_fan_in:
            file_fan_in[tf] = 0
            file_fan_in_sources[tf] = set()
        file_fan_in[tf] += 1
        file_fan_in_sources[tf].add(e["source_module"])

    # Files with outgoing lazy/conditional cross-module imports
    file_has_lazy = {}
    for e in cross_edges:
        sf = e["source_file"]
        if e["import_type"] in ("lazy", "conditional"):
            file_has_lazy[sf] = True

    # Collect all files with cross-module edges (source or target)
    all_cross_files = set()
    for e in cross_edges:
        all_cross_files.add(e["source_file"])
        all_cross_files.add(e["target_file"])

    file_scores = {}
    for filepath in sorted(all_cross_files):
        # Skip non-core files and __init__.py
        if filepath.endswith("__init__.py"):
            continue

        fi_count = file_fan_in.get(filepath, 0)
        is_contract = 1 if filepath in CONTRACT_FILES else 0
        lines = get_file_lines(filepath)
        loc_score = score_range(lines, [200, 500, 1000])
        complexity = get_complexity_proxy(filepath)
        cx_score = score_range(complexity, [200, 500, 1000])
        has_lazy = 1 if file_has_lazy.get(filepath, False) else 0
        is_artifact = 1 if filepath in ARTIFACT_WRITERS else 0

        # Weighted score
        total = (fi_count * 3) + (is_contract * 2) + loc_score + cx_score + (has_lazy * 2) + (is_artifact * 2)

        file_scores[filepath] = {
            "fan_in": fi_count,
            "fan_in_weighted": fi_count * 3,
            "fan_in_sources": sorted(file_fan_in_sources.get(filepath, set())),
            "contract": is_contract,
            "contract_weighted": is_contract * 2,
            "lines": lines,
            "loc_score": loc_score,
            "complexity": complexity,
            "cx_score": cx_score,
            "lazy": has_lazy,
            "lazy_weighted": has_lazy * 2,
            "artifact": is_artifact,
            "artifact_weighted": is_artifact * 2,
            "total": total,
        }

    return file_scores


def main():
    print("Loading dependency edges...")
    edges = load_edges()
    cross_count = sum(1 for e in edges if e["cross_module"])
    print(f"  Total edges: {len(edges)}, Cross-module: {cross_count}")
    print()

    # --- Module Scores ---
    print("=" * 80)
    print("MODULE-LEVEL SCORES")
    print("=" * 80)
    module_scores = compute_module_scores(edges)

    # Sort by total score descending
    ranked = sorted(module_scores.items(), key=lambda x: -x[1]["total"])

    print(f"\n{'Rank':>4} | {'Module':<12} | {'Score':>5} | {'Fan-in':>6} | {'Fan-out':>7} | {'Contract':>8} | {'Changes':>7} | {'Complex':>7} | {'TestGap':>7} | Key Risk")
    print("-" * 120)
    for rank, (mod, s) in enumerate(ranked, 1):
        key_risk = ""
        if mod == "config":
            key_risk = f"Supreme hub: {s['fan_in_edges']} inbound edges from ALL modules"
        elif mod == "autopilot":
            key_risk = f"Highest fan-out ({s['fan_out_edges']} edges), 6 circular api edges"
        elif mod == "features":
            key_risk = f"90+ features flow through pipeline, indicators dependency"
        elif mod == "data":
            key_risk = f"Primary data ingestion, {s['fan_in_raw']} consumer modules"
        elif mod == "models":
            key_risk = f"Training/prediction pipeline, governance contracts"
        elif mod == "backtest":
            key_risk = f"Core simulation, errors invalidate all results"
        elif mod == "api":
            key_risk = f"Largest module ({s['fan_out_edges']} outbound), circular with autopilot"
        elif mod == "regime":
            key_risk = f"7 consumer modules, structural state contracts"
        elif mod == "risk":
            key_risk = f"Position sizing/limits affect all live trading"
        elif mod == "indicators":
            key_risk = f"90+ indicators, 0 dedicated tests, signature changes break features"
        elif mod == "kalshi":
            key_risk = f"Self-contained vertical, lower blast radius"
        elif mod == "validation":
            key_risk = f"Truth Layer contracts, 0 dedicated tests"
        elif mod == "evaluation":
            key_risk = f"Leaf module (0 fan-in), safe to change"
        elif mod == "utils":
            key_risk = f"Minimal module, leaf (0 fan-in)"

        print(f"{rank:>4} | {mod:<12} | {s['total']:>3}/18 | {s['fan_in']:>4}/3 | {s['fan_out']:>5}/3 | {s['contract']:>6}/3 | {s['changes']:>5}/3 | {s['complexity']:>5}/3 | {s['test_gaps']:>5}/3 | {key_risk}")

    # --- File Scores ---
    print()
    print("=" * 80)
    print("FILE-LEVEL SCORES (Top 25)")
    print("=" * 80)
    file_scores = compute_file_scores(edges)

    # Sort by total score descending
    file_ranked = sorted(file_scores.items(), key=lambda x: -x[1]["total"])

    print(f"\n{'Rank':>4} | {'File':<45} | {'Score':>5} | {'FanIn(3x)':>9} | {'Contract(2x)':>12} | {'LOC(1x)':>7} | {'CX(1x)':>6} | {'Lazy(2x)':>8} | {'Artif(2x)':>9}")
    print("-" * 140)
    for rank, (filepath, s) in enumerate(file_ranked[:25], 1):
        print(f"{rank:>4} | {filepath:<45} | {s['total']:>5} | {s['fan_in']:>3}→{s['fan_in_weighted']:>4} | {s['contract']:>1}→{s['contract_weighted']:>3}      | {s['loc_score']:>5}/3 | {s['cx_score']:>4}/3 | {s['lazy']:>1}→{s['lazy_weighted']:>3}  | {s['artifact']:>1}→{s['artifact_weighted']:>3}")

    # --- Detailed info for blast radius ---
    print()
    print("=" * 80)
    print("BLAST RADIUS DATA (Top 15 files)")
    print("=" * 80)
    cross_edges = [e for e in edges if e["cross_module"]]

    for rank, (filepath, s) in enumerate(file_ranked[:15], 1):
        print(f"\n### {rank}. {filepath} (score={s['total']})")
        print(f"  Lines: {s['lines']}, Complexity proxy: {s['complexity']}")
        print(f"  Fan-in sources: {s['fan_in_sources']}")

        # Direct dependents
        print(f"  Direct dependents:")
        for e in cross_edges:
            if e["target_file"] == filepath:
                print(f"    - {e['source_file']}:{e['source_line']} ({e['source_module']}) "
                      f"imports {e['symbols_imported']} [{e['import_type']}]")

        # Files this one imports from
        print(f"  Outgoing cross-module imports:")
        for e in cross_edges:
            if e["source_file"] == filepath:
                print(f"    → {e['target_file']}:{e['source_line']} "
                      f"imports {e['symbols_imported']} [{e['import_type']}]")


if __name__ == "__main__":
    main()
