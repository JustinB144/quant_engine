#!/usr/bin/env python3
"""
JOB 2: Dependency Extraction Script
Extracts every cross-module and same-module import relationship in the
quant_engine codebase using Python AST parsing.

Outputs:
  - docs/audit/DEPENDENCY_EDGES.json
  - docs/audit/DEPENDENCY_MATRIX.md
"""

import ast
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
QUANT_ENGINE = ROOT  # quant_engine is the root package

# Top-level module directories (packages)
MODULE_DIRS = [
    "api", "autopilot", "backtest", "data", "evaluation",
    "features", "indicators", "kalshi", "models", "regime",
    "risk", "utils", "validation",
]

# Root-level files that are considered part of "config" module
CONFIG_FILES = {"config.py", "config_structured.py"}

# Root-level entry points and scripts
ENTRY_POINTS_GLOB = "run_*.py"
SCRIPTS_DIR = "scripts"


def get_module_for_file(filepath: Path) -> str:
    """Determine which top-level module a file belongs to."""
    rel = filepath.relative_to(ROOT)
    parts = rel.parts

    if len(parts) == 1:
        # Root-level file
        name = parts[0]
        if name in CONFIG_FILES:
            return "config"
        if name == "reproducibility.py":
            return "reproducibility"
        if name.startswith("run_"):
            return "entry_points"
        return "root"

    if parts[0] == "scripts":
        return "scripts"

    if parts[0] in MODULE_DIRS:
        return parts[0]

    return "unknown"


def resolve_relative_import(source_file: Path, module: Optional[str], level: int) -> tuple[str, str]:
    """Resolve a relative import to its target module and file path.

    For a file at a/b/c.py with package a.b:
      - level 1 (from .X): stays in package a.b -> a/b/X
      - level 2 (from ..X): goes to parent a -> a/X
      - level 3 (from ...X): goes to grandparent (root) -> X

    Formula: base = directory_parts[:len(directory_parts) - (level - 1)]

    Returns (target_module, target_file_relative) or ("unknown", "unknown").
    """
    source_rel = source_file.relative_to(ROOT)
    source_parts = list(source_rel.parts)

    # Directory parts = package path (remove filename)
    dir_parts = source_parts[:-1]

    # Compute how many directory levels to remove
    levels_up = level - 1
    if levels_up > len(dir_parts):
        return "unknown", "unknown"

    # Base directory after going up
    base_parts = dir_parts[:len(dir_parts) - levels_up]

    # Append the module path
    if module:
        mod_parts = module.split(".")
        target_parts = base_parts + mod_parts
    else:
        target_parts = list(base_parts)

    if not target_parts:
        return "unknown", "unknown"

    # Determine the target top-level module
    first = target_parts[0]
    if first in MODULE_DIRS:
        target_module = first
    elif first in ("config", "config_structured"):
        target_module = "config"
    elif first == "reproducibility":
        target_module = "reproducibility"
    else:
        # Might be a root-level module
        target_module = first

    # Try to resolve to a file
    target_path = Path(*target_parts)
    target_file_py = str(target_path) + ".py"
    target_dir_init = target_path / "__init__.py"

    if (ROOT / target_file_py).exists():
        return target_module, target_file_py
    elif (ROOT / target_dir_init).exists():
        return target_module, str(target_dir_init)
    else:
        # Last part might be a symbol name, try without it
        if len(target_parts) > 1:
            parent_parts = target_parts[:-1]
            parent_path = Path(*parent_parts)
            parent_file = str(parent_path) + ".py"
            if (ROOT / parent_file).exists():
                return target_module, parent_file
            parent_init = parent_path / "__init__.py"
            if (ROOT / parent_init).exists():
                return target_module, str(parent_init)

        return target_module, str(target_path)


def resolve_absolute_import(module_str: str) -> tuple[str, str]:
    """Resolve an absolute import like 'quant_engine.data.loader' to target module and file."""
    parts = module_str.split(".")

    # Strip 'quant_engine.' prefix if present
    if parts[0] == "quant_engine":
        parts = parts[1:]

    if not parts:
        return "unknown", "unknown"

    # Check if first part is a known module or config file
    first = parts[0]

    if first in ("config", "config_structured"):
        return "config", first + ".py"

    if first == "reproducibility":
        return "reproducibility", "reproducibility.py"

    if first not in MODULE_DIRS:
        return "unknown", "unknown"

    target_module = first

    # Try to resolve the rest to a file path
    target_parts = parts
    target_path = Path(*target_parts)
    target_file = str(target_path) + ".py"
    target_dir = target_path / "__init__.py"

    if (ROOT / target_file).exists():
        return target_module, target_file
    elif (ROOT / target_dir).exists():
        return target_module, str(target_path / "__init__.py")
    else:
        # Try parent (last part might be a symbol)
        if len(target_parts) > 1:
            parent_parts = target_parts[:-1]
            parent_path = Path(*parent_parts)
            parent_file = str(parent_path) + ".py"
            if (ROOT / parent_file).exists():
                return target_module, parent_file
            parent_init = parent_path / "__init__.py"
            if (ROOT / parent_init).exists():
                return target_module, str(parent_init)

        return target_module, str(target_path)


def classify_import_type(node: ast.AST, source_lines: list[str]) -> str:
    """Classify an import as top_level, lazy, or conditional."""
    # Check if inside a try/except or if-guard
    # We do this by checking parent context, but AST doesn't store parents
    # So we use a heuristic: check the indentation level and line context

    line_no = node.lineno
    if line_no <= len(source_lines):
        line = source_lines[line_no - 1]
        indent = len(line) - len(line.lstrip())

        if indent == 0:
            return "top_level"

        # Check surrounding context for try/except or if-guard
        # Look backwards for try: or if: at a lower indent level
        for i in range(line_no - 2, max(0, line_no - 20) - 1, -1):
            if i < 0 or i >= len(source_lines):
                continue
            ctx_line = source_lines[i].rstrip()
            if not ctx_line or ctx_line.lstrip().startswith("#"):
                continue
            ctx_indent = len(ctx_line) - len(ctx_line.lstrip())
            if ctx_indent < indent:
                stripped = ctx_line.strip()
                if stripped.startswith("try:") or stripped.startswith("except"):
                    return "conditional"
                if stripped.startswith("if ") or stripped.startswith("elif "):
                    return "conditional"
                break

        # If indented but not conditional, it's lazy (inside a function)
        if indent > 0:
            return "lazy"

    return "top_level"


def extract_imports_from_file(filepath: Path) -> list[dict]:
    """Extract all import edges from a single Python file."""
    edges = []
    try:
        source = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return edges

    source_lines = source.split("\n")

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return edges

    source_rel = str(filepath.relative_to(ROOT))
    source_module = get_module_for_file(filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue

            module_str = node.module or ""
            level = node.level
            names = [alias.name for alias in node.names] if node.names else []

            # Determine import type
            import_type = classify_import_type(node, source_lines)

            # Resolve target
            if level > 0:
                # Relative import
                target_module, target_file = resolve_relative_import(
                    filepath, module_str, level
                )
                # Reconstruct the import statement from the source line
                line_no = node.lineno
                if line_no <= len(source_lines):
                    import_stmt = source_lines[line_no - 1].strip()
                    # Handle multi-line imports
                    if "(" in import_stmt and ")" not in import_stmt:
                        for j in range(line_no, min(line_no + 20, len(source_lines))):
                            import_stmt += " " + source_lines[j].strip()
                            if ")" in source_lines[j]:
                                break
                else:
                    import_stmt = f"from {'.' * level}{module_str} import {', '.join(names)}"
            else:
                # Absolute import
                target_module, target_file = resolve_absolute_import(module_str)
                line_no = node.lineno
                if line_no <= len(source_lines):
                    import_stmt = source_lines[line_no - 1].strip()
                    if "(" in import_stmt and ")" not in import_stmt:
                        for j in range(line_no, min(line_no + 20, len(source_lines))):
                            import_stmt += " " + source_lines[j].strip()
                            if ")" in source_lines[j]:
                                break
                else:
                    import_stmt = f"from {module_str} import {', '.join(names)}"

            if target_module == "unknown":
                # Skip third-party imports
                continue

            # Determine cross-module status
            cross_module = source_module != target_module

            # Special case: api internal imports that look like cross-module
            # api/routers/config_mgmt.py importing from ..config is api's own config
            if source_module == "api" and target_module == "api":
                cross_module = False

            notes = ""
            # Flag known architectural issues
            if source_module == "autopilot" and target_module == "api":
                notes = "Circular: autopilot imports from api (api serves autopilot)"
            if source_module == "data" and target_module == "kalshi":
                notes = "Lazy import inside factory function — optional coupling"

            edge = {
                "source_file": source_rel,
                "source_module": source_module,
                "source_line": node.lineno,
                "target_file": target_file,
                "target_module": target_module,
                "import_statement": import_stmt,
                "symbols_imported": names,
                "import_type": import_type,
                "cross_module": cross_module,
                "notes": notes,
            }
            edges.append(edge)

        elif isinstance(node, ast.Import):
            # Handle `import X` statements (rare in this codebase for internal modules)
            for alias in node.names:
                module_str = alias.name
                if module_str.startswith("quant_engine."):
                    target_module, target_file = resolve_absolute_import(module_str)
                    if target_module == "unknown":
                        continue
                    import_type = classify_import_type(node, source_lines)
                    cross_module = source_module != target_module

                    edge = {
                        "source_file": source_rel,
                        "source_module": source_module,
                        "source_line": node.lineno,
                        "target_file": target_file,
                        "target_module": target_module,
                        "import_statement": source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else f"import {module_str}",
                        "symbols_imported": [alias.asname or alias.name.split(".")[-1]],
                        "import_type": import_type,
                        "cross_module": cross_module,
                        "notes": "",
                    }
                    edges.append(edge)

    return edges


def collect_all_files() -> list[Path]:
    """Collect all Python files to scan."""
    files = []

    # Module directories
    for mod_dir in MODULE_DIRS:
        mod_path = ROOT / mod_dir
        if mod_path.exists():
            for py_file in mod_path.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    files.append(py_file)

    # Root config files
    for cfg in CONFIG_FILES:
        cfg_path = ROOT / cfg
        if cfg_path.exists():
            files.append(cfg_path)

    # reproducibility.py
    repro = ROOT / "reproducibility.py"
    if repro.exists():
        files.append(repro)

    # Entry points
    for entry in ROOT.glob("run_*.py"):
        files.append(entry)

    # Scripts
    scripts_dir = ROOT / "scripts"
    if scripts_dir.exists():
        for py_file in scripts_dir.glob("*.py"):
            if "__pycache__" not in str(py_file):
                files.append(py_file)

    return sorted(set(files))


def build_adjacency_matrix(edges: list[dict]) -> dict[str, dict[str, int]]:
    """Build module-level adjacency matrix from cross-module edges."""
    all_modules = sorted(set(
        [e["source_module"] for e in edges] + [e["target_module"] for e in edges]
    ))

    matrix = {src: {tgt: 0 for tgt in all_modules} for src in all_modules}

    for edge in edges:
        if edge["cross_module"]:
            src = edge["source_module"]
            tgt = edge["target_module"]
            matrix[src][tgt] += 1

    return matrix


def generate_dependency_matrix_md(edges: list[dict], matrix: dict[str, dict[str, int]]) -> str:
    """Generate the DEPENDENCY_MATRIX.md content."""
    # Core modules for the matrix (exclude entry_points, scripts, reproducibility from matrix rows)
    core_modules = [
        "config", "data", "features", "indicators", "regime", "models",
        "backtest", "risk", "evaluation", "validation", "autopilot",
        "kalshi", "api", "utils",
    ]

    # Also include entry_points and scripts if they have cross-module imports
    all_source_modules = sorted(set(e["source_module"] for e in edges if e["cross_module"]))
    extra_modules = [m for m in all_source_modules if m not in core_modules]

    display_modules = core_modules + extra_modules

    lines = []
    lines.append("# DEPENDENCY MATRIX")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")
    lines.append("## Module-Level Adjacency Matrix")
    lines.append("")
    lines.append("Each cell shows the number of cross-module import edges from the row module to the column module.")
    lines.append("")

    # Header
    header = "| Source \u2193 / Target \u2192 |"
    separator = "|---|"
    for mod in core_modules:
        header += f" {mod} |"
        separator += "---:|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for src in display_modules:
        row = f"| **{src}** |"
        for tgt in core_modules:
            count = matrix.get(src, {}).get(tgt, 0)
            cell = str(count) if count > 0 else "."
            row += f" {cell} |"
        lines.append(row)

    lines.append("")

    # Fan-out summary (how many modules each module imports FROM)
    lines.append("## Fan-Out (Outgoing Dependencies)")
    lines.append("")
    lines.append("Number of cross-module import edges originating from each module.")
    lines.append("")
    lines.append("| Module | Fan-Out (edges) | Target Modules |")
    lines.append("|---|---:|---|")

    for mod in display_modules:
        fan_out = sum(matrix.get(mod, {}).get(t, 0) for t in core_modules)
        targets = [t for t in core_modules if matrix.get(mod, {}).get(t, 0) > 0]
        targets_str = ", ".join(targets) if targets else "(none)"
        lines.append(f"| {mod} | {fan_out} | {targets_str} |")

    lines.append("")

    # Fan-in summary (how many modules import INTO each module)
    lines.append("## Fan-In (Incoming Dependencies)")
    lines.append("")
    lines.append("Number of cross-module import edges targeting each module.")
    lines.append("")
    lines.append("| Module | Fan-In (edges) | Source Modules |")
    lines.append("|---|---:|---|")

    for mod in core_modules:
        fan_in = sum(matrix.get(s, {}).get(mod, 0) for s in display_modules)
        sources = [s for s in display_modules if matrix.get(s, {}).get(mod, 0) > 0]
        sources_str = ", ".join(sources) if sources else "(none)"
        lines.append(f"| {mod} | {fan_in} | {sources_str} |")

    lines.append("")

    # Hub identification
    lines.append("## Hub Modules (5+ cross-module connections)")
    lines.append("")

    for mod in display_modules:
        fan_out_targets = [t for t in core_modules if matrix.get(mod, {}).get(t, 0) > 0]
        fan_in_sources = [s for s in display_modules if matrix.get(s, {}).get(mod, 0) > 0]
        total_connections = len(set(fan_out_targets) | set(fan_in_sources))
        if total_connections >= 5:
            lines.append(f"- **{mod}**: {total_connections} distinct module connections "
                         f"(imports from {len(fan_out_targets)} modules, imported by {len(fan_in_sources)} modules)")

    lines.append("")

    # Isolated modules
    lines.append("## Isolated Modules (zero cross-module imports)")
    lines.append("")
    isolated = []
    for mod in core_modules:
        fan_out = sum(matrix.get(mod, {}).get(t, 0) for t in core_modules)
        fan_in = sum(matrix.get(s, {}).get(mod, 0) for s in display_modules)
        if fan_out == 0 and fan_in == 0:
            isolated.append(mod)
    if isolated:
        for mod in isolated:
            lines.append(f"- **{mod}**: No cross-module imports in either direction")
    else:
        lines.append("(None — all modules participate in cross-module imports)")

    lines.append("")

    # Import type breakdown
    lines.append("## Import Type Breakdown")
    lines.append("")

    cross_edges = [e for e in edges if e["cross_module"]]
    top_level = sum(1 for e in cross_edges if e["import_type"] == "top_level")
    lazy = sum(1 for e in cross_edges if e["import_type"] == "lazy")
    conditional = sum(1 for e in cross_edges if e["import_type"] == "conditional")

    lines.append(f"- **Top-level**: {top_level} edges (loaded at import time)")
    lines.append(f"- **Lazy**: {lazy} edges (loaded inside function bodies)")
    lines.append(f"- **Conditional**: {conditional} edges (inside try/except or if-guards)")
    lines.append("")

    # Architectural notes
    lines.append("## Architectural Notes")
    lines.append("")

    noted_edges = [e for e in edges if e.get("notes")]
    if noted_edges:
        for e in noted_edges:
            lines.append(f"- `{e['source_file']}:{e['source_line']}` \u2192 `{e['target_module']}`: {e['notes']}")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"Scanning quant_engine at: {ROOT}")
    print()

    # Collect all files
    files = collect_all_files()
    print(f"Found {len(files)} Python files to scan")

    # Extract all edges
    all_edges = []
    for f in files:
        edges = extract_imports_from_file(f)
        all_edges.extend(edges)

    print(f"Extracted {len(all_edges)} total import edges")

    cross_module = [e for e in all_edges if e["cross_module"]]
    same_module = [e for e in all_edges if not e["cross_module"]]
    cross_module_count = len(cross_module)
    same_module_count = len(same_module)
    print(f"  Cross-module: {cross_module_count}")
    print(f"  Same-module: {same_module_count}")

    output = {
        "metadata": {
            "generated": date.today().isoformat(),
            "total_edges": len(all_edges),
            "cross_module_edges": cross_module_count,
            "same_module_edges": same_module_count,
        },
        "edges": all_edges,
    }

    # Write JSON
    # Canonical audit data layout:
    # docs/audit/data/DEPENDENCY_EDGES.json  — Job 2 output
    # docs/audit/data/INTERFACE_CONTRACTS.yaml — Job 3 output
    output_dir = ROOT / "docs" / "audit" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "DEPENDENCY_EDGES.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {json_path}")

    # Build adjacency matrix
    matrix = build_adjacency_matrix(all_edges)

    # Generate markdown
    md_content = generate_dependency_matrix_md(all_edges, matrix)
    md_path = output_dir / "DEPENDENCY_MATRIX.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Wrote {md_path}")

    # Print summary
    print("\n--- SUMMARY ---")
    for src_mod in sorted(set(e["source_module"] for e in cross_module)):
        targets = {}
        for e in cross_module:
            if e["source_module"] == src_mod:
                tgt = e["target_module"]
                targets[tgt] = targets.get(tgt, 0) + 1
        target_str = ", ".join(f"{t}({c})" for t, c in sorted(targets.items()))
        print(f"  {src_mod} -> {target_str}")


if __name__ == "__main__":
    main()
