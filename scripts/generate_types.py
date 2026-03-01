"""Generate TypeScript interfaces from Pydantic schemas.

Usage:
    python scripts/generate_types.py > frontend/src/types/generated.ts

Introspects all Pydantic models in api/schemas/ and emits TypeScript
interfaces. This ensures the frontend types stay in sync with the
backend schema definitions.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, get_args, get_origin

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

from pydantic import BaseModel


def discover_schema_modules() -> list:
    """Auto-discover all modules in api.schemas package."""
    import api.schemas as schemas_pkg
    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules(schemas_pkg.__path__):
        if modname.startswith("_"):
            continue  # Skip __init__.py etc.
        full_name = f"api.schemas.{modname}"
        modules.append(full_name)
    return sorted(modules)


try:
    SCHEMA_MODULES = discover_schema_modules()
    logger.info("Auto-discovered schema modules: %s", SCHEMA_MODULES)
except Exception:
    logger.warning("Auto-discovery failed; using fallback schema list")
    SCHEMA_MODULES = [
        "api.schemas.envelope",
        "api.schemas.compute",
    ]


def python_type_to_ts(annotation: Any, seen: Set[str] | None = None) -> str:
    """Convert a Python type annotation to a TypeScript type string."""
    if seen is None:
        seen = set()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[X] → X | undefined
    if origin is type(None):
        return "null"

    # Handle None
    if annotation is type(None):
        return "null"

    # Handle Union types (Optional is Union[X, None])
    if origin is type(Optional[str]):
        pass

    # Check for Union explicitly
    try:
        from typing import Union
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                # Optional[X] case
                return f"{python_type_to_ts(non_none[0], seen)} | undefined"
            return " | ".join(python_type_to_ts(a, seen) for a in non_none)
    except ImportError:
        pass

    # Handle List[X] → X[]
    if origin is list or origin is List:
        if args:
            return f"{python_type_to_ts(args[0], seen)}[]"
        return "unknown[]"

    # Handle Dict[K, V] → Record<K, V>
    if origin is dict or origin is Dict:
        if args and len(args) == 2:
            k = python_type_to_ts(args[0], seen)
            v = python_type_to_ts(args[1], seen)
            return f"Record<{k}, {v}>"
        return "Record<string, unknown>"

    # Primitives
    if annotation is str:
        return "string"
    if annotation is int:
        return "number"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is datetime:
        return "string"
    if annotation is Any:
        return "unknown"

    # Pydantic model reference
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    # Fallback
    name = getattr(annotation, "__name__", str(annotation))
    return name


def model_to_ts(model: type[BaseModel]) -> str:
    """Convert a Pydantic model class to a TypeScript interface."""
    lines = [f"export interface {model.__name__} {{"]

    for field_name, field_info in model.model_fields.items():
        ts_type = python_type_to_ts(field_info.annotation)
        optional = "?" if not field_info.is_required() else ""
        lines.append(f"  {field_name}{optional}: {ts_type}")

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    print("/**")
    print(" * AUTO-GENERATED from Pydantic schemas — do not edit manually.")
    print(f" * Run: python scripts/generate_types.py > frontend/src/types/generated.ts")
    print(" */")
    print()

    seen_models: Set[str] = set()

    for mod_path in SCHEMA_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            print(f"// Skipped {mod_path}: {e}")
            continue

        print(f"// ── {mod_path} ──")
        print()

        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
                and name not in seen_models
            ):
                seen_models.add(name)
                print(model_to_ts(obj))
                print()


if __name__ == "__main__":
    main()
