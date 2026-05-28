from __future__ import annotations

from typing import Any


_ALLOWED_BUILTINS = {
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "any": any,
    "all": all,
    "round": round,
    "sorted": sorted,
    "set": set,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}


def build_filter_globals(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    scope: dict[str, Any] = {"__builtins__": dict(_ALLOWED_BUILTINS)}
    if extra:
        scope.update(extra)
    return scope


def allowed_builtin_names() -> tuple[str, ...]:
    return tuple(sorted(_ALLOWED_BUILTINS.keys()))
