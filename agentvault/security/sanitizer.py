from __future__ import annotations

import re


_DUNDER_DENYLIST = (
    "__class__",
    "__mro__",
    "__subclasses__",
    "__import__",
    "__builtins__",
)


_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_filter_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("filter expression must be a string")
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()
    if not collapsed:
        raise ValueError("filter expression is empty")
    for token in _DUNDER_DENYLIST:
        if token in collapsed:
            raise ValueError(f"reserved token in filter: {token}")
    return collapsed


def is_safe_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))
