from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from agentvault.security.sanitizer import normalize_filter_text
from agentvault.runtime.safe_globals import build_filter_globals
from agentvault.types.exceptions import FilterValidationError


class FilterExpressionParser:
    def __init__(self, field_resolver: Callable[[str], Any] | None = None) -> None:
        self._field_resolver = field_resolver
        self._globals = build_filter_globals()

    def compile(self, expression: str) -> Callable[[Mapping[str, Any]], bool]:
        text = normalize_filter_text(expression)
        compiled = compile(text, "<filter>", "eval")

        def predicate(record: Mapping[str, Any]) -> bool:
            local_scope = dict(record)
            return bool(eval(compiled, self._globals, local_scope))

        return predicate

    def evaluate(self, expression: str, record: Mapping[str, Any]) -> Any:
        text = normalize_filter_text(expression)
        local_scope = dict(record)
        if self._field_resolver is not None:
            local_scope.setdefault("__resolve", self._field_resolver)
        return eval(text, self._globals, local_scope)

    def select(
        self,
        expression: str,
        records: Iterable[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        predicate = self.compile(expression)
        return [r for r in records if predicate(r)]
