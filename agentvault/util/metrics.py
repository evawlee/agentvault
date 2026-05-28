from __future__ import annotations

import threading
from typing import Iterator


class CounterRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, by: int = 1) -> int:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by
            return self._counters[name]

    def get(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._counters.clear()
            else:
                self._counters.pop(name, None)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def items(self) -> Iterator[tuple[str, int]]:
        return iter(self.snapshot().items())


_DEFAULT_REGISTRY = CounterRegistry()


def default_registry() -> CounterRegistry:
    return _DEFAULT_REGISTRY
