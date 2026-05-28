from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchQuery:
    text: str
    top_k: int = 5
    filter: str | None = None

    def with_filter(self, expression: str) -> "SearchQuery":
        return SearchQuery(text=self.text, top_k=self.top_k, filter=expression)

    def with_top_k(self, k: int) -> "SearchQuery":
        if k <= 0:
            raise ValueError("top_k must be positive")
        return SearchQuery(text=self.text, top_k=k, filter=self.filter)
