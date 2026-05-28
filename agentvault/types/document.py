from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def with_metadata(self, **kwargs: Any) -> "Document":
        merged = dict(self.metadata)
        merged.update(kwargs)
        return Document(
            id=self.id,
            text=self.text,
            embedding=list(self.embedding),
            metadata=merged,
            tags=list(self.tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": list(self.embedding),
            "metadata": dict(self.metadata),
            "tags": list(self.tags),
        }
