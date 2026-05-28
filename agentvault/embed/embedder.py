from __future__ import annotations

import hashlib

from agentvault.types.exceptions import EmbeddingError


class HashEmbedder:
    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise EmbeddingError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise EmbeddingError("text must be a string")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector: list[float] = []
        for index in range(self._dimension):
            byte = digest[index % len(digest)]
            vector.append((byte / 255.0) * 2.0 - 1.0)
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
