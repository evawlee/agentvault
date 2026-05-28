from __future__ import annotations

from typing import Any, Iterable

from agentvault.filters.expression_parser import FilterExpressionParser
from agentvault.types.document import Document
from agentvault.types.exceptions import StoreError
from agentvault.embed.similarity import cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._parser = FilterExpressionParser()

    def add(self, document: Document) -> None:
        if document.id in self._documents:
            raise StoreError(f"document already present: {document.id}")
        self._documents[document.id] = document

    def add_many(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self.add(doc)

    def get(self, doc_id: str) -> Document:
        if doc_id not in self._documents:
            raise StoreError(f"document not found: {doc_id}")
        return self._documents[doc_id]

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._documents:
            raise StoreError(f"document not found: {doc_id}")
        del self._documents[doc_id]

    def count(self) -> int:
        return len(self._documents)

    def all_documents(self) -> list[Document]:
        return list(self._documents.values())

    def search(
        self,
        query_vector: list[float],
        filter_expression: str | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        candidates: list[Document]
        if filter_expression:
            records = [self._document_to_record(d) for d in self._documents.values()]
            kept_records = self._parser.select(filter_expression, records)
            candidates = [self._documents[r["id"]] for r in kept_records]
        else:
            candidates = list(self._documents.values())

        ranked = sorted(
            candidates,
            key=lambda d: cosine_similarity(query_vector, d.embedding),
            reverse=True,
        )
        return ranked[:top_k]

    def filter_documents(self, filter_expression: str) -> list[Document]:
        records = [self._document_to_record(d) for d in self._documents.values()]
        kept_records = self._parser.select(filter_expression, records)
        return [self._documents[r["id"]] for r in kept_records]

    @staticmethod
    def _document_to_record(document: Document) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": document.id,
            "text": document.text,
            "embedding": document.embedding,
            "metadata": dict(document.metadata),
            "tags": list(document.tags),
        }
        for key, value in document.metadata.items():
            if key not in record:
                record[key] = value
        return record
