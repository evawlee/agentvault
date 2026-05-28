from __future__ import annotations

from typing import Any

from agentvault.filters.expression_parser import FilterExpressionParser
from agentvault.store.inmem_vector_store import InMemoryVectorStore
from agentvault.types.document import Document
from agentvault.types.query import SearchQuery
from agentvault.embed.embedder import HashEmbedder


class SearchPlugin:
    def __init__(
        self,
        store: InMemoryVectorStore | None = None,
        embedder: HashEmbedder | None = None,
    ) -> None:
        self._store = store or InMemoryVectorStore()
        self._embedder = embedder or HashEmbedder(dimension=8)
        self._parser = FilterExpressionParser()

    @property
    def store(self) -> InMemoryVectorStore:
        return self._store

    def ingest(self, doc_id: str, text: str, **metadata: Any) -> Document:
        tags = metadata.pop("tags", []) or []
        embedding = self._embedder.embed(text)
        doc = Document(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
            tags=list(tags),
        )
        self._store.add(doc)
        return doc

    def search(self, query: SearchQuery) -> list[Document]:
        embedding = self._embedder.embed(query.text)
        return self._store.search(
            embedding,
            filter_expression=query.filter,
            top_k=query.top_k,
        )

    def filter_only(self, expression: str) -> list[Document]:
        return self._store.filter_documents(expression)

    def evaluate_against(self, expression: str, doc_id: str) -> Any:
        doc = self._store.get(doc_id)
        record = InMemoryVectorStore._document_to_record(doc)
        return self._parser.evaluate(expression, record)
