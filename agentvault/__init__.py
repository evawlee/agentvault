from agentvault.types.document import Document
from agentvault.types.query import SearchQuery
from agentvault.types.exceptions import (
    AgentvaultError,
    StoreError,
    FilterValidationError,
    EmbeddingError,
)
from agentvault.store.inmem_vector_store import InMemoryVectorStore
from agentvault.search.search_plugin import SearchPlugin
from agentvault.embed.embedder import HashEmbedder

__version__ = "0.4.2"

__all__ = [
    "Document",
    "SearchQuery",
    "AgentvaultError",
    "StoreError",
    "FilterValidationError",
    "EmbeddingError",
    "InMemoryVectorStore",
    "SearchPlugin",
    "HashEmbedder",
    "__version__",
]
