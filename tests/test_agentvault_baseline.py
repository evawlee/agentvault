import pytest

from agentvault import (
    Document,
    HashEmbedder,
    InMemoryVectorStore,
    SearchPlugin,
    SearchQuery,
    StoreError,
    FilterValidationError,
)
from agentvault.embed.similarity import cosine_similarity, euclidean_distance, l2_normalize
from agentvault.runtime.safe_globals import build_filter_globals, allowed_builtin_names
from agentvault.util.metrics import CounterRegistry


@pytest.fixture
def populated_store():
    store = InMemoryVectorStore()
    store.add(Document(id="alpha", text="alpha doc", embedding=[1.0, 0.0, 0.0], metadata={"score": 10, "kind": "blog"}))
    store.add(Document(id="bravo", text="bravo doc", embedding=[0.0, 1.0, 0.0], metadata={"score": 25, "kind": "news"}))
    store.add(Document(id="charlie", text="charlie doc", embedding=[0.0, 0.0, 1.0], metadata={"score": 7, "kind": "blog"}))
    return store


class TestStoreBasics:
    def test_add_then_get(self):
        store = InMemoryVectorStore()
        doc = Document(id="a", text="hello", embedding=[0.5, 0.5])
        store.add(doc)
        assert store.get("a").text == "hello"
        assert store.count() == 1

    def test_add_duplicate_raises(self):
        store = InMemoryVectorStore()
        store.add(Document(id="a", text="x", embedding=[1.0]))
        with pytest.raises(StoreError):
            store.add(Document(id="a", text="y", embedding=[1.0]))

    def test_remove(self):
        store = InMemoryVectorStore()
        store.add(Document(id="a", text="x", embedding=[1.0]))
        store.remove("a")
        assert store.count() == 0
        with pytest.raises(StoreError):
            store.get("a")

    def test_add_many(self):
        store = InMemoryVectorStore()
        store.add_many([
            Document(id="a", text="aaa", embedding=[1.0]),
            Document(id="b", text="bbb", embedding=[0.0]),
        ])
        assert store.count() == 2


class TestFilterBasics:
    def test_simple_score_filter(self, populated_store):
        kept = populated_store.filter_documents("score > 8")
        ids = sorted(d.id for d in kept)
        assert ids == ["alpha", "bravo"]

    def test_kind_equality(self, populated_store):
        kept = populated_store.filter_documents("kind == 'blog'")
        ids = sorted(d.id for d in kept)
        assert ids == ["alpha", "charlie"]

    def test_boolean_combination(self, populated_store):
        kept = populated_store.filter_documents("kind == 'blog' and score >= 10")
        ids = sorted(d.id for d in kept)
        assert ids == ["alpha"]

    def test_empty_filter_raises(self, populated_store):
        with pytest.raises(ValueError):
            populated_store.filter_documents("")

    def test_non_string_filter_raises(self, populated_store):
        with pytest.raises(TypeError):
            populated_store.filter_documents(None)

    def test_normalized_whitespace(self, populated_store):
        kept = populated_store.filter_documents("   score   >   5  ")
        assert len(kept) == 3


class TestSearch:
    def test_search_returns_top_k(self, populated_store):
        result = populated_store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(result) == 2
        assert result[0].id == "alpha"

    def test_search_with_filter(self, populated_store):
        result = populated_store.search(
            [0.5, 0.5, 0.0],
            filter_expression="kind == 'news'",
            top_k=3,
        )
        ids = [d.id for d in result]
        assert ids == ["bravo"]


class TestEmbedder:
    def test_deterministic(self):
        e = HashEmbedder(dimension=8)
        v1 = e.embed("hello")
        v2 = e.embed("hello")
        assert v1 == v2

    def test_different_inputs_differ(self):
        e = HashEmbedder(dimension=8)
        assert e.embed("alpha") != e.embed("beta")

    def test_embed_batch(self):
        e = HashEmbedder(dimension=4)
        vectors = e.embed_batch(["a", "b", "c"])
        assert len(vectors) == 3
        assert all(len(v) == 4 for v in vectors)


class TestSimilarity:
    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_euclidean_zero(self):
        assert euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_l2_normalize(self):
        v = l2_normalize([3.0, 4.0])
        assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


class TestSearchPlugin:
    def test_ingest_and_search(self):
        plugin = SearchPlugin()
        plugin.ingest("a", "the brown fox", topic="animals")
        plugin.ingest("b", "the red ball", topic="toys")
        query = SearchQuery(text="brown fox", top_k=1)
        result = plugin.search(query)
        assert result[0].id == "a"

    def test_filter_only_path(self):
        plugin = SearchPlugin()
        plugin.ingest("a", "doc a", region="us")
        plugin.ingest("b", "doc b", region="eu")
        kept = plugin.filter_only("region == 'eu'")
        assert [d.id for d in kept] == ["b"]


class TestSafeGlobals:
    def test_allowed_builtins_present(self):
        scope = build_filter_globals()
        names = scope["__builtins__"]
        assert "len" in names
        assert "True" in names

    def test_builtin_names_sorted(self):
        names = allowed_builtin_names()
        assert names == tuple(sorted(names))


class TestMetrics:
    def test_inc_returns_running_total(self):
        reg = CounterRegistry()
        reg.inc("hits")
        reg.inc("hits", 3)
        assert reg.get("hits") == 4

    def test_reset_specific(self):
        reg = CounterRegistry()
        reg.inc("a")
        reg.inc("b")
        reg.reset("a")
        assert reg.get("a") == 0
        assert reg.get("b") == 1
