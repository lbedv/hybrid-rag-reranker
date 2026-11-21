import math
from app.services.embedding import EmbeddingService


def test_embedding_exact_unit_norm():
    svc = EmbeddingService(api_key=None)
    emb = svc.get_embedding("Test incident content for unit norm")
    assert len(emb) == 1536
    # Exact L2 norm must be 1.0
    norm_sq = sum(x * x for x in emb)
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-7)


def test_embedding_api_key_whitespace_safety():
    # Whitespace-only keys must not instantiate client or set is_live = True
    svc_empty = EmbeddingService(api_key="")
    assert not svc_empty.is_live
    assert svc_empty.api_key is None

    svc_spaces = EmbeddingService(api_key="   ")
    assert not svc_spaces.is_live
    assert svc_spaces.api_key is None


def test_embedding_deterministic():
    svc = EmbeddingService(api_key=None)
    e1 = svc.get_embedding("customs clearance failure")
    e2 = svc.get_embedding("customs clearance failure")
    assert e1 == e2
