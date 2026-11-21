import pytest
from app.services.fusion import reciprocal_rank_fusion


def test_rrf_basic_fusion_and_scoring():
    dense = [{"id": "doc1", "title": "Doc 1"}, {"id": "doc2", "title": "Doc 2"}]
    sparse = [{"id": "doc2", "title": "Doc 2"}, {"id": "doc3", "title": "Doc 3"}]

    # k=60
    # doc1: dense rank 1 -> 1/61 = 0.016393
    # doc2: dense rank 2 (1/62) + sparse rank 1 (1/61) = 0.016129 + 0.016393 = 0.032522
    # doc3: sparse rank 2 -> 1/62 = 0.016129
    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert len(fused) == 3
    # doc2 must be #1 because it appeared in both lists
    assert fused[0]["id"] == "doc2"
    assert fused[0]["rank_before"] == 1
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["sparse_rank"] == 1
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-4) == (1 / 62) + (1 / 61)

    # doc1 is #2 (dense rank 1: 1/61)
    assert fused[1]["id"] == "doc1"
    assert fused[1]["dense_rank"] == 1
    assert fused[1]["sparse_rank"] is None
    assert pytest.approx(fused[1]["rrf_score"], rel=1e-4) == 1 / 61

    # doc3 is #3 (sparse rank 2: 1/62)
    assert fused[2]["id"] == "doc3"
    assert fused[2]["dense_rank"] is None
    assert fused[2]["sparse_rank"] == 2
    assert pytest.approx(fused[2]["rrf_score"], rel=1e-4) == 1 / 62


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], [], k=60) == []

    dense = [{"id": "doc1"}]
    res = reciprocal_rank_fusion(dense, [], k=60)
    assert len(res) == 1
    assert res[0]["id"] == "doc1"
    assert pytest.approx(res[0]["rrf_score"], rel=1e-4) == 1 / 61

    sparse = [{"id": "doc2"}]
    res2 = reciprocal_rank_fusion([], sparse, k=60)
    assert len(res2) == 1
    assert res2[0]["id"] == "doc2"
    assert pytest.approx(res2[0]["rrf_score"], rel=1e-4) == 1 / 61


def test_rrf_custom_k():
    dense = [{"id": "doc1"}]
    sparse = [{"id": "doc1"}]
    fused = reciprocal_rank_fusion(dense, sparse, k=10)
    # rank 1 in both: 1/11 + 1/11 = 2/11
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-4) == 2 / 11


def test_rrf_invalid_k():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([{"id": "doc1"}], [], k=0)


def test_rrf_tie_breaking_is_deterministic():
    # Two documents with identical RRF score
    dense = [{"id": "b"}, {"id": "a"}]
    sparse = [{"id": "a"}, {"id": "b"}]
    # Both have rank 1 + rank 2 = 1/61 + 1/62
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
    # Sorted stably by id
    assert fused[0]["id"] == "a"
    assert fused[1]["id"] == "b"
