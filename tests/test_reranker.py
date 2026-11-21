from datetime import datetime, timedelta, timezone
import pytest
from app.services.reranker import RerankerWeights, rerank


@pytest.fixture
def base_now():
    return datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)


def test_reranker_half_life(base_now):
    """At exactly half_life_days, T factor should be exactly 0.5."""
    half_life = 30.0
    doc = {
        "id": "doc1",
        "rrf_score": 0.03,
        "occurred_at": base_now - timedelta(days=half_life),
        "category": "customs",
        "severity_level": 3,
    }

    results = rerank([doc], now=base_now, half_life_days=half_life)
    assert len(results) == 1
    factors = results[0]["factors"]
    assert pytest.approx(factors["t"], abs=1e-4) == 0.5


def test_reranker_future_date_age_zero(base_now):
    """If occurred_at is in the future, age is clamped to 0 days (T = 1.0)."""
    doc = {
        "id": "doc1",
        "rrf_score": 0.03,
        "occurred_at": base_now + timedelta(days=5),
        "severity_level": 1,
    }
    results = rerank([doc], now=base_now)
    assert results[0]["factors"]["t"] == 1.0


def test_reranker_category_match_and_absent_renormalization(base_now):
    """When category is provided, match gives C=1.0, mismatch C=0.0.
    When category is absent/None, weight_c is dropped and active weights sum to 1.0."""
    docs = [
        {"id": "d1", "rrf_score": 0.03, "category": "customs", "severity_level": 3},
        {"id": "d2", "rrf_score": 0.03, "category": "delay", "severity_level": 3},
    ]

    # Category provided
    res_matched = rerank(docs, query_category="customs", now=base_now)
    assert res_matched[0]["id"] == "d1"
    assert res_matched[0]["factors"]["c"] == 1.0
    assert res_matched[1]["factors"]["c"] == 0.0

    # Category absent
    res_no_cat = rerank(docs, query_category=None, now=base_now)
    for r in res_no_cat:
        assert r["factors"]["c"] == 0.0
        active_w = r["active_weights"]
        assert active_w["w_c"] == 0.0
        # Active weights sum to 1.0
        total_w = active_w["w_v"] + active_w["w_t"] + active_w["w_s"]
        assert pytest.approx(total_w, abs=1e-3) == 1.0


def test_reranker_severity_levels(base_now):
    """Severity 1 -> S = 0.0; Severity 5 -> S = 1.0; Severity 3 -> S = 0.5."""
    docs = [
        {"id": "low", "rrf_score": 0.02, "severity_level": 1},
        {"id": "mid", "rrf_score": 0.02, "severity_level": 3},
        {"id": "high", "rrf_score": 0.02, "severity_level": 5},
    ]
    res = rerank(docs, now=base_now)
    factors_by_id = {d["id"]: d["factors"]["s"] for d in res}
    assert factors_by_id["low"] == 0.0
    assert pytest.approx(factors_by_id["mid"], abs=1e-4) == 0.5
    assert factors_by_id["high"] == 1.0


def test_reranker_weight_zero_disables_factor(base_now):
    """Weight 0 completely disables a factor."""
    weights = RerankerWeights(weight_v=1.0, weight_t=0.0, weight_c=0.0, weight_s=0.0)
    docs = [
        {"id": "d1", "rrf_score": 0.05, "severity_level": 1, "occurred_at": base_now - timedelta(days=100)},
        {"id": "d2", "rrf_score": 0.01, "severity_level": 5, "occurred_at": base_now},
    ]
    res = rerank(docs, query_category="customs", weights=weights, now=base_now)
    # Only V matters, so d1 with higher rrf_score must be first
    assert res[0]["id"] == "d1"
    assert res[0]["active_weights"]["w_v"] == 1.0
    assert res[0]["active_weights"]["w_t"] == 0.0
    assert res[0]["active_weights"]["w_c"] == 0.0
    assert res[0]["active_weights"]["w_s"] == 0.0


def test_reranker_min_max_normalization_edge_cases(base_now):
    """Single candidate or identical scores should have V = 1.0 without ZeroDivisionError."""
    # Single doc
    single = [{"id": "d1", "rrf_score": 0.04}]
    res_single = rerank(single, now=base_now)
    assert res_single[0]["factors"]["v"] == 1.0

    # Identical scores
    identical = [
        {"id": "d1", "rrf_score": 0.04},
        {"id": "d2", "rrf_score": 0.04},
    ]
    res_ident = rerank(identical, now=base_now)
    assert res_ident[0]["factors"]["v"] == 1.0
    assert res_ident[1]["factors"]["v"] == 1.0


def test_reranker_rank_movement_and_delta(base_now):
    """Verify rank_before, rank_after and rank_delta tracking in a candidate pool."""
    # d1: rank 1 from retrieval, old (-180d), low severity
    # d2: rank 2 from retrieval, fresh (today), high severity, matching category
    # d3: rank 3 baseline
    d1 = {
        "id": "d1",
        "rank_before": 1,
        "rrf_score": 0.033,
        "category": "other",
        "severity_level": 1,
        "occurred_at": base_now - timedelta(days=180),
    }
    d2 = {
        "id": "d2",
        "rank_before": 2,
        "rrf_score": 0.030,
        "category": "customs",
        "severity_level": 5,
        "occurred_at": base_now,
    }
    d3 = {
        "id": "d3",
        "rank_before": 3,
        "rrf_score": 0.010,
        "category": "other",
        "severity_level": 1,
        "occurred_at": base_now - timedelta(days=180),
    }
    res = rerank([d1, d2, d3], query_category="customs", now=base_now)

    # d2 jumps to rank 1
    assert res[0]["id"] == "d2"
    assert res[0]["rank_before"] == 2
    assert res[0]["rank_after"] == 1
    assert res[0]["rank_delta"] == 1  # 2 - 1 = +1 (moved up by 1)

    # d1 dropped to rank 2
    assert res[1]["id"] == "d1"
    assert res[1]["rank_before"] == 1
    assert res[1]["rank_after"] == 2
    assert res[1]["rank_delta"] == -1  # 1 - 2 = -1 (dropped by 1)


def test_reranker_stable_tie_breaking(base_now):
    """When final scores are equal, sort order is stable and deterministic."""
    d1 = {"id": "alpha", "rank_before": 1, "rrf_score": 0.02, "severity_level": 3}
    d2 = {"id": "beta", "rank_before": 2, "rrf_score": 0.02, "severity_level": 3}
    res = rerank([d1, d2], now=base_now)
    assert res[0]["id"] == "alpha"
    assert res[1]["id"] == "beta"
