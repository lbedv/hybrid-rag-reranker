from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class RerankerWeights:
    weight_v: float = 0.50
    weight_t: float = 0.25
    weight_c: float = 0.15
    weight_s: float = 0.10


def _get_val(doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _copy_doc(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        return dict(doc)
    if hasattr(doc, "__dict__"):
        return {k: v for k, v in doc.__dict__.items() if not k.startswith("_")}
    return {"id": getattr(doc, "id", None), "raw": doc}


def rerank(
    candidates: list[Any],
    query_category: str | None = None,
    weights: RerankerWeights | None = None,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Metadata Reranker (Pure Function).

    Scores candidates using 4 signals:
    - V: Min-max normalized RRF score (or base score) in [0, 1]
    - T: Exponential recency decay: 0.5 ** (age_days / half_life_days)
    - C: Category match: 1.0 if category matches query_category, else 0.0
    - S: Severity level: (severity_level - 1) / 4 in [0, 1]

    Dynamic weight re-normalization:
    If query_category is None/empty or any weight is 0.0, inactive factors are excluded,
    and active weights are re-scaled so sum(active_weights) == 1.0.

    Args:
        candidates: Candidate items with scores and metadata.
        query_category: Optional target category to boost.
        weights: RerankerWeights (default 0.50, 0.25, 0.15, 0.10).
        now: Reference datetime (defaults to UTC now). Injected for determinism.
        half_life_days: Half-life for recency decay in days (default 30.0).

    Returns:
        List of candidate dictionaries sorted descending by final_score.
    """
    if not candidates:
        return []

    if weights is None:
        weights = RerankerWeights()

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # 1. Determine active factors & re-normalize weights
    category_active = bool(query_category and query_category.strip())
    
    w_v = max(0.0, weights.weight_v)
    w_t = max(0.0, weights.weight_t)
    w_c = max(0.0, weights.weight_c) if category_active else 0.0
    w_s = max(0.0, weights.weight_s)

    sum_weights = w_v + w_t + w_c + w_s
    if sum_weights > 0:
        w_v /= sum_weights
        w_t /= sum_weights
        w_c /= sum_weights
        w_s /= sum_weights
    else:
        # Fallback if all weights are 0
        w_v = 1.0

    # 2. Min-max normalization for V factor
    raw_scores = [
        float(_get_val(doc, "rrf_score", _get_val(doc, "score", 0.0)))
        for doc in candidates
    ]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_range = max_score - min_score

    # 3. Calculate factors for each candidate
    scored_candidates: list[dict[str, Any]] = []

    for i, orig_doc in enumerate(candidates):
        doc = _copy_doc(orig_doc)
        
        # Original rank before reranking (1-based)
        rank_before = _get_val(orig_doc, "rank_before", i + 1)
        doc["rank_before"] = rank_before

        # Factor V: normalized score in [0, 1]
        if score_range > 0:
            v_factor = (raw_scores[i] - min_score) / score_range
        else:
            v_factor = 1.0

        # Factor T: recency decay
        occurred_at = _get_val(doc, "occurred_at")
        if occurred_at is not None:
            if isinstance(occurred_at, str):
                try:
                    occurred_at = datetime.fromisoformat(occurred_at)
                except ValueError:
                    occurred_at = None
        
        if occurred_at is not None:
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=timezone.utc)
            # Future dates have age 0
            age_seconds = max(0.0, (now - occurred_at).total_seconds())
            age_days = age_seconds / 86400.0
            if half_life_days > 0:
                t_factor = 0.5 ** (age_days / half_life_days)
            else:
                t_factor = 1.0
        else:
            t_factor = 0.5

        # Factor C: category match
        if category_active:
            doc_category = str(_get_val(doc, "category", "")).strip().lower()
            target_category = query_category.strip().lower()
            c_factor = 1.0 if doc_category == target_category else 0.0
        else:
            c_factor = 0.0

        # Factor S: severity level in [0, 1]
        sev = _get_val(doc, "severity_level", 1)
        try:
            sev_num = int(sev)
            s_factor = max(0.0, min(1.0, (sev_num - 1) / 4.0))
        except (ValueError, TypeError):
            s_factor = 0.0

        # Final composite score
        final_score = (w_v * v_factor) + (w_t * t_factor) + (w_c * c_factor) + (w_s * s_factor)

        doc["factors"] = {
            "v": round(v_factor, 4),
            "t": round(t_factor, 4),
            "c": round(c_factor, 4),
            "s": round(s_factor, 4),
        }
        doc["active_weights"] = {
            "w_v": round(w_v, 4),
            "w_t": round(w_t, 4),
            "w_c": round(w_c, 4),
            "w_s": round(w_s, 4),
        }
        doc["final_score"] = round(final_score, 6)
        scored_candidates.append(doc)

    # 4. Sort descending by final_score, breaking ties by rank_before
    scored_candidates.sort(
        key=lambda x: (-x["final_score"], x.get("rank_before", 0), str(x.get("id", "")))
    )

    # 5. Assign rank_after and rank_delta
    for rank_after, doc in enumerate(scored_candidates, start=1):
        doc["rank_after"] = rank_after
        doc["rank_delta"] = doc["rank_before"] - rank_after

    return scored_candidates
