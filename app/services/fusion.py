from typing import Any


def _get_id(doc: Any) -> str:
    """Extract string ID from dict or object."""
    if isinstance(doc, dict):
        val = doc.get("id")
    else:
        val = getattr(doc, "id", None)
    if val is None:
        raise ValueError(f"Document missing 'id': {doc}")
    return str(val)


def _copy_doc(doc: Any) -> dict[str, Any]:
    """Convert candidate document to mutable dict."""
    if isinstance(doc, dict):
        return dict(doc)
    if hasattr(doc, "__dict__"):
        data = {k: v for k, v in doc.__dict__.items() if not k.startswith("_")}
        return data
    return {"id": _get_id(doc), "raw": doc}


def reciprocal_rank_fusion(
    dense_results: list[Any],
    sparse_results: list[Any],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF).

    Formula:
        RRF(d) = sum_{r in {dense, sparse}} 1 / (k + rank_r(d))
    where rank starts at 1.

    Args:
        dense_results: Ranked list from dense (vector) retrieval.
        sparse_results: Ranked list from sparse (FTS) retrieval.
        k: Smoothing constant (default 60).

    Returns:
        Fused list of candidate dictionaries sorted descending by rrf_score.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive")

    fused_docs: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    # 1. Process dense retrieval ranks (1-based)
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = _get_id(doc)
        if doc_id not in fused_docs:
            fused_docs[doc_id] = _copy_doc(doc)
        dense_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # 2. Process sparse retrieval ranks (1-based)
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = _get_id(doc)
        if doc_id not in fused_docs:
            fused_docs[doc_id] = _copy_doc(doc)
        sparse_ranks[doc_id] = rank
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # 3. Assemble results with metadata
    results: list[dict[str, Any]] = []
    for doc_id, doc in fused_docs.items():
        doc["rrf_score"] = rrf_scores[doc_id]
        doc["dense_rank"] = dense_ranks.get(doc_id)
        doc["sparse_rank"] = sparse_ranks.get(doc_id)
        results.append(doc)

    # 4. Sort descending by rrf_score, tie-breaking stably by doc_id
    results.sort(key=lambda x: (-x["rrf_score"], str(x.get("id", ""))))

    # Assign fused rank_before
    for idx, doc in enumerate(results, start=1):
        doc["rank_before"] = idx

    return results
