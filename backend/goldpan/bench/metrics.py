"""Statistical metrics and confidence intervals for GoldPan Bench."""

import hashlib
import math
from pathlib import Path
from typing import Sequence


def wilson_ci_95(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Compute point estimate and 95% Wilson score confidence interval for a binomial proportion.

    Returns:
        (point_estimate, lower_bound, upper_bound) in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = max(0.0, min(1.0, successes / total))
    n = float(total)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return round(p, 4), round(lower, 4), round(upper, 4)


def compute_recall_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int = 3) -> float:
    """Compute binary Recall@k (1.0 if at least one relevant doc is in top-k, else 0.0)."""
    if not relevant_doc_ids:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    rel_set = set(relevant_doc_ids)
    return 1.0 if (top_k & rel_set) else 0.0


def compute_mrr_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int = 5) -> float:
    """Compute Reciprocal Rank at k."""
    if not relevant_doc_ids:
        return 0.0
    rel_set = set(relevant_doc_ids)
    for idx, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in rel_set:
            return 1.0 / float(idx + 1)
    return 0.0


def compute_dataset_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a frozen benchmark dataset file."""
    file_path = Path(path)
    data = file_path.read_bytes()
    return hashlib.sha256(data).hexdigest()
