"""
Search Analytics — Evaluation Metrics and Query Analysis.

Provides IR evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
and query complexity scoring for search quality monitoring.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ─── IR Evaluation Metrics ───────────────────────────────────────────────────

@dataclass
class RelevanceJudgment:
    """A single relevance judgment for a query-document pair."""
    query_id: str
    doc_id: str
    relevance: int = 1  # 0 = not relevant, 1 = relevant, 2+ = highly relevant


@dataclass
class SearchResult:
    """A single search result with its rank position."""
    doc_id: str
    rank: int
    score: float = 0.0


class IRMetrics:
    """
    Information Retrieval evaluation metrics.

    Computes standard metrics for measuring search quality
    against ground-truth relevance judgments.
    """

    @staticmethod
    def precision_at_k(results: list[SearchResult], relevant: set[str], k: int) -> float:
        """
        Precision@K — fraction of top-K results that are relevant.

        Args:
            results: Ranked search results.
            relevant: Set of relevant document IDs.
            k: Number of top results to consider.

        Returns:
            Precision score between 0.0 and 1.0.
        """
        if k <= 0:
            return 0.0
        top_k = results[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for r in top_k if r.doc_id in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(results: list[SearchResult], relevant: set[str], k: int) -> float:
        """
        Recall@K — fraction of relevant docs found in top-K results.

        Args:
            results: Ranked search results.
            relevant: Set of relevant document IDs.
            k: Number of top results to consider.

        Returns:
            Recall score between 0.0 and 1.0.
        """
        if not relevant:
            return 0.0
        top_k = results[:k]
        hits = sum(1 for r in top_k if r.doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def f1_at_k(results: list[SearchResult], relevant: set[str], k: int) -> float:
        """
        F1@K — harmonic mean of Precision@K and Recall@K.

        Returns:
            F1 score between 0.0 and 1.0.
        """
        p = IRMetrics.precision_at_k(results, relevant, k)
        r = IRMetrics.recall_at_k(results, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @staticmethod
    def reciprocal_rank(results: list[SearchResult], relevant: set[str]) -> float:
        """
        Reciprocal Rank — 1/rank of the first relevant result.

        Used in Mean Reciprocal Rank (MRR) across queries.

        Returns:
            RR between 0.0 and 1.0 (0 if no relevant result found).
        """
        for i, result in enumerate(results):
            if result.doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(results: list[SearchResult], relevant: set[str]) -> float:
        """
        Average Precision — area under the precision-recall curve.

        Used in Mean Average Precision (MAP) across queries.

        Returns:
            AP between 0.0 and 1.0.
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precision = 0.0

        for i, result in enumerate(results):
            if result.doc_id in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precision += precision_at_i

        return sum_precision / len(relevant)

    @staticmethod
    def ndcg_at_k(
        results: list[SearchResult],
        relevance_scores: dict[str, int],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        Considers graded relevance: highly relevant docs ranked higher
        contribute more to the score.

        Args:
            results: Ranked search results.
            relevance_scores: Mapping of doc_id → relevance grade (0, 1, 2, ...).
            k: Cutoff rank.

        Returns:
            NDCG score between 0.0 and 1.0.
        """
        if k <= 0:
            return 0.0

        # DCG for actual results
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            rel = relevance_scores.get(result.doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 2)

        # Ideal DCG (sort all relevances descending)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1) / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg


# ─── Batch Evaluation ────────────────────────────────────────────────────────

@dataclass
class QueryEvaluation:
    """Evaluation results for a single query."""
    query_id: str
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    reciprocal_rank: float = 0.0
    average_precision: float = 0.0
    ndcg_at_10: float = 0.0
    num_results: int = 0
    num_relevant: int = 0


def evaluate_queries(
    query_results: dict[str, list[SearchResult]],
    judgments: dict[str, set[str]],
    relevance_grades: Optional[dict[str, dict[str, int]]] = None,
) -> dict:
    """
    Evaluate multiple queries and compute aggregate metrics.

    Args:
        query_results: Mapping query_id → list of SearchResults.
        judgments: Mapping query_id → set of relevant doc_ids.
        relevance_grades: Optional graded relevance per query
                          (query_id → {doc_id: grade}).

    Returns:
        Dict with per-query evaluations and aggregate metrics
        (MAP, MRR, avg NDCG@10).
    """
    evaluations: list[QueryEvaluation] = []

    for qid, results in query_results.items():
        relevant = judgments.get(qid, set())
        grades = (relevance_grades or {}).get(qid, {r: 1 for r in relevant})

        ev = QueryEvaluation(
            query_id=qid,
            precision_at_5=IRMetrics.precision_at_k(results, relevant, 5),
            precision_at_10=IRMetrics.precision_at_k(results, relevant, 10),
            recall_at_10=IRMetrics.recall_at_k(results, relevant, 10),
            reciprocal_rank=IRMetrics.reciprocal_rank(results, relevant),
            average_precision=IRMetrics.average_precision(results, relevant),
            ndcg_at_10=IRMetrics.ndcg_at_k(results, grades, 10),
            num_results=len(results),
            num_relevant=len(relevant),
        )
        evaluations.append(ev)

    n = len(evaluations) or 1

    return {
        "per_query": evaluations,
        "aggregate": {
            "map": sum(e.average_precision for e in evaluations) / n,
            "mrr": sum(e.reciprocal_rank for e in evaluations) / n,
            "avg_precision_at_5": sum(e.precision_at_5 for e in evaluations) / n,
            "avg_precision_at_10": sum(e.precision_at_10 for e in evaluations) / n,
            "avg_recall_at_10": sum(e.recall_at_10 for e in evaluations) / n,
            "avg_ndcg_at_10": sum(e.ndcg_at_10 for e in evaluations) / n,
            "total_queries": len(evaluations),
        },
    }


# ─── Query Complexity Analysis ───────────────────────────────────────────────

@dataclass
class QueryAnalysis:
    """Analysis of a search query's characteristics."""
    raw_query: str
    token_count: int = 0
    has_boolean: bool = False
    has_phrase: bool = False
    has_field_filter: bool = False
    estimated_complexity: str = "simple"
    legal_terms_found: list[str] = field(default_factory=list)


# Legal vocabulary for detection
LEGAL_VOCABULARY = frozenset({
    "jurisdiction", "precedent", "stare decisis", "ratio decidendi",
    "obiter dictum", "habeas corpus", "mandamus", "certiorari",
    "ultra vires", "prima facie", "res judicata", "sub judice",
    "injunction", "tort", "plaintiff", "defendant", "appellant",
    "respondent", "petitioner", "prosecution", "acquittal",
    "conviction", "bail", "remand", "quash", "appeal",
    "constitutional", "fundamental rights", "due process",
    "judicial review", "writ petition", "contempt of court",
})


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a search query for complexity and characteristics.

    Useful for:
    - Routing simple vs complex queries to different search strategies
    - Logging query patterns for search improvement
    - User feedback on query reformulation
    """
    analysis = QueryAnalysis(raw_query=query)

    tokens = query.lower().split()
    analysis.token_count = len(tokens)

    # Boolean operators
    boolean_ops = {"and", "or", "not", "and not", "or not"}
    if any(op in query.lower() for op in boolean_ops):
        analysis.has_boolean = True

    # Quoted phrases
    if '"' in query or "'" in query:
        analysis.has_phrase = True

    # Field filters (e.g., court:supreme, year:2024)
    if ":" in query:
        analysis.has_field_filter = True

    # Legal terms
    query_lower = query.lower()
    for term in LEGAL_VOCABULARY:
        if term in query_lower:
            analysis.legal_terms_found.append(term)

    # Complexity estimation
    complexity_score = 0
    complexity_score += min(analysis.token_count, 5)  # More tokens = more complex
    complexity_score += 2 if analysis.has_boolean else 0
    complexity_score += 2 if analysis.has_phrase else 0
    complexity_score += 1 if analysis.has_field_filter else 0
    complexity_score += len(analysis.legal_terms_found)

    if complexity_score <= 2:
        analysis.estimated_complexity = "simple"
    elif complexity_score <= 5:
        analysis.estimated_complexity = "moderate"
    else:
        analysis.estimated_complexity = "complex"

    return analysis


# ─── Search Session Tracker ──────────────────────────────────────────────────

@dataclass
class SearchEvent:
    """A single event in a search session."""
    query: str
    num_results: int
    top_score: float = 0.0
    clicked_rank: Optional[int] = None


class SearchSessionTracker:
    """
    Track search sessions for analysis.

    Records queries, results, and click-through behavior
    to measure search effectiveness over time.
    """

    def __init__(self):
        self._events: list[SearchEvent] = []

    def record_search(
        self,
        query: str,
        num_results: int,
        top_score: float = 0.0,
    ) -> int:
        """Record a search event. Returns event index."""
        event = SearchEvent(
            query=query,
            num_results=num_results,
            top_score=top_score,
        )
        self._events.append(event)
        return len(self._events) - 1

    def record_click(self, event_index: int, clicked_rank: int):
        """Record a click on a search result."""
        if 0 <= event_index < len(self._events):
            self._events[event_index].clicked_rank = clicked_rank

    def get_summary(self) -> dict:
        """Summarize the search session."""
        if not self._events:
            return {"total_searches": 0}

        zero_results = sum(1 for e in self._events if e.num_results == 0)
        clicks = [e for e in self._events if e.clicked_rank is not None]
        avg_click_rank = (
            sum(e.clicked_rank for e in clicks) / len(clicks)
            if clicks
            else None
        )

        return {
            "total_searches": len(self._events),
            "zero_result_rate": round(zero_results / len(self._events), 3),
            "click_through_rate": round(len(clicks) / len(self._events), 3),
            "avg_click_rank": round(avg_click_rank, 2) if avg_click_rank else None,
            "avg_results_per_query": round(
                sum(e.num_results for e in self._events) / len(self._events), 1
            ),
            "queries": [e.query for e in self._events],
        }

    def reset(self):
        """Clear all tracked events."""
        self._events.clear()
