"""Tests for search analytics and IR evaluation metrics."""

import pytest
from search_analytics import (
    IRMetrics,
    QueryAnalysis,
    SearchResult,
    SearchSessionTracker,
    analyze_query,
    evaluate_queries,
)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _make_results(*doc_ids: str) -> list[SearchResult]:
    """Create SearchResult list from doc IDs in rank order."""
    return [SearchResult(doc_id=did, rank=i + 1, score=10 - i) for i, did in enumerate(doc_ids)]


# ─── Precision@K Tests ───────────────────────────────────────────────────────

class TestPrecisionAtK:

    def test_all_relevant(self):
        results = _make_results("a", "b", "c")
        relevant = {"a", "b", "c"}
        assert IRMetrics.precision_at_k(results, relevant, 3) == 1.0

    def test_none_relevant(self):
        results = _make_results("a", "b", "c")
        relevant = {"x", "y"}
        assert IRMetrics.precision_at_k(results, relevant, 3) == 0.0

    def test_partial_relevant(self):
        results = _make_results("a", "b", "c", "d")
        relevant = {"a", "c"}
        assert IRMetrics.precision_at_k(results, relevant, 4) == 0.5

    def test_k_greater_than_results(self):
        results = _make_results("a", "b")
        relevant = {"a", "b"}
        # Only 2 results but k=5 — precision = 2/5
        assert IRMetrics.precision_at_k(results, relevant, 5) == pytest.approx(0.4)

    def test_k_zero(self):
        results = _make_results("a")
        assert IRMetrics.precision_at_k(results, {"a"}, 0) == 0.0

    def test_empty_results(self):
        assert IRMetrics.precision_at_k([], {"a"}, 5) == 0.0


# ─── Recall@K Tests ──────────────────────────────────────────────────────────

class TestRecallAtK:

    def test_all_found(self):
        results = _make_results("a", "b", "c")
        relevant = {"a", "b"}
        assert IRMetrics.recall_at_k(results, relevant, 3) == 1.0

    def test_none_found(self):
        results = _make_results("x", "y")
        relevant = {"a", "b"}
        assert IRMetrics.recall_at_k(results, relevant, 2) == 0.0

    def test_partial_found(self):
        results = _make_results("a", "x", "b", "y")
        relevant = {"a", "b", "c"}
        assert IRMetrics.recall_at_k(results, relevant, 4) == pytest.approx(2 / 3)

    def test_empty_relevant(self):
        results = _make_results("a")
        assert IRMetrics.recall_at_k(results, set(), 1) == 0.0


# ─── F1@K Tests ──────────────────────────────────────────────────────────────

class TestF1AtK:

    def test_perfect_score(self):
        results = _make_results("a", "b")
        relevant = {"a", "b"}
        assert IRMetrics.f1_at_k(results, relevant, 2) == 1.0

    def test_zero_score(self):
        results = _make_results("x", "y")
        relevant = {"a", "b"}
        assert IRMetrics.f1_at_k(results, relevant, 2) == 0.0

    def test_harmonic_mean(self):
        results = _make_results("a", "x", "y", "z")
        relevant = {"a", "b"}
        f1 = IRMetrics.f1_at_k(results, relevant, 4)
        p = IRMetrics.precision_at_k(results, relevant, 4)
        r = IRMetrics.recall_at_k(results, relevant, 4)
        expected = 2 * p * r / (p + r)
        assert f1 == pytest.approx(expected)


# ─── Reciprocal Rank Tests ───────────────────────────────────────────────────

class TestReciprocalRank:

    def test_first_result_relevant(self):
        results = _make_results("a", "b", "c")
        assert IRMetrics.reciprocal_rank(results, {"a"}) == 1.0

    def test_second_result_relevant(self):
        results = _make_results("x", "a", "b")
        assert IRMetrics.reciprocal_rank(results, {"a"}) == 0.5

    def test_third_result_relevant(self):
        results = _make_results("x", "y", "a")
        assert IRMetrics.reciprocal_rank(results, {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        results = _make_results("x", "y", "z")
        assert IRMetrics.reciprocal_rank(results, {"a"}) == 0.0

    def test_empty_results(self):
        assert IRMetrics.reciprocal_rank([], {"a"}) == 0.0


# ─── Average Precision Tests ─────────────────────────────────────────────────

class TestAveragePrecision:

    def test_perfect_ranking(self):
        results = _make_results("a", "b", "c")
        relevant = {"a", "b", "c"}
        assert IRMetrics.average_precision(results, relevant) == 1.0

    def test_mixed_ranking(self):
        results = _make_results("a", "x", "b", "y")
        relevant = {"a", "b"}
        # P@1=1, P@3=2/3 → AP = (1 + 2/3) / 2 = 5/6
        assert IRMetrics.average_precision(results, relevant) == pytest.approx(5 / 6)

    def test_empty_relevant(self):
        results = _make_results("a", "b")
        assert IRMetrics.average_precision(results, set()) == 0.0


# ─── NDCG Tests ──────────────────────────────────────────────────────────────

class TestNDCG:

    def test_perfect_ndcg(self):
        results = _make_results("a", "b")
        grades = {"a": 2, "b": 1}
        assert IRMetrics.ndcg_at_k(results, grades, 2) == pytest.approx(1.0)

    def test_reversed_ndcg(self):
        # Less relevant doc ranked first
        results = _make_results("b", "a")
        grades = {"a": 2, "b": 1}
        ndcg = IRMetrics.ndcg_at_k(results, grades, 2)
        assert 0 < ndcg < 1.0

    def test_no_relevant_docs(self):
        results = _make_results("x", "y")
        grades = {"a": 2}
        assert IRMetrics.ndcg_at_k(results, grades, 2) == 0.0

    def test_k_zero(self):
        results = _make_results("a")
        assert IRMetrics.ndcg_at_k(results, {"a": 1}, 0) == 0.0


# ─── Batch Evaluation Tests ─────────────────────────────────────────────────

class TestBatchEvaluation:

    def test_single_query(self):
        query_results = {"q1": _make_results("a", "b", "c")}
        judgments = {"q1": {"a", "c"}}
        result = evaluate_queries(query_results, judgments)

        assert result["aggregate"]["total_queries"] == 1
        assert result["aggregate"]["mrr"] == 1.0  # first result relevant
        assert len(result["per_query"]) == 1

    def test_multiple_queries(self):
        query_results = {
            "q1": _make_results("a", "x"),
            "q2": _make_results("y", "b"),
        }
        judgments = {
            "q1": {"a"},
            "q2": {"b"},
        }
        result = evaluate_queries(query_results, judgments)
        assert result["aggregate"]["total_queries"] == 2
        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert result["aggregate"]["mrr"] == pytest.approx(0.75)

    def test_empty_queries(self):
        result = evaluate_queries({}, {})
        assert result["aggregate"]["total_queries"] == 0


# ─── Query Analysis Tests ────────────────────────────────────────────────────

class TestQueryAnalysis:

    def test_simple_query(self):
        analysis = analyze_query("recent cases")
        assert analysis.token_count == 2
        assert analysis.estimated_complexity == "simple"

    def test_complex_query(self):
        analysis = analyze_query(
            '"fundamental rights" AND constitutional jurisdiction NOT habeas corpus'
        )
        assert analysis.has_boolean
        assert analysis.has_phrase
        assert analysis.estimated_complexity == "complex"
        assert len(analysis.legal_terms_found) > 0

    def test_field_filter_detection(self):
        analysis = analyze_query("court:supreme murder case")
        assert analysis.has_field_filter

    def test_legal_terms_detected(self):
        analysis = analyze_query("habeas corpus petition")
        assert "habeas corpus" in analysis.legal_terms_found

    def test_empty_query(self):
        analysis = analyze_query("")
        assert analysis.token_count == 0
        assert analysis.estimated_complexity == "simple"


# ─── Session Tracker Tests ───────────────────────────────────────────────────

class TestSearchSessionTracker:

    def setup_method(self):
        self.tracker = SearchSessionTracker()

    def test_record_search(self):
        idx = self.tracker.record_search("bail murder", num_results=15, top_score=8.5)
        assert idx == 0

    def test_record_click(self):
        idx = self.tracker.record_search("test query", num_results=10)
        self.tracker.record_click(idx, clicked_rank=3)
        summary = self.tracker.get_summary()
        assert summary["click_through_rate"] == 1.0

    def test_zero_result_rate(self):
        self.tracker.record_search("good query", num_results=10)
        self.tracker.record_search("bad query", num_results=0)
        summary = self.tracker.get_summary()
        assert summary["zero_result_rate"] == 0.5

    def test_empty_session(self):
        summary = self.tracker.get_summary()
        assert summary["total_searches"] == 0

    def test_reset(self):
        self.tracker.record_search("q1", num_results=5)
        self.tracker.record_search("q2", num_results=3)
        self.tracker.reset()
        assert self.tracker.get_summary()["total_searches"] == 0

    def test_multiple_searches(self):
        for i in range(5):
            self.tracker.record_search(f"query {i}", num_results=i * 2)
        summary = self.tracker.get_summary()
        assert summary["total_searches"] == 5
        assert summary["avg_results_per_query"] == pytest.approx(4.0)

    def test_avg_click_rank(self):
        idx1 = self.tracker.record_search("q1", num_results=10)
        self.tracker.record_click(idx1, clicked_rank=1)
        idx2 = self.tracker.record_search("q2", num_results=10)
        self.tracker.record_click(idx2, clicked_rank=5)
        summary = self.tracker.get_summary()
        assert summary["avg_click_rank"] == 3.0
