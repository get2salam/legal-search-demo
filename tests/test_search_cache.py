"""
Tests for search_cache module.

Covers LRU eviction, TTL expiry, query frequency tracking,
thread safety, cache statistics, and edge cases.
"""

import threading
import time

import pytest

from search_cache import (
    CacheEntry,
    CacheStats,
    QueryFrequencyTracker,
    SearchCache,
    _estimate_size,
    _make_cache_key,
)


# ─── CacheEntry Tests ────────────────────────────────────────────────────────


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(
            key="abc",
            value="data",
            created_at=time.time(),
            last_accessed=time.time(),
            ttl_seconds=300.0,
        )
        assert not entry.is_expired

    def test_expired_after_ttl(self):
        entry = CacheEntry(
            key="abc",
            value="data",
            created_at=time.time() - 400,
            last_accessed=time.time() - 400,
            ttl_seconds=300.0,
        )
        assert entry.is_expired

    def test_no_expiry_with_zero_ttl(self):
        entry = CacheEntry(
            key="abc",
            value="data",
            created_at=time.time() - 999999,
            last_accessed=time.time(),
            ttl_seconds=0,
        )
        assert not entry.is_expired

    def test_touch_updates_access(self):
        entry = CacheEntry(
            key="abc",
            value="data",
            created_at=time.time() - 10,
            last_accessed=time.time() - 10,
            ttl_seconds=300.0,
            hit_count=0,
        )
        old_access = entry.last_accessed
        entry.touch()
        assert entry.last_accessed > old_access
        assert entry.hit_count == 1

    def test_age_seconds(self):
        entry = CacheEntry(
            key="abc",
            value="data",
            created_at=time.time() - 60,
            last_accessed=time.time(),
            ttl_seconds=300.0,
        )
        assert 59 <= entry.age_seconds <= 62


# ─── CacheStats Tests ────────────────────────────────────────────────────────


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_zero_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0
        assert stats.miss_rate == 0.0

    def test_hit_rate_mixed(self):
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75
        assert stats.miss_rate == 0.25

    def test_total_requests(self):
        stats = CacheStats(hits=10, misses=5)
        assert stats.total_requests == 15

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, evictions=2, max_size=100)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["hit_rate"] == 0.6667
        assert d["max_size"] == 100


# ─── Cache Key Tests ─────────────────────────────────────────────────────────


class TestCacheKey:
    """Tests for cache key generation."""

    def test_same_query_same_key(self):
        k1 = _make_cache_key("murder bail")
        k2 = _make_cache_key("murder bail")
        assert k1 == k2

    def test_case_insensitive(self):
        k1 = _make_cache_key("Murder Bail")
        k2 = _make_cache_key("murder bail")
        assert k1 == k2

    def test_strips_whitespace(self):
        k1 = _make_cache_key("  murder bail  ")
        k2 = _make_cache_key("murder bail")
        assert k1 == k2

    def test_different_query_different_key(self):
        k1 = _make_cache_key("murder")
        k2 = _make_cache_key("theft")
        assert k1 != k2

    def test_filters_affect_key(self):
        k1 = _make_cache_key("murder", court="Supreme Court")
        k2 = _make_cache_key("murder", court="High Court")
        assert k1 != k2

    def test_filter_order_independent(self):
        k1 = _make_cache_key("bail", court="SC", year=2024)
        k2 = _make_cache_key("bail", year=2024, court="SC")
        assert k1 == k2

    def test_none_filters_ignored(self):
        k1 = _make_cache_key("bail")
        k2 = _make_cache_key("bail", court=None, year=None)
        assert k1 == k2

    def test_key_is_hex_string(self):
        key = _make_cache_key("test query")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


# ─── Size Estimation Tests ────────────────────────────────────────────────────


class TestSizeEstimation:
    """Tests for memory size estimation."""

    def test_empty_list(self):
        assert _estimate_size([]) == 64

    def test_list_of_dicts(self):
        data = [{"title": "Test", "text": "x" * 100}] * 10
        size = _estimate_size(data)
        assert size > 500  # Non-trivial

    def test_string(self):
        assert _estimate_size("hello") == 5

    def test_dict(self):
        size = _estimate_size({"key": "value"})
        assert size > 0

    def test_none_fallback(self):
        assert _estimate_size(None) == 128


# ─── QueryFrequencyTracker Tests ─────────────────────────────────────────────


class TestQueryFrequencyTracker:
    """Tests for query frequency tracking."""

    def test_record_and_count(self):
        tracker = QueryFrequencyTracker()
        tracker.record("murder bail")
        tracker.record("murder bail")
        tracker.record("theft case")

        stats = tracker.get_stats()
        assert stats["unique_queries"] == 2
        assert stats["total_searches"] == 3

    def test_case_normalization(self):
        tracker = QueryFrequencyTracker()
        tracker.record("Murder Bail")
        tracker.record("murder bail")

        stats = tracker.get_stats()
        assert stats["unique_queries"] == 1
        assert stats["total_searches"] == 2

    def test_top_queries_ordering(self):
        tracker = QueryFrequencyTracker()
        for _ in range(5):
            tracker.record("popular query")
        for _ in range(2):
            tracker.record("less popular")
        tracker.record("rare query")

        top = tracker.top_queries(limit=3)
        assert len(top) == 3
        assert top[0]["query"] == "popular query"
        assert top[0]["count"] == 5
        assert top[1]["count"] == 2
        assert top[2]["count"] == 1

    def test_cache_hit_tracking(self):
        tracker = QueryFrequencyTracker()
        tracker.record("bail", cache_hit=True)
        tracker.record("bail", cache_hit=False)
        tracker.record("bail", cache_hit=True)

        top = tracker.top_queries(1)
        assert top[0]["cache_hit_rate"] == pytest.approx(0.667, abs=0.01)

    def test_prune_at_capacity(self):
        tracker = QueryFrequencyTracker(max_tracked=5, prune_threshold=1)
        # Add 5 unique single-occurrence queries
        for i in range(5):
            tracker.record(f"query_{i}")

        # Make one popular
        for _ in range(5):
            tracker.record("query_0")

        # This should trigger pruning
        tracker.record("overflow_query")

        stats = tracker.get_stats()
        # Should have pruned some single-occurrence queries
        assert stats["unique_queries"] <= 5

    def test_clear(self):
        tracker = QueryFrequencyTracker()
        tracker.record("test")
        tracker.clear()
        assert tracker.get_stats()["unique_queries"] == 0

    def test_empty_stats(self):
        tracker = QueryFrequencyTracker()
        stats = tracker.get_stats()
        assert stats["unique_queries"] == 0
        assert stats["total_searches"] == 0
        assert stats["avg_frequency"] == 0.0


# ─── SearchCache Core Tests ──────────────────────────────────────────────────


class TestSearchCache:
    """Tests for the main SearchCache class."""

    def test_put_and_get(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("murder bail", ["result1", "result2"])
        result = cache.get("murder bail")
        assert result == ["result1", "result2"]

    def test_miss_returns_none(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        assert cache.get("nonexistent") is None

    def test_filters_matter(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("bail", ["sc_results"], court="Supreme Court")
        cache.put("bail", ["hc_results"], court="High Court")

        assert cache.get("bail", court="Supreme Court") == ["sc_results"]
        assert cache.get("bail", court="High Court") == ["hc_results"]
        assert cache.get("bail") is None  # No filter = different key

    def test_overwrite_existing(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("test", ["old"])
        cache.put("test", ["new"])
        assert cache.get("test") == ["new"]

    def test_size_tracking(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        assert len(cache) == 0
        cache.put("q1", [1])
        cache.put("q2", [2])
        assert len(cache) == 2
        assert cache.size == 2


class TestLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_evicts_when_full(self):
        cache = SearchCache(max_entries=3, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])
        cache.put("q3", [3])
        assert cache.size == 3

        # This should evict q1 (least recently used)
        cache.put("q4", [4])
        assert cache.size == 3
        assert cache.get("q1") is None
        assert cache.get("q4") == [4]

    def test_access_prevents_eviction(self):
        cache = SearchCache(max_entries=3, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])
        cache.put("q3", [3])

        # Access q1 to make it recently used
        cache.get("q1")

        # Insert q4 — should evict q2 (now the LRU)
        cache.put("q4", [4])
        assert cache.get("q1") == [1]  # Still here
        assert cache.get("q2") is None  # Evicted
        assert cache.get("q4") == [4]

    def test_eviction_stats(self):
        cache = SearchCache(max_entries=2, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])
        cache.put("q3", [3])  # Evicts q1

        stats = cache.get_stats()
        assert stats["evictions"] >= 1

    def test_is_full_property(self):
        cache = SearchCache(max_entries=2, default_ttl=60)
        assert not cache.is_full
        cache.put("q1", [1])
        cache.put("q2", [2])
        assert cache.is_full


class TestTTLExpiry:
    """Tests for time-to-live expiry."""

    def test_entry_expires(self):
        cache = SearchCache(max_entries=10, default_ttl=0.1)  # 100ms TTL
        cache.put("test", ["data"])
        assert cache.get("test") == ["data"]

        time.sleep(0.15)
        assert cache.get("test") is None

    def test_custom_ttl_per_entry(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("short", ["data"], ttl=0.1)
        cache.put("long", ["data"], ttl=60)

        time.sleep(0.15)
        assert cache.get("short") is None
        assert cache.get("long") == ["data"]

    def test_purge_expired(self):
        cache = SearchCache(max_entries=10, default_ttl=0.1)
        cache.put("q1", [1])
        cache.put("q2", [2])
        cache.put("q3", [3])

        time.sleep(0.15)
        purged = cache.purge_expired()
        assert purged == 3
        assert cache.size == 0

    def test_expired_entry_counted_in_stats(self):
        cache = SearchCache(max_entries=10, default_ttl=0.1)
        cache.put("test", [1])
        time.sleep(0.15)
        cache.get("test")  # Miss due to expiry

        stats = cache.get_stats()
        assert stats["expired"] >= 1
        assert stats["misses"] >= 1


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    def test_hit_miss_tracking(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("exists", [1])

        cache.get("exists")   # Hit
        cache.get("exists")   # Hit
        cache.get("nope")     # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.6667, abs=0.01)

    def test_memory_estimate(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("q1", [{"title": "test", "text": "x" * 1000}] * 10)

        stats = cache.get_stats()
        assert stats["estimated_memory_bytes"] > 0
        assert "estimated_memory_mb" in stats

    def test_query_stats_enabled(self):
        cache = SearchCache(max_entries=10, track_queries=True)
        cache.put("test", [1])
        cache.get("test")

        qstats = cache.get_query_stats()
        assert qstats is not None
        assert qstats["unique_queries"] >= 1

    def test_query_stats_disabled(self):
        cache = SearchCache(max_entries=10, track_queries=False)
        cache.put("test", [1])
        cache.get("test")

        assert cache.get_query_stats() is None
        assert cache.top_queries() == []


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_specific(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])

        cache.invalidate("q1")
        assert cache.get("q1") is None
        assert cache.get("q2") == [2]

    def test_invalidate_with_filters(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("bail", [1], court="SC")
        cache.put("bail", [2], court="HC")

        cache.invalidate("bail", court="SC")
        assert cache.get("bail", court="SC") is None
        assert cache.get("bail", court="HC") == [2]

    def test_clear_all(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])
        cache.put("q3", [3])

        cache.clear()
        assert cache.size == 0
        assert cache.get("q1") is None

    def test_invalidate_nonexistent(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.invalidate("nonexistent")  # Should not raise
        assert cache.size == 0


class TestCacheEdgeCases:
    """Tests for edge cases and validation."""

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries must be positive"):
            SearchCache(max_entries=0)

    def test_invalid_negative_ttl(self):
        with pytest.raises(ValueError, match="default_ttl must be non-negative"):
            SearchCache(default_ttl=-1)

    def test_cache_single_entry(self):
        cache = SearchCache(max_entries=1, default_ttl=60)
        cache.put("q1", [1])
        cache.put("q2", [2])  # Evicts q1
        assert cache.get("q1") is None
        assert cache.get("q2") == [2]
        assert cache.size == 1

    def test_empty_query(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        cache.put("", ["empty query result"])
        assert cache.get("") == ["empty query result"]

    def test_large_result_set(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        big_result = [{"id": i, "text": f"case_{i}" * 100} for i in range(1000)]
        cache.put("big query", big_result)
        assert cache.get("big query") == big_result

    def test_repr(self):
        cache = SearchCache(max_entries=100, default_ttl=60)
        r = repr(cache)
        assert "SearchCache" in r
        assert "0/100" in r

    def test_capacity_property(self):
        cache = SearchCache(max_entries=42, default_ttl=60)
        assert cache.capacity == 42

    def test_zero_ttl_means_no_expiry(self):
        cache = SearchCache(max_entries=10, default_ttl=0)
        cache.put("forever", [1])
        # Even after "waiting", should still be there
        assert cache.get("forever") == [1]


class TestThreadSafety:
    """Tests for concurrent access safety."""

    def test_concurrent_puts(self):
        cache = SearchCache(max_entries=1000, default_ttl=60)
        errors = []

        def writer(start: int):
            try:
                for i in range(100):
                    cache.put(f"query_{start + i}", [start + i])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 1000  # Never exceeds max

    def test_concurrent_reads_and_writes(self):
        cache = SearchCache(max_entries=100, default_ttl=60)
        errors = []

        # Pre-populate
        for i in range(50):
            cache.put(f"q_{i}", [i])

        def reader():
            try:
                for i in range(50):
                    cache.get(f"q_{i}")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50, 100):
                    cache.put(f"q_{i}", [i])
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=reader) for _ in range(3)]
            + [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_evictions(self):
        cache = SearchCache(max_entries=10, default_ttl=60)
        errors = []

        def flood(offset: int):
            try:
                for i in range(100):
                    cache.put(f"flood_{offset}_{i}", [i])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=flood, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 10


class TestTopQueriesIntegration:
    """Integration tests for query tracking through the cache."""

    def test_popular_queries_via_cache(self):
        cache = SearchCache(max_entries=100, default_ttl=60, track_queries=True)

        # Simulate search pattern
        for _ in range(10):
            cache.put("bail application", ["res"])
            cache.get("bail application")

        for _ in range(3):
            cache.get("murder appeal")  # All misses

        top = cache.top_queries(5)
        assert len(top) >= 1
        # "bail application" should be most frequent
        assert top[0]["query"] == "bail application"

    def test_stats_consistency(self):
        cache = SearchCache(max_entries=50, default_ttl=60)

        # Do 10 puts + 10 gets (all hits) + 5 misses
        for i in range(10):
            cache.put(f"q_{i}", [i])
        for i in range(10):
            cache.get(f"q_{i}")
        for i in range(10, 15):
            cache.get(f"q_{i}")  # Misses

        stats = cache.get_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["current_entries"] == 10
        assert stats["hit_rate"] == pytest.approx(0.6667, abs=0.01)
