"""
Search Cache — LRU Caching with TTL and Query Frequency Tracking.

Provides an in-memory cache layer for search results with:
- LRU eviction when capacity is exceeded
- Time-to-live (TTL) expiry per entry
- Query frequency tracking for popular-query analysis
- Thread-safe operations for concurrent access
- Cache statistics (hit rate, evictions, memory usage)
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Cache Entry ──────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A single cached search result with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    ttl_seconds: float
    hit_count: int = 0
    size_estimate: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        if self.ttl_seconds <= 0:
            return False  # No expiry
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was created."""
        return time.time() - self.created_at

    def touch(self):
        """Update last access time and increment hit count."""
        self.last_accessed = time.time()
        self.hit_count += 1


# ─── Cache Statistics ─────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Aggregate cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    current_size: int = 0
    max_size: int = 0
    total_entries: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as a fraction (0.0 to 1.0)."""
        return 1.0 - self.hit_rate

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expired": self.expired,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "total_requests": self.total_requests,
            "current_entries": self.total_entries,
            "max_size": self.max_size,
        }


# ─── Query Frequency Tracker ─────────────────────────────────────────────────

@dataclass
class QueryRecord:
    """Tracks frequency and timing for a single query."""

    query: str
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class QueryFrequencyTracker:
    """
    Track query frequencies for popular-query analysis.

    Maintains a bounded frequency table with automatic pruning
    of infrequent queries to control memory usage.
    """

    def __init__(self, max_tracked: int = 10_000, prune_threshold: int = 1):
        self._records: dict[str, QueryRecord] = {}
        self._max_tracked = max_tracked
        self._prune_threshold = prune_threshold
        self._lock = threading.Lock()

    def record(self, query: str, cache_hit: bool = False):
        """Record a query occurrence."""
        normalized = query.strip().lower()
        now = time.time()

        with self._lock:
            if normalized not in self._records:
                # Prune if at capacity
                if len(self._records) >= self._max_tracked:
                    self._prune()

                self._records[normalized] = QueryRecord(
                    query=normalized,
                    count=0,
                    first_seen=now,
                )

            record = self._records[normalized]
            record.count += 1
            record.last_seen = now

            if cache_hit:
                record.cache_hits += 1
            else:
                record.cache_misses += 1

    def _prune(self):
        """Remove low-frequency queries to free space."""
        # Remove queries seen only once (or below threshold)
        to_remove = [
            key for key, rec in self._records.items()
            if rec.count <= self._prune_threshold
        ]
        for key in to_remove:
            del self._records[key]

        # If still over capacity, remove oldest entries
        if len(self._records) >= self._max_tracked:
            sorted_by_time = sorted(
                self._records.items(),
                key=lambda x: x[1].last_seen,
            )
            remove_count = len(self._records) - self._max_tracked // 2
            for key, _ in sorted_by_time[:remove_count]:
                del self._records[key]

    def top_queries(self, limit: int = 20) -> list[dict]:
        """Get the most frequent queries."""
        with self._lock:
            sorted_records = sorted(
                self._records.values(),
                key=lambda r: r.count,
                reverse=True,
            )
            return [
                {
                    "query": r.query,
                    "count": r.count,
                    "cache_hit_rate": round(
                        r.cache_hits / max(r.count, 1), 3
                    ),
                    "first_seen": r.first_seen,
                    "last_seen": r.last_seen,
                }
                for r in sorted_records[:limit]
            ]

    def get_stats(self) -> dict:
        """Summary statistics for tracked queries."""
        with self._lock:
            if not self._records:
                return {
                    "unique_queries": 0,
                    "total_searches": 0,
                    "avg_frequency": 0.0,
                }

            total_count = sum(r.count for r in self._records.values())
            return {
                "unique_queries": len(self._records),
                "total_searches": total_count,
                "avg_frequency": round(total_count / len(self._records), 2),
                "max_frequency": max(r.count for r in self._records.values()),
            }

    def clear(self):
        """Reset all query tracking data."""
        with self._lock:
            self._records.clear()


# ─── LRU Search Cache ────────────────────────────────────────────────────────

def _make_cache_key(query: str, **filters) -> str:
    """
    Generate a deterministic cache key from query + filters.

    Uses SHA-256 for uniform key distribution.
    """
    parts = [query.strip().lower()]
    for key in sorted(filters.keys()):
        val = filters[key]
        if val is not None:
            parts.append(f"{key}={val}")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _estimate_size(value: Any) -> int:
    """Rough memory estimate for a cached value (bytes)."""
    if isinstance(value, list):
        # Estimate list of dicts
        if not value:
            return 64
        # Sample first item
        sample = str(value[0]) if value else ""
        return len(sample) * len(value) + 64
    elif isinstance(value, dict):
        return len(str(value))
    elif isinstance(value, str):
        return len(value)
    return 128  # Default estimate


class SearchCache:
    """
    Thread-safe LRU cache for search results with TTL expiry.

    Features:
    - Bounded capacity with LRU eviction
    - Per-entry TTL (time-to-live)
    - Integrated query frequency tracking
    - Hit/miss/eviction statistics
    - Manual and automatic cache invalidation

    Usage:
        cache = SearchCache(max_entries=1000, default_ttl=300)

        # Try cache first
        results = cache.get("murder bail", court="Supreme Court")
        if results is None:
            results = engine.search("murder bail", court="Supreme Court")
            cache.put("murder bail", results, court="Supreme Court")
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: float = 300.0,
        track_queries: bool = True,
        max_tracked_queries: int = 10_000,
    ):
        """
        Initialize the search cache.

        Args:
            max_entries: Maximum cached entries before LRU eviction.
            default_ttl: Default time-to-live in seconds (0 = no expiry).
            track_queries: Whether to track query frequencies.
            max_tracked_queries: Max unique queries to track.
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_entries)

        self._tracker: Optional[QueryFrequencyTracker] = None
        if track_queries:
            self._tracker = QueryFrequencyTracker(
                max_tracked=max_tracked_queries,
            )

    def get(self, query: str, **filters) -> Optional[Any]:
        """
        Retrieve cached results for a query.

        Args:
            query: Search query string.
            **filters: Additional filters (court, year, etc.).

        Returns:
            Cached results if found and not expired, else None.
        """
        key = _make_cache_key(query, **filters)

        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                if self._tracker:
                    self._tracker.record(query, cache_hit=False)
                return None

            entry = self._cache[key]

            # Check TTL
            if entry.is_expired:
                del self._cache[key]
                self._stats.expired += 1
                self._stats.misses += 1
                self._stats.total_entries = len(self._cache)
                if self._tracker:
                    self._tracker.record(query, cache_hit=False)
                return None

            # Cache hit — move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)

            self._stats.hits += 1
            if self._tracker:
                self._tracker.record(query, cache_hit=True)

            return entry.value

    def put(
        self,
        query: str,
        value: Any,
        ttl: Optional[float] = None,
        **filters,
    ):
        """
        Store search results in the cache.

        Args:
            query: Search query string.
            value: Search results to cache.
            ttl: Optional per-entry TTL override (seconds).
            **filters: Additional filters used in the query.
        """
        key = _make_cache_key(query, **filters)
        ttl_seconds = ttl if ttl is not None else self._default_ttl
        now = time.time()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            last_accessed=now,
            ttl_seconds=ttl_seconds,
            size_estimate=_estimate_size(value),
        )

        with self._lock:
            # Update existing or insert new
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Evict if at capacity
                while len(self._cache) >= self._max_entries:
                    self._evict_one()

                self._cache[key] = entry

            self._stats.total_entries = len(self._cache)

    def _evict_one(self):
        """Evict the least recently used entry (internal, lock must be held)."""
        if self._cache:
            # First try to evict an expired entry
            for key, entry in self._cache.items():
                if entry.is_expired:
                    del self._cache[key]
                    self._stats.expired += 1
                    return

            # Otherwise evict LRU (first item in OrderedDict)
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def invalidate(self, query: str, **filters):
        """Remove a specific entry from the cache."""
        key = _make_cache_key(query, **filters)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.total_entries = len(self._cache)

    def invalidate_by_prefix(self, query_prefix: str):
        """
        Invalidate all entries whose original query starts with a prefix.

        Note: This is O(n) as it scans all entries.
        For frequent use, consider a trie-based cache key structure.
        """
        prefix_lower = query_prefix.strip().lower()
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                # We can't reverse the hash, but we can check via stored value
                # This is a limitation — for prefix invalidation, store query in entry
                pass
            # For now, this is a no-op placeholder
            # Full implementation would require storing the original query in CacheEntry

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._stats.total_entries = 0

    def purge_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries purged.
        """
        purged = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                purged += 1

            self._stats.expired += purged
            self._stats.total_entries = len(self._cache)

        return purged

    def get_stats(self) -> dict:
        """Get cache performance statistics."""
        with self._lock:
            stats = self._stats.to_dict()

            # Add memory estimate
            total_size = sum(
                entry.size_estimate for entry in self._cache.values()
            )
            stats["estimated_memory_bytes"] = total_size
            stats["estimated_memory_mb"] = round(total_size / (1024 * 1024), 2)

            return stats

    def get_query_stats(self) -> Optional[dict]:
        """Get query frequency statistics (if tracking is enabled)."""
        if self._tracker:
            return self._tracker.get_stats()
        return None

    def top_queries(self, limit: int = 20) -> list[dict]:
        """Get the most frequently searched queries."""
        if self._tracker:
            return self._tracker.top_queries(limit)
        return []

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def capacity(self) -> int:
        """Maximum cache capacity."""
        return self._max_entries

    @property
    def is_full(self) -> bool:
        """Whether the cache is at capacity."""
        with self._lock:
            return len(self._cache) >= self._max_entries

    def __contains__(self, item: tuple) -> bool:
        """Check if a (query, **filters) is cached (does not update access time)."""
        if isinstance(item, str):
            key = _make_cache_key(item)
        else:
            return False

        with self._lock:
            if key in self._cache:
                return not self._cache[key].is_expired
            return False

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"SearchCache(entries={self.size}/{self._max_entries}, "
            f"hit_rate={self._stats.hit_rate:.1%})"
        )
