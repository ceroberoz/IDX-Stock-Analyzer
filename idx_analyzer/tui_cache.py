"""
SQLite Cache Layer for IDX Terminal
Replaces file-based cache with SQLite for all API responses
"""

import logging

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class TUICache:
    """SQLite-based cache for TUI and CLI"""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: float = 1.0):
        """Initialize SQLite cache

        Args:
            cache_dir: Directory for cache file (default: ~/.cache/idx-analyzer)
            ttl_hours: Cache TTL in hours
        """
        self.ttl_hours = ttl_hours

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "idx-analyzer"

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "cache.db"

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Create cache tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)
            """)

            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, expires_at FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                data_str, expires_at = row

                # Check if expired
                expires = datetime.fromisoformat(expires_at)
                if datetime.now() > expires:
                    # Delete expired entry
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None

                return json.loads(data_str)

        except Exception:
            return None

    def set(self, key: str, data: Any, ttl_hours: Optional[float] = None) -> None:
        """Cache data with TTL

        Args:
            key: Cache key
            data: Data to cache (must be JSON serializable)
            ttl_hours: Override default TTL
        """
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cache (key, data, expires_at)
                       VALUES (?, ?, ?)""",
                    (key, json.dumps(data), expires_at.isoformat()),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache set error: {e}")  # Fail silently but log

    def delete(self, key: str) -> None:
        """Delete cached entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache delete error: {e}")

    def clear(self) -> None:
        """Clear all cached data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache clear error: {e}")

    def clear_expired(self) -> int:
        """Clear expired entries, return count deleted"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),),
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug(f"Cache clear_expired error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),),
                ).fetchone()[0]

                # Calculate size
                size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    "total_entries": total,
                    "expired_entries": expired,
                    "valid_entries": total - expired,
                    "cache_size_mb": round(size / (1024 * 1024), 2),
                    "db_path": str(self.db_path),
                    "ttl_hours": self.ttl_hours,
                }
        except Exception as e:
            return {"error": str(e)}


# Global cache instance
_cache_instance: Optional[TUICache] = None


def get_cache() -> TUICache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TUICache()
    return _cache_instance


def configure_cache(
    cache_dir: Optional[Path] = None, ttl_hours: float = 1.0
) -> TUICache:
    """Configure and get cache instance"""
    global _cache_instance
    _cache_instance = TUICache(cache_dir=cache_dir, ttl_hours=ttl_hours)
    return _cache_instance


# Convenience functions
def cache_get(key: str) -> Optional[Any]:
    """Get from global cache"""
    return get_cache().get(key)


def cache_set(key: str, data: Any, ttl_hours: Optional[float] = None) -> None:
    """Set in global cache"""
    get_cache().set(key, data, ttl_hours)


def cache_clear() -> None:
    """Clear global cache"""
    get_cache().clear()
