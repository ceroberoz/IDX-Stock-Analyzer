"""
HTTP Cache management for Yahoo Finance API requests.

Uses requests-cache with SQLite backend to persist API responses.
Cache duration: 1 day (86400 seconds) by default.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from requests_cache import CachedSession, SQLiteCache

from .config import get_config

logger = logging.getLogger(__name__)

# Global cached session instance
_cached_session: Optional[CachedSession] = None


def get_cache_location() -> Path:
    """Get the default cache location"""
    config = get_config()

    if config.network.cache_location:
        return Path(config.network.cache_location)

    # Default: ~/.cache/idx-analyzer/http_cache.sqlite
    cache_dir = Path.home() / ".cache" / "idx-analyzer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "http_cache.sqlite"


def get_cached_session() -> CachedSession:
    """
    Get or create a cached session for HTTP requests.

    The session uses SQLite backend with 1-day expiration by default.
    This helps avoid Yahoo Finance API rate limits.

    Returns:
        CachedSession: A requests session with caching enabled
    """
    global _cached_session

    if _cached_session is not None:
        return _cached_session

    config = get_config()

    if not config.network.use_cache:
        # Return a regular session if caching is disabled
        from requests import Session

        return Session()

    cache_location = get_cache_location()
    cache_ttl = config.network.cache_ttl

    logger.debug(f"Initializing HTTP cache at: {cache_location}")
    logger.debug(f"Cache TTL: {cache_ttl} seconds ({cache_ttl / 3600:.1f} hours)")

    _cached_session = CachedSession(
        backend=SQLiteCache(cache_location),
        expire_after=timedelta(seconds=cache_ttl),
        # Cache only GET requests (don't cache POST/PUT/DELETE)
        allowable_methods=["GET"],
        # Cache 200 OK and 304 Not Modified responses
        allowable_codes=[200, 304],
        # Don't include headers in cache key (ignore auth tokens, etc.)
        match_headers=False,
        # stale_if_error: Return cached response even if expired when request fails
        stale_if_error=True,
    )

    logger.info(f"HTTP cache initialized: {cache_location}")
    logger.info(f"Cache expiration: {cache_ttl / 3600:.1f} hours")

    return _cached_session


def clear_cache() -> None:
    """Clear the HTTP cache"""
    global _cached_session

    if _cached_session is not None:
        _cached_session.cache.clear()
        logger.info("HTTP cache cleared")
    else:
        # Clear cache file directly
        cache_location = get_cache_location()
        if cache_location.exists():
            cache_location.unlink()
            logger.info(f"Cache file removed: {cache_location}")


def get_cache_info() -> dict:
    """Get information about the current cache"""
    global _cached_session

    cache_location = get_cache_location()
    config = get_config()

    info = {
        "cache_location": str(cache_location),
        "cache_ttl_seconds": config.network.cache_ttl,
        "cache_ttl_hours": config.network.cache_ttl / 3600,
        "cache_enabled": config.network.use_cache,
        "cache_exists": cache_location.exists(),
    }

    if _cached_session is not None and cache_location.exists():
        try:
            # Get cache statistics if available
            responses = _cached_session.cache.responses_count()
            info["cached_responses"] = responses
        except Exception:
            info["cached_responses"] = "unknown"

    return info


def configure_yfinance_cache() -> None:
    """
    Configure yfinance to use our cached session.

    This should be called once at application startup.
    """
    import yfinance as yf

    session = get_cached_session()

    # Set the session for yfinance to use
    # Note: yfinance uses requests internally, and we can override the session
    yf.utils._requests_session = session

    logger.info("yfinance configured to use cached session")


# For backward compatibility
init_cache = configure_yfinance_cache
