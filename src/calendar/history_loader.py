"""
history_loader.py — Daemon-level singleton loader for the 1-year calendar
history cache.

Purpose
-------
Sub-AC 4b: fetch and cache the past 1 year of Google Calendar events so that
the event classifier (``event_classifier.py``) has reliable historical context
for deciding whether an upcoming meeting is a *first* or *follow-up* external
encounter.

This module provides three public functions:

``initialize(calendar_client, ...)``
    Called once at daemon startup.  Tries to warm the cache from disk first
    (fast path).  Falls back to a live API fetch if no fresh cache exists.
    Errors are *isolated* — a failure here logs a warning but never prevents
    the daemon from starting.

``get_cache() → CalendarHistoryCache | None``
    Returns the live singleton.  Returns ``None`` when initialisation failed or
    ``initialize()`` has not been called yet; callers must handle ``None``
    gracefully (event classifier already does this).

``refresh(calendar_client, ...)``
    Unconditionally rebuilds the cache from the live API and replaces the
    singleton.  Intended to be called by the APScheduler daily refresh job so
    that the history window slides forward automatically.

Thread safety
-------------
All singleton mutations go through ``_lock`` so the scheduler's background
thread can call ``refresh()`` safely while the briefing pipeline reads via
``get_cache()`` from another thread.

Typical usage (daemon startup)::

    from src.calendar.google_calendar import GoogleCalendarClient
    from src.calendar.history_loader import initialize as init_history, get_cache

    cal = GoogleCalendarClient()
    cal.connect()
    init_history(cal)                       # warm cache (fast on re-start)

    # Later, in the briefing pipeline:
    from src.calendar.history_loader import get_cache
    cache = get_cache()                     # may be None – handle gracefully
    category = classify_event(meeting, cache)

Scheduled daily refresh (scheduler.py)::

    from src.calendar.history_loader import refresh as refresh_history
    refresh_history(calendar_client)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from src.config import (
    CALENDAR_CACHE_FILE,
    CALENDAR_HISTORY_LOOKBACK_DAYS,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────────

_cache: Optional["CalendarHistoryCache"] = None  # type: ignore[name-defined]
_lock = threading.Lock()
_initialized: bool = False


# ── Public API ─────────────────────────────────────────────────────────────────

def initialize(
    calendar_client: "GoogleCalendarClient",  # type: ignore[name-defined]
    lookback_days: int = CALENDAR_HISTORY_LOOKBACK_DAYS,
    max_cache_age_hours: float = 12.0,
) -> bool:
    """
    Warm the module-level history cache singleton.

    Tries to load a fresh-enough cache from ``CALENDAR_CACHE_FILE`` first; if
    the file is missing or stale the calendar API is queried for the full
    *lookback_days* window (default 365 days).

    Errors are caught and logged as warnings; the function returns ``False``
    if the cache could not be populated.  The daemon **continues running** —
    the event classifier degrades gracefully when the cache is unavailable
    (external meetings default to ``EXTERNAL_FIRST``).

    Args:
        calendar_client:    An authenticated ``GoogleCalendarClient`` instance.
        lookback_days:      How many days of history to fetch (default 365).
        max_cache_age_hours: Treat a cached file as stale after this many hours.

    Returns:
        ``True`` if the cache was successfully loaded or built; ``False`` on
        any error.
    """
    global _cache, _initialized

    logger.info(
        "history_loader.initialize: warming %d-day calendar history cache …",
        lookback_days,
    )

    try:
        from src.calendar.history_cache import CalendarHistoryCache

        cache = CalendarHistoryCache.load_or_build(
            calendar_client=calendar_client,
            lookback_days=lookback_days,
            max_cache_age_hours=max_cache_age_hours,
            path=CALENDAR_CACHE_FILE,
        )
        with _lock:
            _cache = cache
            _initialized = True

        summary = cache.summary()
        logger.info(
            "history_loader.initialize: cache ready — %d events, %d external domains "
            "(built_at=%s)",
            summary["total_events"],
            summary["known_external_domains"],
            summary["built_at"],
        )
        return True

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "history_loader.initialize: failed to build history cache — "
            "event classifier will run without history context. Error: %s",
            exc,
        )
        with _lock:
            _initialized = True  # mark attempted so refresh() works
        return False


def get_cache() -> Optional["CalendarHistoryCache"]:  # type: ignore[name-defined]
    """
    Return the module-level ``CalendarHistoryCache`` singleton, or ``None``.

    Returns ``None`` when:
      - ``initialize()`` has not been called yet, OR
      - ``initialize()`` was called but the API/disk load failed.

    Callers must handle ``None`` gracefully.  The event classifier already
    does this — external meetings default to ``EXTERNAL_FIRST`` when the
    cache is unavailable.

    Thread-safe: acquires ``_lock`` before reading.
    """
    with _lock:
        return _cache


def refresh(
    calendar_client: "GoogleCalendarClient",  # type: ignore[name-defined]
    lookback_days: int = CALENDAR_HISTORY_LOOKBACK_DAYS,
) -> bool:
    """
    Unconditionally rebuild the history cache from the live API and replace
    the module-level singleton.

    Unlike ``initialize()``, this function bypasses the age check and always
    fetches fresh data.  It is designed to be called by the APScheduler daily
    refresh job to slide the 1-year history window forward.

    Retries the API call up to ``API_RETRY_ATTEMPTS`` times with
    ``API_RETRY_DELAY_SECONDS`` delay between attempts (matching the project-
    wide retry policy).

    Args:
        calendar_client:  An authenticated ``GoogleCalendarClient`` instance.
        lookback_days:    History window in days (default 365).

    Returns:
        ``True`` if the cache was rebuilt successfully; ``False`` on failure.
    """
    global _cache

    logger.info(
        "history_loader.refresh: rebuilding %d-day calendar history cache …",
        lookback_days,
    )

    last_exc: Optional[Exception] = None

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            from src.calendar.history_cache import CalendarHistoryCache

            new_cache = CalendarHistoryCache.build(
                calendar_client=calendar_client,
                lookback_days=lookback_days,
            )
            new_cache.save(CALENDAR_CACHE_FILE)

            with _lock:
                _cache = new_cache

            summary = new_cache.summary()
            logger.info(
                "history_loader.refresh: cache refreshed — %d events, "
                "%d external domains (attempt %d/%d)",
                summary["total_events"],
                summary["known_external_domains"],
                attempt,
                API_RETRY_ATTEMPTS,
            )
            return True

        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logger.warning(
                "history_loader.refresh: attempt %d/%d failed: %s",
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)

    logger.error(
        "history_loader.refresh: all %d attempts failed. Last error: %s",
        API_RETRY_ATTEMPTS,
        last_exc,
    )
    return False


def reset() -> None:
    """
    Clear the module-level singleton (for testing and clean re-initialisation).

    This should not be called in production code.
    """
    global _cache, _initialized
    with _lock:
        _cache = None
        _initialized = False
    logger.debug("history_loader.reset: singleton cleared.")


def is_initialized() -> bool:
    """Return True if ``initialize()`` has been called (regardless of success)."""
    return _initialized
