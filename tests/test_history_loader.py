"""
Tests for history_loader.py — daemon-level singleton loader for the
1-year calendar history cache (Sub-AC 4a).

Covers:
- initialize(): returns True when cache loads/builds successfully
- initialize(): returns False (but does not raise) on API failure
- initialize(): populates the module-level singleton
- get_cache(): returns None before initialize() is called
- get_cache(): returns the cache after initialize()
- refresh(): rebuilds cache unconditionally and replaces singleton
- refresh(): retries up to API_RETRY_ATTEMPTS times on failure
- refresh(): returns False after all retries exhausted
- reset(): clears singleton state (used in test teardown)
- is_initialized(): reflects whether initialize() has been called
- Thread safety: concurrent get_cache() calls do not raise

Note on patching:
    history_loader.py imports CALENDAR_CACHE_FILE at module load time:
        from src.config import CALENDAR_CACHE_FILE, ...
    Therefore we must patch `src.calendar.history_loader.CALENDAR_CACHE_FILE`
    (the local reference in the loader module), not `src.config.CALENDAR_CACHE_FILE`.
"""
from __future__ import annotations

import sys
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.calendar.history_loader as loader
from src.calendar.history_cache import CalendarHistoryCache, CachedEvent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_calendar_client(events=None):
    """Return a mock GoogleCalendarClient that returns *events* from list_all_historical_events."""
    client = MagicMock()
    client.list_all_historical_events.return_value = events or []
    return client


def _make_cached_event(event_id: str = "e1") -> CachedEvent:
    return CachedEvent(
        event_id=event_id,
        title="Test Meeting",
        start_iso="2025-06-01T10:00:00+00:00",
        end_iso="2025-06-01T11:00:00+00:00",
        all_day=False,
        organizer_email="me@kakaoventures.co.kr",
        is_external=True,
        attendee_emails=["me@kakaoventures.co.kr", "partner@acme.com"],
        attendee_domains=["kakaoventures.co.kr", "acme.com"],
    )


def _make_meeting_mock(event_id: str = "m1"):
    mock = MagicMock()
    mock.event_id = event_id
    mock.summary = "Meeting"
    mock.start = datetime(2025, 6, 1, 10, tzinfo=timezone.utc)
    mock.end = datetime(2025, 6, 1, 11, tzinfo=timezone.utc)
    mock.all_day = False
    mock.organizer_email = "me@kakaoventures.co.kr"
    mock.is_external = True
    att = MagicMock()
    att.email = "partner@acme.com"
    mock.attendees = [att]
    return mock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Always reset the loader singleton before and after each test."""
    loader.reset()
    yield
    loader.reset()


# ── initialize() ─────────────────────────────────────────────────────────────

class TestInitialize:

    def test_returns_true_on_success(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            result = loader.initialize(client)
        assert result is True

    def test_sets_cache_singleton(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client([_make_meeting_mock()])
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        cache = loader.get_cache()
        assert cache is not None
        assert isinstance(cache, CalendarHistoryCache)

    def test_cache_has_events_from_api(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        meetings = [_make_meeting_mock(f"m{i}") for i in range(4)]
        client = _make_calendar_client(meetings)
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        cache = loader.get_cache()
        assert cache.total_events == 4

    def test_marks_initialized_on_success(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        assert not loader.is_initialized()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.is_initialized()

    def test_returns_false_on_api_failure(self, tmp_path):
        """Returns False when no cached file exists AND API raises."""
        cache_path = tmp_path / "nonexistent_cache.json"
        # File does NOT exist → load_or_build must call API → API fails
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("API down")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            result = loader.initialize(client)
        assert result is False

    def test_does_not_raise_on_api_failure(self, tmp_path):
        cache_path = tmp_path / "nonexistent_cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = Exception("unexpected error")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            # Should NOT raise — daemon must continue on failure
            try:
                loader.initialize(client)
            except Exception:
                pytest.fail("initialize() should not raise on failure")

    def test_marks_initialized_even_on_failure(self, tmp_path):
        cache_path = tmp_path / "nonexistent_cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("fail")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.is_initialized()

    def test_cache_is_none_on_failure(self, tmp_path):
        cache_path = tmp_path / "nonexistent_cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("fail")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.get_cache() is None

    def test_loads_from_disk_when_fresh_cache_exists(self, tmp_path):
        """initialize() should use disk cache if it exists and is fresh."""
        cache_path = tmp_path / "cache.json"
        # Pre-populate a fresh cache
        events = [_make_cached_event("disk_e1")]
        fresh_cache = CalendarHistoryCache(
            events=events,
            built_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        fresh_cache.save(cache_path)

        client = _make_calendar_client()  # API should NOT be called
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client, max_cache_age_hours=12.0)

        assert loader.get_cache() is not None
        assert loader.get_cache().total_events == 1
        # Since cache was fresh, list_all_historical_events should NOT have been called
        client.list_all_historical_events.assert_not_called()

    def test_rebuilds_stale_cache(self, tmp_path):
        """initialize() should rebuild when cached file is stale."""
        cache_path = tmp_path / "cache.json"
        # Write a stale cache
        stale = CalendarHistoryCache(
            events=[_make_cached_event("old_e1")],
            built_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        stale.save(cache_path)

        new_meetings = [_make_meeting_mock(f"new_{i}") for i in range(2)]
        client = _make_calendar_client(new_meetings)
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client, max_cache_age_hours=12.0)

        assert loader.get_cache().total_events == 2
        client.list_all_historical_events.assert_called_once()


# ── get_cache() ───────────────────────────────────────────────────────────────

class TestGetCache:

    def test_returns_none_before_initialize(self):
        assert loader.get_cache() is None

    def test_returns_cache_after_initialize(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.get_cache() is not None

    def test_returns_same_instance_on_repeated_calls(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        c1 = loader.get_cache()
        c2 = loader.get_cache()
        assert c1 is c2


# ── refresh() ─────────────────────────────────────────────────────────────────

class TestRefresh:

    def test_returns_true_on_success(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            result = loader.refresh(client)
        assert result is True

    def test_replaces_old_cache(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        # Initialize with empty cache
        empty_client = _make_calendar_client([])
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(empty_client)
        assert loader.get_cache().total_events == 0

        # Refresh with 3 new events
        new_meetings = [_make_meeting_mock(f"nm{i}") for i in range(3)]
        new_client = _make_calendar_client(new_meetings)
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.refresh(new_client)

        assert loader.get_cache().total_events == 3

    def test_saves_to_disk_on_success(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client([_make_meeting_mock()])
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.refresh(client)
        assert cache_path.exists()

    def test_returns_false_after_all_retries_fail(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("API down")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path), \
             patch("src.calendar.history_loader.API_RETRY_ATTEMPTS", 3), \
             patch("src.calendar.history_loader.API_RETRY_DELAY_SECONDS", 0):
            result = loader.refresh(client)
        assert result is False

    def test_retries_specified_times(self, tmp_path):
        """refresh() should retry API_RETRY_ATTEMPTS times on failure."""
        cache_path = tmp_path / "cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("fail")
        with patch("src.calendar.history_loader.API_RETRY_ATTEMPTS", 3), \
             patch("src.calendar.history_loader.API_RETRY_DELAY_SECONDS", 0), \
             patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.refresh(client)
        assert client.list_all_historical_events.call_count == 3

    def test_does_not_raise_on_repeated_failure(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("fail")
        with patch("src.calendar.history_loader.API_RETRY_ATTEMPTS", 2), \
             patch("src.calendar.history_loader.API_RETRY_DELAY_SECONDS", 0), \
             patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            try:
                loader.refresh(client)
            except Exception:
                pytest.fail("refresh() should not raise after all retries fail")

    def test_succeeds_on_second_attempt(self, tmp_path):
        """If first call fails but second succeeds, refresh() returns True."""
        cache_path = tmp_path / "cache.json"
        meetings = [_make_meeting_mock()]
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient failure")
            return meetings

        client = MagicMock()
        client.list_all_historical_events.side_effect = _side_effect

        with patch("src.calendar.history_loader.API_RETRY_ATTEMPTS", 3), \
             patch("src.calendar.history_loader.API_RETRY_DELAY_SECONDS", 0), \
             patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            result = loader.refresh(client)

        assert result is True
        assert loader.get_cache() is not None
        assert loader.get_cache().total_events == 1


# ── reset() ──────────────────────────────────────────────────────────────────

class TestReset:

    def test_clears_cache(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client([_make_meeting_mock()])
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.get_cache() is not None
        loader.reset()
        assert loader.get_cache() is None

    def test_clears_initialized_flag(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.is_initialized()
        loader.reset()
        assert not loader.is_initialized()


# ── is_initialized() ─────────────────────────────────────────────────────────

class TestIsInitialized:

    def test_false_before_initialize(self):
        assert not loader.is_initialized()

    def test_true_after_successful_initialize(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client()
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.is_initialized()

    def test_true_even_after_failed_initialize(self, tmp_path):
        cache_path = tmp_path / "nonexistent_cache.json"
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("fail")
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)
        assert loader.is_initialized()


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_get_cache_calls(self, tmp_path):
        """Multiple threads calling get_cache() concurrently should not raise."""
        cache_path = tmp_path / "cache.json"
        client = _make_calendar_client([_make_meeting_mock()])
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)

        errors = []

        def _worker():
            try:
                c = loader.get_cache()
                assert c is not None
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_refresh_while_reading(self, tmp_path):
        """refresh() running in one thread should not corrupt cache read in another."""
        cache_path = tmp_path / "cache.json"
        initial_meetings = [_make_meeting_mock(f"init_{i}") for i in range(5)]
        client = _make_calendar_client(initial_meetings)
        with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
            loader.initialize(client)

        errors = []

        def _refresh_worker():
            new_meetings = [_make_meeting_mock(f"new_{i}") for i in range(3)]
            new_client = _make_calendar_client(new_meetings)
            try:
                with patch("src.calendar.history_loader.CALENDAR_CACHE_FILE", cache_path):
                    loader.refresh(new_client)
            except Exception as exc:
                errors.append(exc)

        def _read_worker():
            try:
                c = loader.get_cache()
                # Either old or new cache is acceptable — it must be a valid object
                if c is not None:
                    assert isinstance(c, CalendarHistoryCache)
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=_refresh_worker)]
            + [threading.Thread(target=_read_worker) for _ in range(10)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
