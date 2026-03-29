"""
Tests for CalendarHistoryCache (Sub-AC 4a).

Covers:
- CachedEvent.from_dict / to_dict round-trip
- CachedEvent.from_meeting conversion
- CachedEvent.external_domains derived property
- CalendarHistoryCache index building (by email, by domain)
- get_meetings_by_email / get_meetings_by_domain query methods
- is_known_external_domain
- past_meeting_count_for_email / past_meeting_count_for_domain
- last_meeting_with_email / last_meeting_with_domain
- known_external_domains set
- summary() stats dict
- CalendarHistoryCache.save() + load() round-trip (with tmp file)
- load() raises FileNotFoundError for missing file
- load() raises ValueError for unsupported version
- load_or_build(): returns cached when fresh
- load_or_build(): rebuilds when stale
- load_or_build(): rebuilds when file missing
- build() calls list_all_historical_events on the client
- GoogleCalendarClient.list_all_historical_events pagination (unit)
"""
from __future__ import annotations

import json
import sys
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar.history_cache import (
    CalendarHistoryCache,
    CachedEvent,
    _email_domain,
)
from src.config import INTERNAL_DOMAIN


# ── Helpers / Factories ───────────────────────────────────────────────────────

def _make_cached_event(
    event_id: str = "evt1",
    title: str = "Test Meeting",
    start_iso: str = "2025-06-01T10:00:00+00:00",
    end_iso: str = "2025-06-01T11:00:00+00:00",
    all_day: bool = False,
    organizer_email: str = "me@kakaoventures.co.kr",
    is_external: bool = True,
    attendee_emails: list[str] | None = None,
    attendee_domains: list[str] | None = None,
) -> CachedEvent:
    if attendee_emails is None:
        attendee_emails = ["me@kakaoventures.co.kr", "partner@acme.com"]
    if attendee_domains is None:
        attendee_domains = ["kakaoventures.co.kr", "acme.com"]
    return CachedEvent(
        event_id=event_id,
        title=title,
        start_iso=start_iso,
        end_iso=end_iso,
        all_day=all_day,
        organizer_email=organizer_email,
        is_external=is_external,
        attendee_emails=attendee_emails,
        attendee_domains=attendee_domains,
    )


def _make_meeting_mock(
    event_id: str = "m1",
    summary: str = "Demo Meeting",
    start: datetime | None = None,
    end: datetime | None = None,
    attendee_emails: list[str] | None = None,
    is_external: bool = True,
) -> MagicMock:
    """Return a mock object that mimics src.calendar.google_calendar.Meeting."""
    if start is None:
        start = datetime(2025, 9, 15, 9, 0, tzinfo=timezone.utc)
    if end is None:
        end = start + timedelta(hours=1)
    if attendee_emails is None:
        attendee_emails = ["a@kakaoventures.co.kr", "b@partner.com"]

    mock = MagicMock()
    mock.event_id = event_id
    mock.summary = summary
    mock.start = start
    mock.end = end
    mock.all_day = False
    mock.organizer_email = attendee_emails[0] if attendee_emails else ""
    mock.is_external = is_external

    attendees = []
    for email in attendee_emails:
        att = MagicMock()
        att.email = email
        attendees.append(att)
    mock.attendees = attendees
    return mock


# ── CachedEvent ───────────────────────────────────────────────────────────────

class TestCachedEvent:

    def test_to_dict_round_trip(self):
        ev = _make_cached_event()
        d = ev.to_dict()
        restored = CachedEvent.from_dict(d)
        assert restored.event_id == ev.event_id
        assert restored.title == ev.title
        assert restored.start_iso == ev.start_iso
        assert restored.attendee_emails == ev.attendee_emails
        assert restored.attendee_domains == ev.attendee_domains
        assert restored.is_external == ev.is_external

    def test_start_property_parses_iso(self):
        ev = _make_cached_event(start_iso="2025-06-01T10:00:00+00:00")
        assert ev.start == datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc)

    def test_external_domains_excludes_internal(self):
        ev = _make_cached_event(
            attendee_domains=["kakaoventures.co.kr", "acme.com", "beta.io"]
        )
        assert INTERNAL_DOMAIN not in ev.external_domains
        assert "acme.com" in ev.external_domains
        assert "beta.io" in ev.external_domains

    def test_external_domains_deduplicates(self):
        ev = _make_cached_event(
            attendee_domains=["acme.com", "acme.com", "kakaoventures.co.kr"]
        )
        assert ev.external_domains.count("acme.com") == 1

    def test_from_meeting_populates_all_fields(self):
        meeting = _make_meeting_mock(
            event_id="x42",
            summary="Board Review",
            attendee_emails=["a@kakaoventures.co.kr", "ceo@bigcorp.com"],
        )
        ev = CachedEvent.from_meeting(meeting)
        assert ev.event_id == "x42"
        assert ev.title == "Board Review"
        assert "a@kakaoventures.co.kr" in ev.attendee_emails
        assert "ceo@bigcorp.com" in ev.attendee_emails
        assert "bigcorp.com" in ev.attendee_domains

    def test_from_meeting_lowercases_emails(self):
        meeting = _make_meeting_mock(
            attendee_emails=["Upper@Case.COM", "normal@domain.io"]
        )
        ev = CachedEvent.from_meeting(meeting)
        assert "upper@case.com" in ev.attendee_emails
        assert "normal@domain.io" in ev.attendee_emails

    def test_from_meeting_skips_empty_email(self):
        meeting = _make_meeting_mock(attendee_emails=["", "valid@corp.com"])
        ev = CachedEvent.from_meeting(meeting)
        assert "" not in ev.attendee_emails

    def test_all_day_event_preserved(self):
        meeting = _make_meeting_mock()
        meeting.all_day = True
        ev = CachedEvent.from_meeting(meeting)
        assert ev.all_day is True


# ── CalendarHistoryCache — Index & Queries ────────────────────────────────────

class TestCalendarHistoryCacheQueries:

    def _build_cache(self) -> CalendarHistoryCache:
        events = [
            _make_cached_event(
                event_id="e1",
                title="Acme Intro",
                start_iso="2025-01-10T09:00:00+00:00",
                attendee_emails=["alice@kakaoventures.co.kr", "bob@acme.com"],
                attendee_domains=["kakaoventures.co.kr", "acme.com"],
            ),
            _make_cached_event(
                event_id="e2",
                title="Acme Follow-up",
                start_iso="2025-03-20T10:00:00+00:00",
                attendee_emails=["alice@kakaoventures.co.kr", "bob@acme.com", "carol@acme.com"],
                attendee_domains=["kakaoventures.co.kr", "acme.com"],
            ),
            _make_cached_event(
                event_id="e3",
                title="Beta Corp Meeting",
                start_iso="2025-06-01T11:00:00+00:00",
                attendee_emails=["dave@kakaoventures.co.kr", "eve@beta.io"],
                attendee_domains=["kakaoventures.co.kr", "beta.io"],
            ),
            _make_cached_event(
                event_id="e4",
                title="Internal Sync",
                start_iso="2025-07-15T14:00:00+00:00",
                is_external=False,
                attendee_emails=["alice@kakaoventures.co.kr", "dave@kakaoventures.co.kr"],
                attendee_domains=["kakaoventures.co.kr"],
            ),
        ]
        return CalendarHistoryCache(events=events)

    def test_get_meetings_by_email_returns_matching(self):
        cache = self._build_cache()
        results = cache.get_meetings_by_email("bob@acme.com")
        assert len(results) == 2
        titles = {r.title for r in results}
        assert "Acme Intro" in titles
        assert "Acme Follow-up" in titles

    def test_get_meetings_by_email_case_insensitive(self):
        cache = self._build_cache()
        assert len(cache.get_meetings_by_email("BOB@ACME.COM")) == 2

    def test_get_meetings_by_email_returns_empty_for_unknown(self):
        cache = self._build_cache()
        assert cache.get_meetings_by_email("nobody@xyz.com") == []

    def test_get_meetings_by_email_sorted_most_recent_first(self):
        cache = self._build_cache()
        results = cache.get_meetings_by_email("bob@acme.com")
        assert results[0].start_iso > results[1].start_iso

    def test_get_meetings_by_domain_returns_matching(self):
        cache = self._build_cache()
        results = cache.get_meetings_by_domain("acme.com")
        assert len(results) == 2

    def test_get_meetings_by_domain_case_insensitive(self):
        cache = self._build_cache()
        assert len(cache.get_meetings_by_domain("ACME.COM")) == 2

    def test_get_meetings_by_domain_empty_for_unknown(self):
        cache = self._build_cache()
        assert cache.get_meetings_by_domain("unknown.org") == []

    def test_is_known_external_domain_true(self):
        cache = self._build_cache()
        assert cache.is_known_external_domain("acme.com") is True
        assert cache.is_known_external_domain("beta.io") is True

    def test_is_known_external_domain_false_for_internal(self):
        cache = self._build_cache()
        assert cache.is_known_external_domain(INTERNAL_DOMAIN) is False

    def test_is_known_external_domain_false_for_unknown(self):
        cache = self._build_cache()
        assert cache.is_known_external_domain("newco.example") is False

    def test_past_meeting_count_for_email(self):
        cache = self._build_cache()
        assert cache.past_meeting_count_for_email("bob@acme.com") == 2

    def test_past_meeting_count_for_email_zero(self):
        cache = self._build_cache()
        assert cache.past_meeting_count_for_email("nobody@xyz.com") == 0

    def test_past_meeting_count_for_domain(self):
        cache = self._build_cache()
        assert cache.past_meeting_count_for_domain("acme.com") == 2

    def test_last_meeting_with_email_returns_most_recent(self):
        cache = self._build_cache()
        latest = cache.last_meeting_with_email("bob@acme.com")
        assert latest is not None
        assert latest.title == "Acme Follow-up"

    def test_last_meeting_with_email_none_for_unknown(self):
        cache = self._build_cache()
        assert cache.last_meeting_with_email("nobody@xyz.com") is None

    def test_last_meeting_with_domain(self):
        cache = self._build_cache()
        latest = cache.last_meeting_with_domain("acme.com")
        assert latest is not None
        assert latest.title == "Acme Follow-up"

    def test_known_external_domains_excludes_internal(self):
        cache = self._build_cache()
        ext = cache.known_external_domains
        assert INTERNAL_DOMAIN not in ext
        assert "acme.com" in ext
        assert "beta.io" in ext

    def test_external_events_property(self):
        cache = self._build_cache()
        external = cache.external_events
        ids = {e.event_id for e in external}
        assert "e1" in ids
        assert "e2" in ids
        assert "e3" in ids
        assert "e4" not in ids  # internal sync

    def test_total_events_property(self):
        cache = self._build_cache()
        assert cache.total_events == 4

    def test_summary_dict_contains_expected_keys(self):
        cache = self._build_cache()
        s = cache.summary()
        for key in (
            "total_events",
            "external_events",
            "unique_attendee_emails",
            "unique_attendee_domains",
            "known_external_domains",
            "lookback_days",
            "built_at",
        ):
            assert key in s, f"Missing key in summary: {key}"

    def test_summary_counts_are_accurate(self):
        cache = self._build_cache()
        s = cache.summary()
        assert s["total_events"] == 4
        assert s["external_events"] == 3


# ── CalendarHistoryCache — Persistence ───────────────────────────────────────

class TestCalendarHistoryCachePersistence:

    def _make_cache(self, n: int = 3) -> CalendarHistoryCache:
        events = [
            _make_cached_event(
                event_id=f"e{i}",
                title=f"Meeting {i}",
                start_iso=f"2025-0{i+1}-01T10:00:00+00:00",
                attendee_emails=[f"a{i}@kakaoventures.co.kr", f"b{i}@company{i}.com"],
                attendee_domains=["kakaoventures.co.kr", f"company{i}.com"],
            )
            for i in range(1, n + 1)
        ]
        return CalendarHistoryCache(events=events, lookback_days=365)

    def test_save_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = self._make_cache(2)
            cache.save(path)
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_round_trip_event_count(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = self._make_cache(5)
            cache.save(path)
            loaded = CalendarHistoryCache.load(path)
            assert loaded.total_events == 5
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_round_trip_event_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = self._make_cache(2)
            cache.save(path)
            loaded = CalendarHistoryCache.load(path)
            original = cache.events[0]
            restored = loaded.events[0]
            assert restored.event_id == original.event_id
            assert restored.title == original.title
            assert restored.attendee_emails == original.attendee_emails
            assert restored.attendee_domains == original.attendee_domains
            assert restored.is_external == original.is_external
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_preserves_lookback_days(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = CalendarHistoryCache(events=[], lookback_days=180)
            cache.save(path)
            loaded = CalendarHistoryCache.load(path)
            assert loaded.lookback_days == 180
        finally:
            path.unlink(missing_ok=True)

    def test_load_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CalendarHistoryCache.load(Path("/tmp/nonexistent_cache_xyz.json"))

    def test_load_raises_for_wrong_version(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"version": 99, "events": [], "built_at": "2025-01-01T00:00:00+00:00", "lookback_days": 365}, f)
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="version"):
                CalendarHistoryCache.load(path)
        finally:
            path.unlink(missing_ok=True)

    def test_loaded_cache_indexes_correctly(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = self._make_cache(3)
            cache.save(path)
            loaded = CalendarHistoryCache.load(path)
            # Should be able to query by email after load
            assert loaded.past_meeting_count_for_email("a1@kakaoventures.co.kr") == 1
        finally:
            path.unlink(missing_ok=True)

    def test_save_is_atomic_on_success(self):
        """save() should not leave a .tmp file behind on success."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            cache = self._make_cache(1)
            cache.save(path)
            tmp = Path(str(path) + ".tmp")
            assert not tmp.exists()
        finally:
            path.unlink(missing_ok=True)


# ── CalendarHistoryCache — load_or_build ─────────────────────────────────────

class TestLoadOrBuild:

    def _mock_client(self, meetings: list[MagicMock] | None = None) -> MagicMock:
        client = MagicMock()
        client.list_all_historical_events.return_value = meetings or []
        return client

    def test_builds_from_api_when_no_file(self):
        meetings = [_make_meeting_mock(event_id=f"m{i}") for i in range(3)]
        client = self._mock_client(meetings)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = CalendarHistoryCache.load_or_build(client, path=path)
        assert cache.total_events == 3
        client.list_all_historical_events.assert_called_once()

    def test_saves_after_build(self):
        client = self._mock_client([_make_meeting_mock()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            CalendarHistoryCache.load_or_build(client, path=path)
            assert path.exists()

    def test_loads_cache_when_fresh(self):
        """When cached file is fresh (< max_cache_age_hours), API not called."""
        client = self._mock_client()
        events = [_make_cached_event(event_id="cached1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            # Write a fresh cache manually
            fresh_cache = CalendarHistoryCache(
                events=events,
                built_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            )
            fresh_cache.save(path)

            loaded = CalendarHistoryCache.load_or_build(
                client, max_cache_age_hours=12.0, path=path
            )

        assert loaded.total_events == 1
        client.list_all_historical_events.assert_not_called()

    def test_rebuilds_when_cache_stale(self):
        """When cached file is stale (> max_cache_age_hours), API IS called."""
        meetings = [_make_meeting_mock(event_id="fresh_m")]
        client = self._mock_client(meetings)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            # Write a stale cache
            stale_cache = CalendarHistoryCache(
                events=[_make_cached_event(event_id="old1")],
                built_at=datetime.now(timezone.utc) - timedelta(hours=25),
            )
            stale_cache.save(path)

            fresh = CalendarHistoryCache.load_or_build(
                client, max_cache_age_hours=12.0, path=path
            )

        client.list_all_historical_events.assert_called_once()
        assert fresh.total_events == 1

    def test_rebuilds_on_corrupt_cache(self):
        """A corrupt cache file should trigger a rebuild, not an exception."""
        client = self._mock_client([_make_meeting_mock()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            path.write_text("this is not valid json {{{{")
            cache = CalendarHistoryCache.load_or_build(client, path=path)
        assert cache.total_events == 1
        client.list_all_historical_events.assert_called_once()


# ── CalendarHistoryCache.build ────────────────────────────────────────────────

class TestCacheBuild:

    def test_build_calls_list_all_historical_events(self):
        client = MagicMock()
        client.list_all_historical_events.return_value = []
        CalendarHistoryCache.build(client, lookback_days=365)
        client.list_all_historical_events.assert_called_once_with(lookback_days=365)

    def test_build_converts_all_meetings(self):
        meetings = [_make_meeting_mock(event_id=f"m{i}") for i in range(5)]
        client = MagicMock()
        client.list_all_historical_events.return_value = meetings
        cache = CalendarHistoryCache.build(client)
        assert cache.total_events == 5

    def test_build_sets_lookback_days(self):
        client = MagicMock()
        client.list_all_historical_events.return_value = []
        cache = CalendarHistoryCache.build(client, lookback_days=180)
        assert cache.lookback_days == 180

    def test_build_propagates_api_error(self):
        client = MagicMock()
        client.list_all_historical_events.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="API down"):
            CalendarHistoryCache.build(client)


# ── GoogleCalendarClient.list_all_historical_events (unit) ───────────────────

class TestListAllHistoricalEvents:
    """
    Unit tests for the paginated list_all_historical_events method
    on GoogleCalendarClient.  The Google API service is mocked.
    """

    def _make_raw_event(self, idx: int) -> dict:
        """Create a minimal raw event dict as returned by the Google API."""
        return {
            "id": f"event_{idx}",
            "summary": f"Meeting {idx}",
            "status": "confirmed",
            "start": {"dateTime": f"2025-0{(idx % 9) + 1}-01T10:00:00Z"},
            "end": {"dateTime": f"2025-0{(idx % 9) + 1}-01T11:00:00Z"},
            "attendees": [
                {"email": f"person{idx}@ext.com", "responseStatus": "accepted"},
                {"email": "me@kakaoventures.co.kr", "responseStatus": "accepted"},
            ],
        }

    def _make_client_with_mock_service(self, pages: list[list[dict]]) -> "GoogleCalendarClient":
        """
        Return a GoogleCalendarClient whose internal _service is mocked to
        return the provided pages of raw events.
        """
        from src.calendar.google_calendar import GoogleCalendarClient

        client = GoogleCalendarClient.__new__(GoogleCalendarClient)
        client._creds = MagicMock()
        client._creds.expired = False

        # Build mock service that returns pages sequentially
        mock_service = MagicMock()
        mock_events_resource = MagicMock()
        mock_service.events.return_value = mock_events_resource

        page_responses = []
        for i, page in enumerate(pages):
            response = {"items": page}
            if i < len(pages) - 1:
                response["nextPageToken"] = f"token_{i+1}"
            page_responses.append(response)

        mock_list = MagicMock()
        mock_request = MagicMock()
        mock_list.return_value = mock_request
        mock_request.execute.side_effect = page_responses
        mock_events_resource.list = mock_list

        client._service = mock_service
        return client

    def test_single_page(self):
        events_page = [self._make_raw_event(i) for i in range(5)]
        client = self._make_client_with_mock_service([events_page])
        meetings = client.list_all_historical_events(lookback_days=365, page_size=500)
        assert len(meetings) == 5

    def test_multi_page_aggregates_all(self):
        page1 = [self._make_raw_event(i) for i in range(3)]
        page2 = [self._make_raw_event(i) for i in range(3, 7)]
        client = self._make_client_with_mock_service([page1, page2])
        meetings = client.list_all_historical_events(lookback_days=365, page_size=500)
        assert len(meetings) == 7

    def test_empty_result(self):
        client = self._make_client_with_mock_service([[]])
        meetings = client.list_all_historical_events(lookback_days=365)
        assert meetings == []

    def test_returns_meeting_objects(self):
        from src.calendar.google_calendar import Meeting
        client = self._make_client_with_mock_service([[self._make_raw_event(0)]])
        meetings = client.list_all_historical_events()
        assert len(meetings) == 1
        assert isinstance(meetings[0], Meeting)

    def test_meeting_titles_populated(self):
        client = self._make_client_with_mock_service([[self._make_raw_event(3)]])
        meetings = client.list_all_historical_events()
        assert meetings[0].summary == "Meeting 3"

    def test_attendee_domains_accessible(self):
        client = self._make_client_with_mock_service([[self._make_raw_event(0)]])
        meetings = client.list_all_historical_events()
        emails = [a.email for a in meetings[0].attendees]
        domains = {e.split("@")[1] for e in emails if "@" in e}
        assert "ext.com" in domains
        assert "kakaoventures.co.kr" in domains


# ── Helper ────────────────────────────────────────────────────────────────────

class TestEmailDomainHelper:

    def test_normal(self):
        assert _email_domain("user@example.com") == "example.com"

    def test_uppercase(self):
        assert _email_domain("USER@EXAMPLE.COM") == "example.com"

    def test_empty_string(self):
        assert _email_domain("") == ""

    def test_no_at_sign(self):
        assert _email_domain("nodomain") == ""

    def test_subdomain(self):
        assert _email_domain("a@sub.corp.io") == "sub.corp.io"


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
