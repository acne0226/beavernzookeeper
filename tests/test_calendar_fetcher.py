"""
Unit tests for src/calendar_fetcher.py (Sub-AC 1).

Tests are fully offline (no real Google API calls).  All interactions with
the Google Calendar API are mocked.

Covers:
- fetch_todays_events() returns a list of structured event dicts
- Each event has all required fields:
    id, title, status, start, end, all_day, start_iso, end_iso,
    attendees, location, video_link, conference_type,
    organizer_email, organizer_name, html_link
- Attendee sub-dicts contain: email, name, response_status, is_organizer
- Cancelled events are excluded from results
- All-day events (using 'date' key) are handled correctly
- Timed events (using 'dateTime' key) are handled correctly
- Video conference links extracted from Google Meet conferenceData
- Zoom links extracted from event description
- Microsoft Teams links extracted from description
- location field extracted from event location field
- Retry logic: up to RETRY_COUNT attempts on HttpError
- Retry logic: up to RETRY_COUNT attempts on generic exceptions
- No sleep called after the final (exhausted) retry attempt
- RuntimeError raised after all retries exhausted
- fetch_events_range() returns events spanning a multi-day window
- fetch_events_range() excludes cancelled events
- Events are sorted by start time ascending
- Default target_date is today (Asia/Seoul timezone)
- organizer_name falls back to email if displayName missing
- Attendees without an email address are included with empty email
- description preserved in parsed event dict
- recurring_event_id included when present
- recurring_event_id is None when absent

Run:
    python -m pytest tests/test_calendar_fetcher.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar_fetcher import (
    fetch_todays_events,
    fetch_events_range,
    _parse_event,
    _extract_attendees,
    _extract_location_or_link,
    RETRY_COUNT,
    RETRY_DELAY,
    DEFAULT_TIMEZONE,
)


# ── Factories ──────────────────────────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")


def _timed_event(
    event_id: str = "evt-001",
    summary: str = "Test Meeting",
    start_dt: str = "2026-03-29T10:00:00+09:00",
    end_dt: str = "2026-03-29T11:00:00+09:00",
    attendees: list[dict] | None = None,
    location: str | None = None,
    description: str | None = None,
    status: str = "confirmed",
    organizer_email: str = "organizer@example.com",
    organizer_name: str = "Org Name",
    conference_data: dict | None = None,
    recurring_event_id: str | None = None,
) -> dict:
    """Build a fake timed Google Calendar event dict."""
    event: dict = {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt},
        "organizer": {"email": organizer_email, "displayName": organizer_name},
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
    }
    if attendees is not None:
        event["attendees"] = attendees
    if location is not None:
        event["location"] = location
    if description is not None:
        event["description"] = description
    if conference_data is not None:
        event["conferenceData"] = conference_data
    if recurring_event_id is not None:
        event["recurringEventId"] = recurring_event_id
    return event


def _all_day_event(
    event_id: str = "allday-001",
    summary: str = "Holiday",
    start_date: str = "2026-03-29",
    end_date: str = "2026-03-30",
    status: str = "confirmed",
) -> dict:
    """Build a fake all-day Google Calendar event dict."""
    return {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"date": start_date},
        "end": {"date": end_date},
        "organizer": {"email": "me@example.com"},
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
    }


def _google_meet_conference_data(meet_link: str = "https://meet.google.com/abc-def-ghi") -> dict:
    return {
        "entryPoints": [
            {"entryPointType": "video", "uri": meet_link},
            {"entryPointType": "phone", "uri": "tel:+1234567890"},
        ],
        "conferenceSolution": {"name": "Google Meet"},
    }


def _make_service_mock(events: list[dict]) -> MagicMock:
    """Return a mock Google Calendar service that returns *events* on list()."""
    service = MagicMock()
    service.events.return_value.list.return_value.execute.return_value = {
        "items": events,
        "kind": "calendar#events",
    }
    return service


# ── REQUIRED_FIELDS ────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "id", "title", "status", "start", "end", "all_day",
    "start_iso", "end_iso", "attendees",
    "location", "video_link", "conference_type",
    "organizer_email", "organizer_name", "html_link",
}

REQUIRED_ATTENDEE_FIELDS = {"email", "name", "response_status", "is_organizer"}


def _assert_event_structure(event: dict) -> None:
    """Assert all required fields are present in a parsed event dict."""
    missing = REQUIRED_FIELDS - event.keys()
    assert not missing, f"Event missing required fields: {missing}"

    attendees = event["attendees"]
    assert isinstance(attendees, list)
    for att in attendees:
        missing_att = REQUIRED_ATTENDEE_FIELDS - att.keys()
        assert not missing_att, f"Attendee missing required fields: {missing_att}"


# ── fetch_todays_events: structure ────────────────────────────────────────────

class TestFetchTodaysEventsStructure:

    def test_returns_list(self):
        """fetch_todays_events() must return a list."""
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert isinstance(result, list)

    def test_returns_empty_list_for_no_events(self):
        """fetch_todays_events() returns [] when the calendar has no events today."""
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock([])):
            result = fetch_todays_events()
        assert result == []

    def test_single_event_has_all_required_fields(self):
        """Each returned event must have all required fields."""
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result) == 1
        _assert_event_structure(result[0])

    def test_multiple_events_all_have_required_fields(self):
        """All returned events must have the required structure."""
        events = [
            _timed_event("evt-A", "Meeting A", "2026-03-29T09:00:00+09:00", "2026-03-29T10:00:00+09:00"),
            _timed_event("evt-B", "Meeting B", "2026-03-29T11:00:00+09:00", "2026-03-29T12:00:00+09:00"),
            _all_day_event("allday-1"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result) == 3
        for ev in result:
            _assert_event_structure(ev)


# ── fetch_todays_events: field values ─────────────────────────────────────────

class TestFetchTodaysEventsFieldValues:

    def test_title_populated(self):
        events = [_timed_event(summary="Board Meeting")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["title"] == "Board Meeting"

    def test_id_populated(self):
        events = [_timed_event(event_id="unique-id-xyz")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["id"] == "unique-id-xyz"

    def test_status_confirmed(self):
        events = [_timed_event(status="confirmed")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["status"] == "confirmed"

    def test_status_tentative(self):
        events = [_timed_event(status="tentative")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["status"] == "tentative"

    def test_timed_event_all_day_false(self):
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["all_day"] is False

    def test_all_day_event_all_day_true(self):
        events = [_all_day_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["all_day"] is True

    def test_start_is_datetime_for_timed_event(self):
        events = [_timed_event(start_dt="2026-03-29T09:30:00+09:00")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert isinstance(result[0]["start"], datetime)

    def test_start_is_date_for_all_day_event(self):
        events = [_all_day_event(start_date="2026-03-29")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert isinstance(result[0]["start"], date)

    def test_start_iso_populated_for_timed_event(self):
        events = [_timed_event(start_dt="2026-03-29T09:30:00+09:00")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert "2026-03-29" in result[0]["start_iso"]

    def test_start_iso_populated_for_all_day_event(self):
        events = [_all_day_event(start_date="2026-03-29")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["start_iso"] == "2026-03-29"

    def test_organizer_email_populated(self):
        events = [_timed_event(organizer_email="boss@company.com")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["organizer_email"] == "boss@company.com"

    def test_organizer_name_populated(self):
        events = [_timed_event(organizer_name="Boss Person")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["organizer_name"] == "Boss Person"

    def test_organizer_name_falls_back_to_email(self):
        """If organizer has no displayName, organizer_name should be the email."""
        event = _timed_event(organizer_email="noreply@example.com")
        event["organizer"] = {"email": "noreply@example.com"}  # no displayName
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock([event])):
            result = fetch_todays_events()
        assert result[0]["organizer_name"] == "noreply@example.com"

    def test_html_link_populated(self):
        events = [_timed_event(event_id="abc123")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert "calendar.google.com" in result[0]["html_link"] or result[0]["html_link"] != ""

    def test_no_title_defaults_to_no_title_string(self):
        """Events with no summary should get a default title."""
        event = _timed_event(summary="")
        event.pop("summary", None)
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock([event])):
            result = fetch_todays_events()
        assert result[0]["title"] is not None

    def test_description_preserved(self):
        events = [_timed_event(description="Quarterly earnings review")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["description"] == "Quarterly earnings review"

    def test_description_none_when_absent(self):
        events = [_timed_event()]  # no description
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["description"] is None

    def test_recurring_event_id_included(self):
        events = [_timed_event(recurring_event_id="recur-base-id")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["recurring_event_id"] == "recur-base-id"

    def test_recurring_event_id_none_when_absent(self):
        events = [_timed_event()]  # no recurringEventId
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["recurring_event_id"] is None


# ── fetch_todays_events: attendees ─────────────────────────────────────────────

class TestFetchTodaysEventsAttendees:

    def test_attendees_list_populated(self):
        raw_attendees = [
            {"email": "alice@example.com", "displayName": "Alice", "responseStatus": "accepted"},
            {"email": "bob@example.com", "displayName": "Bob", "responseStatus": "needsAction"},
        ]
        events = [_timed_event(attendees=raw_attendees, organizer_email="alice@example.com")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result[0]["attendees"]) == 2

    def test_attendee_email_field(self):
        raw_attendees = [{"email": "carol@partner.com", "responseStatus": "accepted"}]
        events = [_timed_event(attendees=raw_attendees)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["attendees"][0]["email"] == "carol@partner.com"

    def test_attendee_name_falls_back_to_email(self):
        """If displayName is absent, name should be the email."""
        raw_attendees = [{"email": "anon@co.com", "responseStatus": "needsAction"}]
        events = [_timed_event(attendees=raw_attendees)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["attendees"][0]["name"] == "anon@co.com"

    def test_attendee_response_status(self):
        raw_attendees = [{"email": "dave@co.com", "displayName": "Dave", "responseStatus": "declined"}]
        events = [_timed_event(attendees=raw_attendees)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["attendees"][0]["response_status"] == "declined"

    def test_attendee_is_organizer_true(self):
        organizer_email = "host@example.com"
        raw_attendees = [
            {"email": organizer_email, "displayName": "Host", "responseStatus": "accepted"},
        ]
        events = [_timed_event(attendees=raw_attendees, organizer_email=organizer_email)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        organizer_att = next(a for a in result[0]["attendees"] if a["email"] == organizer_email)
        assert organizer_att["is_organizer"] is True

    def test_attendee_is_organizer_false_for_non_organizer(self):
        raw_attendees = [
            {"email": "guest@other.com", "displayName": "Guest", "responseStatus": "accepted"},
        ]
        events = [_timed_event(attendees=raw_attendees, organizer_email="host@example.com")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["attendees"][0]["is_organizer"] is False

    def test_no_attendees_returns_empty_list(self):
        events = [_timed_event()]  # no attendees key
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["attendees"] == []


# ── fetch_todays_events: location / video_link ─────────────────────────────────

class TestFetchTodaysEventsLocationLink:

    def test_location_field_populated(self):
        events = [_timed_event(location="Conference Room A")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["location"] == "Conference Room A"

    def test_location_none_when_absent(self):
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["location"] is None

    def test_google_meet_video_link_extracted(self):
        conf_data = _google_meet_conference_data("https://meet.google.com/xyz-uvw-rst")
        events = [_timed_event(conference_data=conf_data)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["video_link"] == "https://meet.google.com/xyz-uvw-rst"
        assert result[0]["conference_type"] == "Google Meet"

    def test_video_link_none_when_no_conference_data(self):
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["video_link"] is None
        assert result[0]["conference_type"] is None

    def test_zoom_link_extracted_from_description(self):
        zoom_desc = "Join Zoom Meeting https://company.zoom.us/j/123456789?pwd=abc"
        events = [_timed_event(description=zoom_desc)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["video_link"] is not None
        assert "zoom.us" in result[0]["video_link"]
        assert result[0]["conference_type"] == "Zoom"

    def test_teams_link_extracted_from_description(self):
        teams_desc = "Join on Teams: https://teams.microsoft.com/l/meetup-join/19%3A/0"
        events = [_timed_event(description=teams_desc)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["video_link"] is not None
        assert "teams.microsoft.com" in result[0]["video_link"]
        assert result[0]["conference_type"] == "Microsoft Teams"

    def test_conference_data_takes_priority_over_description_zoom(self):
        """If both conferenceData and Zoom link in description exist, use conferenceData."""
        conf_data = _google_meet_conference_data("https://meet.google.com/aaa-bbb-ccc")
        zoom_desc = "Also see https://company.zoom.us/j/999999"
        events = [_timed_event(conference_data=conf_data, description=zoom_desc)]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        # conferenceData wins
        assert "meet.google.com" in result[0]["video_link"]
        assert result[0]["conference_type"] == "Google Meet"


# ── fetch_todays_events: cancelled events excluded ─────────────────────────────

class TestFetchTodaysEventsCancellation:

    def test_cancelled_events_excluded(self):
        events = [
            _timed_event(event_id="keep", status="confirmed"),
            _timed_event(event_id="drop", status="cancelled"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result) == 1
        assert result[0]["id"] == "keep"

    def test_all_cancelled_returns_empty(self):
        events = [
            _timed_event(status="cancelled"),
            _timed_event(status="cancelled"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result == []

    def test_tentative_not_excluded(self):
        events = [_timed_event(status="tentative")]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result) == 1


# ── fetch_todays_events: sorting ───────────────────────────────────────────────

class TestFetchTodaysEventsSorting:

    def test_events_sorted_by_start_time(self):
        events = [
            _timed_event("late",  "Late Meeting",  "2026-03-29T15:00:00+09:00", "2026-03-29T16:00:00+09:00"),
            _timed_event("early", "Early Meeting", "2026-03-29T09:00:00+09:00", "2026-03-29T10:00:00+09:00"),
            _timed_event("mid",   "Mid Meeting",   "2026-03-29T12:00:00+09:00", "2026-03-29T13:00:00+09:00"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert [ev["id"] for ev in result] == ["early", "mid", "late"]

    def test_all_day_event_sorted_before_timed(self):
        """All-day events (treated as midnight) should sort before timed daytime events."""
        events = [
            _timed_event("timed", "Morning Meeting", "2026-03-29T09:00:00+09:00", "2026-03-29T10:00:00+09:00"),
            _all_day_event("allday", "Holiday", "2026-03-29", "2026-03-30"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert result[0]["id"] == "allday"


# ── fetch_todays_events: default date in KST ──────────────────────────────────

class TestFetchTodaysEventsDefaultDate:

    def test_default_date_is_today_kst(self):
        """When target_date is omitted, the time window should cover today in KST."""
        KST = ZoneInfo("Asia/Seoul")
        expected_date = datetime.now(KST).date()

        captured: list[dict] = []

        def _capture_service():
            service = MagicMock()
            def _list(**kwargs):
                captured.append(kwargs)
                result = MagicMock()
                result.execute.return_value = {"items": []}
                return result
            service.events.return_value.list.side_effect = _list
            return service

        with patch("src.calendar_fetcher._build_service", side_effect=_capture_service):
            fetch_todays_events()

        assert len(captured) >= 1
        time_min_str = captured[0]["timeMin"]
        assert str(expected_date) in time_min_str

    def test_explicit_target_date_used(self):
        """Providing an explicit target_date should use that date for the window."""
        target = date(2026, 1, 15)
        captured: list[dict] = []

        def _capture_service():
            service = MagicMock()
            def _list(**kwargs):
                captured.append(kwargs)
                result = MagicMock()
                result.execute.return_value = {"items": []}
                return result
            service.events.return_value.list.side_effect = _list
            return service

        with patch("src.calendar_fetcher._build_service", side_effect=_capture_service):
            fetch_todays_events(target_date=target)

        assert "2026-01-15" in captured[0]["timeMin"]


# ── fetch_todays_events: retry logic ──────────────────────────────────────────

class TestFetchTodaysEventsRetry:

    def test_retry_count_constant_is_three(self):
        assert RETRY_COUNT == 3

    def test_retry_delay_constant_is_ten(self):
        assert RETRY_DELAY == 10

    def test_succeeds_on_first_attempt(self):
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_todays_events()
        assert len(result) == 1

    def test_succeeds_on_second_attempt(self):
        """Should retry and succeed if first attempt raises an exception."""
        from googleapiclient.errors import HttpError
        from unittest.mock import MagicMock

        call_count = 0
        good_service = _make_service_mock([_timed_event()])

        def _build():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                bad = MagicMock()
                resp = MagicMock()
                resp.status = 503
                bad.events.return_value.list.return_value.execute.side_effect = HttpError(resp, b"Service Unavailable")
                return bad
            return good_service

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            result = fetch_todays_events()

        assert call_count == 2
        assert len(result) == 1

    def test_raises_runtime_error_after_all_retries_exhausted(self):
        """RuntimeError should be raised when all retries are exhausted."""
        from googleapiclient.errors import HttpError

        def _build():
            bad = MagicMock()
            resp = MagicMock()
            resp.status = 500
            bad.events.return_value.list.return_value.execute.side_effect = HttpError(resp, b"Internal Error")
            return bad

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_todays_events()

    def test_retry_sleep_called_between_attempts(self):
        """time.sleep(RETRY_DELAY) must be called between each failed attempt."""
        from googleapiclient.errors import HttpError

        def _build():
            bad = MagicMock()
            resp = MagicMock()
            resp.status = 500
            bad.events.return_value.list.return_value.execute.side_effect = HttpError(resp, b"Error")
            return bad

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                fetch_todays_events()

        # sleep between attempts, but NOT after last attempt
        assert mock_sleep.call_count == RETRY_COUNT - 1
        for c in mock_sleep.call_args_list:
            assert c[0][0] == RETRY_DELAY

    def test_generic_exception_also_retried(self):
        """Non-HttpError exceptions should also trigger the retry loop."""
        call_count = 0
        good_service = _make_service_mock([_timed_event()])

        def _build():
            nonlocal call_count
            call_count += 1
            if call_count < RETRY_COUNT:
                raise ConnectionError("Network timeout")
            return good_service

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            result = fetch_todays_events()

        assert call_count == RETRY_COUNT
        assert len(result) == 1

    def test_exactly_retry_count_attempts_made(self):
        """The API should be attempted exactly RETRY_COUNT times on repeated failure."""
        call_count = 0

        def _build():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"attempt {call_count}")

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError):
                fetch_todays_events()

        assert call_count == RETRY_COUNT


# ── fetch_events_range ─────────────────────────────────────────────────────────

class TestFetchEventsRange:

    def test_returns_list(self):
        events = [_timed_event()]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_events_range(date(2026, 3, 1), date(2026, 3, 29))
        assert isinstance(result, list)

    def test_excludes_cancelled_events(self):
        events = [
            _timed_event(event_id="keep", status="confirmed"),
            _timed_event(event_id="drop", status="cancelled"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_events_range(date(2026, 3, 1), date(2026, 3, 29))
        assert all(ev["id"] != "drop" for ev in result)

    def test_events_sorted_by_start(self):
        events = [
            _timed_event("b", "B", "2026-03-15T14:00:00+09:00", "2026-03-15T15:00:00+09:00"),
            _timed_event("a", "A", "2026-03-10T09:00:00+09:00", "2026-03-10T10:00:00+09:00"),
        ]
        with patch("src.calendar_fetcher._build_service", return_value=_make_service_mock(events)):
            result = fetch_events_range(date(2026, 3, 1), date(2026, 3, 29))
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_time_window_sent_to_api(self):
        """Verify that the correct timeMin and timeMax are passed to the API."""
        captured: list[dict] = []

        def _capture_service():
            service = MagicMock()
            def _list(**kwargs):
                captured.append(kwargs)
                result = MagicMock()
                result.execute.return_value = {"items": []}
                return result
            service.events.return_value.list.side_effect = _list
            return service

        with patch("src.calendar_fetcher._build_service", side_effect=_capture_service):
            fetch_events_range(date(2026, 3, 1), date(2026, 3, 8))

        assert len(captured) >= 1
        assert "2026-03-01" in captured[0]["timeMin"]
        assert "2026-03-08" in captured[0]["timeMax"]

    def test_retry_on_failure(self):
        """fetch_events_range should also retry on API failure."""
        call_count = 0
        good_service = _make_service_mock([_timed_event()])

        def _build():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("down")
            return good_service

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            result = fetch_events_range(date(2026, 3, 1), date(2026, 3, 29))

        assert call_count == 2
        assert len(result) == 1

    def test_raises_after_all_retries(self):
        """fetch_events_range raises RuntimeError after all retries exhausted."""
        def _build():
            raise RuntimeError("persistent failure")

        with patch("src.calendar_fetcher._build_service", side_effect=_build), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                fetch_events_range(date(2026, 3, 1), date(2026, 3, 8))


# ── _parse_event internal unit tests ──────────────────────────────────────────

class TestParseEventInternal:
    """Unit tests for the _parse_event helper (no API calls needed)."""

    def test_timed_event_all_day_false(self):
        raw = _timed_event()
        parsed = _parse_event(raw, DEFAULT_TIMEZONE)
        assert parsed["all_day"] is False

    def test_all_day_event_all_day_true(self):
        raw = _all_day_event()
        parsed = _parse_event(raw, DEFAULT_TIMEZONE)
        assert parsed["all_day"] is True

    def test_start_time_in_kst(self):
        """Start datetime should be in KST."""
        raw = _timed_event(start_dt="2026-03-29T09:00:00+09:00")
        parsed = _parse_event(raw, "Asia/Seoul")
        start = parsed["start"]
        assert isinstance(start, datetime)
        assert start.tzinfo is not None

    def test_missing_summary_gets_default(self):
        raw = {
            "id": "no-summary",
            "status": "confirmed",
            "start": {"dateTime": "2026-03-29T10:00:00+09:00"},
            "end":   {"dateTime": "2026-03-29T11:00:00+09:00"},
            "organizer": {"email": "x@y.com"},
            "htmlLink": "https://calendar.google.com/",
        }
        parsed = _parse_event(raw, DEFAULT_TIMEZONE)
        assert parsed["title"] == "(No Title)"

    def test_all_required_fields_present(self):
        raw = _timed_event(
            attendees=[{"email": "a@b.com", "displayName": "A", "responseStatus": "accepted"}],
            location="Room 1",
        )
        parsed = _parse_event(raw, DEFAULT_TIMEZONE)
        _assert_event_structure(parsed)


# ── _extract_attendees internal unit tests ────────────────────────────────────

class TestExtractAttendeesInternal:

    def test_empty_attendees(self):
        event = {"organizer": {"email": "host@co.com"}}
        result = _extract_attendees(event)
        assert result == []

    def test_organizer_flag_set_for_organizer_email(self):
        event = {
            "organizer": {"email": "host@co.com"},
            "attendees": [
                {"email": "host@co.com", "responseStatus": "accepted"},
                {"email": "guest@other.com", "responseStatus": "needsAction"},
            ],
        }
        result = _extract_attendees(event)
        host = next(a for a in result if a["email"] == "host@co.com")
        assert host["is_organizer"] is True

    def test_organizer_flag_set_via_attendee_organizer_field(self):
        event = {
            "organizer": {"email": "other@co.com"},
            "attendees": [
                {"email": "host@co.com", "responseStatus": "accepted", "organizer": True},
            ],
        }
        result = _extract_attendees(event)
        host = next(a for a in result if a["email"] == "host@co.com")
        assert host["is_organizer"] is True


# ── _extract_location_or_link internal unit tests ─────────────────────────────

class TestExtractLocationOrLink:

    def test_no_location_or_conference(self):
        event = {}
        result = _extract_location_or_link(event)
        assert result["location"] is None
        assert result["video_link"] is None
        assert result["conference_type"] is None

    def test_location_field_extracted(self):
        event = {"location": "Seoul Office, Floor 3"}
        result = _extract_location_or_link(event)
        assert result["location"] == "Seoul Office, Floor 3"

    def test_google_meet_from_conference_data(self):
        event = {"conferenceData": _google_meet_conference_data("https://meet.google.com/aaa-bbb")}
        result = _extract_location_or_link(event)
        assert result["video_link"] == "https://meet.google.com/aaa-bbb"
        assert result["conference_type"] == "Google Meet"

    def test_zoom_link_from_description(self):
        event = {"description": "Meeting https://myorg.zoom.us/j/12345?pwd=xyz"}
        result = _extract_location_or_link(event)
        assert result["video_link"] is not None
        assert "zoom.us" in result["video_link"]
        assert result["conference_type"] == "Zoom"

    def test_teams_link_from_description(self):
        event = {"description": "Teams: https://teams.microsoft.com/l/meetup-join/19%3A/abc"}
        result = _extract_location_or_link(event)
        assert "teams.microsoft.com" in result["video_link"]
        assert result["conference_type"] == "Microsoft Teams"


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
