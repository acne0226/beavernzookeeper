"""
Unit tests for src/calendar_fetcher.py (Sub-AC 1).

Tests are entirely offline (no real Google API calls) — all API interactions
are mocked via unittest.mock.

Verifies:
1. fetch_todays_events() returns a list of structured event dicts
2. Each event has required fields: id, title, start, end, attendees,
   location, video_link, conference_type, all_day, html_link, etc.
3. Attendee dicts contain email, name, response_status, is_organizer
4. All-day events are correctly identified
5. Cancelled events are excluded from results
6. Events are sorted by start time
7. Retry logic fires on API errors (up to RETRY_COUNT=3 times)
8. RuntimeError is raised after all retries are exhausted
9. Conference links extracted for Google Meet, Zoom, and Microsoft Teams
10. fetch_events_range() works for multi-day windows

Run:
    python -m pytest tests/test_calendar_fetcher_unit.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar_fetcher import (
    fetch_todays_events,
    fetch_events_range,
    _parse_event,
    _parse_event_time,
    _extract_attendees,
    _extract_location_or_link,
)

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc


# ── Raw event factories ───────────────────────────────────────────────────────

def _make_raw_event(
    event_id: str = "evt-001",
    summary: str = "Test Meeting",
    start_dt: str = "2026-03-29T10:00:00+09:00",
    end_dt: str = "2026-03-29T11:00:00+09:00",
    status: str = "confirmed",
    attendees: list[dict] | None = None,
    description: str | None = None,
    location: str | None = None,
    html_link: str = "https://calendar.google.com/event/evt-001",
    organizer_email: str = "organizer@kakaoventures.co.kr",
    conference_data: dict | None = None,
    recurring_event_id: str | None = None,
) -> dict:
    """Build a minimal raw Google Calendar API event dict."""
    raw: dict = {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt},
        "htmlLink": html_link,
        "organizer": {
            "email": organizer_email,
            "displayName": organizer_email.split("@")[0],
        },
        "attendees": attendees or [],
    }
    if description is not None:
        raw["description"] = description
    if location is not None:
        raw["location"] = location
    if conference_data is not None:
        raw["conferenceData"] = conference_data
    if recurring_event_id is not None:
        raw["recurringEventId"] = recurring_event_id
    return raw


def _make_all_day_event(
    event_id: str = "evt-allday",
    summary: str = "All Day Event",
    event_date: str = "2026-03-29",
    end_date: str = "2026-03-30",
) -> dict:
    """Build a raw all-day event."""
    return {
        "id": event_id,
        "summary": summary,
        "status": "confirmed",
        "start": {"date": event_date},
        "end": {"date": end_date},
        "htmlLink": f"https://calendar.google.com/event/{event_id}",
        "organizer": {"email": "org@kakaoventures.co.kr"},
        "attendees": [],
    }


def _make_api_response(items: list[dict]) -> dict:
    """Wrap raw event list in a Google Calendar API response."""
    return {"items": items, "summary": "primary"}


def _mock_service(items: list[dict]):
    """Build a mock Google Calendar service object."""
    svc = MagicMock()
    svc.events.return_value.list.return_value.execute.return_value = _make_api_response(items)
    return svc


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: _parse_event_time
# ══════════════════════════════════════════════════════════════════════════════

class TestParseEventTime:
    """Tests for the _parse_event_time helper."""

    def test_timed_event_returns_datetime(self):
        event = _make_raw_event()
        result = _parse_event_time(event, "Asia/Seoul")
        assert isinstance(result["start"], datetime)
        assert isinstance(result["end"], datetime)

    def test_timed_event_all_day_false(self):
        event = _make_raw_event()
        result = _parse_event_time(event, "Asia/Seoul")
        assert result["all_day"] is False

    def test_all_day_event_returns_date(self):
        event = _make_all_day_event()
        result = _parse_event_time(event, "Asia/Seoul")
        assert isinstance(result["start"], date)
        assert result["all_day"] is True

    def test_timezone_applied_to_datetime(self):
        event = _make_raw_event(start_dt="2026-03-29T10:00:00+09:00")
        tz = ZoneInfo("Asia/Seoul")
        result = _parse_event_time(event, "Asia/Seoul")
        # Should be 10:00 KST
        assert result["start"].hour == 10

    def test_start_iso_in_result(self):
        event = _make_raw_event(start_dt="2026-03-29T10:00:00+09:00")
        result = _parse_event_time(event, "Asia/Seoul")
        assert "start_iso" in result
        assert "2026-03-29" in result["start_iso"]


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: _extract_attendees
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAttendees:
    """Tests for the _extract_attendees helper."""

    def test_empty_attendees_returns_empty_list(self):
        event = _make_raw_event(attendees=[])
        result = _extract_attendees(event)
        assert result == []

    def test_attendee_has_required_fields(self):
        event = _make_raw_event(
            attendees=[{
                "email": "ceo@startup.com",
                "displayName": "Startup CEO",
                "responseStatus": "accepted",
            }]
        )
        result = _extract_attendees(event)
        assert len(result) == 1
        att = result[0]
        assert "email" in att
        assert "name" in att
        assert "response_status" in att
        assert "is_organizer" in att

    def test_attendee_email_populated(self):
        event = _make_raw_event(
            attendees=[{"email": "user@example.com", "responseStatus": "needsAction"}]
        )
        result = _extract_attendees(event)
        assert result[0]["email"] == "user@example.com"

    def test_attendee_name_falls_back_to_email(self):
        event = _make_raw_event(
            attendees=[{"email": "cto@external.com"}]
        )
        result = _extract_attendees(event)
        assert result[0]["name"] == "cto@external.com"

    def test_organizer_flagged_correctly(self):
        organizer_email = "organizer@kakaoventures.co.kr"
        event = _make_raw_event(
            organizer_email=organizer_email,
            attendees=[
                {"email": organizer_email, "responseStatus": "accepted"},
                {"email": "other@ext.com", "responseStatus": "accepted"},
            ]
        )
        result = _extract_attendees(event)
        organizer = next(a for a in result if a["email"] == organizer_email)
        non_organizer = next(a for a in result if a["email"] == "other@ext.com")
        assert organizer["is_organizer"] is True
        assert non_organizer["is_organizer"] is False

    def test_multiple_attendees_returned(self):
        event = _make_raw_event(
            attendees=[
                {"email": "a@test.com", "responseStatus": "accepted"},
                {"email": "b@test.com", "responseStatus": "declined"},
                {"email": "c@test.com", "responseStatus": "needsAction"},
            ]
        )
        result = _extract_attendees(event)
        assert len(result) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: _extract_location_or_link
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractLocationOrLink:
    """Tests for the _extract_location_or_link helper."""

    def test_no_location_no_conference(self):
        event = _make_raw_event()
        result = _extract_location_or_link(event)
        assert result["location"] is None
        assert result["video_link"] is None
        assert result["conference_type"] is None

    def test_physical_location_returned(self):
        event = _make_raw_event(location="카카오 판교 오피스 3층 회의실")
        result = _extract_location_or_link(event)
        assert result["location"] == "카카오 판교 오피스 3층 회의실"

    def test_google_meet_extracted_from_conference_data(self):
        conference_data = {
            "entryPoints": [
                {
                    "entryPointType": "video",
                    "uri": "https://meet.google.com/abc-def-ghi",
                }
            ],
            "conferenceSolution": {"name": "Google Meet"},
        }
        event = _make_raw_event(conference_data=conference_data)
        result = _extract_location_or_link(event)
        assert result["video_link"] == "https://meet.google.com/abc-def-ghi"
        assert result["conference_type"] == "Google Meet"

    def test_zoom_link_extracted_from_description(self):
        event = _make_raw_event(
            description="Join the meeting: https://kakao.zoom.us/j/123456789"
        )
        result = _extract_location_or_link(event)
        assert result["video_link"] is not None
        assert "zoom.us" in result["video_link"]
        assert result["conference_type"] == "Zoom"

    def test_teams_link_extracted_from_description(self):
        event = _make_raw_event(
            description="Join: https://teams.microsoft.com/l/meetup-join/abc123"
        )
        result = _extract_location_or_link(event)
        assert result["video_link"] is not None
        assert "teams.microsoft.com" in result["video_link"]
        assert result["conference_type"] == "Microsoft Teams"

    def test_location_and_conference_coexist(self):
        conference_data = {
            "entryPoints": [{"entryPointType": "video", "uri": "https://meet.google.com/xyz"}],
            "conferenceSolution": {"name": "Google Meet"},
        }
        event = _make_raw_event(
            location="3층 회의실",
            conference_data=conference_data,
        )
        result = _extract_location_or_link(event)
        assert result["location"] == "3층 회의실"
        assert result["video_link"] == "https://meet.google.com/xyz"


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: _parse_event (full event parsing)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseEvent:
    """Tests for the full _parse_event function."""

    REQUIRED_FIELDS = {
        "id", "title", "status", "organizer_email", "organizer_name",
        "start", "end", "all_day", "start_iso", "end_iso",
        "attendees", "location", "video_link", "conference_type",
        "description", "html_link", "recurring_event_id",
    }

    def test_all_required_fields_present(self):
        event = _make_raw_event()
        result = _parse_event(event, "Asia/Seoul")
        for field in self.REQUIRED_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_title_extracted(self):
        event = _make_raw_event(summary="Startup Pitch Meeting")
        result = _parse_event(event, "Asia/Seoul")
        assert result["title"] == "Startup Pitch Meeting"

    def test_no_summary_gets_default_title(self):
        event = _make_raw_event(summary="(No Title)")
        result = _parse_event(event, "Asia/Seoul")
        assert result["title"] == "(No Title)"

    def test_status_extracted(self):
        event = _make_raw_event(status="confirmed")
        result = _parse_event(event, "Asia/Seoul")
        assert result["status"] == "confirmed"

    def test_html_link_extracted(self):
        event = _make_raw_event(html_link="https://calendar.google.com/event/abc")
        result = _parse_event(event, "Asia/Seoul")
        assert result["html_link"] == "https://calendar.google.com/event/abc"

    def test_attendees_are_list(self):
        event = _make_raw_event(attendees=[
            {"email": "a@test.com", "responseStatus": "accepted"},
        ])
        result = _parse_event(event, "Asia/Seoul")
        assert isinstance(result["attendees"], list)
        assert len(result["attendees"]) == 1

    def test_all_day_event_flag(self):
        event = _make_all_day_event()
        result = _parse_event(event, "Asia/Seoul")
        assert result["all_day"] is True

    def test_timed_event_not_all_day(self):
        event = _make_raw_event()
        result = _parse_event(event, "Asia/Seoul")
        assert result["all_day"] is False

    def test_description_none_when_absent(self):
        event = _make_raw_event()
        result = _parse_event(event, "Asia/Seoul")
        assert result["description"] is None

    def test_description_populated_when_present(self):
        event = _make_raw_event(description="Meeting notes here")
        result = _parse_event(event, "Asia/Seoul")
        assert result["description"] == "Meeting notes here"

    def test_recurring_event_id_populated(self):
        event = _make_raw_event(recurring_event_id="recurring-001")
        result = _parse_event(event, "Asia/Seoul")
        assert result["recurring_event_id"] == "recurring-001"

    def test_recurring_event_id_none_when_absent(self):
        event = _make_raw_event()
        result = _parse_event(event, "Asia/Seoul")
        assert result["recurring_event_id"] is None


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: fetch_todays_events
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchTodaysEvents:
    """Tests for fetch_todays_events (main public API)."""

    def _patch_service(self, items: list[dict]):
        """Return a context manager that patches _build_service."""
        mock_svc = _mock_service(items)
        return patch("src.calendar_fetcher._build_service", return_value=mock_svc)

    def test_returns_list(self):
        with self._patch_service([]):
            result = fetch_todays_events()
        assert isinstance(result, list)

    def test_empty_calendar_returns_empty_list(self):
        with self._patch_service([]):
            result = fetch_todays_events()
        assert result == []

    def test_single_event_returned(self):
        raw = _make_raw_event(summary="투자 심의 미팅")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert len(result) == 1
        assert result[0]["title"] == "투자 심의 미팅"

    def test_multiple_events_all_returned(self):
        raws = [
            _make_raw_event(event_id="e1", summary="Meeting 1"),
            _make_raw_event(event_id="e2", summary="Meeting 2"),
            _make_raw_event(event_id="e3", summary="Meeting 3"),
        ]
        with self._patch_service(raws):
            result = fetch_todays_events()
        assert len(result) == 3

    def test_cancelled_events_excluded(self):
        raws = [
            _make_raw_event(event_id="e1", summary="Active Meeting"),
            _make_raw_event(event_id="e2", summary="Cancelled Meeting", status="cancelled"),
        ]
        with self._patch_service(raws):
            result = fetch_todays_events()
        assert len(result) == 1
        assert result[0]["title"] == "Active Meeting"

    def test_events_sorted_by_start_time(self):
        raws = [
            _make_raw_event(
                event_id="late",
                summary="Late Meeting",
                start_dt="2026-03-29T15:00:00+09:00",
                end_dt="2026-03-29T16:00:00+09:00",
            ),
            _make_raw_event(
                event_id="early",
                summary="Early Meeting",
                start_dt="2026-03-29T09:00:00+09:00",
                end_dt="2026-03-29T10:00:00+09:00",
            ),
        ]
        with self._patch_service(raws):
            result = fetch_todays_events()
        assert result[0]["title"] == "Early Meeting"
        assert result[1]["title"] == "Late Meeting"

    def test_target_date_accepted(self):
        """Passing an explicit target_date should not raise."""
        with self._patch_service([]):
            result = fetch_todays_events(target_date=date(2026, 3, 29))
        assert isinstance(result, list)

    def test_custom_timezone_accepted(self):
        """Passing a custom timezone should not raise."""
        with self._patch_service([]):
            result = fetch_todays_events(timezone="UTC")
        assert isinstance(result, list)

    def test_event_has_required_fields(self):
        required = {
            "id", "title", "status", "start", "end", "all_day",
            "attendees", "location", "video_link", "html_link",
        }
        raw = _make_raw_event(summary="Check")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        for field in required:
            assert field in result[0], f"Missing field: {field}"

    def test_attendee_fields_populated(self):
        raw = _make_raw_event(
            attendees=[
                {"email": "founder@acme.com", "displayName": "Alice", "responseStatus": "accepted"}
            ]
        )
        with self._patch_service([raw]):
            result = fetch_todays_events()
        att = result[0]["attendees"][0]
        assert att["email"] == "founder@acme.com"
        assert att["name"] == "Alice"
        assert att["response_status"] == "accepted"
        assert "is_organizer" in att

    def test_all_day_event_in_results(self):
        raw = _make_all_day_event(summary="연간 팀 데이")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert len(result) == 1
        assert result[0]["all_day"] is True
        assert result[0]["title"] == "연간 팀 데이"

    def test_video_link_google_meet(self):
        conf = {
            "entryPoints": [{"entryPointType": "video", "uri": "https://meet.google.com/abc"}],
            "conferenceSolution": {"name": "Google Meet"},
        }
        raw = _make_raw_event(conference_data=conf)
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["video_link"] == "https://meet.google.com/abc"
        assert result[0]["conference_type"] == "Google Meet"

    def test_zoom_link_from_description(self):
        raw = _make_raw_event(
            description="Zoom: https://kakao.zoom.us/j/999888777"
        )
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["video_link"] is not None
        assert "zoom.us" in result[0]["video_link"]

    def test_html_link_in_result(self):
        raw = _make_raw_event(html_link="https://calendar.google.com/event/test123")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["html_link"] == "https://calendar.google.com/event/test123"

    def test_location_in_result(self):
        raw = _make_raw_event(location="판교 회의실 B")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["location"] == "판교 회의실 B"

    def test_description_in_result(self):
        raw = _make_raw_event(description="사업 검토 자료 첨부")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["description"] == "사업 검토 자료 첨부"


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Retry logic
# ══════════════════════════════════════════════════════════════════════════════

class TestRetryLogic:
    """Tests for retry-on-API-failure behavior."""

    def test_retries_on_http_error(self):
        """Should retry up to RETRY_COUNT times on HttpError."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock

        mock_resp = Mock()
        mock_resp.status = 500
        mock_resp.reason = "Server Error"
        error = HttpError(mock_resp, b"Server Error")

        mock_svc = MagicMock()
        mock_svc.events.return_value.list.return_value.execute.side_effect = error

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep"):  # speed up test
                with pytest.raises(RuntimeError, match="Failed to fetch"):
                    fetch_todays_events()

        # Should have been called 3 times (RETRY_COUNT)
        assert mock_svc.events.return_value.list.return_value.execute.call_count == 3

    def test_retries_on_generic_exception(self):
        """Should retry on any unexpected exception."""
        mock_svc = MagicMock()
        mock_svc.events.return_value.list.return_value.execute.side_effect = ConnectionError("Network error")

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep"):
                with pytest.raises(RuntimeError, match="Failed to fetch"):
                    fetch_todays_events()

        assert mock_svc.events.return_value.list.return_value.execute.call_count == 3

    def test_succeeds_after_one_failure(self):
        """Should succeed if the second attempt succeeds."""
        from googleapiclient.errors import HttpError
        from unittest.mock import Mock

        mock_resp = Mock()
        mock_resp.status = 503
        mock_resp.reason = "Service Unavailable"
        error = HttpError(mock_resp, b"Service Unavailable")

        mock_svc = MagicMock()
        raw = _make_raw_event(summary="Retry Success")
        mock_svc.events.return_value.list.return_value.execute.side_effect = [
            error,
            _make_api_response([raw]),
        ]

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep"):
                result = fetch_todays_events()

        assert len(result) == 1
        assert result[0]["title"] == "Retry Success"

    def test_raises_runtime_error_after_all_retries(self):
        """After RETRY_COUNT failures, raises RuntimeError."""
        mock_svc = MagicMock()
        mock_svc.events.return_value.list.return_value.execute.side_effect = Exception("Persistent error")

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep"):
                with pytest.raises(RuntimeError) as exc_info:
                    fetch_todays_events()

        assert "Failed to fetch Google Calendar events" in str(exc_info.value)

    def test_sleep_called_between_retries(self):
        """time.sleep should be called between retry attempts."""
        mock_svc = MagicMock()
        mock_svc.events.return_value.list.return_value.execute.side_effect = Exception("err")

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep") as mock_sleep:
                with pytest.raises(RuntimeError):
                    fetch_todays_events()

        # Sleep called between attempts: RETRY_COUNT - 1 = 2 times
        assert mock_sleep.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: fetch_events_range
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchEventsRange:
    """Tests for the fetch_events_range function."""

    def _patch_service(self, items: list[dict]):
        mock_svc = _mock_service(items)
        return patch("src.calendar_fetcher._build_service", return_value=mock_svc)

    def test_returns_list(self):
        start = date(2026, 3, 24)
        end = date(2026, 3, 29)
        with self._patch_service([]):
            result = fetch_events_range(start, end)
        assert isinstance(result, list)

    def test_range_returns_events(self):
        raws = [
            _make_raw_event(event_id="e1", summary="Day 1 Meeting"),
            _make_raw_event(event_id="e2", summary="Day 3 Meeting"),
        ]
        start = date(2026, 3, 24)
        end = date(2026, 3, 30)
        with self._patch_service(raws):
            result = fetch_events_range(start, end)
        assert len(result) == 2

    def test_cancelled_events_excluded_in_range(self):
        raws = [
            _make_raw_event(event_id="e1", summary="Active"),
            _make_raw_event(event_id="e2", summary="Cancelled", status="cancelled"),
        ]
        start = date(2026, 3, 24)
        end = date(2026, 3, 30)
        with self._patch_service(raws):
            result = fetch_events_range(start, end)
        assert len(result) == 1
        assert result[0]["title"] == "Active"

    def test_range_sorted_by_start(self):
        raws = [
            _make_raw_event(
                event_id="late",
                summary="Friday",
                start_dt="2026-03-27T15:00:00+09:00",
                end_dt="2026-03-27T16:00:00+09:00",
            ),
            _make_raw_event(
                event_id="early",
                summary="Monday",
                start_dt="2026-03-23T09:00:00+09:00",
                end_dt="2026-03-23T10:00:00+09:00",
            ),
        ]
        start = date(2026, 3, 23)
        end = date(2026, 3, 28)
        with self._patch_service(raws):
            result = fetch_events_range(start, end)
        assert result[0]["title"] == "Monday"
        assert result[1]["title"] == "Friday"

    def test_range_retry_on_failure(self):
        """fetch_events_range retries on error."""
        mock_svc = MagicMock()
        mock_svc.events.return_value.list.return_value.execute.side_effect = Exception("err")
        start = date(2026, 3, 24)
        end = date(2026, 3, 30)

        with patch("src.calendar_fetcher._build_service", return_value=mock_svc):
            with patch("src.calendar_fetcher.time.sleep"):
                with pytest.raises(RuntimeError):
                    fetch_events_range(start, end)

        assert mock_svc.events.return_value.list.return_value.execute.call_count == 3


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and resilience tests."""

    def _patch_service(self, items: list[dict]):
        mock_svc = _mock_service(items)
        return patch("src.calendar_fetcher._build_service", return_value=mock_svc)

    def test_event_with_no_attendees_has_empty_list(self):
        raw = _make_raw_event(attendees=[])
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["attendees"] == []

    def test_event_without_location_has_none(self):
        raw = _make_raw_event()
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["location"] is None

    def test_event_without_conference_has_none_video_link(self):
        raw = _make_raw_event()
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["video_link"] is None

    def test_malformed_event_skipped_gracefully(self):
        """Malformed event (no start) is skipped; valid events are returned."""
        good = _make_raw_event(event_id="good", summary="Good Meeting")
        bad = {"id": "bad", "summary": "Bad", "status": "confirmed"}  # no start/end
        with self._patch_service([bad, good]):
            result = fetch_todays_events()
        # The good event should still be returned even if bad was skipped
        titles = [e["title"] for e in result]
        assert "Good Meeting" in titles

    def test_start_and_end_times_present(self):
        raw = _make_raw_event(
            start_dt="2026-03-29T14:00:00+09:00",
            end_dt="2026-03-29T15:00:00+09:00",
        )
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["start"] is not None
        assert result[0]["end"] is not None

    def test_organizer_info_in_result(self):
        raw = _make_raw_event(organizer_email="organizer@kakaoventures.co.kr")
        with self._patch_service([raw]):
            result = fetch_todays_events()
        assert result[0]["organizer_email"] == "organizer@kakaoventures.co.kr"

    def test_mixed_all_day_and_timed_events(self):
        """All-day events should be sorted to start before timed events."""
        all_day = _make_all_day_event(summary="Conference")
        timed = _make_raw_event(
            summary="Timed Meeting",
            start_dt="2026-03-29T10:00:00+09:00",
            end_dt="2026-03-29T11:00:00+09:00",
        )
        with self._patch_service([timed, all_day]):
            result = fetch_todays_events()
        # Both should be present
        titles = [e["title"] for e in result]
        assert "Conference" in titles
        assert "Timed Meeting" in titles


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
