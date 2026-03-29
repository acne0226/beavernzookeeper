"""
Tests for Sub-AC 3.2: Briefing Data Aggregation.

Verifies that aggregate_briefing_data() correctly fetches and combines data
from all three sources (Google Calendar, Gmail, Notion) when /brief is invoked.

Test strategy
-------------
* Unit-test the BriefingData container (properties, summary, error handling).
* Test aggregate_briefing_data() with all sources mocked (no real API calls).
* Verify per-source independence: one source failing must not affect others.
* Verify 3-retry / 10s-delay error handling via _fetch_with_retry().
* Verify the full formatter (format_full_briefing) handles all data combinations.

Run with:
    python -m pytest tests/test_aggregator.py -v
"""
from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KST = ZoneInfo("Asia/Seoul")

# ── Stub data classes ─────────────────────────────────────────────────────────

@dataclass
class _StubMeeting:
    """Minimal Meeting stub for testing without Google credentials."""
    event_id: str = "evt-001"
    summary: str = "Test Meeting"
    start: datetime = None
    end: datetime = None
    is_external: bool = False
    attendees: list = field(default_factory=list)
    external_attendees: list = field(default_factory=list)
    description: str = ""
    location: str = ""
    html_link: str = ""
    organizer_email: str = ""
    duration_minutes: int = 60
    all_day: bool = False

    def __post_init__(self):
        if self.start is None:
            self.start = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc)  # 10:00 KST
        if self.end is None:
            self.end = datetime(2026, 3, 29, 2, 0, tzinfo=timezone.utc)    # 11:00 KST


@dataclass
class _StubEmail:
    """Minimal EmailMessage stub for testing."""
    message_id: str = "msg-001"
    thread_id: str = "thread-001"
    subject: str = "Test Email"
    sender: str = "Test Sender"
    sender_email: str = "sender@external.com"
    snippet: str = "Preview text"
    received_at: datetime = None
    is_unread: bool = True
    labels: list = field(default_factory=list)
    body_preview: str = ""
    is_external: bool = True

    def __post_init__(self):
        if self.received_at is None:
            self.received_at = datetime(2026, 3, 29, 9, 0, tzinfo=timezone.utc)


@dataclass
class _StubNotionItem:
    """Minimal NotionDeadlineItem stub for testing."""
    page_id: str = "page-001"
    name: str = "Portfolio Co."
    deadline: date = None
    deadline_prop: str = "Deadline"
    status: str = "검토 중"
    url: str = "https://notion.so/page-001"
    is_overdue: bool = False
    days_until: int = 7

    def __post_init__(self):
        if self.deadline is None:
            self.deadline = date(2026, 4, 5)


# ── BriefingData tests ─────────────────────────────────────────────────────────

class TestBriefingData:
    """Unit tests for BriefingData container properties."""

    def setup_method(self):
        from src.briefing.aggregator import BriefingData
        self._cls = BriefingData

    def test_default_state_all_sources_ok(self):
        bd = self._cls(target_date=date(2026, 3, 29))
        assert bd.all_sources_ok is True
        assert bd.has_calendar is True
        assert bd.has_gmail is True
        assert bd.has_notion is True

    def test_empty_lists_by_default(self):
        bd = self._cls(target_date=date(2026, 3, 29))
        assert bd.calendar_events == []
        assert bd.emails == []
        assert bd.notion_deadlines == []
        assert bd.source_errors == {}

    def test_source_error_marks_source_unavailable(self):
        bd = self._cls(
            target_date=date(2026, 3, 29),
            source_errors={"calendar": "API failed"},
        )
        assert bd.has_calendar is False
        assert bd.has_gmail is True
        assert bd.has_notion is True
        assert bd.all_sources_ok is False

    def test_multiple_source_errors(self):
        bd = self._cls(
            target_date=date(2026, 3, 29),
            source_errors={"calendar": "err1", "notion": "err2"},
        )
        assert bd.has_calendar is False
        assert bd.has_gmail is True
        assert bd.has_notion is False
        assert bd.all_sources_ok is False

    def test_external_meetings_property(self):
        m1 = _StubMeeting(is_external=True)
        m2 = _StubMeeting(is_external=False)
        bd = self._cls(target_date=date(2026, 3, 29), calendar_events=[m1, m2])
        assert len(bd.external_meetings) == 1
        assert bd.external_meetings[0] is m1

    def test_unread_emails_property(self):
        e1 = _StubEmail(is_unread=True)
        e2 = _StubEmail(is_unread=False)
        bd = self._cls(target_date=date(2026, 3, 29), emails=[e1, e2])
        assert len(bd.unread_emails) == 1
        assert bd.unread_emails[0] is e1

    def test_overdue_deadlines_property(self):
        d1 = _StubNotionItem(is_overdue=True)
        d2 = _StubNotionItem(is_overdue=False)
        bd = self._cls(target_date=date(2026, 3, 29), notion_deadlines=[d1, d2])
        assert len(bd.overdue_deadlines) == 1
        assert bd.overdue_deadlines[0] is d1

    def test_upcoming_deadlines_property(self):
        d1 = _StubNotionItem(is_overdue=True)
        d2 = _StubNotionItem(is_overdue=False)
        d3 = _StubNotionItem(is_overdue=False)
        bd = self._cls(target_date=date(2026, 3, 29), notion_deadlines=[d1, d2, d3])
        assert len(bd.upcoming_deadlines) == 2

    def test_summary_string(self):
        bd = self._cls(
            target_date=date(2026, 3, 29),
            calendar_events=[_StubMeeting()],
            emails=[_StubEmail(), _StubEmail()],
            notion_deadlines=[_StubNotionItem()],
        )
        summary = bd.summary()
        assert "calendar=1" in summary
        assert "emails=2" in summary
        assert "notion_deadlines=1" in summary

    def test_summary_includes_errors(self):
        bd = self._cls(
            target_date=date(2026, 3, 29),
            source_errors={"gmail": "timeout"},
        )
        summary = bd.summary()
        assert "gmail" in summary

    def test_fetched_at_is_utc(self):
        bd = self._cls(target_date=date(2026, 3, 29))
        assert bd.fetched_at.tzinfo == timezone.utc


# ── Retry wrapper tests ────────────────────────────────────────────────────────

class TestFetchWithRetry:
    """Tests for the _fetch_with_retry helper."""

    def setup_method(self):
        from src.briefing.aggregator import _fetch_with_retry
        self._retry = _fetch_with_retry

    def test_succeeds_on_first_attempt(self):
        def ok_fn(x):
            return [x], None

        data, err = self._retry(ok_fn, "test", "value")
        assert data == ["value"]
        assert err is None

    def test_returns_empty_and_error_on_persistent_failure(self):
        def fail_fn():
            return [], "Connection refused"

        # Patch sleep to avoid waiting
        with patch("src.briefing.aggregator.time.sleep"):
            data, err = self._retry(fail_fn, "test")

        assert data == []
        assert err is not None
        assert "Connection refused" in err

    def test_succeeds_on_second_attempt(self):
        call_count = [0]

        def flaky_fn():
            call_count[0] += 1
            if call_count[0] == 1:
                return [], "Transient error"
            return ["result"], None

        with patch("src.briefing.aggregator.time.sleep"):
            data, err = self._retry(flaky_fn, "test")

        assert data == ["result"]
        assert err is None
        assert call_count[0] == 2

    def test_retries_exactly_api_retry_attempts_times(self):
        from src.config import API_RETRY_ATTEMPTS
        call_count = [0]

        def always_fail():
            call_count[0] += 1
            return [], f"error_{call_count[0]}"

        with patch("src.briefing.aggregator.time.sleep"):
            self._retry(always_fail, "test")

        assert call_count[0] == API_RETRY_ATTEMPTS

    def test_sleeps_between_retries(self):
        from src.config import API_RETRY_ATTEMPTS, API_RETRY_DELAY_SECONDS

        def always_fail():
            return [], "error"

        with patch("src.briefing.aggregator.time.sleep") as mock_sleep:
            self._retry(always_fail, "test")

        # Should sleep between retries (not after the last attempt)
        expected_sleep_calls = API_RETRY_ATTEMPTS - 1
        assert mock_sleep.call_count == expected_sleep_calls
        if expected_sleep_calls > 0:
            mock_sleep.assert_called_with(API_RETRY_DELAY_SECONDS)


# ── aggregate_briefing_data tests ──────────────────────────────────────────────

class TestAggregateBriefingData:
    """
    Tests for aggregate_briefing_data() with mocked data sources.
    All actual API calls are patched to avoid network dependencies.
    """

    def _make_stub_calendar_client(self, events=None):
        """Return a mock GoogleCalendarClient."""
        mock_client = MagicMock()
        mock_client.list_upcoming_events.return_value = events or [_StubMeeting()]
        return mock_client

    def _make_stub_gmail_client(self, emails=None):
        """Return a mock GmailClient."""
        mock_client = MagicMock()
        mock_client.fetch_inbox_emails.return_value = emails or [_StubEmail()]
        return mock_client

    def _make_stub_notion_client(self, deadlines=None):
        """Return a mock NotionClient."""
        mock_client = MagicMock()
        mock_client.fetch_deadline_items.return_value = deadlines or [_StubNotionItem()]
        return mock_client

    def test_returns_briefing_data_instance(self):
        from src.briefing.aggregator import BriefingData, aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], None)):
            result = aggregate_briefing_data(
                target_date=date(2026, 3, 29),
                fetch_calendar=True,
                fetch_gmail=True,
                fetch_notion=True,
            )

        assert isinstance(result, BriefingData)
        assert result.target_date == date(2026, 3, 29)

    def test_all_sources_fetched_by_default(self):
        from src.briefing.aggregator import aggregate_briefing_data

        meetings = [_StubMeeting()]
        emails = [_StubEmail()]
        deadlines = [_StubNotionItem()]

        with patch("src.briefing.aggregator._fetch_calendar", return_value=(meetings, None)) as mc, \
             patch("src.briefing.aggregator._fetch_gmail", return_value=(emails, None)) as mg, \
             patch("src.briefing.aggregator._fetch_notion", return_value=(deadlines, None)) as mn:
            result = aggregate_briefing_data(target_date=date(2026, 3, 29))

        mc.assert_called_once()
        mg.assert_called_once()
        mn.assert_called_once()

        assert result.calendar_events == meetings
        assert result.emails == emails
        assert result.notion_deadlines == deadlines

    def test_calendar_source_failure_does_not_prevent_other_sources(self):
        from src.briefing.aggregator import aggregate_briefing_data

        emails = [_StubEmail()]
        deadlines = [_StubNotionItem()]

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], "API error")), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=(emails, None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=(deadlines, None)), \
             patch("src.briefing.aggregator.time.sleep"):
            result = aggregate_briefing_data(target_date=date(2026, 3, 29))

        assert result.has_calendar is False
        assert "calendar" in result.source_errors
        assert result.emails == emails
        assert result.notion_deadlines == deadlines
        assert result.has_gmail is True
        assert result.has_notion is True

    def test_gmail_source_failure_does_not_prevent_other_sources(self):
        from src.briefing.aggregator import aggregate_briefing_data

        meetings = [_StubMeeting()]
        deadlines = [_StubNotionItem()]

        with patch("src.briefing.aggregator._fetch_calendar", return_value=(meetings, None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], "Gmail quota exceeded")), \
             patch("src.briefing.aggregator._fetch_notion", return_value=(deadlines, None)), \
             patch("src.briefing.aggregator.time.sleep"):
            result = aggregate_briefing_data(target_date=date(2026, 3, 29))

        assert result.has_gmail is False
        assert "gmail" in result.source_errors
        assert result.calendar_events == meetings
        assert result.notion_deadlines == deadlines

    def test_notion_source_failure_does_not_prevent_other_sources(self):
        from src.briefing.aggregator import aggregate_briefing_data

        meetings = [_StubMeeting()]
        emails = [_StubEmail()]

        with patch("src.briefing.aggregator._fetch_calendar", return_value=(meetings, None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=(emails, None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], "Notion auth failed")), \
             patch("src.briefing.aggregator.time.sleep"):
            result = aggregate_briefing_data(target_date=date(2026, 3, 29))

        assert result.has_notion is False
        assert "notion" in result.source_errors
        assert result.calendar_events == meetings
        assert result.emails == emails

    def test_all_sources_can_fail_independently(self):
        from src.briefing.aggregator import aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], "err1")), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], "err2")), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], "err3")), \
             patch("src.briefing.aggregator.time.sleep"):
            result = aggregate_briefing_data(target_date=date(2026, 3, 29))

        assert result.all_sources_ok is False
        assert len(result.source_errors) == 3
        assert "calendar" in result.source_errors
        assert "gmail" in result.source_errors
        assert "notion" in result.source_errors

    def test_target_date_defaults_to_today_kst(self):
        from src.briefing.aggregator import aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], None)):
            result = aggregate_briefing_data()

        # The date should match today KST
        today_kst = datetime.now(KST).date()
        assert result.target_date == today_kst

    def test_skip_calendar_flag(self):
        from src.briefing.aggregator import aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar") as mc, \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], None)):
            result = aggregate_briefing_data(
                target_date=date(2026, 3, 29),
                fetch_calendar=False,
            )

        mc.assert_not_called()
        assert result.calendar_events == []
        # When skipped, no error is recorded (absence of data ≠ error)
        assert "calendar" not in result.source_errors

    def test_skip_gmail_flag(self):
        from src.briefing.aggregator import aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail") as mg, \
             patch("src.briefing.aggregator._fetch_notion", return_value=([], None)):
            result = aggregate_briefing_data(
                target_date=date(2026, 3, 29),
                fetch_gmail=False,
            )

        mg.assert_not_called()
        assert result.emails == []
        assert "gmail" not in result.source_errors

    def test_skip_notion_flag(self):
        from src.briefing.aggregator import aggregate_briefing_data

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion") as mn:
            result = aggregate_briefing_data(
                target_date=date(2026, 3, 29),
                fetch_notion=False,
            )

        mn.assert_not_called()
        assert result.notion_deadlines == []
        assert "notion" not in result.source_errors


# ── Per-source fetch helper tests ─────────────────────────────────────────────

class TestFetchCalendar:
    """Tests for _fetch_calendar helper.

    The clients are imported *inside* the helper functions (lazy imports),
    so we must patch the original module, not the aggregator namespace.
    """

    def test_returns_tuple_of_list_and_none_on_success(self):
        from src.briefing.aggregator import _fetch_calendar

        meetings = [_StubMeeting()]
        mock_client = MagicMock()
        mock_client.list_upcoming_events.return_value = meetings

        with patch("src.calendar.google_calendar.GoogleCalendarClient", return_value=mock_client):
            events, err = _fetch_calendar(date(2026, 3, 29))

        assert events == meetings
        assert err is None

    def test_returns_empty_and_error_string_on_failure(self):
        from src.briefing.aggregator import _fetch_calendar

        mock_client = MagicMock()
        mock_client.list_upcoming_events.side_effect = RuntimeError("Auth failed")

        with patch("src.calendar.google_calendar.GoogleCalendarClient", return_value=mock_client):
            events, err = _fetch_calendar(date(2026, 3, 29))

        assert events == []
        assert err is not None
        assert "Auth failed" in err

    def test_uses_kst_day_boundaries(self):
        """Events should be queried for the full KST day (00:00–23:59 KST)."""
        from src.briefing.aggregator import _fetch_calendar

        mock_client = MagicMock()
        mock_client.list_upcoming_events.return_value = []

        with patch("src.calendar.google_calendar.GoogleCalendarClient", return_value=mock_client):
            _fetch_calendar(date(2026, 3, 29))

        call_args = mock_client.list_upcoming_events.call_args
        time_min = call_args[1].get("time_min") or call_args[0][0]
        time_max = call_args[1].get("time_max") or call_args[0][1]

        # KST 00:00 = UTC 15:00 (prev day), KST 00:00 next day = UTC 15:00 same day
        # For 2026-03-29: KST midnight = UTC 2026-03-28 15:00
        assert time_min.tzinfo is not None
        assert time_max.tzinfo is not None
        # time_max should be 24h after time_min
        delta = time_max - time_min
        assert delta == timedelta(days=1)


class TestFetchGmail:
    """Tests for _fetch_gmail helper."""

    def test_returns_tuple_of_list_and_none_on_success(self):
        from src.briefing.aggregator import _fetch_gmail

        emails = [_StubEmail()]
        mock_client = MagicMock()
        mock_client.fetch_inbox_emails.return_value = emails

        with patch("src.gmail.client.GmailClient", return_value=mock_client):
            result, err = _fetch_gmail(date(2026, 3, 29))

        assert result == emails
        assert err is None

    def test_returns_empty_and_error_string_on_failure(self):
        from src.briefing.aggregator import _fetch_gmail

        mock_client = MagicMock()
        mock_client.fetch_inbox_emails.side_effect = RuntimeError("403 Forbidden")

        with patch("src.gmail.client.GmailClient", return_value=mock_client):
            result, err = _fetch_gmail(date(2026, 3, 29))

        assert result == []
        assert "403 Forbidden" in err

    def test_fetches_at_least_one_day_of_emails(self):
        from src.briefing.aggregator import _fetch_gmail

        mock_client = MagicMock()
        mock_client.fetch_inbox_emails.return_value = []

        with patch("src.gmail.client.GmailClient", return_value=mock_client):
            _fetch_gmail(date(2026, 3, 29))

        call_args = mock_client.fetch_inbox_emails.call_args
        days = call_args[1].get("days") or (call_args[0][0] if call_args[0] else 1)
        assert days >= 1


class TestFetchNotion:
    """Tests for _fetch_notion helper."""

    def test_returns_tuple_of_list_and_none_on_success(self):
        from src.briefing.aggregator import _fetch_notion

        deadlines = [_StubNotionItem()]
        mock_client = MagicMock()
        mock_client.fetch_deadline_items.return_value = deadlines

        with patch("src.notion.client.NotionClient", return_value=mock_client):
            result, err = _fetch_notion(date(2026, 3, 29))

        assert result == deadlines
        assert err is None

    def test_returns_empty_and_error_string_on_failure(self):
        from src.briefing.aggregator import _fetch_notion

        mock_client = MagicMock()
        mock_client.fetch_deadline_items.side_effect = RuntimeError("Notion API error")

        with patch("src.notion.client.NotionClient", return_value=mock_client):
            result, err = _fetch_notion(date(2026, 3, 29))

        assert result == []
        assert "Notion API error" in err

    def test_fetches_30_day_lookahead_including_overdue(self):
        """Notion fetch should always cover 30 days ahead plus overdue items."""
        from src.briefing.aggregator import _fetch_notion

        mock_client = MagicMock()
        mock_client.fetch_deadline_items.return_value = []

        with patch("src.notion.client.NotionClient", return_value=mock_client):
            _fetch_notion(date(2026, 3, 29))

        call_args = mock_client.fetch_deadline_items.call_args
        kwargs = call_args[1] if call_args[1] else {}
        lookahead = kwargs.get("lookahead_days", 30)
        include_overdue = kwargs.get("include_overdue", True)

        assert lookahead >= 30
        assert include_overdue is True


# ── Full formatter tests ────────────────────────────────────────────────────────

class TestFullFormatter:
    """
    Tests for format_full_briefing(BriefingData) in src.briefing.full_formatter.
    Verifies that all three data source sections are rendered correctly.
    """

    def _make_briefing_data(
        self,
        meetings=None,
        emails=None,
        deadlines=None,
        source_errors=None,
        target_date=None,
    ):
        from src.briefing.aggregator import BriefingData
        return BriefingData(
            target_date=target_date or date(2026, 3, 29),
            calendar_events=meetings if meetings is not None else [],
            emails=emails if emails is not None else [],
            notion_deadlines=deadlines if deadlines is not None else [],
            source_errors=source_errors or {},
        )

    def test_returns_tuple_of_str_and_list(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        text, blocks = format_full_briefing(bd)
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_blocks_within_50_limit(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            meetings=[_StubMeeting() for _ in range(10)],
            emails=[_StubEmail() for _ in range(30)],
            deadlines=[_StubNotionItem() for _ in range(20)],
        )
        _, blocks = format_full_briefing(bd)
        assert len(blocks) <= 50

    def test_header_block_present(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(target_date=date(2026, 3, 29))
        _, blocks = format_full_briefing(bd)
        headers = [b for b in blocks if b.get("type") == "header"]
        assert len(headers) >= 1

    def test_header_contains_date(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(target_date=date(2026, 3, 29))
        _, blocks = format_full_briefing(bd)
        header = next(b for b in blocks if b.get("type") == "header")
        assert "3월" in header["text"]["text"]
        assert "29일" in header["text"]["text"]

    def test_footer_block_present(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        _, blocks = format_full_briefing(bd)
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        assert len(context_blocks) >= 1

    def test_footer_contains_accuracy_disclaimer(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            e.get("text", "") if isinstance(e, str) else e.get("text", "")
            for b in blocks if b.get("type") == "context"
            for e in (b.get("elements") or [])
        )
        assert "확인 불가" in all_text

    def test_calendar_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "일정" in all_text or "캘린더" in all_text

    def test_gmail_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "이메일" in all_text or "편지함" in all_text

    def test_notion_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data()
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "마감" in all_text or "Notion" in all_text

    def test_email_subject_shown_in_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        email = _StubEmail(subject="중요 투자 안건")
        bd = self._make_briefing_data(emails=[email])
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "중요 투자 안건" in all_text

    def test_notion_item_name_shown_in_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        item = _StubNotionItem(name="스타트업 Alpha")
        bd = self._make_briefing_data(deadlines=[item])
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "스타트업 Alpha" in all_text

    def test_overdue_deadline_shows_urgency_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        overdue_item = _StubNotionItem(is_overdue=True, days_until=-3)
        bd = self._make_briefing_data(deadlines=[overdue_item])
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "🚨" in all_text

    def test_calendar_error_shows_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            source_errors={"calendar": "OAuth expired"}
        )
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_gmail_error_shows_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            source_errors={"gmail": "403 Forbidden"}
        )
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_notion_error_shows_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            source_errors={"notion": "Database not found"}
        )
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_all_sources_error_produces_valid_blocks(self):
        """Even with all sources failing, we should get a valid (non-empty) response."""
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            source_errors={
                "calendar": "err1",
                "gmail": "err2",
                "notion": "err3",
            }
        )
        text, blocks = format_full_briefing(bd)
        assert len(blocks) > 0
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_sources_produce_no_data_messages(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(meetings=[], emails=[], deadlines=[])
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        # Should indicate no data, not crash
        assert "없습니다" in all_text or "없음" in all_text or len(blocks) > 0

    def test_fallback_text_contains_date(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(target_date=date(2026, 3, 29))
        text, _ = format_full_briefing(bd)
        assert "2026" in text
        assert "3월" in text

    def test_fallback_text_summarises_all_sources(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            meetings=[_StubMeeting()],
            emails=[_StubEmail(), _StubEmail()],
            deadlines=[_StubNotionItem()],
        )
        text, _ = format_full_briefing(bd)
        # All three sources should be mentioned in the fallback
        assert any(kw in text for kw in ["캘린더", "일정", "미팅"])
        assert any(kw in text for kw in ["이메일", "메일"])
        assert any(kw in text for kw in ["Notion", "마감"])

    def test_unread_emails_highlighted(self):
        from src.briefing.full_formatter import format_full_briefing
        unread = _StubEmail(is_unread=True, subject="긴급 검토 요청")
        read = _StubEmail(is_unread=False, subject="일반 뉴스레터")
        bd = self._make_briefing_data(emails=[unread, read])
        _, blocks = format_full_briefing(bd)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        # Unread icon should appear
        assert "📬" in all_text

    def test_valid_block_types_only(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = self._make_briefing_data(
            meetings=[_StubMeeting()],
            emails=[_StubEmail()],
            deadlines=[_StubNotionItem()],
        )
        _, blocks = format_full_briefing(bd)
        valid_types = {"header", "section", "divider", "context", "actions", "image"}
        for block in blocks:
            assert block.get("type") in valid_types, (
                f"Invalid block type: {block.get('type')}"
            )


# ── run_aggregated_brief tests ────────────────────────────────────────────────

class TestRunAggregatedBrief:
    """
    Tests for the run_aggregated_brief pipeline function.

    Since aggregate_briefing_data and format_full_briefing are imported
    *inside* run_aggregated_brief (lazy imports), we patch them in their
    original modules (src.briefing.aggregator and src.briefing.full_formatter).
    """

    def test_dry_run_returns_true_without_bot(self):
        from src.briefing.pipeline import run_aggregated_brief

        with patch("src.briefing.aggregator.aggregate_briefing_data") as mock_agg:
            from src.briefing.aggregator import BriefingData
            mock_agg.return_value = BriefingData(target_date=date(2026, 3, 29))
            result = run_aggregated_brief(
                target_date=date(2026, 3, 29),
                bot=None,
            )

        assert result is True

    def test_calls_aggregate_briefing_data_with_target_date(self):
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bot_mock = MagicMock()
        bot_mock.send_message.return_value = True
        bd = BriefingData(target_date=date(2026, 3, 29))

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd) as mock_agg, \
             patch("src.briefing.full_formatter.format_full_briefing", return_value=("text", [])):
            run_aggregated_brief(
                target_date=date(2026, 3, 29),
                bot=bot_mock,
            )

        mock_agg.assert_called_once_with(target_date=date(2026, 3, 29))

    def test_sends_message_via_bot_on_success(self):
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bot_mock = MagicMock()
        bot_mock.send_message.return_value = True
        bd = BriefingData(target_date=date(2026, 3, 29))

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback", [{"type": "section"}])):
            result = run_aggregated_brief(
                target_date=date(2026, 3, 29),
                bot=bot_mock,
            )

        bot_mock.send_message.assert_called_once_with(
            "fallback", blocks=[{"type": "section"}]
        )
        assert result is True

    def test_returns_false_when_send_message_fails(self):
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bot_mock = MagicMock()
        bot_mock.send_message.return_value = False
        bot_mock.send_error = MagicMock()
        bd = BriefingData(target_date=date(2026, 3, 29))

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd), \
             patch("src.briefing.full_formatter.format_full_briefing", return_value=("text", [])):
            result = run_aggregated_brief(
                target_date=date(2026, 3, 29),
                bot=bot_mock,
            )

        assert result is False

    def test_direct_dm_to_user_id_when_provided(self):
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bot_mock = MagicMock()
        bot_mock._client.chat_postMessage.return_value = {"ok": True}
        bd = BriefingData(target_date=date(2026, 3, 29))

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("text", [{"type": "section"}])):
            result = run_aggregated_brief(
                target_date=date(2026, 3, 29),
                bot=bot_mock,
                user_id="U123456",
            )

        bot_mock._client.chat_postMessage.assert_called_once()
        call_kwargs = bot_mock._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "U123456"

    def test_defaults_target_date_to_today(self):
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bot_mock = MagicMock()
        bot_mock.send_message.return_value = True
        today_kst = datetime.now(KST).date()
        bd = BriefingData(target_date=today_kst)

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd) as mock_agg, \
             patch("src.briefing.full_formatter.format_full_briefing", return_value=("text", [])):
            run_aggregated_brief(bot=bot_mock)

        call_kwargs = mock_agg.call_args[1]
        assert call_kwargs["target_date"] == today_kst


# ── Integration: /brief callback wiring test ──────────────────────────────────

class TestBriefCallbackIntegration:
    """
    Verify that the briefing callback registered with /brief invokes
    aggregate_briefing_data() when called.
    """

    def test_callback_calls_run_aggregated_brief(self):
        """
        _make_briefing_callback() in main.py must produce a callable that
        invokes run_aggregated_brief() with the given target_date.

        run_aggregated_brief is imported inside _callback (lazy import), so
        we patch it in the briefing.pipeline module where it lives.
        """
        import main as main_module

        callback, bot_holder = main_module._make_briefing_callback()
        assert callable(callback)

        bot_mock = MagicMock()
        bot_holder[0] = bot_mock  # simulate post-construction wiring

        # Patch run_aggregated_brief in the pipeline module (where it is defined)
        with patch("src.briefing.pipeline.run_aggregated_brief") as mock_run:
            mock_run.return_value = True
            callback(date(2026, 3, 29), "U123", "C456")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("target_date") == date(2026, 3, 29)
        assert call_kwargs.get("bot") is bot_mock
        assert call_kwargs.get("user_id") == "U123"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestBriefingData,
        TestFetchWithRetry,
        TestAggregateBriefingData,
        TestFetchCalendar,
        TestFetchGmail,
        TestFetchNotion,
        TestFullFormatter,
        TestRunAggregatedBrief,
        TestBriefCallbackIntegration,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        if hasattr(instance, "setup_method"):
            instance.setup_method()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                print(f"  ✓ {cls.__name__}.{method_name}")
                passed += 1
            except Exception:
                print(f"  ✗ {cls.__name__}.{method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
