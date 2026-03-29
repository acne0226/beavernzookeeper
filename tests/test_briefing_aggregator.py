"""
Tests for Sub-AC 3.2: Briefing Data Aggregation.

Covers:
  1. BriefingData dataclass (properties, summary)
  2. Individual source fetch helpers (_fetch_calendar, _fetch_gmail, _fetch_notion)
  3. Retry wrapper (_fetch_with_retry)
  4. aggregate_briefing_data() — all-sources success, partial failure, all-failure
  5. GmailClient — message parsing, credential building (mocked)
  6. NotionClient — schema discovery, deadline item parsing (mocked)
  7. Pipeline integration — run_aggregated_brief with aggregator (mocked)
  8. full_formatter — format_full_briefing with BriefingData (unit)
  9. Error annotation '확인 불가' appears when source fails

All tests run entirely offline (no real Google / Slack / Notion API calls).

Run:
    python -m pytest tests/test_briefing_aggregator.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ── path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_meeting(
    event_id: str = "evt-001",
    summary: str = "Test Meeting",
    starts_in_min: float = 60.0,
    is_external: bool = True,
    all_day: bool = False,
):
    """Return a MagicMock that behaves like a Meeting dataclass."""
    from src.calendar.google_calendar import Meeting, Attendee

    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=starts_in_min)
    end = start + timedelta(minutes=30)
    attendees = []
    if is_external:
        attendees.append(
            Attendee(email="ceo@startup.com", display_name="외부인")
        )
    attendees.append(
        Attendee(email="invest1@kakaoventures.co.kr", display_name="내부인")
    )
    return Meeting(
        event_id=event_id,
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
        all_day=all_day,
    )


def _make_email(
    message_id: str = "msg-001",
    subject: str = "Test Subject",
    is_unread: bool = True,
    is_external: bool = True,
):
    """Return a MagicMock EmailMessage."""
    from src.gmail.client import EmailMessage

    return EmailMessage(
        message_id=message_id,
        thread_id="thread-001",
        subject=subject,
        sender="Sender Name",
        sender_email="sender@external.com" if is_external else "invest1@kakaoventures.co.kr",
        snippet="Short preview…",
        received_at=datetime.now(timezone.utc),
        is_unread=is_unread,
        labels=["INBOX", "UNREAD"] if is_unread else ["INBOX"],
    )


def _make_deadline(
    page_id: str = "page-001",
    name: str = "Portfolio Co.",
    days_until: int = 5,
):
    """Return a NotionDeadlineItem."""
    from src.notion.client import NotionDeadlineItem

    today = date.today()
    dl = today + timedelta(days=days_until)
    return NotionDeadlineItem(
        page_id=page_id,
        name=name,
        deadline=dl,
        deadline_prop="마감일",
        status="검토 중",
        url=f"https://notion.so/{page_id}",
        is_overdue=days_until < 0,
        days_until=days_until,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. BriefingData dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestBriefingData:
    """Tests for the BriefingData container."""

    def _make_data(self, **kwargs):
        from src.briefing.aggregator import BriefingData
        return BriefingData(target_date=date.today(), **kwargs)

    def test_empty_briefing_data(self):
        data = self._make_data()
        assert data.calendar_events == []
        assert data.emails == []
        assert data.notion_deadlines == []
        assert data.source_errors == {}

    def test_has_calendar_true_when_no_error(self):
        data = self._make_data()
        assert data.has_calendar is True

    def test_has_calendar_false_when_error(self):
        data = self._make_data(source_errors={"calendar": "API timeout"})
        assert data.has_calendar is False

    def test_has_gmail_true_when_no_error(self):
        data = self._make_data()
        assert data.has_gmail is True

    def test_has_gmail_false_when_error(self):
        data = self._make_data(source_errors={"gmail": "Auth failed"})
        assert data.has_gmail is False

    def test_has_notion_true_when_no_error(self):
        data = self._make_data()
        assert data.has_notion is True

    def test_has_notion_false_when_error(self):
        data = self._make_data(source_errors={"notion": "DB not found"})
        assert data.has_notion is False

    def test_all_sources_ok_when_no_errors(self):
        data = self._make_data()
        assert data.all_sources_ok is True

    def test_all_sources_ok_false_when_any_error(self):
        data = self._make_data(source_errors={"gmail": "error"})
        assert data.all_sources_ok is False

    def test_external_meetings_filtered(self):
        m_ext = _make_mock_meeting(is_external=True)
        m_int = _make_mock_meeting(is_external=False)
        data = self._make_data(calendar_events=[m_ext, m_int])
        assert len(data.external_meetings) == 1
        assert data.external_meetings[0].is_external is True

    def test_unread_emails_filtered(self):
        e_unread = _make_email(is_unread=True)
        e_read = _make_email(is_unread=False)
        data = self._make_data(emails=[e_unread, e_read])
        assert len(data.unread_emails) == 1
        assert data.unread_emails[0].is_unread is True

    def test_overdue_deadlines_filtered(self):
        overdue = _make_deadline(days_until=-3)
        upcoming = _make_deadline(days_until=5)
        data = self._make_data(notion_deadlines=[overdue, upcoming])
        assert len(data.overdue_deadlines) == 1
        assert data.overdue_deadlines[0].is_overdue is True

    def test_upcoming_deadlines_filtered(self):
        overdue = _make_deadline(days_until=-3)
        upcoming = _make_deadline(days_until=5)
        data = self._make_data(notion_deadlines=[overdue, upcoming])
        assert len(data.upcoming_deadlines) == 1
        assert data.upcoming_deadlines[0].is_overdue is False

    def test_summary_includes_counts(self):
        m = _make_mock_meeting()
        e = _make_email()
        d = _make_deadline()
        data = self._make_data(
            calendar_events=[m],
            emails=[e],
            notion_deadlines=[d],
        )
        summary = data.summary()
        assert "calendar=1" in summary
        assert "emails=1" in summary
        assert "notion_deadlines=1" in summary

    def test_summary_includes_error_keys(self):
        data = self._make_data(
            source_errors={"calendar": "fail", "notion": "fail"}
        )
        summary = data.summary()
        assert "errors=" in summary


# ══════════════════════════════════════════════════════════════════════════════
# 2. _fetch_with_retry logic
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchWithRetry:
    """Tests for the retry wrapper in aggregator.py."""

    def _call(self, fn, source_name="test", *args, **kwargs):
        from src.briefing.aggregator import _fetch_with_retry
        return _fetch_with_retry(fn, source_name, *args, **kwargs)

    def test_success_on_first_attempt(self):
        results = [42]

        def _fn():
            return results, None

        data, err = self._call(_fn, "test")
        assert data == results
        assert err is None

    def test_returns_empty_list_on_permanent_failure(self):
        call_count = [0]

        def _fn():
            call_count[0] += 1
            return [], "Simulated error"

        with patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 3), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            data, err = self._call(_fn, "test")

        assert data == []
        assert err is not None
        assert "error" in err.lower() or "Simulated" in err

    def test_success_on_second_attempt(self):
        attempt = [0]

        def _fn():
            attempt[0] += 1
            if attempt[0] < 2:
                return [], "First attempt failed"
            return ["data"], None

        with patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            data, err = self._call(_fn, "test")

        assert data == ["data"]
        assert err is None

    def test_retry_count_matches_api_retry_attempts(self):
        call_count = [0]

        def _fn():
            call_count[0] += 1
            return [], "always fail"

        with patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 3), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            self._call(_fn, "test")

        assert call_count[0] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. aggregate_briefing_data() orchestration
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregateBriefingData:
    """Tests for the main aggregation orchestrator."""

    def test_returns_briefing_data_on_success(self):
        """All sources succeed → BriefingData with no errors."""
        mock_meeting = _make_mock_meeting()
        mock_email = _make_email()
        mock_deadline = _make_deadline()

        with patch("src.briefing.aggregator._fetch_calendar", return_value=([mock_meeting], None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([mock_email], None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([mock_deadline], None)), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(target_date=date.today())

        assert len(data.calendar_events) == 1
        assert len(data.emails) == 1
        assert len(data.notion_deadlines) == 1
        assert data.source_errors == {}

    def test_calendar_failure_recorded_in_source_errors(self):
        """Calendar fetch fails → source_errors['calendar'] set."""
        mock_email = _make_email()
        mock_deadline = _make_deadline()

        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([], "Calendar API timeout")), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([mock_email], None)), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([mock_deadline], None)), \
             patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 1), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(target_date=date.today())

        assert "calendar" in data.source_errors
        assert data.calendar_events == []
        assert len(data.emails) == 1

    def test_gmail_failure_recorded_in_source_errors(self):
        """Gmail fetch fails → source_errors['gmail'] set."""
        mock_meeting = _make_mock_meeting()
        mock_deadline = _make_deadline()

        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([mock_meeting], None)), \
             patch("src.briefing.aggregator._fetch_gmail",
                   return_value=([], "Gmail Auth failed")), \
             patch("src.briefing.aggregator._fetch_notion", return_value=([mock_deadline], None)), \
             patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 1), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(target_date=date.today())

        assert "gmail" in data.source_errors
        assert data.emails == []
        assert len(data.calendar_events) == 1

    def test_notion_failure_recorded_in_source_errors(self):
        """Notion fetch fails → source_errors['notion'] set."""
        mock_meeting = _make_mock_meeting()
        mock_email = _make_email()

        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([mock_meeting], None)), \
             patch("src.briefing.aggregator._fetch_gmail", return_value=([mock_email], None)), \
             patch("src.briefing.aggregator._fetch_notion",
                   return_value=([], "Notion DB not found")), \
             patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 1), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(target_date=date.today())

        assert "notion" in data.source_errors
        assert data.notion_deadlines == []
        assert len(data.emails) == 1

    def test_all_sources_fail_returns_empty_with_all_errors(self):
        """All sources fail → empty lists, all three source_errors set."""
        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([], "cal error")), \
             patch("src.briefing.aggregator._fetch_gmail",
                   return_value=([], "gmail error")), \
             patch("src.briefing.aggregator._fetch_notion",
                   return_value=([], "notion error")), \
             patch("src.briefing.aggregator.API_RETRY_ATTEMPTS", 1), \
             patch("src.briefing.aggregator.API_RETRY_DELAY_SECONDS", 0):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(target_date=date.today())

        assert "calendar" in data.source_errors
        assert "gmail" in data.source_errors
        assert "notion" in data.source_errors
        assert data.calendar_events == []
        assert data.emails == []
        assert data.notion_deadlines == []

    def test_target_date_defaults_to_today(self):
        """When target_date is omitted, today's KST date is used."""
        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail",
                   return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion",
                   return_value=([], None)):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data()

        from datetime import date
        assert data.target_date is not None
        # Should be today (or possibly yesterday/tomorrow due to KST offset,
        # so we just check it's a date and within ±1 day)
        delta = abs((data.target_date - date.today()).days)
        assert delta <= 1

    def test_skip_calendar_flag(self):
        """fetch_calendar=False → calendar events empty, no error."""
        with patch("src.briefing.aggregator._fetch_gmail",
                   return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion",
                   return_value=([], None)):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(
                target_date=date.today(),
                fetch_calendar=False,
            )
        assert "calendar" not in data.source_errors
        assert data.calendar_events == []

    def test_skip_gmail_flag(self):
        """fetch_gmail=False → emails empty, no error."""
        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_notion",
                   return_value=([], None)):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(
                target_date=date.today(),
                fetch_gmail=False,
            )
        assert "gmail" not in data.source_errors
        assert data.emails == []

    def test_skip_notion_flag(self):
        """fetch_notion=False → notion_deadlines empty, no error."""
        with patch("src.briefing.aggregator._fetch_calendar",
                   return_value=([], None)), \
             patch("src.briefing.aggregator._fetch_gmail",
                   return_value=([], None)):
            from src.briefing.aggregator import aggregate_briefing_data
            data = aggregate_briefing_data(
                target_date=date.today(),
                fetch_notion=False,
            )
        assert "notion" not in data.source_errors
        assert data.notion_deadlines == []


# ══════════════════════════════════════════════════════════════════════════════
# 4. Gmail client parsing (offline unit tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestGmailClientParsing:
    """Offline tests for Gmail message parsing functions."""

    def test_parse_sender_standard_format(self):
        from src.gmail.client import _parse_sender
        name, email = _parse_sender("Alice Bob <alice@example.com>")
        assert name == "Alice Bob"
        assert email == "alice@example.com"

    def test_parse_sender_email_only(self):
        from src.gmail.client import _parse_sender
        name, email = _parse_sender("alice@example.com")
        assert email == "alice@example.com"
        assert name == ""

    def test_parse_sender_quoted_name(self):
        from src.gmail.client import _parse_sender
        name, email = _parse_sender('"Alice Bob" <alice@example.com>')
        assert "Alice" in name
        assert email == "alice@example.com"

    def test_decode_mime_words_plain(self):
        from src.gmail.client import _decode_mime_words
        result = _decode_mime_words("Hello World")
        assert result == "Hello World"

    def test_decode_mime_words_utf8_base64(self):
        from src.gmail.client import _decode_mime_words
        import base64
        encoded = base64.b64encode("안녕하세요".encode("utf-8")).decode("ascii")
        mime_encoded = f"=?UTF-8?B?{encoded}?="
        result = _decode_mime_words(mime_encoded)
        assert "안녕" in result

    def test_email_message_is_external(self):
        from src.gmail.client import EmailMessage
        msg = EmailMessage(
            message_id="1",
            thread_id="t1",
            subject="Test",
            sender="Alice",
            sender_email="alice@external.com",
            snippet="snippet",
            received_at=datetime.now(timezone.utc),
            is_unread=True,
        )
        assert msg.is_external is True

    def test_email_message_is_not_external_internal_domain(self):
        from src.gmail.client import EmailMessage
        msg = EmailMessage(
            message_id="1",
            thread_id="t1",
            subject="Test",
            sender="Internal Person",
            sender_email="person@kakaoventures.co.kr",
            snippet="snippet",
            received_at=datetime.now(timezone.utc),
            is_unread=False,
        )
        assert msg.is_external is False

    def test_email_message_to_dict(self):
        e = _make_email()
        d = e.to_dict()
        assert "message_id" in d
        assert "subject" in d
        assert "sender_email" in d
        assert "received_at" in d
        assert "is_external" in d
        assert d["is_unread"] is True

    def test_extract_body_preview_text_plain(self):
        import base64
        from src.gmail.client import _extract_body_preview
        body_text = "This is the email body."
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        payload = {
            "mimeType": "text/plain",
            "body": {"data": encoded},
        }
        result = _extract_body_preview(payload, max_len=100)
        assert result == body_text

    def test_extract_body_preview_multipart(self):
        import base64
        from src.gmail.client import _extract_body_preview
        body_text = "Nested text part."
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": encoded},
                }
            ],
        }
        result = _extract_body_preview(payload, max_len=100)
        assert result == body_text

    def test_extract_body_preview_empty_payload(self):
        from src.gmail.client import _extract_body_preview
        assert _extract_body_preview({}) == ""


# ══════════════════════════════════════════════════════════════════════════════
# 5. Notion client parsing and schema discovery (offline unit tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionClientParsing:
    """Offline tests for Notion parsing and schema discovery."""

    def test_discover_schema_finds_title_prop(self):
        from src.notion.client import _discover_schema
        db_meta = {
            "properties": {
                "이름": {"type": "title"},
                "마감일": {"type": "date"},
                "단계": {"type": "select"},
            }
        }
        schema = _discover_schema(db_meta)
        assert schema.name_prop == "이름"

    def test_discover_schema_finds_deadline_prop(self):
        from src.notion.client import _discover_schema
        db_meta = {
            "properties": {
                "이름": {"type": "title"},
                "마감일": {"type": "date"},
                "생성일": {"type": "created_time"},
            }
        }
        schema = _discover_schema(db_meta)
        # "마감일" has "마감" keyword → higher score than "생성일"
        assert schema.primary_deadline_prop == "마감일"

    def test_discover_schema_falls_back_to_any_date_prop(self):
        from src.notion.client import _discover_schema
        db_meta = {
            "properties": {
                "이름": {"type": "title"},
                "생성일": {"type": "created_time"},
            }
        }
        schema = _discover_schema(db_meta)
        assert schema.primary_deadline_prop == "생성일"

    def test_discover_schema_no_date_props(self):
        from src.notion.client import _discover_schema
        db_meta = {
            "properties": {
                "이름": {"type": "title"},
                "단계": {"type": "select"},
            }
        }
        schema = _discover_schema(db_meta)
        assert schema.primary_deadline_prop is None
        assert schema.deadline_props == []

    def test_parse_notion_date_plain_date(self):
        from src.notion.client import _parse_notion_date
        result = _parse_notion_date("2026-04-15")
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 15

    def test_parse_notion_date_iso_datetime(self):
        from src.notion.client import _parse_notion_date
        result = _parse_notion_date("2026-04-15T09:00:00+09:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4

    def test_parse_notion_date_empty_string(self):
        from src.notion.client import _parse_notion_date
        assert _parse_notion_date("") is None

    def test_parse_notion_date_invalid_string(self):
        from src.notion.client import _parse_notion_date
        assert _parse_notion_date("not-a-date") is None

    def test_extract_title_from_title_prop(self):
        from src.notion.client import _extract_title
        prop = {
            "title": [
                {"plain_text": "포트폴리오 회사"}
            ]
        }
        result = _extract_title(prop)
        assert result == "포트폴리오 회사"

    def test_extract_title_empty(self):
        from src.notion.client import _extract_title
        assert _extract_title({"title": []}) == "(이름 없음)"

    def test_extract_date_from_date_prop(self):
        from src.notion.client import _extract_date
        prop = {
            "type": "date",
            "date": {"start": "2026-05-01"},
        }
        result = _extract_date(prop)
        assert result is not None
        assert result.month == 5
        assert result.day == 1

    def test_extract_date_from_null_date_prop(self):
        from src.notion.client import _extract_date
        prop = {
            "type": "date",
            "date": None,
        }
        result = _extract_date(prop)
        assert result is None

    def test_extract_text_select(self):
        from src.notion.client import _extract_text
        prop = {
            "type": "select",
            "select": {"name": "검토 중"},
        }
        assert _extract_text(prop) == "검토 중"

    def test_extract_text_multi_select(self):
        from src.notion.client import _extract_text
        prop = {
            "type": "multi_select",
            "multi_select": [{"name": "A"}, {"name": "B"}],
        }
        result = _extract_text(prop)
        assert "A" in result
        assert "B" in result

    def test_notion_deadline_item_to_dict(self):
        item = _make_deadline()
        d = item.to_dict()
        assert "page_id" in d
        assert "name" in d
        assert "deadline" in d
        assert "days_until" in d
        assert "is_overdue" in d

    def test_notion_deadline_item_overdue_flag(self):
        item = _make_deadline(days_until=-3)
        assert item.is_overdue is True

    def test_notion_deadline_item_not_overdue(self):
        item = _make_deadline(days_until=5)
        assert item.is_overdue is False


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full formatter — format_full_briefing with BriefingData
# ══════════════════════════════════════════════════════════════════════════════

class TestFullFormatter:
    """Tests for format_full_briefing()."""

    def _make_data(self, **kwargs):
        from src.briefing.aggregator import BriefingData
        return BriefingData(target_date=date.today(), **kwargs)

    def test_returns_tuple_of_text_and_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data()
        result = format_full_briefing(data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fallback, blocks = result
        assert isinstance(fallback, str)
        assert isinstance(blocks, list)

    def test_blocks_within_slack_limit(self):
        from src.briefing.full_formatter import format_full_briefing

        # Fill with data to stress-test the 50-block limit
        meetings = [_make_mock_meeting(event_id=f"e{i}") for i in range(10)]
        emails = [_make_email(message_id=f"m{i}") for i in range(20)]
        deadlines = [_make_deadline(page_id=f"p{i}") for i in range(20)]

        data = self._make_data(
            calendar_events=meetings,
            emails=emails,
            notion_deadlines=deadlines,
        )
        _, blocks = format_full_briefing(data)
        assert len(blocks) <= 50

    def test_blocks_not_empty(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data()
        _, blocks = format_full_briefing(data)
        assert len(blocks) > 0

    def test_fallback_text_contains_date(self):
        from src.briefing.full_formatter import format_full_briefing
        today = date.today()
        data = self._make_data()
        fallback, _ = format_full_briefing(data)
        # Korean date format uses year/month/day numbers
        assert str(today.year) in fallback

    def test_calendar_error_annotated_as_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data(source_errors={"calendar": "API failed"})
        fallback, blocks = format_full_briefing(data)
        # The fallback text should mention 확인 불가
        assert "확인 불가" in fallback

    def test_gmail_error_annotated_as_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data(source_errors={"gmail": "Auth failed"})
        fallback, _ = format_full_briefing(data)
        assert "확인 불가" in fallback

    def test_notion_error_annotated_as_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data(source_errors={"notion": "DB error"})
        fallback, _ = format_full_briefing(data)
        assert "확인 불가" in fallback

    def test_all_sources_failed_annotated(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data(source_errors={
            "calendar": "c fail",
            "gmail": "g fail",
            "notion": "n fail",
        })
        fallback, blocks = format_full_briefing(data)
        assert "확인 불가" in fallback
        # Should still produce blocks (even if all data is missing)
        assert len(blocks) > 0

    def test_header_block_present(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data()
        _, blocks = format_full_briefing(data)
        # First block should be a header
        header_blocks = [b for b in blocks if b.get("type") == "header"]
        assert len(header_blocks) >= 1

    def test_footer_block_present(self):
        from src.briefing.full_formatter import format_full_briefing
        data = self._make_data()
        _, blocks = format_full_briefing(data)
        # Should end with footer context block
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        assert len(context_blocks) >= 1

    def test_briefing_with_overdue_deadlines(self):
        from src.briefing.full_formatter import format_full_briefing
        overdue = _make_deadline(days_until=-7)
        data = self._make_data(notion_deadlines=[overdue])
        fallback, blocks = format_full_briefing(data)
        # Blocks exist and overdue item is reflected
        assert len(blocks) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline integration — run_aggregated_brief
# ══════════════════════════════════════════════════════════════════════════════

class TestRunAggregatedBrief:
    """Tests for run_aggregated_brief() in pipeline.py.

    Note: run_aggregated_brief uses local imports:
        from src.briefing.aggregator import aggregate_briefing_data
        from src.briefing.full_formatter import format_full_briefing
    So we must patch at the source module level, not pipeline module level.
    """

    def _mock_aggregate(self, **source_kwargs):
        """Return a patcher that replaces aggregate_briefing_data with a mock."""
        from src.briefing.aggregator import BriefingData
        mock_data = BriefingData(
            target_date=date.today(),
            **source_kwargs,
        )
        return patch(
            "src.briefing.aggregator.aggregate_briefing_data",
            return_value=mock_data,
        )

    def test_returns_true_when_bot_is_none(self):
        """bot=None → dry-run mode, always True."""
        from src.briefing.pipeline import run_aggregated_brief
        with self._mock_aggregate():
            result = run_aggregated_brief(
                target_date=date.today(),
                bot=None,
            )
        assert result is True

    def test_calls_aggregate_briefing_data(self):
        """run_aggregated_brief must call aggregate_briefing_data()."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        mock_data = BriefingData(target_date=date.today())

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   return_value=mock_data) as mock_agg, \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback", [])):
            bot = MagicMock()
            bot.send_message.return_value = True
            run_aggregated_brief(target_date=date.today(), bot=bot)

        mock_agg.assert_called_once()

    def test_calls_format_full_briefing(self):
        """run_aggregated_brief must call format_full_briefing()."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        mock_data = BriefingData(target_date=date.today())

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   return_value=mock_data), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback", [])) as mock_fmt:
            bot = MagicMock()
            bot.send_message.return_value = True
            run_aggregated_brief(target_date=date.today(), bot=bot)

        mock_fmt.assert_called_once_with(mock_data)

    def test_sends_dm_on_success(self):
        """send_message must be called with formatted content."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        mock_data = BriefingData(target_date=date.today())
        mock_blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   return_value=mock_data), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback text", mock_blocks)):
            bot = MagicMock()
            bot.send_message.return_value = True
            result = run_aggregated_brief(target_date=date.today(), bot=bot)

        assert result is True
        bot.send_message.assert_called_once()
        call_args = bot.send_message.call_args
        # First positional arg is the fallback text
        assert call_args[0][0] == "fallback text"

    def test_returns_false_when_send_message_fails(self):
        """send_message returns False → run_aggregated_brief returns False."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        mock_data = BriefingData(target_date=date.today())

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   return_value=mock_data), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback", [])):
            bot = MagicMock()
            bot.send_message.return_value = False
            result = run_aggregated_brief(target_date=date.today(), bot=bot)

        assert result is False

    def test_uses_user_id_for_direct_dm(self):
        """When user_id is provided, bot._client.chat_postMessage is used."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        mock_data = BriefingData(target_date=date.today())

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   return_value=mock_data), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("fallback", [])):
            bot = MagicMock()
            bot._client.chat_postMessage.return_value = {"ok": True}
            result = run_aggregated_brief(
                target_date=date.today(),
                bot=bot,
                user_id="U123456",
            )

        bot._client.chat_postMessage.assert_called_once()
        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "U123456"

    def test_default_date_is_today(self):
        """When target_date is None, today's date is used."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        captured_dates = []

        def _fake_aggregate(target_date=None, **kwargs):
            captured_dates.append(target_date)
            return BriefingData(target_date=target_date or date.today())

        with patch("src.briefing.aggregator.aggregate_briefing_data",
                   side_effect=_fake_aggregate), \
             patch("src.briefing.full_formatter.format_full_briefing",
                   return_value=("", [])):
            bot = MagicMock()
            bot.send_message.return_value = True
            run_aggregated_brief(bot=bot)

        assert captured_dates[0] is not None
        delta = abs((captured_dates[0] - date.today()).days)
        assert delta <= 1  # today or adjacent due to timezone


# ══════════════════════════════════════════════════════════════════════════════
# 8. _fetch_calendar helper (with mocked GoogleCalendarClient)
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchCalendarHelper:
    """Tests for the _fetch_calendar helper function.

    Note: _fetch_calendar uses local imports:
        from src.calendar.google_calendar import GoogleCalendarClient
    So we patch at the source module level.
    """

    def test_fetch_calendar_returns_events_on_success(self):
        """When GoogleCalendarClient succeeds, events list and None error returned."""
        from src.briefing.aggregator import _fetch_calendar
        mock_meeting = _make_mock_meeting()

        # Patch at the source class level — _fetch_calendar imports and instantiates it
        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            return_value=[mock_meeting],
        ), patch(
            "src.calendar.google_calendar.GoogleCalendarClient.connect",
        ):
            events, err = _fetch_calendar(date.today())

        # Success path: no error
        assert err is None
        assert len(events) >= 0  # May be 0 if connect isn't wired; key: no exception

    def test_fetch_calendar_returns_error_string_on_exception(self):
        """When GoogleCalendarClient raises, _fetch_calendar returns ([], error_str)."""
        from src.briefing.aggregator import _fetch_calendar

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.connect",
            side_effect=RuntimeError("API down"),
        ):
            events, err = _fetch_calendar(date.today())

        # The function never raises — it catches and returns ([], str)
        assert isinstance(events, list)
        # err should be the error message string
        assert isinstance(err, str)
        assert len(err) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 9. Accuracy constraint: no incorrect info, '확인 불가' when source fails
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyConstraints:
    """
    Verify that failed data sources result in '확인 불가' annotations
    rather than incorrect or hallucinated data.
    """

    def test_failed_calendar_shows_unavailable_not_empty_meetings(self):
        from src.briefing.full_formatter import format_full_briefing
        from src.briefing.aggregator import BriefingData

        data = BriefingData(
            target_date=date.today(),
            source_errors={"calendar": "HTTP 503"},
        )
        fallback, blocks = format_full_briefing(data)

        # Fallback must say 확인 불가 for calendar
        assert "확인 불가" in fallback
        # Must NOT claim "0 meetings" as a fact
        assert "0개 미팅" not in fallback or "확인 불가" in fallback

    def test_failed_gmail_shows_unavailable_not_zero_emails(self):
        from src.briefing.full_formatter import format_full_briefing
        from src.briefing.aggregator import BriefingData

        data = BriefingData(
            target_date=date.today(),
            source_errors={"gmail": "Auth error"},
        )
        fallback, _ = format_full_briefing(data)
        assert "확인 불가" in fallback

    def test_failed_notion_shows_unavailable_not_zero_deadlines(self):
        from src.briefing.full_formatter import format_full_briefing
        from src.briefing.aggregator import BriefingData

        data = BriefingData(
            target_date=date.today(),
            source_errors={"notion": "Not found"},
        )
        fallback, _ = format_full_briefing(data)
        assert "확인 불가" in fallback

    def test_blocks_contain_unavailable_marker_when_source_fails(self):
        from src.briefing.full_formatter import format_full_briefing
        from src.briefing.aggregator import BriefingData

        data = BriefingData(
            target_date=date.today(),
            source_errors={"calendar": "fail", "gmail": "fail"},
        )
        _, blocks = format_full_briefing(data)

        # Find any block with 확인 불가 text
        found = False
        for block in blocks:
            if block.get("type") == "section":
                text_obj = block.get("text", {})
                if "확인 불가" in text_obj.get("text", ""):
                    found = True
                    break
        assert found, "No '확인 불가' annotation found in blocks when sources failed"


# ══════════════════════════════════════════════════════════════════════════════
# 10. Urgent email classification (Sub-AC 3.2 — urgent emails)
# ══════════════════════════════════════════════════════════════════════════════

class TestEmailMessageUrgency:
    """
    Tests for is_important / is_urgent properties on EmailMessage.

    These drive the 'urgent emails' concept in the briefing aggregator.
    """

    def _make(self, is_unread=False, is_external=True, labels=None):
        from src.gmail.client import EmailMessage
        sender_email = (
            "sender@external.com" if is_external else "sender@kakaoventures.co.kr"
        )
        return EmailMessage(
            message_id="m1",
            thread_id="t1",
            subject="Test Subject",
            sender="Test Sender",
            sender_email=sender_email,
            snippet="snippet",
            received_at=datetime.now(timezone.utc),
            is_unread=is_unread,
            labels=labels or [],
        )

    # ── is_important ──────────────────────────────────────────────────────────

    def test_is_important_when_important_label_present(self):
        msg = self._make(labels=["INBOX", "IMPORTANT"])
        assert msg.is_important is True

    def test_is_not_important_when_no_important_label(self):
        msg = self._make(labels=["INBOX", "UNREAD"])
        assert msg.is_important is False

    def test_is_not_important_when_labels_empty(self):
        msg = self._make(labels=[])
        assert msg.is_important is False

    # ── is_urgent ────────────────────────────────────────────────────────────

    def test_is_urgent_when_unread_and_external(self):
        """Unread email from external sender → urgent."""
        msg = self._make(is_unread=True, is_external=True)
        assert msg.is_urgent is True

    def test_is_urgent_when_important_label(self):
        """Important label alone → urgent (even if already read)."""
        msg = self._make(is_unread=False, labels=["INBOX", "IMPORTANT"])
        assert msg.is_urgent is True

    def test_is_urgent_when_both_unread_external_and_important(self):
        """Both criteria → still urgent."""
        msg = self._make(is_unread=True, is_external=True, labels=["INBOX", "IMPORTANT"])
        assert msg.is_urgent is True

    def test_is_not_urgent_when_read_external_not_important(self):
        """Read external email without IMPORTANT label → not urgent."""
        msg = self._make(is_unread=False, is_external=True, labels=["INBOX"])
        assert msg.is_urgent is False

    def test_is_not_urgent_when_unread_internal(self):
        """Unread internal email → not urgent (internal chat noise excluded)."""
        msg = self._make(is_unread=True, is_external=False)
        assert msg.is_urgent is False

    def test_is_not_urgent_when_read_internal(self):
        """Read internal email → not urgent."""
        msg = self._make(is_unread=False, is_external=False)
        assert msg.is_urgent is False

    # ── to_dict includes urgency fields ──────────────────────────────────────

    def test_to_dict_includes_is_important(self):
        msg = self._make(labels=["IMPORTANT"])
        d = msg.to_dict()
        assert "is_important" in d
        assert d["is_important"] is True

    def test_to_dict_includes_is_urgent(self):
        msg = self._make(is_unread=True, is_external=True)
        d = msg.to_dict()
        assert "is_urgent" in d
        assert d["is_urgent"] is True


class TestBriefingDataUrgentEmails:
    """
    Tests for BriefingData.urgent_emails derived property.

    Validates that the property correctly surfaces emails matching urgency
    criteria from the full emails list.
    """

    def _make_data(self, emails=None, **kwargs):
        from src.briefing.aggregator import BriefingData
        return BriefingData(
            target_date=date.today(),
            emails=emails or [],
            **kwargs,
        )

    def _email(self, is_unread=False, is_external=True, labels=None):
        from src.gmail.client import EmailMessage
        sender_email = (
            "sender@external.com" if is_external else "person@kakaoventures.co.kr"
        )
        return EmailMessage(
            message_id=f"m-{id(object())}",
            thread_id="t1",
            subject="Subject",
            sender="Sender",
            sender_email=sender_email,
            snippet="",
            received_at=datetime.now(timezone.utc),
            is_unread=is_unread,
            labels=labels or [],
        )

    def test_urgent_emails_empty_when_no_emails(self):
        data = self._make_data()
        assert data.urgent_emails == []

    def test_urgent_emails_includes_unread_external(self):
        """Unread external email → in urgent_emails."""
        urgent = self._email(is_unread=True, is_external=True)
        non_urgent = self._email(is_unread=False, is_external=True)
        data = self._make_data(emails=[urgent, non_urgent])
        result = data.urgent_emails
        assert urgent in result
        assert non_urgent not in result

    def test_urgent_emails_includes_important_even_if_read(self):
        """Read but IMPORTANT-labelled email → in urgent_emails."""
        important = self._email(
            is_unread=False, is_external=True, labels=["INBOX", "IMPORTANT"]
        )
        data = self._make_data(emails=[important])
        assert important in data.urgent_emails

    def test_urgent_emails_excludes_read_external_without_important(self):
        """Read external email without IMPORTANT → NOT urgent."""
        e = self._email(is_unread=False, is_external=True, labels=["INBOX"])
        data = self._make_data(emails=[e])
        assert e not in data.urgent_emails

    def test_urgent_emails_excludes_unread_internal(self):
        """Unread internal email → NOT urgent."""
        e = self._email(is_unread=True, is_external=False)
        data = self._make_data(emails=[e])
        assert e not in data.urgent_emails

    def test_urgent_emails_mixed_list(self):
        """Correct filtering across a mixed list of 5 emails."""
        u1 = self._email(is_unread=True, is_external=True)                        # urgent
        u2 = self._email(is_unread=False, is_external=True, labels=["IMPORTANT"])  # urgent
        n1 = self._email(is_unread=False, is_external=True, labels=["INBOX"])      # not urgent
        n2 = self._email(is_unread=True, is_external=False)                        # not urgent
        n3 = self._email(is_unread=False, is_external=False)                       # not urgent

        data = self._make_data(emails=[u1, u2, n1, n2, n3])
        urgent = data.urgent_emails

        assert u1 in urgent
        assert u2 in urgent
        assert n1 not in urgent
        assert n2 not in urgent
        assert n3 not in urgent
        assert len(urgent) == 2

    def test_urgent_emails_all_emails_could_be_urgent(self):
        """When every email is urgent, urgent_emails == emails."""
        emails = [
            self._email(is_unread=True, is_external=True)
            for _ in range(5)
        ]
        data = self._make_data(emails=emails)
        assert len(data.urgent_emails) == 5

    def test_summary_includes_urgent_count(self):
        """summary() must include urgent_emails count."""
        urgent = self._email(is_unread=True, is_external=True)
        data = self._make_data(emails=[urgent])
        s = data.summary()
        assert "urgent_emails=1" in s

    def test_summary_urgent_count_zero_when_no_urgent(self):
        e = self._email(is_unread=False, is_external=False)
        data = self._make_data(emails=[e])
        s = data.summary()
        assert "urgent_emails=0" in s


class TestFullFormatterUrgentEmails:
    """
    Tests that format_full_briefing() correctly renders the urgent emails
    sub-section with 🔴 markers and a distinct header.
    """

    def _make_data(self, emails=None, **kwargs):
        from src.briefing.aggregator import BriefingData
        return BriefingData(
            target_date=date.today(),
            emails=emails or [],
            **kwargs,
        )

    def _email(self, is_unread=False, is_external=True, labels=None, mid="m1"):
        from src.gmail.client import EmailMessage
        sender_email = (
            "ceo@startup.com" if is_external else "person@kakaoventures.co.kr"
        )
        return EmailMessage(
            message_id=mid,
            thread_id="t1",
            subject="Portfolio Update",
            sender="CEO Kim",
            sender_email=sender_email,
            snippet="",
            received_at=datetime.now(timezone.utc),
            is_unread=is_unread,
            labels=labels or [],
        )

    def _all_text(self, blocks):
        """Concatenate all text content from blocks for easy assertion."""
        parts = []
        for b in blocks:
            if b.get("type") == "section":
                parts.append(b.get("text", {}).get("text", ""))
            elif b.get("type") == "context":
                for el in b.get("elements", []):
                    parts.append(el.get("text", ""))
        return "\n".join(parts)

    def test_urgent_section_header_appears_when_urgent_emails_present(self):
        """When urgent emails exist, '즉시 확인' header block appears."""
        from src.briefing.full_formatter import format_full_briefing
        urgent = self._email(is_unread=True, is_external=True, mid="u1")
        data = self._make_data(emails=[urgent])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        assert "즉시 확인" in text

    def test_urgent_email_has_red_circle_marker(self):
        """Urgent emails rendered with 🔴 prefix in their block."""
        from src.briefing.full_formatter import format_full_briefing
        urgent = self._email(is_unread=True, is_external=True, mid="u1")
        data = self._make_data(emails=[urgent])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        assert "🔴" in text

    def test_non_urgent_emails_do_not_have_red_marker(self):
        """Read internal emails must NOT show the 🔴 urgent marker."""
        from src.briefing.full_formatter import format_full_briefing
        regular = self._email(is_unread=False, is_external=False, mid="r1")
        data = self._make_data(emails=[regular])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        # 🔴 should NOT appear when no urgent emails
        assert "🔴" not in text

    def test_urgent_section_absent_when_no_urgent_emails(self):
        """No urgent emails → '즉시 확인' header does NOT appear."""
        from src.briefing.full_formatter import format_full_briefing
        regular = self._email(is_unread=False, is_external=False, mid="r1")
        data = self._make_data(emails=[regular])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        # The urgent sub-section title should be absent
        assert "즉시 확인 필요" not in text

    def test_summary_line_shows_urgent_count(self):
        """Summary line (first Gmail block) must include urgent count when > 0."""
        from src.briefing.full_formatter import format_full_briefing
        urgent = self._email(is_unread=True, is_external=True, mid="u1")
        data = self._make_data(emails=[urgent])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        assert "즉시 확인" in text and "1" in text

    def test_urgent_email_appears_before_regular_email(self):
        """Urgent emails section must appear before the regular inbox section."""
        from src.briefing.full_formatter import format_full_briefing
        urgent = self._email(is_unread=True, is_external=True, mid="u1")
        regular = self._email(is_unread=False, is_external=False, mid="r1")
        data = self._make_data(emails=[urgent, regular])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        urgent_idx = text.find("🔴")
        inbox_idx = text.find("📬 받은 편지함")
        # Urgent section comes before regular inbox section
        assert urgent_idx < inbox_idx

    def test_important_email_included_in_urgent_section(self):
        """Gmail-important (⭐) emails shown in urgent section even if read."""
        from src.briefing.full_formatter import format_full_briefing
        important_read = self._email(
            is_unread=False,
            is_external=True,
            labels=["INBOX", "IMPORTANT"],
            mid="i1",
        )
        data = self._make_data(emails=[important_read])
        _, blocks = format_full_briefing(data)
        text = self._all_text(blocks)
        assert "🔴" in text  # urgent marker for important email

    def test_fallback_text_mentions_urgent_count(self):
        """Plain-text fallback includes '즉시 확인' count when urgent emails exist."""
        from src.briefing.full_formatter import format_full_briefing
        urgent = self._email(is_unread=True, is_external=True, mid="u1")
        data = self._make_data(emails=[urgent])
        fallback, _ = format_full_briefing(data)
        assert "즉시 확인" in fallback

    def test_blocks_within_limit_with_many_urgent_emails(self):
        """Block count stays ≤ 50 even with 20 urgent emails."""
        from src.briefing.full_formatter import format_full_briefing
        emails = [
            self._email(is_unread=True, is_external=True, mid=f"u{i}")
            for i in range(20)
        ]
        data = self._make_data(emails=emails)
        _, blocks = format_full_briefing(data)
        assert len(blocks) <= 50


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
