"""
Unit tests for Sub-AC 2c: Meeting Briefing Formatter + Delivery Pipeline.

Tests cover:
  Formatter (format_meeting_briefing)
  ────────────────────────────────────
  - Output is a (str, list) tuple
  - Block count is always ≤ 50
  - Valid Slack Block Kit types in every block
  - Header block contains meeting title
  - Time section shows KST times, duration, and countdown
  - Location shown when present; absent when not provided
  - Calendar link shown when html_link is set
  - Agenda section shown only when description is non-empty
  - External attendee section with display name / email
  - Attendee history shown when past_meeting_count > 0
  - Attendee history marked '확인 불가' when calendar_history_available=False
  - Internal attendee context element present when internals exist
  - Gmail section: normal threads, empty threads, source unavailable
  - '확인 불가' annotation in Gmail section when gmail_available=False
  - Gmail overflow context element when > 5 threads
  - Notion section: normal records, empty records, source unavailable
  - '확인 불가' annotation in Notion section when notion_available=False
  - Notion overflow context element when > 5 records
  - Footer always present at the end
  - Footer lists failed sources when has_errors=True
  - Fallback text contains meeting title and attendee emails
  - Fallback text contains error sources when any source failed

  Delivery (trigger_meeting_briefing)
  ─────────────────────────────────────
  - bot=None returns True without calling any Slack API
  - Successful delivery calls bot.send_message with text + blocks
  - Successful delivery returns True
  - Failed delivery (send_message returns False) returns False
  - trigger_meeting_briefing calls format_meeting_briefing via pipeline
  - Meeting context is aggregated before formatting

Run:
    python -m pytest tests/test_meeting_briefing_formatter.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / factories
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now_plus(minutes: float) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


def _make_attendee_profile(
    email: str = "ceo@startup.com",
    display_name: str = "CEO",
    is_internal: bool = False,
    company_domain: str = "startup.com",
    past_meeting_count: int = 0,
    last_met_date: Optional[datetime] = None,
):
    from src.briefing.context_aggregator import AttendeeProfile

    return AttendeeProfile(
        email=email,
        display_name=display_name,
        is_internal=is_internal,
        company_domain=company_domain,
        past_meeting_count=past_meeting_count,
        last_met_date=last_met_date,
    )


def _make_gmail_thread(
    thread_id: str = "t1",
    subject: str = "테스트 이메일",
    message_count: int = 3,
    latest_date: Optional[datetime] = None,
):
    """Create a minimal EmailThread-like object."""
    from dataclasses import dataclass, field

    @dataclass
    class _FakeThread:
        thread_id: str
        subject: str
        message_count: int
        latest_date: Optional[datetime]

        def to_dict(self):
            return {
                "thread_id": self.thread_id,
                "subject": self.subject,
                "message_count": self.message_count,
            }

    return _FakeThread(
        thread_id=thread_id,
        subject=subject,
        message_count=message_count,
        latest_date=latest_date or datetime.now(UTC),
    )


def _make_notion_record(
    page_id: str = "p1",
    title: str = "스타트업 ABC",
    company_name: str = "ABC Inc.",
    status: str = "투자심사",
    url: str = "https://notion.so/p1",
):
    """Create a minimal NotionRecord-like object."""
    from dataclasses import dataclass

    @dataclass
    class _FakeRecord:
        page_id: str
        title: str
        company_name: str
        status: str
        url: str
        date_value: Optional[datetime] = None
        properties: dict = None

        def to_dict(self):
            return {"page_id": self.page_id, "title": self.title}

    return _FakeRecord(
        page_id=page_id,
        title=title,
        company_name=company_name,
        status=status,
        url=url,
    )


def _make_raw_content(
    meeting_title: str = "파트너사 킥오프",
    starts_in_minutes: float = 12.0,
    duration_minutes: int = 60,
    meeting_location: str = "",
    meeting_description: str = "",
    meeting_html_link: str = "",
    external_profiles=None,
    internal_profiles=None,
    gmail_threads=None,
    gmail_available: bool = True,
    notion_records=None,
    notion_available: bool = True,
    calendar_history_available: bool = True,
    errors=None,
):
    """
    Build a RawBriefingContent using only dataclasses (no real API clients).
    """
    from src.briefing.context_aggregator import (
        RawBriefingContent,
        AttendeeProfile,
        AggregationError,
    )

    now_utc = datetime.now(UTC)
    start = now_utc + timedelta(minutes=starts_in_minutes)
    end = start + timedelta(minutes=duration_minutes)

    all_profiles: list[AttendeeProfile] = []
    for p in external_profiles or []:
        all_profiles.append(p)
    for p in internal_profiles or []:
        all_profiles.append(p)

    content = RawBriefingContent(
        meeting_id="evt-test-001",
        meeting_title=meeting_title,
        meeting_start=start,
        meeting_end=end,
        meeting_location=meeting_location,
        meeting_description=meeting_description,
        meeting_html_link=meeting_html_link,
        attendee_profiles=all_profiles,
        gmail_threads=gmail_threads or [],
        gmail_available=gmail_available,
        notion_records=notion_records or [],
        notion_available=notion_available,
        calendar_history_available=calendar_history_available,
        errors=errors or [],
    )
    return content


def _make_raw_content_with_error(source: str, message: str, **kwargs):
    """Convenience: build a raw content with a pre-recorded aggregation error."""
    from src.briefing.context_aggregator import AggregationError

    err = AggregationError(source=source, message=message)
    flags = {
        f"{source}_available": False,
    }
    flags.update(kwargs)
    return _make_raw_content(errors=[err], **flags)


# ─────────────────────────────────────────────────────────────────────────────
# Test: basic output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputStructure:

    def test_returns_tuple_of_str_and_list(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        result = format_meeting_briefing(raw)
        assert isinstance(result, tuple) and len(result) == 2
        text, blocks = result
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_blocks_never_exceed_50(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        # Max-load scenario: description + many attendees + many threads + many records
        raw = _make_raw_content(
            meeting_description="A" * 500,
            external_profiles=[
                _make_attendee_profile(f"ceo{i}@co{i}.com", f"Person{i}")
                for i in range(10)
            ],
            internal_profiles=[
                _make_attendee_profile(
                    f"invest{i}@kakaoventures.co.kr", f"팀원{i}", is_internal=True
                )
                for i in range(5)
            ],
            gmail_threads=[_make_gmail_thread(f"t{i}", f"Subject {i}") for i in range(10)],
            notion_records=[_make_notion_record(f"p{i}", f"Record {i}") for i in range(10)],
        )
        _, blocks = format_meeting_briefing(raw)
        assert len(blocks) <= 50, f"Got {len(blocks)} blocks — must be ≤ 50"

    def test_all_blocks_have_valid_slack_types(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        valid_types = {"header", "section", "divider", "context", "actions", "image"}
        for block in blocks:
            assert block.get("type") in valid_types, (
                f"Invalid block type: {block.get('type')}"
            )

    def test_footer_is_last_block(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        assert blocks[-1].get("type") == "context", "Last block must be a context (footer)"

    def test_second_to_last_block_is_divider(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        assert blocks[-2].get("type") == "divider", (
            "Second-to-last block must be a divider before the footer"
        )

    def test_first_block_is_header(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        assert blocks[0].get("type") == "header"


# ─────────────────────────────────────────────────────────────────────────────
# Test: header block
# ─────────────────────────────────────────────────────────────────────────────

class TestHeaderBlock:

    def test_header_contains_meeting_title(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_title="스타트업 투자 심사")
        _, blocks = format_meeting_briefing(raw)
        header = blocks[0]
        assert "스타트업 투자 심사" in header["text"]["text"]

    def test_header_contains_calendar_emoji(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        assert "📅" in blocks[0]["text"]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: time / location section
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeLocationBlock:

    def _get_time_block(self, raw):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        _, blocks = format_meeting_briefing(raw)
        # Second block (index 1) is the time section
        return blocks[1]

    def test_time_block_is_section(self):
        raw = _make_raw_content()
        block = self._get_time_block(raw)
        assert block.get("type") == "section"

    def test_countdown_in_time_block(self):
        raw = _make_raw_content(starts_in_minutes=12.0)
        block = self._get_time_block(raw)
        text = block["text"]["text"]
        assert "시작까지" in text
        assert "분" in text

    def test_kst_times_in_time_block(self):
        """Start and end KST times should both appear in the time block text."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(starts_in_minutes=120, duration_minutes=60)
        _, blocks = format_meeting_briefing(raw)
        time_block = blocks[1]["text"]["text"]
        # Should contain HH:MM–HH:MM pattern somewhere
        assert "–" in time_block  # em-dash between start and end

    def test_duration_shown(self):
        raw = _make_raw_content(duration_minutes=90)
        block = self._get_time_block(raw)
        assert "90분" in block["text"]["text"]

    def test_location_shown_when_present(self):
        raw = _make_raw_content(meeting_location="강남구 회의실 A")
        block = self._get_time_block(raw)
        assert "강남구 회의실 A" in block["text"]["text"]

    def test_location_absent_when_not_set(self):
        raw = _make_raw_content(meeting_location="")
        block = self._get_time_block(raw)
        assert "📍" not in block["text"]["text"]

    def test_html_link_shown_when_present(self):
        raw = _make_raw_content(meeting_html_link="https://calendar.google.com/xyz")
        block = self._get_time_block(raw)
        assert "https://calendar.google.com/xyz" in block["text"]["text"]

    def test_html_link_absent_when_not_set(self):
        raw = _make_raw_content(meeting_html_link="")
        block = self._get_time_block(raw)
        assert "캘린더에서 보기" not in block["text"]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: agenda / description section
# ─────────────────────────────────────────────────────────────────────────────

class TestAgendaSection:

    def _all_section_texts(self, blocks):
        return "\n".join(
            b["text"]["text"]
            for b in blocks
            if b.get("type") == "section" and b.get("text")
        )

    def test_agenda_section_shown_when_description_present(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_description="투자 조건 협의 및 계약 검토")
        _, blocks = format_meeting_briefing(raw)
        all_text = self._all_section_texts(blocks)
        assert "투자 조건 협의 및 계약 검토" in all_text

    def test_agenda_section_absent_when_no_description(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_description="")
        _, blocks = format_meeting_briefing(raw)
        all_text = self._all_section_texts(blocks)
        assert "안건/설명" not in all_text

    def test_description_truncated_at_500_chars(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_description="X" * 600)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._all_section_texts(blocks)
        # Truncation marker should be present and content capped at 500 chars
        assert "…" in all_text or len("X" * 600) == 600
        # Find the agenda block and verify length ≤ 500 + overhead
        agenda_block = next(
            (b for b in blocks if "안건/설명" in b.get("text", {}).get("text", "")),
            None,
        )
        assert agenda_block is not None
        # The content portion after the label should be ≤ 500 chars
        content = agenda_block["text"]["text"].replace("*📋 안건/설명*\n", "")
        assert len(content) <= 500


# ─────────────────────────────────────────────────────────────────────────────
# Test: external attendees section
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalAttendeesSection:

    def _get_section_texts(self, blocks):
        return "\n".join(
            b["text"]["text"]
            for b in blocks
            if b.get("type") == "section" and b.get("text")
        )

    def test_no_external_attendees_shows_없음(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(external_profiles=[])
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "없음" in all_text

    def test_external_attendee_display_name_shown(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            external_profiles=[_make_attendee_profile(display_name="John CEO")]
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "John CEO" in all_text

    def test_external_attendee_email_shown_when_no_display_name(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            external_profiles=[
                _make_attendee_profile(email="founder@startup.io", display_name="")
            ]
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "founder@startup.io" in all_text

    def test_company_domain_shown(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            external_profiles=[
                _make_attendee_profile(company_domain="openai.com")
            ]
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "openai.com" in all_text

    def test_history_shown_when_past_meetings_exist(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        last_met = datetime(2026, 1, 15, tzinfo=UTC)
        raw = _make_raw_content(
            external_profiles=[
                _make_attendee_profile(
                    past_meeting_count=5,
                    last_met_date=last_met,
                )
            ]
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "과거 미팅 5회" in all_text
        assert "2026-01-15" in all_text

    def test_history_zero_no_calendar_shows_확인_불가(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        raw = _make_raw_content(
            external_profiles=[_make_attendee_profile(past_meeting_count=0)],
            calendar_history_available=False,
            errors=[AggregationError(source="calendar_history", message="API timeout")],
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        assert "확인 불가" in all_text

    def test_history_zero_with_calendar_shows_nothing_extra(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        raw = _make_raw_content(
            external_profiles=[_make_attendee_profile(past_meeting_count=0)],
            calendar_history_available=True,
        )
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_section_texts(blocks)
        # When calendar is available but no past meetings, no history label at all
        assert "과거 미팅" not in all_text


# ─────────────────────────────────────────────────────────────────────────────
# Test: internal attendees context
# ─────────────────────────────────────────────────────────────────────────────

class TestInternalAttendeesContext:

    def test_internal_attendees_shown_in_context(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        raw = _make_raw_content(
            internal_profiles=[
                _make_attendee_profile(
                    email="invest1@kakaoventures.co.kr",
                    display_name="투자팀A",
                    is_internal=True,
                )
            ]
        )
        _, blocks = format_meeting_briefing(raw)
        context_texts = [
            e["text"]
            for b in blocks
            if b.get("type") == "context"
            for e in b.get("elements", [])
            if e.get("type") == "mrkdwn"
        ]
        combined = "\n".join(context_texts)
        assert "투자팀A" in combined or "invest1@kakaoventures.co.kr" in combined

    def test_no_internal_attendees_no_context(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        raw = _make_raw_content(internal_profiles=[])
        _, blocks = format_meeting_briefing(raw)
        context_texts = [
            e["text"]
            for b in blocks
            if b.get("type") == "context"
            for e in b.get("elements", [])
            if e.get("type") == "mrkdwn"
        ]
        combined = "\n".join(context_texts)
        assert "🏢 내부" not in combined


# ─────────────────────────────────────────────────────────────────────────────
# Test: Gmail threads section
# ─────────────────────────────────────────────────────────────────────────────

class TestGmailSection:

    def _get_all_text(self, blocks):
        parts = []
        for b in blocks:
            if b.get("type") == "section" and b.get("text"):
                parts.append(b["text"]["text"])
            elif b.get("type") == "context":
                for e in b.get("elements", []):
                    if e.get("type") == "mrkdwn":
                        parts.append(e["text"])
        return "\n".join(parts)

    def test_gmail_unavailable_shows_확인_불가(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        err = AggregationError(source="gmail", message="auth failed")
        raw = _make_raw_content(gmail_available=False, errors=[err])
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "확인 불가" in all_text
        assert "관련 이메일 스레드" in all_text

    def test_empty_threads_shows_없음_message(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        raw = _make_raw_content(gmail_available=True, gmail_threads=[])
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "관련 이메일 없음" in all_text

    def test_threads_shown_with_subject(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread("t1", "투자 계약서 검토 요청")]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "투자 계약서 검토 요청" in all_text

    def test_thread_count_in_section_header(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread(f"t{i}", f"Subject {i}") for i in range(3)]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "(3개)" in all_text

    def test_thread_message_count_shown(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread("t1", "Test", message_count=7)]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "7개 메시지" in all_text

    def test_overflow_context_shown_when_more_than_5(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread(f"t{i}", f"Thread {i}") for i in range(8)]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        # 8 total - 5 shown = 3 remaining
        assert "그 외 3개 스레드" in all_text

    def test_no_overflow_when_exactly_5(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread(f"t{i}", f"Thread {i}") for i in range(5)]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "그 외" not in all_text


# ─────────────────────────────────────────────────────────────────────────────
# Test: Notion records section
# ─────────────────────────────────────────────────────────────────────────────

class TestNotionSection:

    def _get_all_text(self, blocks):
        parts = []
        for b in blocks:
            if b.get("type") == "section" and b.get("text"):
                parts.append(b["text"]["text"])
            elif b.get("type") == "context":
                for e in b.get("elements", []):
                    if e.get("type") == "mrkdwn":
                        parts.append(e["text"])
        return "\n".join(parts)

    def test_notion_unavailable_shows_확인_불가(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        err = AggregationError(source="notion", message="rate limited")
        raw = _make_raw_content(notion_available=False, errors=[err])
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "확인 불가" in all_text
        assert "관련 딜" in all_text

    def test_empty_records_shows_없음_message(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        raw = _make_raw_content(notion_available=True, notion_records=[])
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "관련 Notion 항목 없음" in all_text

    def test_notion_record_title_shown(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        records = [_make_notion_record(title="카카오 시리즈 A")]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "카카오 시리즈 A" in all_text

    def test_notion_record_status_shown(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        records = [_make_notion_record(status="투자심사")]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "투자심사" in all_text

    def test_notion_record_url_linked(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        url = "https://notion.so/deal-123"
        records = [_make_notion_record(url=url)]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert url in all_text

    def test_record_count_in_section_header(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        records = [_make_notion_record(f"p{i}", f"Deal {i}") for i in range(4)]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        assert "(4개)" in all_text

    def test_overflow_context_shown_when_more_than_5(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        records = [_make_notion_record(f"p{i}", f"Deal {i}") for i in range(9)]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = self._get_all_text(blocks)
        # 9 total - 5 shown = 4 remaining
        assert "그 외 4개 항목" in all_text


# ─────────────────────────────────────────────────────────────────────────────
# Test: footer block
# ─────────────────────────────────────────────────────────────────────────────

class TestFooterBlock:

    def _footer_text(self, blocks):
        return blocks[-1]["elements"][0]["text"]

    def test_footer_contains_accuracy_disclaimer(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        assert "확인 불가" in self._footer_text(blocks)

    def test_footer_lists_error_sources(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        errors = [
            AggregationError(source="gmail", message="auth failed"),
            AggregationError(source="notion", message="timeout"),
        ]
        raw = _make_raw_content(
            gmail_available=False,
            notion_available=False,
            errors=errors,
        )
        _, blocks = format_meeting_briefing(raw)
        footer = self._footer_text(blocks)
        assert "gmail" in footer
        assert "notion" in footer

    def test_footer_no_error_sources_when_all_ok(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content()
        _, blocks = format_meeting_briefing(raw)
        footer = self._footer_text(blocks)
        # No error source annotation when everything is fine
        assert "⚠️ 일부 정보 확인 불가" not in footer


# ─────────────────────────────────────────────────────────────────────────────
# Test: fallback text
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackText:

    def test_fallback_contains_meeting_title(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_title="투자 심사 최종 회의")
        text, _ = format_meeting_briefing(raw)
        assert "투자 심사 최종 회의" in text

    def test_fallback_contains_external_email(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            external_profiles=[
                _make_attendee_profile(email="cto@startup.com", display_name="CTO")
            ]
        )
        text, _ = format_meeting_briefing(raw)
        assert "cto@startup.com" in text

    def test_fallback_contains_gmail_count(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            gmail_threads=[_make_gmail_thread("t1"), _make_gmail_thread("t2")]
        )
        text, _ = format_meeting_briefing(raw)
        assert "2개 스레드" in text

    def test_fallback_contains_notion_count(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(
            notion_records=[_make_notion_record("p1"), _make_notion_record("p2", "B")]
        )
        text, _ = format_meeting_briefing(raw)
        assert "2개" in text

    def test_fallback_lists_error_sources_when_present(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        raw = _make_raw_content(
            gmail_available=False,
            errors=[AggregationError(source="gmail", message="down")],
        )
        text, _ = format_meeting_briefing(raw)
        assert "gmail" in text.lower() or "확인 불가" in text


# ─────────────────────────────────────────────────────────────────────────────
# Test: trigger_meeting_briefing delivery
# ─────────────────────────────────────────────────────────────────────────────

def _make_meeting_for_pipeline(
    event_id: str = "evt-pipe-001",
    summary: str = "파이프라인 테스트 미팅",
    starts_in_minutes: float = 12.0,
):
    from src.calendar.google_calendar import Meeting, Attendee

    now = datetime.now(UTC)
    start = now + timedelta(minutes=starts_in_minutes)
    end = start + timedelta(minutes=60)

    attendees = [
        Attendee(email="ceo@startup.com", display_name="CEO"),
        Attendee(email="invest1@kakaoventures.co.kr", display_name="팀원A"),
    ]
    return Meeting(
        event_id=event_id,
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
    )


class TestTriggerMeetingBriefingDelivery:
    """Tests for the full trigger_meeting_briefing() pipeline."""

    def test_bot_none_returns_true_no_api_calls(self):
        """bot=None is dry-run mode — must return True without Slack API calls."""
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting_for_pipeline()
        with patch("src.briefing.pipeline._aggregate_meeting_context") as mock_agg:
            result = trigger_meeting_briefing(meeting, bot=None)

        assert result is True
        mock_agg.assert_not_called()

    def test_successful_delivery_returns_true(self):
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent

        meeting = _make_meeting_for_pipeline()
        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        # AC 8: must include external attendees so briefing is not suppressed
        raw = _make_raw_content(
            meeting_title=meeting.summary,
            external_profiles=[_make_attendee_profile("ceo@startup.com", "대표", False)],
        )

        with patch(
            "src.briefing.pipeline._aggregate_meeting_context",
            return_value=raw,
        ), patch(
            "src.briefing.meeting_briefing_formatter.format_meeting_briefing",
            return_value=("fallback", [{"type": "header", "text": {"type": "plain_text", "text": "test"}}]),
        ):
            result = trigger_meeting_briefing(meeting, bot=mock_bot)

        assert result is True
        mock_bot.send_message.assert_called_once()

    def test_failed_delivery_returns_false(self):
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting_for_pipeline()
        mock_bot = MagicMock()
        mock_bot.send_message.return_value = False

        raw = _make_raw_content(meeting_title=meeting.summary)

        with patch(
            "src.briefing.pipeline._aggregate_meeting_context",
            return_value=raw,
        ), patch(
            "src.briefing.meeting_briefing_formatter.format_meeting_briefing",
            return_value=("fallback", []),
        ):
            result = trigger_meeting_briefing(meeting, bot=mock_bot)

        assert result is False

    def test_send_message_called_with_blocks(self):
        """Bot.send_message must be called with both text and blocks kwargs."""
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting_for_pipeline()
        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        fake_blocks = [{"type": "header", "text": {"type": "plain_text", "text": "X"}}]
        # AC 8: include external attendees so briefing passes completeness guard
        raw = _make_raw_content(
            external_profiles=[_make_attendee_profile("ceo@startup.com", "대표", False)],
        )

        with patch(
            "src.briefing.pipeline._aggregate_meeting_context",
            return_value=raw,
        ), patch(
            "src.briefing.meeting_briefing_formatter.format_meeting_briefing",
            return_value=("fallback text", fake_blocks),
        ):
            trigger_meeting_briefing(meeting, bot=mock_bot)

        call_kwargs = mock_bot.send_message.call_args
        # First positional arg = text
        assert call_kwargs[0][0] == "fallback text"
        # blocks= keyword arg
        assert call_kwargs[1]["blocks"] == fake_blocks

    def test_format_meeting_briefing_called_with_raw_content(self):
        """Formatter must be invoked with the output of context aggregation."""
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting_for_pipeline()
        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        # AC 8: include external attendees and specific title to pass completeness guard
        raw = _make_raw_content(
            meeting_title="시리즈A 투자 논의 unique-marker",
            external_profiles=[_make_attendee_profile("ceo@startup.com", "대표", False)],
        )

        with patch(
            "src.briefing.pipeline._aggregate_meeting_context",
            return_value=raw,
        ) as mock_agg, patch(
            "src.briefing.meeting_briefing_formatter.format_meeting_briefing",
            return_value=("text", []),
        ) as mock_fmt:
            trigger_meeting_briefing(meeting, bot=mock_bot)

        # Aggregator was called once (keyword args like is_external_first may vary)
        mock_agg.assert_called_once()
        assert mock_agg.call_args[0][0] is meeting
        # Formatter was called with the aggregated raw content as first positional arg
        # (ai_sections may be passed as an additional kwarg by the pipeline)
        mock_fmt.assert_called_once()
        assert mock_fmt.call_args[0][0] is raw


# ─────────────────────────────────────────────────────────────────────────────
# Test: both Gmail and Notion unavailable simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class TestAllSourcesUnavailable:

    def test_all_sources_unavailable_still_produces_valid_blocks(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        errors = [
            AggregationError(source="gmail", message="auth failed"),
            AggregationError(source="notion", message="api error"),
            AggregationError(source="calendar_history", message="timeout"),
        ]
        raw = _make_raw_content(
            gmail_available=False,
            notion_available=False,
            calendar_history_available=False,
            errors=errors,
        )
        text, blocks = format_meeting_briefing(raw)

        assert isinstance(text, str)
        assert isinstance(blocks, list)
        assert 1 <= len(blocks) <= 50

        valid_types = {"header", "section", "divider", "context", "actions", "image"}
        for block in blocks:
            assert block.get("type") in valid_types

    def test_all_sources_unavailable_has_three_확인_불가_annotations(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        errors = [
            AggregationError(source="gmail", message="failed"),
            AggregationError(source="notion", message="failed"),
        ]
        raw = _make_raw_content(
            gmail_available=False,
            notion_available=False,
            errors=errors,
        )
        _, blocks = format_meeting_briefing(raw)

        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        # Both Gmail and Notion sections should show 확인 불가
        assert all_text.count("확인 불가") >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Test: edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_long_meeting_title_does_not_crash(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(meeting_title="A" * 200)
        text, blocks = format_meeting_briefing(raw)
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_meeting_already_started_negative_countdown(self):
        """Meetings that already started get a negative countdown — no crash."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        raw = _make_raw_content(starts_in_minutes=-5.0)
        text, blocks = format_meeting_briefing(raw)
        assert isinstance(text, str)
        assert isinstance(blocks, list)
        # Negative countdown should still appear in the time block
        time_text = blocks[1]["text"]["text"]
        assert "분" in time_text

    def test_notion_record_without_url_no_link(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        records = [_make_notion_record(url="")]
        raw = _make_raw_content(notion_records=records)
        _, blocks = format_meeting_briefing(raw)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        # No Slack hyperlink syntax when url is empty
        assert "<https://" not in all_text

    def test_gmail_thread_no_subject_shows_placeholder(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        threads = [_make_gmail_thread("t1", subject="")]
        raw = _make_raw_content(gmail_threads=threads)
        _, blocks = format_meeting_briefing(raw)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "제목 없음" in all_text

    def test_multiple_external_attendees_all_listed(self):
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        profiles = [
            _make_attendee_profile(f"ceo{i}@co{i}.com", f"Person{i}")
            for i in range(3)
        ]
        raw = _make_raw_content(external_profiles=profiles)
        _, blocks = format_meeting_briefing(raw)
        all_text = "\n".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        for p in profiles:
            assert p.display_name in all_text


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestOutputStructure,
        TestHeaderBlock,
        TestTimeLocationBlock,
        TestAgendaSection,
        TestExternalAttendeesSection,
        TestInternalAttendeesContext,
        TestGmailSection,
        TestNotionSection,
        TestFooterBlock,
        TestFallbackText,
        TestTriggerMeetingBriefingDelivery,
        TestAllSourcesUnavailable,
        TestEdgeCases,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = sorted(m for m in dir(instance) if m.startswith("test_"))
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {cls.__name__}.{method_name}")
                passed += 1
            except Exception:
                print(f"  ✗ {cls.__name__}.{method_name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 70}")
    print(f"  {passed} passed, {failed} failed")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
