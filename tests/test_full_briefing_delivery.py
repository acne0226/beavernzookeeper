"""
Tests for Sub-AC 3.3: Format aggregated briefing data into a structured
Slack Block Kit message and deliver it to the requesting user via DM.

This test suite verifies the complete pipeline:
  1. format_full_briefing(BriefingData) → (fallback_text, blocks)
     - Structured Block Kit output with header / calendar / gmail / notion / footer
     - Correct Korean date formatting
     - fetched_at timestamp context block
     - '확인 불가' annotations for failed data sources
     - Slack 50-block hard limit respected
     - Valid block types only

  2. run_aggregated_brief(target_date, bot, user_id)
     - Delivers formatted briefing directly to the requesting user (user_id)
     - Falls back to bot.send_message() when direct DM fails
     - Returns True on success, False on failure
     - Sends error notification on delivery failure

  3. /brief slash command → callback → run_aggregated_brief
     - user_id from the Slack event is threaded through to the DM delivery

All tests run entirely offline — no real Google / Slack / Notion API calls.

Run with:
    python -m pytest tests/test_full_briefing_delivery.py -v
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KST = ZoneInfo("Asia/Seoul")
TARGET_DATE = date(2026, 3, 29)


# ══════════════════════════════════════════════════════════════════════════════
# Stub data classes (offline — no API credentials required)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _StubMeeting:
    event_id: str = "evt-001"
    summary: str = "스타트업 미팅"
    start: datetime = None
    end: datetime = None
    is_external: bool = True
    attendees: list = field(default_factory=list)
    external_attendees: list = field(default_factory=list)
    description: str = "안건: 투자 검토"
    location: str = "강남 회의실 A"
    html_link: str = "https://calendar.google.com/event/test"
    organizer_email: str = "invest1@kakaoventures.co.kr"
    duration_minutes: int = 60
    all_day: bool = False

    def __post_init__(self):
        if self.start is None:
            self.start = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc)  # 10:00 KST
        if self.end is None:
            self.end = datetime(2026, 3, 29, 2, 0, tzinfo=timezone.utc)    # 11:00 KST


@dataclass
class _StubEmail:
    message_id: str = "msg-001"
    thread_id: str = "thread-001"
    subject: str = "투자 검토 요청"
    sender: str = "외부 대표"
    sender_email: str = "ceo@startup.com"
    snippet: str = "투자 관련 문의드립니다…"
    received_at: datetime = None
    is_unread: bool = True
    is_external: bool = True
    labels: list = field(default_factory=lambda: ["INBOX", "UNREAD"])

    def __post_init__(self):
        if self.received_at is None:
            self.received_at = datetime(2026, 3, 29, 6, 0, tzinfo=timezone.utc)  # 15:00 KST


@dataclass
class _StubDeadline:
    page_id: str = "page-001"
    name: str = "포트폴리오 A"
    deadline: date = None
    status: str = "진행 중"
    url: str = "https://notion.so/page-001"
    is_overdue: bool = False
    days_until: int = 5

    def __post_init__(self):
        if self.deadline is None:
            self.deadline = TARGET_DATE + timedelta(days=self.days_until)


def _make_briefing_data(
    meetings=None,
    emails=None,
    deadlines=None,
    source_errors=None,
    target_date=None,
    fetched_at=None,
):
    """Build a BriefingData instance without touching any real API."""
    from src.briefing.aggregator import BriefingData

    return BriefingData(
        target_date=target_date or TARGET_DATE,
        calendar_events=meetings if meetings is not None else [],
        emails=emails if emails is not None else [],
        notion_deadlines=deadlines if deadlines is not None else [],
        source_errors=source_errors or {},
        fetched_at=fetched_at or datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc),
    )


def _make_real_meeting(
    summary: str = "스타트업 미팅",
    is_external: bool = True,
    start: datetime = None,
    end: datetime = None,
):
    """
    Create a real Meeting dataclass instance (from src.calendar.google_calendar).

    The calendar formatter requires a real Meeting object (not a stub), so any
    test that exercises the calendar rendering section must use this factory.
    """
    from src.calendar.google_calendar import Meeting, Attendee

    if start is None:
        start = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc)   # 10:00 KST
    if end is None:
        end = datetime(2026, 3, 29, 2, 0, tzinfo=timezone.utc)     # 11:00 KST

    attendees = []
    if is_external:
        attendees.append(Attendee(email="ceo@startup.com", display_name="외부 대표"))
    attendees.append(Attendee(email="invest1@kakaoventures.co.kr", display_name="내부 팀원"))

    return Meeting(
        event_id="evt-test-001",
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
        html_link="https://calendar.google.com/event/test",
    )


def _make_bot_mock(send_message_return: bool = True) -> MagicMock:
    bot = MagicMock()
    bot.send_message.return_value = send_message_return
    bot._client.chat_postMessage.return_value = {"ok": True}
    return bot


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: format_full_briefing — output structure
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatFullBriefingStructure:
    """Verify the structural validity of the Block Kit output."""

    def test_returns_tuple_of_str_and_list(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data()
        result = format_full_briefing(bd)
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, blocks = result
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_blocks_not_empty(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        assert len(blocks) > 0

    def test_fallback_text_not_empty(self):
        from src.briefing.full_formatter import format_full_briefing
        text, _ = format_full_briefing(_make_briefing_data())
        assert len(text.strip()) > 0

    def test_block_count_within_slack_limit(self):
        """50 is Slack's hard block limit — must never exceed it."""
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(
            meetings=[_StubMeeting() for _ in range(10)],
            emails=[_StubEmail() for _ in range(30)],
            deadlines=[_StubDeadline(days_until=i + 1) for i in range(20)],
        )
        _, blocks = format_full_briefing(bd)
        assert len(blocks) <= 50

    def test_all_block_types_are_valid_slack_types(self):
        """Block Kit only supports a finite set of block types."""
        from src.briefing.full_formatter import format_full_briefing
        valid_types = {"header", "section", "divider", "context", "actions", "image", "input"}
        bd = _make_briefing_data(
            meetings=[_StubMeeting()],
            emails=[_StubEmail()],
            deadlines=[_StubDeadline()],
        )
        _, blocks = format_full_briefing(bd)
        for block in blocks:
            assert block.get("type") in valid_types, (
                f"Invalid Slack block type: {block.get('type')!r}"
            )

    def test_each_block_has_type_field(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        for block in blocks:
            assert "type" in block, f"Block missing 'type' key: {block}"


class TestFormatFullBriefingHeader:
    """Verify header and timestamp blocks."""

    def test_header_block_is_first_block(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        assert blocks[0]["type"] == "header"

    def test_header_contains_korean_date(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data(target_date=date(2026, 3, 29)))
        header = blocks[0]
        text = header["text"]["text"]
        assert "2026" in text
        assert "3월" in text
        assert "29일" in text

    def test_header_contains_briefing_emoji(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        header_text = blocks[0]["text"]["text"]
        assert "📋" in header_text

    def test_header_text_type_is_plain_text(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        assert blocks[0]["text"]["type"] == "plain_text"

    def test_korean_weekday_in_header(self):
        """2026-03-29 is a Sunday (일요일)."""
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data(target_date=date(2026, 3, 29)))
        header_text = blocks[0]["text"]["text"]
        assert "(일)" in header_text

    def test_fetched_at_context_block_present(self):
        """A context block with the briefing generation time should appear after the header."""
        from src.briefing.full_formatter import format_full_briefing
        fetched_utc = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc)  # 09:30 KST
        bd = _make_briefing_data(fetched_at=fetched_utc)
        _, blocks = format_full_briefing(bd)
        # The fetched_at context block should be the second block
        context_blocks = [b for b in blocks[:5] if b.get("type") == "context"]
        assert len(context_blocks) >= 1
        context_text = " ".join(
            e.get("text", "")
            for b in context_blocks
            for e in (b.get("elements") or [])
            if isinstance(e, dict)
        )
        # "09:30" is the KST time for UTC 00:30
        assert "09:30" in context_text
        assert "KST" in context_text

    def test_fetched_at_shows_kst_time(self):
        """fetched_at stored in UTC must be displayed in KST."""
        from src.briefing.full_formatter import format_full_briefing
        # UTC 15:00 = KST 00:00 (next day) — use a more typical time
        fetched_utc = datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc)  # 10:00 KST
        bd = _make_briefing_data(fetched_at=fetched_utc)
        _, blocks = format_full_briefing(bd)
        context_texts = []
        for b in blocks[:5]:
            if b.get("type") == "context":
                for e in (b.get("elements") or []):
                    if isinstance(e, dict):
                        context_texts.append(e.get("text", ""))
        combined = " ".join(context_texts)
        assert "10:00" in combined


class TestFormatFullBriefingSections:
    """Verify that all three data source sections appear correctly."""

    def test_calendar_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        section_texts = [
            b["text"]["text"]
            for b in blocks if b.get("type") == "section" and b.get("text")
        ]
        all_text = "\n".join(section_texts)
        assert "일정" in all_text or "캘린더" in all_text

    def test_gmail_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        section_texts = [
            b["text"]["text"]
            for b in blocks if b.get("type") == "section" and b.get("text")
        ]
        all_text = "\n".join(section_texts)
        assert "이메일" in all_text or "편지함" in all_text

    def test_notion_section_header_present(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        section_texts = [
            b["text"]["text"]
            for b in blocks if b.get("type") == "section" and b.get("text")
        ]
        all_text = "\n".join(section_texts)
        assert "마감" in all_text or "Notion" in all_text or "포트폴리오" in all_text

    def test_three_dividers_between_sections(self):
        """There should be at least 3 dividers separating the four sections."""
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        dividers = [b for b in blocks if b.get("type") == "divider"]
        assert len(dividers) >= 3

    def test_footer_context_block_at_end(self):
        """The last block should be a context footer."""
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        assert blocks[-1]["type"] == "context"

    def test_footer_contains_accuracy_disclaimer(self):
        """Footer must reference '확인 불가' for accuracy guarantee."""
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        footer = blocks[-1]
        footer_text = " ".join(
            e.get("text", "")
            for e in (footer.get("elements") or [])
            if isinstance(e, dict)
        )
        assert "확인 불가" in footer_text

    def test_second_to_last_block_is_divider(self):
        """Divider before footer is always present."""
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data())
        assert blocks[-2]["type"] == "divider"


class TestFormatFullBriefingCalendarSection:
    """Verify calendar meeting data appears correctly in the output."""

    def test_meeting_title_appears_in_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        # Use a real Meeting dataclass — the calendar formatter requires it
        meeting = _make_real_meeting(summary="포트폴리오 킥오프 미팅")
        _, blocks = format_full_briefing(_make_briefing_data(meetings=[meeting]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "포트폴리오 킥오프 미팅" in all_text

    def test_no_meetings_shows_empty_message(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data(meetings=[]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "없습니다" in all_text or "없음" in all_text

    def test_external_meeting_shows_globe_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        meeting = _make_real_meeting(is_external=True)
        _, blocks = format_full_briefing(_make_briefing_data(meetings=[meeting]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🌐" in all_text

    def test_multiple_meetings_all_shown(self):
        from src.briefing.full_formatter import format_full_briefing
        meetings = [
            _make_real_meeting(
                summary=f"미팅 {i}",
                start=datetime(2026, 3, 29, i + 1, 0, tzinfo=timezone.utc),
                end=datetime(2026, 3, 29, i + 2, 0, tzinfo=timezone.utc),
            )
            for i in range(3)
        ]
        _, blocks = format_full_briefing(_make_briefing_data(meetings=meetings))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "미팅 0" in all_text
        assert "미팅 1" in all_text
        assert "미팅 2" in all_text


class TestFormatFullBriefingGmailSection:
    """Verify email data appears correctly in the output."""

    def test_email_subject_appears_in_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        email = _StubEmail(subject="긴급 투자 요청 안건")
        _, blocks = format_full_briefing(_make_briefing_data(emails=[email]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "긴급 투자 요청 안건" in all_text

    def test_unread_email_shows_mailbox_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        email = _StubEmail(is_unread=True)
        _, blocks = format_full_briefing(_make_briefing_data(emails=[email]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "📬" in all_text

    def test_read_email_shows_empty_mailbox_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        email = _StubEmail(is_unread=False)
        _, blocks = format_full_briefing(_make_briefing_data(emails=[email]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "📭" in all_text

    def test_external_email_shows_globe_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        email = _StubEmail(is_external=True)
        _, blocks = format_full_briefing(_make_briefing_data(emails=[email]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🌐" in all_text

    def test_email_summary_shows_total_count(self):
        from src.briefing.full_formatter import format_full_briefing
        emails = [_StubEmail(message_id=f"msg-{i}") for i in range(5)]
        _, blocks = format_full_briefing(_make_briefing_data(emails=emails))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "*5개*" in all_text

    def test_no_emails_shows_empty_message(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data(emails=[]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "없습니다" in all_text or "없음" in all_text

    def test_emails_capped_at_10_shown(self):
        """More than 10 emails should show an overflow notice."""
        from src.briefing.full_formatter import format_full_briefing
        emails = [_StubEmail(message_id=f"msg-{i}", subject=f"이메일 {i}") for i in range(15)]
        _, blocks = format_full_briefing(_make_briefing_data(emails=emails))
        # Overflow should be mentioned in context elements
        context_texts = []
        for b in blocks:
            if b.get("type") == "context":
                for e in (b.get("elements") or []):
                    if isinstance(e, dict):
                        context_texts.append(e.get("text", ""))
        all_context = " ".join(context_texts)
        # Should mention remaining emails
        assert "이메일" in all_context or "5개" in all_context or "그 외" in all_context


class TestFormatFullBriefingNotionSection:
    """Verify Notion deadline data appears correctly in the output."""

    def test_deadline_item_name_appears_in_blocks(self):
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(name="스타트업 베타 투자 검토")
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "스타트업 베타 투자 검토" in all_text

    def test_overdue_item_shows_alarm_icon(self):
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(is_overdue=True, days_until=-3)
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🚨" in all_text

    def test_today_deadline_shows_red_circle(self):
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(is_overdue=False, days_until=0)
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🔴" in all_text

    def test_near_deadline_shows_orange_circle(self):
        """1-3 days until deadline shows orange circle."""
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(is_overdue=False, days_until=2)
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🟠" in all_text

    def test_this_week_deadline_shows_yellow_circle(self):
        """4-7 days until deadline shows yellow circle."""
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(is_overdue=False, days_until=5)
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🟡" in all_text

    def test_upcoming_deadline_shows_green_circle(self):
        """8+ days until deadline shows green circle."""
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(is_overdue=False, days_until=10)
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🟢" in all_text

    def test_no_deadlines_shows_empty_message(self):
        from src.briefing.full_formatter import format_full_briefing
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "없습니다" in all_text or "없음" in all_text

    def test_deadline_summary_shows_overdue_count(self):
        from src.briefing.full_formatter import format_full_briefing
        items = [
            _StubDeadline(name=f"항목 {i}", is_overdue=True, days_until=-i-1)
            for i in range(3)
        ]
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=items))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🚨" in all_text

    def test_notion_url_rendered_as_link(self):
        """When a Notion URL is available, the item name should be a link."""
        from src.briefing.full_formatter import format_full_briefing
        item = _StubDeadline(
            name="투자 포트폴리오 A",
            url="https://notion.so/abc123",
        )
        _, blocks = format_full_briefing(_make_briefing_data(deadlines=[item]))
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "https://notion.so/abc123" in all_text
        assert "투자 포트폴리오 A" in all_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Error Annotations ('확인 불가')
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorAnnotations:
    """
    Verify that failed data sources are annotated '확인 불가' and never
    silently omitted or replaced with fabricated data.
    """

    def test_calendar_failure_annotated_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"calendar": "OAuth token expired"})
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "확인 불가" in all_text

    def test_gmail_failure_annotated_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"gmail": "403 Forbidden"})
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "확인 불가" in all_text

    def test_notion_failure_annotated_unavailable(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"notion": "Database not found"})
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "확인 불가" in all_text

    def test_calendar_error_message_included_in_block(self):
        """The error cause should be visible in the '확인 불가' block."""
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"calendar": "token_revoked"})
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "token_revoked" in all_text

    def test_all_sources_failed_still_produces_valid_output(self):
        """Even total failure must produce a valid, non-empty response."""
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(
            source_errors={
                "calendar": "err1",
                "gmail": "err2",
                "notion": "err3",
            }
        )
        text, blocks = format_full_briefing(bd)
        assert isinstance(text, str) and len(text) > 0
        assert isinstance(blocks, list) and len(blocks) > 0
        # Block limit still respected
        assert len(blocks) <= 50

    def test_all_sources_failed_shows_three_unavailable_annotations(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(
            source_errors={"calendar": "e1", "gmail": "e2", "notion": "e3"}
        )
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        # All three sources should appear as unavailable
        unavailable_count = all_text.count("확인 불가")
        assert unavailable_count >= 3

    def test_partial_failure_does_not_affect_other_sections(self):
        """Gmail failure must not hide Calendar or Notion sections."""
        from src.briefing.full_formatter import format_full_briefing
        # Use a real Meeting dataclass for the calendar formatter
        meeting = _make_real_meeting(summary="일정이 있는 미팅")
        item = _StubDeadline(name="포트폴리오 항목")
        bd = _make_briefing_data(
            meetings=[meeting],
            deadlines=[item],
            source_errors={"gmail": "network error"},
        )
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        # Calendar and Notion must still appear
        assert "일정이 있는 미팅" in all_text
        assert "포트폴리오 항목" in all_text
        # Gmail section should be annotated
        assert "확인 불가" in all_text

    def test_error_annotation_warning_icon_present(self):
        """Error annotations should include a visible warning emoji."""
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"gmail": "timeout"})
        _, blocks = format_full_briefing(bd)
        all_text = " ".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "⚠️" in all_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Fallback text
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackText:
    """Verify the plain-text fallback sent as notification text."""

    def test_fallback_contains_target_date(self):
        from src.briefing.full_formatter import format_full_briefing
        text, _ = format_full_briefing(_make_briefing_data(target_date=date(2026, 3, 29)))
        assert "2026" in text
        assert "3월" in text
        assert "29일" in text

    def test_fallback_mentions_calendar(self):
        from src.briefing.full_formatter import format_full_briefing
        text, _ = format_full_briefing(_make_briefing_data(meetings=[_StubMeeting()]))
        assert any(kw in text for kw in ["캘린더", "일정", "미팅"])

    def test_fallback_mentions_gmail(self):
        from src.briefing.full_formatter import format_full_briefing
        text, _ = format_full_briefing(_make_briefing_data(emails=[_StubEmail()]))
        assert any(kw in text for kw in ["이메일", "메일", "편지함"])

    def test_fallback_mentions_notion(self):
        from src.briefing.full_formatter import format_full_briefing
        text, _ = format_full_briefing(_make_briefing_data(deadlines=[_StubDeadline()]))
        assert any(kw in text for kw in ["Notion", "마감"])

    def test_fallback_notes_unavailable_sources(self):
        from src.briefing.full_formatter import format_full_briefing
        bd = _make_briefing_data(source_errors={"gmail": "error"})
        text, _ = format_full_briefing(bd)
        assert "확인 불가" in text

    def test_fallback_shows_event_count_when_available(self):
        from src.briefing.full_formatter import format_full_briefing
        meetings = [_StubMeeting(summary=f"미팅 {i}") for i in range(3)]
        text, _ = format_full_briefing(_make_briefing_data(meetings=meetings))
        assert "3" in text


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: DM delivery to requesting user via run_aggregated_brief
# ══════════════════════════════════════════════════════════════════════════════

class TestDeliveryToRequestingUser:
    """
    Verify that run_aggregated_brief() delivers the formatted briefing
    directly to the requesting user's Slack user ID.
    """

    def _patch_aggregator(self, bd):
        return patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd)

    def _patch_formatter(self, text="fallback", blocks=None):
        return patch(
            "src.briefing.full_formatter.format_full_briefing",
            return_value=(text, blocks or [{"type": "section", "text": {"type": "mrkdwn", "text": "테스트"}}]),
        )

    def test_sends_dm_to_requesting_user_id(self):
        """When user_id is provided, the briefing must go directly to that user."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock()
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            result = run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_REQUESTING_USER",
            )

        bot._client.chat_postMessage.assert_called_once()
        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "U_REQUESTING_USER"
        assert result is True

    def test_dm_includes_blocks(self):
        """The DM to requesting user must include the Block Kit blocks."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock()
        bd = _make_briefing_data()
        expected_blocks = [{"type": "header", "text": {"type": "plain_text", "text": "브리핑"}}]

        with self._patch_aggregator(bd), self._patch_formatter(blocks=expected_blocks):
            run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_TEST",
            )

        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["blocks"] == expected_blocks

    def test_dm_includes_fallback_text(self):
        """The DM must include fallback text for notifications."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock()
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter(text="📋 브리핑 요약"):
            run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_TEST",
            )

        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["text"] == "📋 브리핑 요약"

    def test_fallback_to_send_message_when_direct_dm_fails(self):
        """When direct DM to user_id fails, bot.send_message() is used as fallback."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock(send_message_return=True)
        bot._client.chat_postMessage.side_effect = Exception("channel_not_found")
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            result = run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_TEST",
            )

        # Fallback send_message should be called
        bot.send_message.assert_called_once()
        assert result is True

    def test_returns_true_on_successful_delivery(self):
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock()
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            result = run_aggregated_brief(TARGET_DATE, bot=bot, user_id="U_TESTER")

        assert result is True

    def test_returns_false_when_all_delivery_attempts_fail(self):
        """Both direct DM and send_message fail → return False."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock(send_message_return=False)
        bot._client.chat_postMessage.side_effect = Exception("all failed")
        bot.send_error = MagicMock()
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            result = run_aggregated_brief(TARGET_DATE, bot=bot, user_id="U_TEST")

        assert result is False

    def test_error_dm_sent_on_delivery_failure(self):
        """On delivery failure, a best-effort error DM is sent."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock(send_message_return=False)
        bot._client.chat_postMessage.side_effect = Exception("network error")
        bot.send_error = MagicMock()
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            run_aggregated_brief(TARGET_DATE, bot=bot, user_id="U_TEST")

        # send_error should be called to notify about the failure
        bot.send_error.assert_called_once()

    def test_without_user_id_uses_bot_send_message(self):
        """When user_id is None, falls back to bot.send_message() (default target user)."""
        from src.briefing.pipeline import run_aggregated_brief

        bot = _make_bot_mock(send_message_return=True)
        bd = _make_briefing_data()

        with self._patch_aggregator(bd), self._patch_formatter():
            result = run_aggregated_brief(TARGET_DATE, bot=bot, user_id=None)

        bot.send_message.assert_called_once()
        assert result is True

    def test_dry_run_returns_true_without_bot(self):
        """When bot=None, function runs in dry-run mode and returns True."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        bd = BriefingData(target_date=TARGET_DATE)
        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd):
            result = run_aggregated_brief(TARGET_DATE, bot=None)

        assert result is True

    def test_defaults_target_date_to_today_kst(self):
        """When target_date is None, today's date in KST is used."""
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        today_kst = datetime.now(KST).date()
        bd = BriefingData(target_date=today_kst)
        bot = _make_bot_mock()

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd) as mock_agg, \
             self._patch_formatter():
            run_aggregated_brief(bot=bot)

        call_kwargs = mock_agg.call_args[1]
        assert call_kwargs["target_date"] == today_kst


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: /brief slash command → user_id threading
# ══════════════════════════════════════════════════════════════════════════════

class TestBriefCommandUserIdThreading:
    """
    Verify that the user_id from the /brief slash command is correctly passed
    through the callback chain to run_aggregated_brief, so the DM is sent to
    the requesting user, not the global target user.
    """

    def test_callback_passes_user_id_to_run_aggregated_brief(self):
        import main as main_module

        callback, bot_holder = main_module._make_briefing_callback()
        bot_mock = _make_bot_mock()
        bot_holder[0] = bot_mock

        with patch("src.briefing.pipeline.run_aggregated_brief") as mock_run:
            mock_run.return_value = True
            callback(TARGET_DATE, "U_REQUESTER_123", "C_CHANNEL")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("user_id") == "U_REQUESTER_123"

    def test_callback_passes_target_date_to_run_aggregated_brief(self):
        import main as main_module

        callback, bot_holder = main_module._make_briefing_callback()
        bot_mock = _make_bot_mock()
        bot_holder[0] = bot_mock

        test_date = date(2026, 4, 1)
        with patch("src.briefing.pipeline.run_aggregated_brief") as mock_run:
            mock_run.return_value = True
            callback(test_date, "U_USER", "C_CHANNEL")

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("target_date") == test_date

    def test_callback_passes_bot_to_run_aggregated_brief(self):
        import main as main_module

        callback, bot_holder = main_module._make_briefing_callback()
        bot_mock = _make_bot_mock()
        bot_holder[0] = bot_mock

        with patch("src.briefing.pipeline.run_aggregated_brief") as mock_run:
            mock_run.return_value = True
            callback(TARGET_DATE, "U_USER", "C_CHANNEL")

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("bot") is bot_mock

    def test_callback_is_callable(self):
        import main as main_module
        callback, _ = main_module._make_briefing_callback()
        assert callable(callback)


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: /brief slash command date parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestBriefCommandDateParsing:
    """Verify the /brief command parses date arguments correctly."""

    def test_no_argument_returns_today(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("")
        assert error is None
        assert result_date == date.today()

    def test_today_returns_today(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("today")
        assert error is None
        assert result_date == date.today()

    def test_tomorrow_returns_tomorrow(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("tomorrow")
        assert error is None
        from datetime import timedelta
        assert result_date == date.today() + timedelta(days=1)

    def test_iso_date_format(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("2026-04-01")
        assert error is None
        assert result_date == date(2026, 4, 1)

    def test_slash_date_format(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("2026/04/01")
        assert error is None
        assert result_date == date(2026, 4, 1)

    def test_invalid_argument_returns_error(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("next week")
        assert result_date is None
        assert error is not None
        assert len(error) > 0

    def test_case_insensitive_today(self):
        from src.slack.commands.brief import _parse_brief_date
        result_date, error = _parse_brief_date("TODAY")
        assert error is None
        assert result_date == date.today()


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Integration — full pipeline with mocked API clients
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """
    End-to-end integration: aggregate_briefing_data → format_full_briefing
    → DM delivery with mocked Slack / Google / Notion clients.
    """

    def test_full_pipeline_all_sources_success(self):
        """
        All three sources return data → formatted briefing delivered as DM.
        """
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData

        # Use real Meeting dataclass for correct calendar rendering
        meeting = _make_real_meeting(summary="통합 테스트 미팅")
        email = _StubEmail(subject="통합 테스트 이메일")
        deadline = _StubDeadline(name="통합 테스트 포트폴리오")
        bd = _make_briefing_data(
            meetings=[meeting],
            emails=[email],
            deadlines=[deadline],
        )

        bot = _make_bot_mock()

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd):
            result = run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_INTEGRATION_TEST",
            )

        assert result is True
        # Should have attempted DM delivery to the requesting user
        bot._client.chat_postMessage.assert_called_once()
        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "U_INTEGRATION_TEST"

    def test_full_pipeline_partial_source_failure(self):
        """
        Gmail fails → briefing still delivered with '확인 불가' annotation.
        """
        from src.briefing.pipeline import run_aggregated_brief

        bd = _make_briefing_data(
            meetings=[_make_real_meeting()],
            emails=[],
            deadlines=[_StubDeadline()],
            source_errors={"gmail": "API timeout"},
        )
        bot = _make_bot_mock()

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd):
            result = run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_USER",
            )

        assert result is True
        # Delivery should have been attempted
        bot._client.chat_postMessage.assert_called_once()
        # The blocks should contain '확인 불가' for gmail
        call_kwargs = bot._client.chat_postMessage.call_args[1]
        blocks = call_kwargs.get("blocks", [])
        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_full_pipeline_all_sources_failed(self):
        """
        All sources fail → briefing still delivered with all sections annotated.
        """
        from src.briefing.pipeline import run_aggregated_brief

        bd = _make_briefing_data(
            source_errors={
                "calendar": "calendar error",
                "gmail": "gmail error",
                "notion": "notion error",
            }
        )
        bot = _make_bot_mock()

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd):
            result = run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_USER",
            )

        assert result is True
        call_kwargs = bot._client.chat_postMessage.call_args[1]
        blocks = call_kwargs.get("blocks", [])
        # Block count still within Slack limit
        assert len(blocks) <= 50
        # At least one '확인 불가' annotation
        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_format_full_briefing_output_passthrough(self):
        """
        The blocks produced by format_full_briefing must reach chat_postMessage
        verbatim — no transformation in run_aggregated_brief.
        """
        from src.briefing.pipeline import run_aggregated_brief
        from src.briefing.aggregator import BriefingData
        from src.briefing.full_formatter import format_full_briefing

        bd = _make_briefing_data(
            meetings=[_make_real_meeting()],
            emails=[_StubEmail()],
            deadlines=[_StubDeadline()],
        )
        # Get the actual formatted output
        expected_text, expected_blocks = format_full_briefing(bd)

        bot = _make_bot_mock()

        with patch("src.briefing.aggregator.aggregate_briefing_data", return_value=bd):
            run_aggregated_brief(
                target_date=TARGET_DATE,
                bot=bot,
                user_id="U_TEST",
            )

        call_kwargs = bot._client.chat_postMessage.call_args[1]
        assert call_kwargs["text"] == expected_text
        assert call_kwargs["blocks"] == expected_blocks


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestFormatFullBriefingStructure,
        TestFormatFullBriefingHeader,
        TestFormatFullBriefingSections,
        TestFormatFullBriefingCalendarSection,
        TestFormatFullBriefingGmailSection,
        TestFormatFullBriefingNotionSection,
        TestErrorAnnotations,
        TestFallbackText,
        TestDeliveryToRequestingUser,
        TestBriefCommandUserIdThreading,
        TestBriefCommandDateParsing,
        TestFullPipelineIntegration,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            try:
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
