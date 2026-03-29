"""
Tests for Sub-AC 2: Daily Briefing Formatter.

Verifies that format_daily_briefing correctly transforms calendar event data
(both Meeting dataclass and calendar_fetcher dict formats) into Slack Block Kit
messages with proper Korean formatting and accurate content.

Run with:
    python -m pytest tests/test_formatter.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone, timedelta
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.briefing.formatter import format_daily_briefing, _fmt_date_kr, _is_external_email

# ── Korean timezone offset ─────────────────────────────────────────────────────
from zoneinfo import ZoneInfo
KST = ZoneInfo("Asia/Seoul")

INTERNAL_DOMAIN = "kakaoventures.co.kr"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc(hour: int, minute: int = 0, day: int = 29, month: int = 3, year: int = 2026) -> datetime:
    """Build a UTC datetime with given components."""
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_meeting_dataclass(
    summary: str = "Test Meeting",
    start_hour: int = 10,
    end_hour: int = 11,
    external_emails: list[str] | None = None,
    internal_emails: list[str] | None = None,
    location: str = "",
    description: str = "",
    html_link: str = "https://calendar.google.com/event/123",
    all_day: bool = False,
):
    """Build a Meeting dataclass for testing."""
    from src.calendar.google_calendar import Meeting, Attendee

    start = _utc(start_hour)
    end = _utc(end_hour)
    attendees = []
    for email in (external_emails or []):
        attendees.append(Attendee(email=email, display_name=email.split("@")[0]))
    for email in (internal_emails or ["invest1@kakaoventures.co.kr"]):
        attendees.append(Attendee(email=email, display_name=email.split("@")[0]))

    return Meeting(
        event_id="evt-001",
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
        description=description,
        location=location,
        html_link=html_link,
        all_day=all_day,
    )


def _make_dict_event(
    title: str = "Dict Meeting",
    start_hour: int = 14,
    end_hour: int = 15,
    external_emails: list[str] | None = None,
    internal_emails: list[str] | None = None,
    location: str | None = None,
    video_link: str | None = None,
    conference_type: str | None = None,
    html_link: str = "https://calendar.google.com/event/456",
    all_day: bool = False,
) -> dict:
    """Build a calendar_fetcher-style dict event."""
    start = _utc(start_hour)
    end = _utc(end_hour)

    attendees = []
    for email in (external_emails or []):
        attendees.append({"email": email, "name": email.split("@")[0]})
    for email in (internal_emails or []):
        attendees.append({"email": email, "name": email.split("@")[0]})

    ev = {
        "title": title,
        "start": start,
        "end": end,
        "all_day": all_day,
        "attendees": attendees,
        "html_link": html_link,
    }
    if location is not None:
        ev["location"] = location
    if video_link is not None:
        ev["video_link"] = video_link
    if conference_type is not None:
        ev["conference_type"] = conference_type
    return ev


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Return type and structure
# ══════════════════════════════════════════════════════════════════════════════

class TestReturnTypeAndStructure:
    """format_daily_briefing returns (str, list[dict]) in all cases."""

    def test_returns_tuple_of_two(self):
        result = format_daily_briefing([])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fallback_text_is_str(self):
        text, _ = format_daily_briefing([])
        assert isinstance(text, str)
        assert len(text) > 0

    def test_blocks_is_list(self):
        _, blocks = format_daily_briefing([])
        assert isinstance(blocks, list)

    def test_blocks_not_empty(self):
        _, blocks = format_daily_briefing([])
        assert len(blocks) > 0

    def test_each_block_is_dict(self):
        _, blocks = format_daily_briefing([])
        for b in blocks:
            assert isinstance(b, dict)

    def test_each_block_has_type_field(self):
        _, blocks = format_daily_briefing([])
        for b in blocks:
            assert "type" in b

    def test_blocks_within_slack_limit(self):
        """Slack Block Kit limit is 50 blocks."""
        events = [
            _make_dict_event(title=f"Meeting {i}", start_hour=i % 12 + 1)
            for i in range(30)
        ]
        _, blocks = format_daily_briefing(events)
        assert len(blocks) <= 50


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Header block
# ══════════════════════════════════════════════════════════════════════════════

class TestHeaderBlock:
    """The first block is always a header with the date."""

    def test_first_block_is_header_type(self):
        _, blocks = format_daily_briefing([])
        assert blocks[0]["type"] == "header"

    def test_header_contains_calendar_emoji(self):
        _, blocks = format_daily_briefing([])
        header_text = blocks[0]["text"]["text"]
        assert "📅" in header_text

    def test_header_contains_briefing_word(self):
        _, blocks = format_daily_briefing([])
        header_text = blocks[0]["text"]["text"]
        assert "브리핑" in header_text

    def test_header_contains_target_date(self):
        """Header must reflect the target_date, not today's date."""
        target = date(2026, 5, 15)
        _, blocks = format_daily_briefing([], target_date=target)
        header_text = blocks[0]["text"]["text"]
        assert "2026" in header_text
        assert "5" in header_text
        assert "15" in header_text

    def test_header_contains_korean_month(self):
        target = date(2026, 3, 29)
        _, blocks = format_daily_briefing([], target_date=target)
        header_text = blocks[0]["text"]["text"]
        # Korean date format: 2026년 3월 29일
        assert "년" in header_text
        assert "월" in header_text
        assert "일" in header_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Empty schedule
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptySchedule:
    """When events=[], the message should gracefully say there are no events."""

    def test_empty_fallback_has_date(self):
        target = date(2026, 3, 29)
        text, _ = format_daily_briefing([], target_date=target)
        assert "2026" in text

    def test_empty_blocks_contain_no_events_text(self):
        _, blocks = format_daily_briefing([])
        full_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        # Should mention no events
        assert "없" in full_text  # "없습니다" (there are none)

    def test_empty_has_footer(self):
        """Footer context block should always be present."""
        _, blocks = format_daily_briefing([])
        context_blocks = [b for b in blocks if b["type"] == "context"]
        assert len(context_blocks) >= 1

    def test_summary_block_mentions_zero_meetings(self):
        """Summary should mention there are no scheduled events."""
        _, blocks = format_daily_briefing([])
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section" and isinstance(b.get("text"), dict)
        ]
        full = " ".join(section_texts)
        assert "없" in full


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Meeting dataclass input
# ══════════════════════════════════════════════════════════════════════════════

class TestMeetingDataclassInput:
    """Accepts Meeting dataclass from google_calendar.py."""

    def test_accepts_meeting_dataclass_no_error(self):
        meeting = _make_meeting_dataclass(external_emails=["ceo@startup.com"])
        text, blocks = format_daily_briefing([meeting])
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_meeting_title_appears_in_output(self):
        meeting = _make_meeting_dataclass(summary="투자 검토 미팅")
        text, blocks = format_daily_briefing([meeting])
        # Title should appear in fallback text
        assert "투자 검토 미팅" in text

    def test_external_meeting_marked_external(self):
        meeting = _make_meeting_dataclass(
            summary="외부 미팅",
            external_emails=["founder@startup.com"],
        )
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "외부" in block_texts

    def test_internal_meeting_marked_internal(self):
        meeting = _make_meeting_dataclass(
            summary="내부 팀 미팅",
            external_emails=[],
            internal_emails=["invest1@kakaoventures.co.kr", "invest2@kakaoventures.co.kr"],
        )
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "내부" in block_texts

    def test_meeting_time_appears_in_blocks(self):
        """Event block should show HH:MM time in KST."""
        # 10:00 UTC = 19:00 KST
        meeting = _make_meeting_dataclass(start_hour=10, end_hour=11)
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "19:00" in block_texts

    def test_html_link_in_block(self):
        meeting = _make_meeting_dataclass(html_link="https://cal.google.com/evt/abc123")
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "abc123" in block_texts

    def test_location_shown_when_present(self):
        meeting = _make_meeting_dataclass(location="카카오 판교 오피스 3층")
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "판교" in block_texts

    def test_zoom_link_in_description_extracted(self):
        meeting = _make_meeting_dataclass(
            description="Join: https://kakao.zoom.us/j/987654321 for the call"
        )
        _, blocks = format_daily_briefing([meeting])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "Zoom" in block_texts or "zoom" in block_texts.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Dict event input (calendar_fetcher format)
# ══════════════════════════════════════════════════════════════════════════════

class TestDictEventInput:
    """Accepts calendar_fetcher dict events."""

    def test_accepts_dict_event_no_error(self):
        ev = _make_dict_event(external_emails=["ceo@startup.com"])
        text, blocks = format_daily_briefing([ev])
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_dict_event_title_in_output(self):
        ev = _make_dict_event(title="투자 심의 위원회")
        text, blocks = format_daily_briefing([ev])
        assert "투자 심의 위원회" in text

    def test_dict_external_attendees_classified(self):
        ev = _make_dict_event(external_emails=["cto@external.com"])
        _, blocks = format_daily_briefing([ev])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "외부" in block_texts

    def test_dict_video_link_passed_through(self):
        ev = _make_dict_event(
            video_link="https://meet.google.com/abc-def-ghi",
            conference_type="Google Meet",
        )
        _, blocks = format_daily_briefing([ev])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "Google Meet" in block_texts or "abc-def-ghi" in block_texts

    def test_dict_all_day_event_classified_correctly(self):
        all_day_start = date(2026, 3, 29)
        ev = {
            "title": "종일 공휴일",
            "start": datetime(2026, 3, 29, 0, 0, tzinfo=KST),
            "end": datetime(2026, 3, 30, 0, 0, tzinfo=KST),
            "all_day": True,
            "attendees": [],
            "html_link": "",
        }
        text, blocks = format_daily_briefing([ev], target_date=date(2026, 3, 29))
        assert "종일 공휴일" in text


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Mixed input (Meeting + dict in same list)
# ══════════════════════════════════════════════════════════════════════════════

class TestMixedInput:
    """Accepts both Meeting dataclasses and dicts in the same events list."""

    def test_mixed_input_no_error(self):
        m = _make_meeting_dataclass(summary="Dataclass 미팅", start_hour=9)
        d = _make_dict_event(title="Dict 미팅", start_hour=14)
        text, blocks = format_daily_briefing([m, d])
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_both_titles_in_fallback(self):
        m = _make_meeting_dataclass(summary="Alpha 미팅", start_hour=9)
        d = _make_dict_event(title="Beta 미팅", start_hour=14)
        text, _ = format_daily_briefing([m, d])
        assert "Alpha 미팅" in text
        assert "Beta 미팅" in text

    def test_summary_count_is_correct(self):
        """Summary block should count both events."""
        m = _make_meeting_dataclass(summary="M1", start_hour=9, external_emails=["x@ext.com"])
        d = _make_dict_event(title="M2", start_hour=14)
        _, blocks = format_daily_briefing([m, d])
        # blocks[0]=header, blocks[1]=greeting, blocks[2]=summary
        section_blocks = [b for b in blocks if b.get("type") == "section"]
        summary_text = section_blocks[0]["text"]["text"]
        assert "2" in summary_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: All-day events
# ══════════════════════════════════════════════════════════════════════════════

class TestAllDayEvents:
    """All-day events should appear in a separate section at the top."""

    def test_all_day_section_present(self):
        ev = {
            "title": "Conference Day",
            "start": datetime(2026, 3, 29, 0, 0, tzinfo=KST),
            "end": datetime(2026, 3, 30, 0, 0, tzinfo=KST),
            "all_day": True,
            "attendees": [],
            "html_link": "",
        }
        _, blocks = format_daily_briefing([ev], target_date=date(2026, 3, 29))
        all_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "종일" in all_texts

    def test_all_day_event_title_in_blocks(self):
        ev = {
            "title": "연간 전략 회의",
            "start": datetime(2026, 3, 29, 0, 0, tzinfo=KST),
            "end": datetime(2026, 3, 30, 0, 0, tzinfo=KST),
            "all_day": True,
            "attendees": [],
            "html_link": "",
        }
        text, blocks = format_daily_briefing([ev], target_date=date(2026, 3, 29))
        all_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "연간 전략 회의" in all_texts

    def test_all_day_from_meeting_dataclass(self):
        meeting = _make_meeting_dataclass(summary="All Day Summit", all_day=True)
        _, blocks = format_daily_briefing([meeting])
        all_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "종일" in all_texts


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: Multiple events ordering / count
# ══════════════════════════════════════════════════════════════════════════════

class TestMultipleEvents:
    """Multiple events are all rendered; summary counts are accurate."""

    def test_three_events_all_in_fallback(self):
        events = [
            _make_dict_event(title="미팅 A", start_hour=9),
            _make_dict_event(title="미팅 B", start_hour=13),
            _make_dict_event(title="미팅 C", start_hour=16),
        ]
        text, _ = format_daily_briefing(events)
        assert "미팅 A" in text
        assert "미팅 B" in text
        assert "미팅 C" in text

    def test_summary_shows_total_count(self):
        events = [
            _make_dict_event(title="미팅 A", start_hour=9),
            _make_dict_event(title="미팅 B", start_hour=13),
        ]
        _, blocks = format_daily_briefing(events)
        # blocks[0]=header, blocks[1]=greeting context, blocks[2]=summary section
        section_blocks = [b for b in blocks if b.get("type") == "section"]
        summary_text = section_blocks[0]["text"]["text"]
        assert "2" in summary_text

    def test_external_count_in_summary(self):
        events = [
            _make_dict_event(title="외부1", start_hour=9, external_emails=["x@ext.com"]),
            _make_dict_event(title="내부1", start_hour=13, internal_emails=["y@kakaoventures.co.kr"]),
        ]
        _, blocks = format_daily_briefing(events)
        section_blocks = [b for b in blocks if b.get("type") == "section"]
        summary_text = section_blocks[0]["text"]["text"]
        assert "외부" in summary_text
        assert "1" in summary_text

    def test_dividers_between_events(self):
        events = [
            _make_dict_event(title="미팅 A", start_hour=9),
            _make_dict_event(title="미팅 B", start_hour=13),
        ]
        _, blocks = format_daily_briefing(events)
        divider_count = sum(1 for b in blocks if b["type"] == "divider")
        assert divider_count >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Part 9: target_date inference
# ══════════════════════════════════════════════════════════════════════════════

class TestTargetDateInference:
    """When target_date is None, it's inferred from the first event."""

    def test_date_inferred_from_dict_event(self):
        ev = _make_dict_event(start_hour=10)  # 2026-03-29 10:00 UTC
        text, blocks = format_daily_briefing([ev])
        header_text = blocks[0]["text"]["text"]
        assert "2026" in header_text

    def test_date_inferred_from_meeting_dataclass(self):
        m = _make_meeting_dataclass(start_hour=9)
        text, blocks = format_daily_briefing([m])
        header_text = blocks[0]["text"]["text"]
        assert "2026" in header_text

    def test_explicit_target_date_overrides_inference(self):
        m = _make_meeting_dataclass(start_hour=9)
        target = date(2026, 12, 25)
        _, blocks = format_daily_briefing([m], target_date=target)
        header_text = blocks[0]["text"]["text"]
        assert "12" in header_text
        assert "25" in header_text

    def test_empty_list_falls_back_to_today(self):
        from datetime import date as date_cls
        today = date_cls.today()
        _, blocks = format_daily_briefing([])
        header_text = blocks[0]["text"]["text"]
        assert str(today.year) in header_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 10: Fallback text structure
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackText:
    """Fallback text is used by non-Block-Kit Slack clients."""

    def test_fallback_contains_date(self):
        text, _ = format_daily_briefing([], target_date=date(2026, 3, 29))
        assert "2026" in text

    def test_fallback_contains_meeting_count(self):
        events = [_make_dict_event(title="미팅1")]
        text, _ = format_daily_briefing(events)
        assert "1" in text

    def test_fallback_contains_all_event_titles_for_small_list(self):
        events = [
            _make_dict_event(title="스타트업 피칭", start_hour=9),
            _make_dict_event(title="투자 심의", start_hour=14),
        ]
        text, _ = format_daily_briefing(events)
        assert "스타트업 피칭" in text
        assert "투자 심의" in text

    def test_fallback_contains_external_internal_labels(self):
        events = [
            _make_dict_event(title="외부 미팅", start_hour=9, external_emails=["x@ext.com"]),
        ]
        text, _ = format_daily_briefing(events)
        assert "[외부]" in text

    def test_fallback_has_briefing_emoji(self):
        text, _ = format_daily_briefing([])
        assert "📅" in text


# ══════════════════════════════════════════════════════════════════════════════
# Part 11: Footer block
# ══════════════════════════════════════════════════════════════════════════════

class TestFooterBlock:
    """A footer context block is always the last block."""

    def test_last_block_is_context(self):
        _, blocks = format_daily_briefing([])
        assert blocks[-1]["type"] == "context"

    def test_footer_mentions_assistant(self):
        _, blocks = format_daily_briefing([])
        footer_text = blocks[-1]["elements"][0]["text"]
        assert "Work Assistant" in footer_text or "봇" in footer_text or "assistant" in footer_text.lower()

    def test_footer_mentions_unconfirmed_annotation(self):
        """Footer should mention the '확인 불가' annotation policy."""
        _, blocks = format_daily_briefing([])
        footer_text = blocks[-1]["elements"][0]["text"]
        assert "확인 불가" in footer_text

    def test_last_block_is_context_with_events(self):
        events = [_make_dict_event()]
        _, blocks = format_daily_briefing(events)
        assert blocks[-1]["type"] == "context"


# ══════════════════════════════════════════════════════════════════════════════
# Part 12: Attendee display
# ══════════════════════════════════════════════════════════════════════════════

class TestAttendeeDisplay:
    """Attendees are shown with reasonable limits and labels."""

    def test_external_attendee_shown_in_block(self):
        m = _make_meeting_dataclass(external_emails=["founder@acme.com"])
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "founder" in block_texts

    def test_internal_attendee_shown_in_block(self):
        m = _make_meeting_dataclass(
            external_emails=[],
            internal_emails=["invest3@kakaoventures.co.kr"],
        )
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "invest3" in block_texts

    def test_many_external_attendees_capped(self):
        """More than 5 external attendees should be summarised."""
        m = _make_meeting_dataclass(
            external_emails=[f"person{i}@ext.com" for i in range(10)],
        )
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        # Should mention overflow ("외 N명")
        assert "외" in block_texts and "명" in block_texts

    def test_no_attendee_section_for_empty_attendees(self):
        """Meeting with no attendees should not crash."""
        m = _make_meeting_dataclass(external_emails=[], internal_emails=[])
        text, blocks = format_daily_briefing([m])
        assert isinstance(blocks, list)
        assert len(blocks) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Part 13: Title truncation
# ══════════════════════════════════════════════════════════════════════════════

class TestTitleTruncation:
    """Titles longer than 60 chars should be truncated with '…'."""

    def test_long_title_truncated(self):
        long_title = "A" * 80
        m = _make_meeting_dataclass(summary=long_title)
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "…" in block_texts

    def test_short_title_not_truncated(self):
        m = _make_meeting_dataclass(summary="짧은 제목")
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "짧은 제목" in block_texts
        # No truncation ellipsis
        assert "…" not in block_texts


# ══════════════════════════════════════════════════════════════════════════════
# Part 14: Video conference links
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoConferenceLinks:
    """Zoom, Teams, Meet links are extracted and shown."""

    def test_zoom_link_from_description(self):
        m = _make_meeting_dataclass(
            description="Zoom meeting: https://kakao.zoom.us/j/123456789"
        )
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "Zoom" in block_texts

    def test_google_meet_link_from_location(self):
        m = _make_meeting_dataclass(
            location="https://meet.google.com/xyz-uvw-abc"
        )
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "Google Meet" in block_texts or "xyz-uvw-abc" in block_texts

    def test_teams_link_from_description(self):
        m = _make_meeting_dataclass(
            description="Join: https://teams.microsoft.com/l/meetup-join/abc123"
        )
        _, blocks = format_daily_briefing([m])
        block_texts = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "Teams" in block_texts or "Microsoft Teams" in block_texts

    def test_no_video_link_no_crash(self):
        m = _make_meeting_dataclass(description="일반 회의입니다.", location="3층 회의실")
        text, blocks = format_daily_briefing([m])
        assert isinstance(blocks, list)


# ══════════════════════════════════════════════════════════════════════════════
# Part 15: Helper function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:
    """Unit tests for internal helper functions."""

    def test_fmt_date_kr_format(self):
        d = date(2026, 3, 29)
        result = _fmt_date_kr(d)
        assert "2026년" in result
        assert "3월" in result
        assert "29일" in result
        assert "일" in result  # Sunday in Korean

    def test_fmt_date_kr_weekday_monday(self):
        d = date(2026, 3, 30)  # Monday
        result = _fmt_date_kr(d)
        assert "월" in result

    def test_is_external_email_internal(self):
        assert _is_external_email("user@kakaoventures.co.kr") is False

    def test_is_external_email_external(self):
        assert _is_external_email("founder@startup.com") is True

    def test_is_external_email_empty(self):
        assert _is_external_email("") is False

    def test_is_external_email_case_insensitive(self):
        assert _is_external_email("USER@KakaoVentures.CO.KR") is False


# ══════════════════════════════════════════════════════════════════════════════
# Part 16: Error resilience
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorResilience:
    """Malformed events should be skipped, not crash the formatter."""

    def test_unknown_type_skipped_no_crash(self):
        """An unsupported event type is skipped with a warning."""
        good_event = _make_dict_event(title="정상 미팅")
        bad_event = object()  # unsupported type
        text, blocks = format_daily_briefing([bad_event, good_event])
        assert "정상 미팅" in text

    def test_dict_missing_title_uses_fallback(self):
        """Dict event with no 'title' key uses default title."""
        ev = {
            "start": _utc(10),
            "end": _utc(11),
            "all_day": False,
            "attendees": [],
            "html_link": "",
        }
        text, blocks = format_daily_briefing([ev])
        # Should not crash; default title used
        assert isinstance(blocks, list)

    def test_none_values_in_dict_handled(self):
        """Dict with None location/video_link should not crash."""
        ev = _make_dict_event(location=None, video_link=None)
        text, blocks = format_daily_briefing([ev])
        assert isinstance(blocks, list)


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
