"""
Tests for Sub-AC 1.2: Briefing Formatter — Greeting, Conflict & Back-to-Back Flagging.

Verifies the three new capabilities added to format_daily_briefing():
  1. Time-aware greeting prepended to every briefing
  2. Overlapping-event (conflict) detection and flagging
  3. Back-to-back meeting detection and flagging

Run with:
    python -m pytest tests/test_formatter_sub_ac_1_2.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.briefing.formatter import (
    format_daily_briefing,
    _greeting,
    _detect_conflicts_and_back_to_backs,
    _warnings_block,
    _NEvent,
    BACK_TO_BACK_THRESHOLD_MINUTES,
)

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc
TARGET_DATE = date(2026, 3, 29)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _kst(hour: int, minute: int = 0, day: int = 29) -> datetime:
    """KST-aware datetime on 2026-03-29 by default."""
    return datetime(2026, 3, day, hour, minute, tzinfo=KST)


def _utc(hour: int, minute: int = 0, day: int = 29) -> datetime:
    """UTC datetime on 2026-03-29 by default."""
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


def _make_event(
    title: str = "테스트 미팅",
    start: datetime | None = None,
    end: datetime | None = None,
    all_day: bool = False,
) -> dict:
    """Build a minimal calendar_fetcher-style dict event."""
    if start is None:
        start = _kst(10)
    if end is None:
        end = _kst(11)
    return {
        "title": title,
        "start": start,
        "end": end,
        "all_day": all_day,
        "attendees": [],
        "html_link": "",
    }


def _make_nevent(
    title: str = "이벤트",
    start: datetime | None = None,
    end: datetime | None = None,
    all_day: bool = False,
) -> _NEvent:
    """Build a minimal _NEvent directly for unit-testing detection logic."""
    if start is None:
        start = _kst(10)
    if end is None:
        end = _kst(11)
    duration = int((end - start).total_seconds() // 60)
    return _NEvent(
        title=title,
        start=start,
        end=end,
        all_day=all_day,
        is_external=False,
        external_attendees=[],
        internal_attendees=[],
        location=None,
        video_link=None,
        conference_type=None,
        html_link="",
        duration_minutes=duration,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Time-aware greeting
# ══════════════════════════════════════════════════════════════════════════════

class TestGreetingFunction:
    """Unit tests for the _greeting() helper."""

    def test_morning_greeting_5am(self):
        dt = datetime(2026, 3, 29, 5, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "아침" in result
        assert "☀️" in result

    def test_morning_greeting_9am(self):
        dt = datetime(2026, 3, 29, 9, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "아침" in result

    def test_morning_greeting_11_59(self):
        dt = datetime(2026, 3, 29, 11, 59, tzinfo=KST)
        result = _greeting(dt)
        assert "아침" in result

    def test_afternoon_greeting_noon(self):
        dt = datetime(2026, 3, 29, 12, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "안녕" in result
        assert "👋" in result

    def test_afternoon_greeting_3pm(self):
        dt = datetime(2026, 3, 29, 15, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "👋" in result or "안녕" in result

    def test_afternoon_greeting_5_59pm(self):
        dt = datetime(2026, 3, 29, 17, 59, tzinfo=KST)
        result = _greeting(dt)
        assert "👋" in result or "오후" in result

    def test_evening_greeting_6pm(self):
        dt = datetime(2026, 3, 29, 18, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "저녁" in result
        assert "🌙" in result

    def test_evening_greeting_9pm(self):
        dt = datetime(2026, 3, 29, 21, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "저녁" in result

    def test_late_night_greeting_midnight(self):
        dt = datetime(2026, 3, 29, 0, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "🌙" in result

    def test_late_night_greeting_4am(self):
        dt = datetime(2026, 3, 29, 4, 0, tzinfo=KST)
        result = _greeting(dt)
        assert "🌙" in result

    def test_returns_string(self):
        result = _greeting(_kst(9))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_arg_returns_string(self):
        # Default (uses datetime.now) — just check it doesn't raise
        result = _greeting()
        assert isinstance(result, str)
        assert len(result) > 0


class TestGreetingInBriefing:
    """Greeting is embedded in the briefing output."""

    def test_greeting_block_present_in_blocks(self):
        """A context block with the greeting text appears after the header."""
        now = _kst(9)  # morning
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        greeting_texts = [
            elem["text"]
            for b in context_blocks
            for elem in b.get("elements", [])
            if "아침" in elem.get("text", "") or "안녕" in elem.get("text", "") or "저녁" in elem.get("text", "")
        ]
        assert len(greeting_texts) >= 1, "Expected at least one greeting text in context blocks"

    def test_morning_greeting_in_briefing(self):
        now = _kst(8, 30)
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        all_context_text = " ".join(
            elem["text"]
            for b in blocks
            if b.get("type") == "context"
            for elem in b.get("elements", [])
        )
        assert "아침" in all_context_text

    def test_afternoon_greeting_in_briefing(self):
        now = _kst(14, 0)
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        all_context_text = " ".join(
            elem["text"]
            for b in blocks
            if b.get("type") == "context"
            for elem in b.get("elements", [])
        )
        assert "안녕" in all_context_text or "오후" in all_context_text

    def test_greeting_in_fallback_text(self):
        now = _kst(9)
        text, _ = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        assert "아침" in text

    def test_greeting_block_is_second_block(self):
        """The greeting context block should follow the header block immediately."""
        now = _kst(10)
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "context"
        # Verify it's the greeting (has greeting text, not footer)
        greeting_elem = blocks[1]["elements"][0]["text"]
        assert "Work Assistant" not in greeting_elem  # not the footer

    def test_greeting_present_with_events(self):
        """Greeting appears even when there are calendar events."""
        ev = _make_event("킥오프 미팅", start=_kst(10), end=_kst(11))
        now = _kst(9)
        _, blocks = format_daily_briefing([ev], target_date=TARGET_DATE, now_kst=now)
        all_context_text = " ".join(
            elem["text"]
            for b in blocks
            if b.get("type") == "context"
            for elem in b.get("elements", [])
        )
        assert "아침" in all_context_text


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Conflict detection — _detect_conflicts_and_back_to_backs()
# ══════════════════════════════════════════════════════════════════════════════

class TestConflictDetection:
    """Unit tests for the conflict detection algorithm."""

    def test_no_conflict_sequential_events(self):
        a = _make_nevent("A", start=_kst(9), end=_kst(10))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.has_conflict
        assert not b.has_conflict

    def test_no_conflict_single_event(self):
        a = _make_nevent("A", start=_kst(9), end=_kst(10))
        _detect_conflicts_and_back_to_backs([a])
        assert not a.has_conflict

    def test_no_conflict_empty_list(self):
        _detect_conflicts_and_back_to_backs([])  # should not raise

    def test_conflict_full_overlap(self):
        """Event B completely overlaps with event A."""
        a = _make_nevent("A", start=_kst(10), end=_kst(12))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.has_conflict
        assert b.has_conflict

    def test_conflict_partial_overlap(self):
        """Event B starts before A ends."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.has_conflict
        assert b.has_conflict

    def test_conflict_exact_same_time(self):
        """Two events at exactly the same time."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10), end=_kst(11))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.has_conflict
        assert b.has_conflict

    def test_no_conflict_adjacent_events(self):
        """Event A ends at 11:00, event B starts at 11:00 — no overlap."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.has_conflict
        assert not b.has_conflict

    def test_conflict_records_other_event_title(self):
        a = _make_nevent("미팅 A", start=_kst(10), end=_kst(11))
        b = _make_nevent("미팅 B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert "미팅 B" in a.conflict_with
        assert "미팅 A" in b.conflict_with

    def test_conflict_three_way(self):
        """Three events that all overlap with each other."""
        a = _make_nevent("A", start=_kst(10), end=_kst(13))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(12))
        c = _make_nevent("C", start=_kst(11), end=_kst(12, 30))
        _detect_conflicts_and_back_to_backs([a, b, c])
        assert a.has_conflict
        assert b.has_conflict
        assert c.has_conflict

    def test_conflict_utc_aware_events(self):
        """Detection works for UTC-aware datetimes (normalised to UTC internally)."""
        # 01:00 UTC = 10:00 KST; 01:30 UTC = 10:30 KST
        a = _make_nevent("A", start=_utc(1), end=_utc(2))
        b = _make_nevent("B", start=_utc(1, 30), end=_utc(2, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.has_conflict
        assert b.has_conflict

    def test_all_day_events_not_flagged_as_conflict(self):
        """All-day events should never be marked as conflicting."""
        a = _make_nevent("All Day", start=_kst(0), end=_kst(0, day=30), all_day=True)
        b = _make_nevent("A", start=_kst(10), end=_kst(11))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.has_conflict
        assert not b.has_conflict


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Back-to-back detection
# ══════════════════════════════════════════════════════════════════════════════

class TestBackToBackDetection:
    """Unit tests for back-to-back meeting detection."""

    def test_back_to_back_zero_gap(self):
        """Event A ends at 11:00, event B starts at 11:00 (0 min gap)."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.is_back_to_back
        assert b.is_back_to_back

    def test_back_to_back_within_threshold(self):
        """Gap of 5 min = threshold — should be flagged."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11, BACK_TO_BACK_THRESHOLD_MINUTES), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        assert a.is_back_to_back
        assert b.is_back_to_back

    def test_not_back_to_back_gap_exceeds_threshold(self):
        """Gap of threshold+1 minutes — NOT back-to-back."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11, BACK_TO_BACK_THRESHOLD_MINUTES + 1), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.is_back_to_back
        assert not b.is_back_to_back

    def test_not_back_to_back_30_min_gap(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11, 30), end=_kst(12, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.is_back_to_back
        assert not b.is_back_to_back

    def test_back_to_back_records_adjacent_title(self):
        a = _make_nevent("미팅 A", start=_kst(10), end=_kst(11))
        b = _make_nevent("미팅 B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        assert "미팅 B" in a.back_to_back_with
        assert "미팅 A" in b.back_to_back_with

    def test_back_to_back_not_flagged_for_overlapping_events(self):
        """Overlapping events should be conflicts, not back-to-back."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11, 30))
        b = _make_nevent("B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        # a and b overlap: a.end > b.start → conflict, not back-to-back
        assert a.has_conflict
        # back-to-back only for non-negative gaps
        assert not a.is_back_to_back

    def test_three_consecutive_meetings(self):
        """A → B → C all back-to-back."""
        a = _make_nevent("A", start=_kst(9), end=_kst(10))
        b = _make_nevent("B", start=_kst(10), end=_kst(11))
        c = _make_nevent("C", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b, c])
        assert a.is_back_to_back
        assert b.is_back_to_back
        assert c.is_back_to_back

    def test_back_to_back_custom_threshold(self):
        """Custom threshold=10 should flag a 10-min gap."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11, 10), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b], threshold_minutes=10)
        assert a.is_back_to_back
        assert b.is_back_to_back

    def test_back_to_back_single_event_no_flag(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        _detect_conflicts_and_back_to_backs([a])
        assert not a.is_back_to_back

    def test_all_day_not_flagged_back_to_back(self):
        a = _make_nevent("All Day", all_day=True)
        b = _make_nevent("Timed", start=_kst(10), end=_kst(11))
        _detect_conflicts_and_back_to_backs([a, b])
        assert not a.is_back_to_back
        assert not b.is_back_to_back


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Warnings block builder
# ══════════════════════════════════════════════════════════════════════════════

class TestWarningsBlock:
    """Unit tests for _warnings_block()."""

    def test_returns_none_when_no_flags(self):
        events = [
            _make_nevent("A", start=_kst(10), end=_kst(11)),
            _make_nevent("B", start=_kst(12), end=_kst(13)),
        ]
        _detect_conflicts_and_back_to_backs(events)
        result = _warnings_block(events)
        assert result is None

    def test_returns_dict_for_conflict(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        assert result is not None
        assert result["type"] == "section"

    def test_conflict_warning_text_contains_event_titles(self):
        a = _make_nevent("알파 미팅", start=_kst(10), end=_kst(11))
        b = _make_nevent("베타 미팅", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        text = result["text"]["text"]
        assert "알파 미팅" in text or "베타 미팅" in text

    def test_conflict_warning_contains_conflict_emoji(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        assert "⚠️" in result["text"]["text"]

    def test_back_to_back_warning_contains_bell_emoji(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        assert "🔔" in result["text"]["text"]

    def test_conflict_pair_reported_only_once(self):
        """A→B conflict should appear once, not twice (once for A, once for B)."""
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        text = result["text"]["text"]
        # Count occurrences of the conflict warning keyword
        conflict_count = text.count("시간 충돌")
        assert conflict_count == 1

    def test_back_to_back_pair_reported_only_once(self):
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(11), end=_kst(12))
        _detect_conflicts_and_back_to_backs([a, b])
        result = _warnings_block([a, b])
        text = result["text"]["text"]
        count = text.count("연속 미팅")
        assert count == 1

    def test_both_conflict_and_b2b_in_same_block(self):
        """When different events have both types of warnings, both appear."""
        # Events A and B overlap (conflict)
        a = _make_nevent("A", start=_kst(10), end=_kst(11))
        b = _make_nevent("B", start=_kst(10, 30), end=_kst(11, 30))
        # Event C starts right after B (back-to-back)
        c = _make_nevent("C", start=_kst(11, 30), end=_kst(12, 30))
        events = [a, b, c]
        _detect_conflicts_and_back_to_backs(events)
        result = _warnings_block(events)
        assert result is not None
        text = result["text"]["text"]
        assert "⚠️" in text
        assert "🔔" in text

    def test_empty_list_returns_none(self):
        assert _warnings_block([]) is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: End-to-end integration — format_daily_briefing() with new features
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatDailyBriefingConflicts:
    """Integration tests: conflict/back-to-back flags appear in full briefing output."""

    def test_conflict_warning_block_present(self):
        """Overlapping events produce a warning section block."""
        ev_a = _make_event("미팅 A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("미팅 B", start=_kst(10, 30), end=_kst(11, 30))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        section_texts = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "⚠️" in section_texts or "충돌" in section_texts

    def test_no_warning_block_for_non_overlapping(self):
        """No warning section when there are no conflicts or back-to-backs."""
        ev_a = _make_event("미팅 A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("미팅 B", start=_kst(13), end=_kst(14))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        section_texts = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        # "충돌" and "연속" should not appear in any section
        assert "충돌" not in section_texts
        assert "연속 미팅" not in section_texts

    def test_back_to_back_warning_in_briefing(self):
        """Consecutive meetings produce a 🔔 연속 미팅 warning."""
        ev_a = _make_event("스탠드업", start=_kst(9), end=_kst(10))
        ev_b = _make_event("킥오프", start=_kst(10), end=_kst(11))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(8))
        section_texts = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "🔔" in section_texts or "연속" in section_texts

    def test_inline_conflict_flag_on_event_block(self):
        """The ⚠️ 충돌 inline flag appears in the event block itself."""
        ev_a = _make_event("미팅 A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("미팅 B", start=_kst(10, 30), end=_kst(11, 30))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        all_section_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        # Inline flag on the event line
        assert "충돌" in all_section_text

    def test_inline_back_to_back_flag_on_event_block(self):
        """The 🔔 연속 inline flag appears in the event block itself."""
        ev_a = _make_event("A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("B", start=_kst(11), end=_kst(12))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        all_section_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "연속" in all_section_text

    def test_conflict_in_fallback_text(self):
        ev_a = _make_event("A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("B", start=_kst(10, 30), end=_kst(11, 30))
        text, _ = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        assert "충돌" in text

    def test_back_to_back_in_fallback_text(self):
        ev_a = _make_event("A", start=_kst(9), end=_kst(10))
        ev_b = _make_event("B", start=_kst(10), end=_kst(11))
        text, _ = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        assert "연속" in text

    def test_50_block_limit_still_respected_with_warnings(self):
        """Warnings + many events must still stay within 50 blocks."""
        # Create many overlapping events
        events = [
            _make_event(f"미팅 {i}", start=_kst(9), end=_kst(10))
            for i in range(25)
        ]
        _, blocks = format_daily_briefing(events, target_date=TARGET_DATE, now_kst=_kst(8))
        assert len(blocks) <= 50

    def test_no_events_still_has_greeting_and_no_warnings(self):
        """Empty day: greeting present, no warning block."""
        now = _kst(10)
        text, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=now)
        assert "아침" in text
        # No warnings section (no conflicting events)
        section_texts = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "충돌" not in section_texts
        assert "연속 미팅" not in section_texts

    def test_single_event_no_warnings(self):
        """A single event can't conflict or be back-to-back with anything."""
        ev = _make_event("혼자 미팅", start=_kst(10), end=_kst(11))
        text, blocks = format_daily_briefing([ev], target_date=TARGET_DATE, now_kst=_kst(9))
        section_texts = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "충돌" not in section_texts
        assert "연속 미팅" not in section_texts


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Block structure validation
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockStructureWithNewFeatures:
    """Validate Slack Block Kit structure after adding greeting + warnings."""

    def test_blocks_have_valid_types(self):
        ev_a = _make_event("A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("B", start=_kst(10, 30), end=_kst(11, 30))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        valid_types = {"header", "section", "divider", "context", "actions", "image"}
        for block in blocks:
            assert block.get("type") in valid_types

    def test_header_still_first_block(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=_kst(9))
        assert blocks[0]["type"] == "header"

    def test_greeting_context_is_second_block(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=_kst(9))
        assert blocks[1]["type"] == "context"

    def test_footer_always_last(self):
        ev_a = _make_event("A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("B", start=_kst(10, 30), end=_kst(11, 30))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        assert blocks[-1]["type"] == "context"
        footer_text = blocks[-1]["elements"][0]["text"]
        assert "확인 불가" in footer_text

    def test_warnings_section_has_mrkdwn_text(self):
        ev_a = _make_event("A", start=_kst(10), end=_kst(11))
        ev_b = _make_event("B", start=_kst(10, 30), end=_kst(11, 30))
        _, blocks = format_daily_briefing([ev_a, ev_b], target_date=TARGET_DATE, now_kst=_kst(9))
        # Find the warnings block — a section with "충돌" in it
        warning_blocks = [
            b for b in blocks
            if b.get("type") == "section"
            and "충돌" in b.get("text", {}).get("text", "")
        ]
        assert len(warning_blocks) >= 1
        wb = warning_blocks[0]
        assert wb["text"]["type"] == "mrkdwn"

    def test_greeting_context_has_mrkdwn_element(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE, now_kst=_kst(9))
        greeting_block = blocks[1]
        assert "elements" in greeting_block
        assert greeting_block["elements"][0]["type"] == "mrkdwn"


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
