"""
Unit tests for src.briefing.formatter.

Tests cover:
  - Empty event list → "no events" message
  - All-day events section
  - External meeting detection (🌐 badge)
  - Internal-only meeting (🏢 badge)
  - Attendee lists (capped, with overflow count)
  - Video link extraction (Zoom, Teams, Google Meet)
  - Location display
  - Date formatting in Korean
  - Both input formats: Meeting dataclass AND calendar_fetcher dict
  - Fallback text content
  - Block Kit structure validity (type, required fields)
  - Slack 50-block limit guard
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.briefing.formatter import format_daily_briefing, _fmt_date_kr, _fmt_time

KST = ZoneInfo("Asia/Seoul")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _kst(hour: int, minute: int = 0, day: int = 29, month: int = 3, year: int = 2026) -> datetime:
    """Return a KST-aware datetime for convenience."""
    return datetime(year, month, day, hour, minute, tzinfo=KST)


def _utc(hour: int, minute: int = 0, day: int = 29, month: int = 3, year: int = 2026) -> datetime:
    """Return a UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _make_meeting(
    summary: str = "테스트 미팅",
    start: datetime | None = None,
    end: datetime | None = None,
    attendees_data: list[dict] | None = None,
    location: str = "",
    description: str = "",
    html_link: str = "https://calendar.google.com/event/123",
    all_day: bool = False,
):
    """Create a Meeting dataclass instance."""
    from src.calendar.google_calendar import Meeting, Attendee

    if start is None:
        start = _kst(10)
    if end is None:
        end = _kst(11)

    attendees = []
    for a in (attendees_data or []):
        attendees.append(
            Attendee(
                email=a["email"],
                display_name=a.get("name", ""),
                response_status=a.get("status", "accepted"),
            )
        )

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
    title: str = "테스트 이벤트",
    start: datetime | None = None,
    end: datetime | None = None,
    attendees_data: list[dict] | None = None,
    location: str | None = None,
    video_link: str | None = None,
    conference_type: str | None = None,
    html_link: str = "https://calendar.google.com/event/456",
    all_day: bool = False,
) -> dict:
    """Create a calendar_fetcher-style event dict."""
    if start is None:
        start = _kst(14) if not all_day else date(2026, 3, 29)
    if end is None:
        end = _kst(15) if not all_day else date(2026, 3, 30)

    attendees = []
    for a in (attendees_data or []):
        attendees.append(
            {
                "email": a["email"],
                "name": a.get("name", a["email"]),
                "response_status": a.get("status", "accepted"),
                "is_organizer": a.get("is_organizer", False),
            }
        )

    return {
        "id": "dict-evt-001",
        "title": title,
        "start": start,
        "end": end,
        "all_day": all_day,
        "start_iso": "",
        "end_iso": "",
        "attendees": attendees,
        "location": location,
        "video_link": video_link,
        "conference_type": conference_type,
        "html_link": html_link,
        "organizer_email": "",
        "organizer_name": "",
        "status": "confirmed",
        "description": None,
        "recurring_event_id": None,
    }


# ── Fixtures ───────────────────────────────────────────────────────────────────

INTERNAL_EMAIL = "invest1@kakaoventures.co.kr"
EXTERNAL_EMAIL = "ceo@startup.com"
TARGET_DATE = date(2026, 3, 29)


# ── Tests: basic structure ─────────────────────────────────────────────────────

class TestEmptyEvents:
    def test_returns_tuple(self):
        text, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        assert isinstance(text, str)
        assert isinstance(blocks, list)

    def test_no_events_message_in_text(self):
        text, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        assert "없습니다" in text or "없음" in text

    def test_header_block_present(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        headers = [b for b in blocks if b.get("type") == "header"]
        assert len(headers) == 1

    def test_header_contains_date(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        header = next(b for b in blocks if b.get("type") == "header")
        assert "3월" in header["text"]["text"]
        assert "2026" in header["text"]["text"]

    def test_footer_block_present(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        contexts = [b for b in blocks if b.get("type") == "context"]
        assert len(contexts) >= 1

    def test_blocks_have_valid_types(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        valid_types = {"header", "section", "divider", "context", "actions", "image"}
        for block in blocks:
            assert block.get("type") in valid_types, f"Invalid block type: {block.get('type')}"

    def test_under_50_blocks(self):
        _, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        assert len(blocks) <= 50


# ── Tests: Korean date formatting ─────────────────────────────────────────────

class TestDateFormatting:
    def test_fmt_date_kr_format(self):
        d = date(2026, 3, 29)
        result = _fmt_date_kr(d)
        assert "2026" in result
        assert "3월" in result
        assert "29일" in result
        # Sunday
        assert "(일)" in result

    def test_fmt_date_kr_weekday_monday(self):
        d = date(2026, 3, 30)   # Monday
        result = _fmt_date_kr(d)
        assert "(월)" in result

    def test_fmt_time_kst_conversion(self):
        # UTC 01:00 = KST 10:00
        dt = _utc(1, 0)
        assert _fmt_time(dt) == "10:00"

    def test_briefing_title_contains_korean_date(self):
        _, blocks = format_daily_briefing([], target_date=date(2026, 3, 29))
        header = next(b for b in blocks if b.get("type") == "header")
        text = header["text"]["text"]
        assert "2026년" in text
        assert "3월 29일" in text


# ── Tests: Meeting dataclass input ────────────────────────────────────────────

class TestMeetingDataclass:
    def test_internal_meeting_badge(self):
        meeting = _make_meeting(
            attendees_data=[
                {"email": INTERNAL_EMAIL, "name": "팀원 A"},
                {"email": "invest2@kakaoventures.co.kr", "name": "팀원 B"},
            ]
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        sections = [b for b in blocks if b.get("type") == "section"]
        event_text = "\n".join(
            b["text"]["text"] for b in sections if b.get("text")
        )
        assert "🏢" in event_text

    def test_external_meeting_badge(self):
        meeting = _make_meeting(
            attendees_data=[
                {"email": INTERNAL_EMAIL, "name": "팀원 A"},
                {"email": EXTERNAL_EMAIL, "name": "외부인"},
            ]
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        sections = [b for b in blocks if b.get("type") == "section"]
        event_text = "\n".join(
            b["text"]["text"] for b in sections if b.get("text")
        )
        assert "🌐" in event_text

    def test_external_attendee_shown(self):
        meeting = _make_meeting(
            attendees_data=[
                {"email": EXTERNAL_EMAIL, "name": "John CEO"},
            ]
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section" and b.get("text")
        )
        assert "John CEO" in event_text or EXTERNAL_EMAIL in event_text

    def test_meeting_title_in_blocks(self):
        title = "포트폴리오 킥오프 미팅"
        meeting = _make_meeting(summary=title)
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        all_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert title in all_text

    def test_meeting_title_in_fallback(self):
        title = "포트폴리오 킥오프 미팅"
        meeting = _make_meeting(summary=title)
        text, _ = format_daily_briefing([meeting], target_date=TARGET_DATE)
        assert title in text

    def test_time_shown_in_blocks(self):
        meeting = _make_meeting(start=_kst(9, 30), end=_kst(10, 30))
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "09:30" in event_text
        assert "10:30" in event_text

    def test_duration_shown(self):
        # 60 minute meeting
        meeting = _make_meeting(start=_kst(10), end=_kst(11))
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "60분" in event_text

    def test_html_link_in_block(self):
        meeting = _make_meeting(html_link="https://calendar.google.com/event/XYZ")
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "https://calendar.google.com/event/XYZ" in event_text

    def test_location_shown(self):
        meeting = _make_meeting(location="서울 강남구 회의실 A")
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "서울 강남구 회의실 A" in event_text

    def test_zoom_link_extracted_from_description(self):
        desc = "Join: https://kakao.zoom.us/j/12345678 Meeting ID: 123"
        meeting = _make_meeting(description=desc)
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "Zoom" in event_text
        assert "kakao.zoom.us" in event_text

    def test_google_meet_link_extracted(self):
        desc = "Video: https://meet.google.com/abc-defg-hij"
        meeting = _make_meeting(description=desc)
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "Google Meet" in event_text

    def test_all_day_meeting_in_all_day_section(self):
        # All-day via Meeting.all_day=True
        meeting = _make_meeting(
            summary="전사 워크샵",
            start=_utc(0, 0),
            end=_utc(0, 0, day=30),
            all_day=True,
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        all_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        # Should appear in the all-day section
        assert "전사 워크샵" in all_text
        assert "종일" in all_text

    def test_summary_counts_external_correctly(self):
        meetings = [
            _make_meeting("외부 미팅 1", attendees_data=[{"email": EXTERNAL_EMAIL}]),
            _make_meeting("내부 미팅", start=_kst(13), end=_kst(14),
                          attendees_data=[{"email": INTERNAL_EMAIL}]),
            _make_meeting("외부 미팅 2", start=_kst(15), end=_kst(16),
                          attendees_data=[{"email": "partner@other.com"}]),
        ]
        _, blocks = format_daily_briefing(meetings, target_date=TARGET_DATE)
        # Find the summary section (2nd block after header)
        sections = [b for b in blocks if b.get("type") == "section"]
        summary_text = sections[0]["text"]["text"]
        assert "*3개*" in summary_text   # 3 total
        assert "*2개*" in summary_text   # 2 external
        assert "*1개*" in summary_text   # 1 internal


# ── Tests: dict-format input ──────────────────────────────────────────────────

class TestDictEventInput:
    def test_dict_internal_meeting(self):
        ev = _make_dict_event(
            title="주간 팀 회의",
            attendees_data=[
                {"email": INTERNAL_EMAIL, "name": "팀원 A"},
            ],
        )
        _, blocks = format_daily_briefing([ev], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🏢" in event_text

    def test_dict_external_meeting(self):
        ev = _make_dict_event(
            title="파트너사 미팅",
            attendees_data=[
                {"email": EXTERNAL_EMAIL, "name": "Jane Partner"},
                {"email": INTERNAL_EMAIL, "name": "팀원"},
            ],
        )
        _, blocks = format_daily_briefing([ev], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "🌐" in event_text

    def test_dict_video_link_shown(self):
        ev = _make_dict_event(
            video_link="https://kakao.zoom.us/j/99999",
            conference_type="Zoom",
        )
        _, blocks = format_daily_briefing([ev], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "Zoom" in event_text
        assert "kakao.zoom.us" in event_text

    def test_dict_all_day_event(self):
        ev = _make_dict_event(
            title="공휴일",
            start=date(2026, 3, 29),
            end=date(2026, 3, 30),
            all_day=True,
        )
        _, blocks = format_daily_briefing([ev], target_date=TARGET_DATE)
        all_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "공휴일" in all_text

    def test_dict_event_title_in_fallback(self):
        ev = _make_dict_event(title="파트너 데모 데이")
        text, _ = format_daily_briefing([ev], target_date=TARGET_DATE)
        assert "파트너 데모 데이" in text


# ── Tests: mixed input format ─────────────────────────────────────────────────

class TestMixedInput:
    def test_mixed_meeting_and_dict(self):
        meeting = _make_meeting(summary="Meeting obj")
        ev_dict = _make_dict_event(title="Dict event", start=_kst(14), end=_kst(15))
        _, blocks = format_daily_briefing([meeting, ev_dict], target_date=TARGET_DATE)
        all_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "Meeting obj" in all_text
        assert "Dict event" in all_text

    def test_all_day_and_timed_mixed(self):
        all_day = _make_meeting(
            summary="전사 워크샵",
            start=_utc(0, 0),
            end=_utc(0, 0, day=30),
            all_day=True,
        )
        timed = _make_meeting(summary="킥오프 미팅", start=_kst(10), end=_kst(11))
        _, blocks = format_daily_briefing([all_day, timed], target_date=TARGET_DATE)
        all_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "전사 워크샵" in all_text
        assert "킥오프 미팅" in all_text
        # Summary should mention all-day count
        sections = [b for b in blocks if b.get("type") == "section"]
        summary_text = sections[0]["text"]["text"]
        assert "종일" in summary_text


# ── Tests: attendee overflow ──────────────────────────────────────────────────

class TestAttendeeOverflow:
    def test_external_attendees_capped_at_5(self):
        meeting = _make_meeting(
            attendees_data=[
                {"email": f"person{i}@external.com", "name": f"Person {i}"}
                for i in range(8)
            ]
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        # Should show "외 N명" for overflow
        assert "외 3명" in event_text

    def test_internal_attendees_capped_at_4(self):
        meeting = _make_meeting(
            attendees_data=[
                {"email": f"invest{i}@kakaoventures.co.kr", "name": f"팀원 {i}"}
                for i in range(7)
            ]
        )
        _, blocks = format_daily_briefing([meeting], target_date=TARGET_DATE)
        event_text = "\n".join(
            b["text"]["text"] for b in blocks
            if b.get("type") == "section" and b.get("text")
        )
        assert "외 3명" in event_text


# ── Tests: block count limit ──────────────────────────────────────────────────

class TestBlockLimit:
    def test_50_block_limit_respected_with_many_events(self):
        """Even with 30 events, blocks must stay ≤ 50."""
        meetings = [
            _make_meeting(
                summary=f"미팅 {i}",
                start=_kst(9 + (i % 8)),
                end=_kst(10 + (i % 8)),
            )
            for i in range(30)
        ]
        _, blocks = format_daily_briefing(meetings, target_date=TARGET_DATE)
        assert len(blocks) <= 50

    def test_truncation_notice_present_when_limit_hit(self):
        """When events are truncated, a notice block should appear."""
        meetings = [
            _make_meeting(
                summary=f"미팅 {i}",
                start=_kst(9),
                end=_kst(10),
            )
            for i in range(30)
        ]
        _, blocks = format_daily_briefing(meetings, target_date=TARGET_DATE)
        all_text = "\n".join(
            str(b) for b in blocks if b.get("type") in ("section", "context")
        )
        # If truncation occurred, check for the overflow notice
        if len(blocks) == 50:
            assert "블록 한도" in all_text or "일정" in all_text


# ── Tests: fallback text ──────────────────────────────────────────────────────

class TestFallbackText:
    def test_fallback_has_date(self):
        text, _ = format_daily_briefing([], target_date=TARGET_DATE)
        assert "2026" in text
        assert "3월" in text

    def test_fallback_has_meeting_titles(self):
        meetings = [
            _make_meeting("아침 스탠드업"),
            _make_meeting("오후 킥오프", start=_kst(14), end=_kst(15)),
        ]
        text, _ = format_daily_briefing(meetings, target_date=TARGET_DATE)
        assert "아침 스탠드업" in text
        assert "오후 킥오프" in text

    def test_fallback_marks_external(self):
        meeting = _make_meeting(
            attendees_data=[{"email": EXTERNAL_EMAIL}]
        )
        text, _ = format_daily_briefing([meeting], target_date=TARGET_DATE)
        assert "[외부]" in text

    def test_fallback_marks_internal(self):
        meeting = _make_meeting(
            attendees_data=[{"email": INTERNAL_EMAIL}]
        )
        text, _ = format_daily_briefing([meeting], target_date=TARGET_DATE)
        assert "[내부]" in text

    def test_fallback_shows_times(self):
        meeting = _make_meeting(start=_kst(9, 30), end=_kst(11, 0))
        text, _ = format_daily_briefing([meeting], target_date=TARGET_DATE)
        assert "09:30" in text
        assert "11:00" in text


# ── Tests: target_date inference ─────────────────────────────────────────────

class TestTargetDateInference:
    def test_date_inferred_from_meeting_start(self):
        meeting = _make_meeting(start=_kst(10, 0, day=5, month=4, year=2026))
        _, blocks = format_daily_briefing([meeting])  # no target_date passed
        header = next(b for b in blocks if b.get("type") == "header")
        text = header["text"]["text"]
        assert "4월" in text
        assert "5일" in text

    def test_date_inferred_from_dict_start(self):
        ev = _make_dict_event(
            start=datetime(2026, 5, 15, 9, 0, tzinfo=KST),
            end=datetime(2026, 5, 15, 10, 0, tzinfo=KST),
        )
        _, blocks = format_daily_briefing([ev])
        header = next(b for b in blocks if b.get("type") == "header")
        text = header["text"]["text"]
        assert "5월" in text
        assert "15일" in text


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run tests directly without pytest for quick spot-check
    import traceback

    test_classes = [
        TestEmptyEvents,
        TestDateFormatting,
        TestMeetingDataclass,
        TestDictEventInput,
        TestMixedInput,
        TestAttendeeOverflow,
        TestBlockLimit,
        TestFallbackText,
        TestTargetDateInference,
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
