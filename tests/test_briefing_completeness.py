"""
AC 8 Tests: Incomplete briefings (missing attendees or meeting purpose)
are never sent.

Tests cover:
  validate_briefing_content()
  ───────────────────────────
  - Returns (True, []) for a complete briefing (attendees + purpose)
  - Returns (False, reasons) when external attendees list is empty
  - Returns (False, reasons) when meeting has no description AND generic title
  - Returns (False, reasons) when both attendees and purpose are missing
  - Returns (True, []) when description is present even with generic title
  - Returns (True, []) when title is specific even without description
  - Title-only purpose: accepts specific titles without description
  - Rejects all defined generic title keywords (Korean and English)
  - Whitespace-only description treated as missing
  - Data-source failures (gmail/notion unavailable) do NOT trigger the guard
  - External attendees count, not internal; internal-only → suppressed

  trigger_meeting_briefing() integration
  ───────────────────────────────────────
  - Incomplete briefing (no attendees) → returns False, never calls format
  - Incomplete briefing (no purpose)  → returns False, never calls format
  - Sends suppression notice via bot when briefing is suppressed
  - Complete briefing → validation passes, format + send called normally
  - bot=None always returns True (test/dry-run mode)

Run:
    python -m pytest tests/test_briefing_completeness.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

UTC = timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / factories
# ─────────────────────────────────────────────────────────────────────────────

def _utc_plus(minutes: float) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


def _make_attendee_profile(
    email: str = "ceo@startup.com",
    display_name: str = "CEO",
    is_internal: bool = False,
    company_domain: str = "startup.com",
):
    from src.briefing.context_aggregator import AttendeeProfile
    return AttendeeProfile(
        email=email,
        display_name=display_name,
        is_internal=is_internal,
        company_domain=company_domain,
    )


def _make_raw_content(
    meeting_title: str = "스타트업 킥오프 미팅",
    meeting_description: str = "신규 투자 검토를 위한 첫 번째 미팅입니다.",
    external_profiles=None,
    internal_profiles=None,
    gmail_available: bool = True,
    notion_available: bool = True,
    calendar_history_available: bool = True,
    errors=None,
):
    """Build a RawBriefingContent using only dataclasses (no real API clients)."""
    from src.briefing.context_aggregator import (
        RawBriefingContent,
        AttendeeProfile,
        AggregationError,
    )

    now_utc = datetime.now(UTC)
    start = now_utc + timedelta(minutes=12)
    end = start + timedelta(minutes=60)

    all_profiles: list[AttendeeProfile] = []
    for p in external_profiles or []:
        all_profiles.append(p)
    for p in internal_profiles or []:
        all_profiles.append(p)

    return RawBriefingContent(
        meeting_id="evt-ac8-test",
        meeting_title=meeting_title,
        meeting_start=start,
        meeting_end=end,
        meeting_description=meeting_description,
        attendee_profiles=all_profiles,
        gmail_available=gmail_available,
        notion_available=notion_available,
        calendar_history_available=calendar_history_available,
        errors=errors or [],
    )


def _external(email="ceo@ext.com", name="외부인"):
    return _make_attendee_profile(email=email, display_name=name, is_internal=False)


def _internal(email="me@kakaoventures.co.kr", name="내부인"):
    return _make_attendee_profile(email=email, display_name=name, is_internal=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_briefing_content
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateBriefingContent:

    def test_complete_briefing_passes(self):
        """Complete briefing with external attendees + description passes."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="시리즈A 투자 검토",
            meeting_description="A사 시리즈A 라운드 투자 의향 논의",
            external_profiles=[_external()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is True
        assert missing == []

    def test_missing_external_attendees_fails(self):
        """No external attendees → incomplete."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="시리즈A 투자 검토",
            meeting_description="투자 논의를 위한 미팅입니다.",
            external_profiles=[],  # no external attendees
            internal_profiles=[_internal()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False
        assert len(missing) >= 1
        # Should mention attendees
        combined = " ".join(missing).lower()
        assert "attendee" in combined or "참석자" in combined

    def test_no_attendees_at_all_fails(self):
        """Completely empty attendee list → incomplete."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="신규 파트너 미팅",
            meeting_description="파트너십 논의",
            external_profiles=[],
            internal_profiles=[],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False

    def test_internal_only_attendees_fails(self):
        """Only internal attendees for an external meeting briefing → incomplete."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="투자 심사 내부 리뷰",
            meeting_description="내부 투자 심사 미팅입니다.",
            external_profiles=[],
            internal_profiles=[_internal()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False

    def test_generic_title_no_description_fails(self):
        """Generic Korean title with no description → missing purpose."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="미팅",
            meeting_description="",
            external_profiles=[_external()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False
        combined = " ".join(missing).lower()
        assert "purpose" in combined or "목적" in combined or "설명" in combined

    def test_generic_english_title_no_description_fails(self):
        """Generic English title with no description → missing purpose."""
        from src.briefing.pipeline import validate_briefing_content
        for generic in ["Meeting", "Call", "Zoom", "Sync", "1:1", "Catch-up", "Chat"]:
            raw = _make_raw_content(
                meeting_title=generic,
                meeting_description="",
                external_profiles=[_external()],
            )
            is_complete, _ = validate_briefing_content(raw)
            assert is_complete is False, f"Expected failure for generic title: '{generic}'"

    def test_generic_korean_titles_all_fail(self):
        """All defined generic Korean title keywords should trigger missing purpose."""
        from src.briefing.pipeline import validate_briefing_content, _GENERIC_TITLES
        korean_generics = [t for t in _GENERIC_TITLES if any(ord(c) > 127 for c in t)]
        for generic in korean_generics:
            raw = _make_raw_content(
                meeting_title=generic,
                meeting_description="",
                external_profiles=[_external()],
            )
            is_complete, _ = validate_briefing_content(raw)
            assert is_complete is False, (
                f"Generic Korean title '{generic}' should fail without description"
            )

    def test_generic_title_with_description_passes(self):
        """Generic title is OK if description provides the purpose."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="미팅",
            meeting_description="스타트업 A의 제품 데모 및 투자 논의를 위한 미팅입니다.",
            external_profiles=[_external()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is True
        assert missing == []

    def test_specific_title_without_description_passes(self):
        """A specific, non-generic title counts as meeting purpose even without description."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="시리즈B 투자계약서 검토 미팅",
            meeting_description="",  # no description
            external_profiles=[_external()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is True
        assert missing == []

    def test_whitespace_description_treated_as_missing(self):
        """Whitespace-only description should not count as a valid purpose."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="회의",
            meeting_description="   \n\t  ",  # only whitespace
            external_profiles=[_external()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False

    def test_both_missing_returns_two_reasons(self):
        """When both attendees AND purpose are missing, both reasons returned."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="미팅",
            meeting_description="",
            external_profiles=[],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is False
        assert len(missing) == 2

    def test_data_source_failure_does_not_trigger_guard(self):
        """
        AC 8 guard must NOT fire when Gmail/Notion sources are unavailable.
        Source failures get '확인 불가' annotations but should not block the briefing.
        """
        from src.briefing.pipeline import validate_briefing_content
        from src.briefing.context_aggregator import AggregationError

        errors = [
            AggregationError(source="gmail", message="Gmail auth failed"),
            AggregationError(source="notion", message="Notion API timeout"),
        ]
        raw = _make_raw_content(
            meeting_title="투자 대상 기업 첫 미팅",
            meeting_description="초기 투자 검토를 위한 창업팀 미팅",
            external_profiles=[_external()],
            gmail_available=False,
            notion_available=False,
            errors=errors,
        )
        is_complete, missing = validate_briefing_content(raw)
        # Should still pass — data source failures are a separate concern
        assert is_complete is True
        assert missing == []

    def test_calendar_history_unavailable_does_not_trigger_guard(self):
        """Calendar history fetch failure must not block the briefing."""
        from src.briefing.pipeline import validate_briefing_content
        from src.briefing.context_aggregator import AggregationError

        errors = [AggregationError(source="calendar_history", message="timeout")]
        raw = _make_raw_content(
            meeting_title="VC 네트워킹 미팅",
            meeting_description="파트너 미팅 및 딜 소싱",
            external_profiles=[_external()],
            calendar_history_available=False,
            errors=errors,
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is True
        assert missing == []

    def test_multiple_external_attendees_complete(self):
        """Multiple external attendees → no attendee issue."""
        from src.briefing.pipeline import validate_briefing_content
        raw = _make_raw_content(
            meeting_title="컨소시엄 투자 논의",
            meeting_description="복수 VC 컨소시엄 투자 조건 협의",
            external_profiles=[
                _external("ceo@startupA.com", "A대표"),
                _external("cto@startupA.com", "A CTO"),
            ],
            internal_profiles=[_internal()],
        )
        is_complete, missing = validate_briefing_content(raw)
        assert is_complete is True
        assert missing == []

    def test_title_case_insensitive_generic_check(self):
        """Generic title check is case-insensitive."""
        from src.briefing.pipeline import validate_briefing_content
        for title in ["MEETING", "Meeting", "mEeTiNg", "ZOOM", "Zoom"]:
            raw = _make_raw_content(
                meeting_title=title,
                meeting_description="",
                external_profiles=[_external()],
            )
            is_complete, _ = validate_briefing_content(raw)
            assert is_complete is False, (
                f"Case-insensitive generic title check failed for '{title}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: trigger_meeting_briefing integration (AC 8 hook)
# ─────────────────────────────────────────────────────────────────────────────

def _make_meeting(
    summary: str = "시리즈A 투자 검토",
    description: str = "초기 투자 논의를 위한 외부 미팅",
    external_count: int = 1,
    internal_count: int = 1,
):
    """Build a minimal Meeting-like mock."""
    from unittest.mock import MagicMock
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    meeting = MagicMock()
    meeting.event_id = "evt-trigger-test"
    meeting.summary = summary
    meeting.description = description
    meeting.start = now + timedelta(minutes=12)
    meeting.end = now + timedelta(minutes=72)
    meeting.location = ""
    meeting.html_link = "https://cal.google.com/test"
    meeting.organizer_email = "organizer@kakaoventures.co.kr"
    meeting.starts_in_minutes = 12.0

    # Build external attendees
    ext_att = []
    for i in range(external_count):
        att = MagicMock()
        att.email = f"ceo{i}@startup{i}.com"
        att.display_name = f"외부인{i}"
        att.response_status = "accepted"
        att.is_internal = False
        ext_att.append(att)

    # Build internal attendees
    int_att = []
    for i in range(internal_count):
        att = MagicMock()
        att.email = f"invest{i}@kakaoventures.co.kr"
        att.display_name = f"내부인{i}"
        att.response_status = "accepted"
        att.is_internal = True
        int_att.append(att)

    meeting.attendees = ext_att + int_att
    meeting.external_attendees = ext_att
    return meeting


class TestTriggerMeetingBriefingCompleteness:

    def test_bot_none_always_returns_true(self):
        """bot=None is test/dry-run mode — always returns True regardless of content."""
        from src.briefing.pipeline import trigger_meeting_briefing
        meeting = _make_meeting(summary="미팅", description="", external_count=0)
        result = trigger_meeting_briefing(meeting, bot=None)
        assert result is True

    def test_complete_briefing_sends_message(self):
        """
        When briefing is complete (attendees + specific title/description),
        format_meeting_briefing is called and bot.send_message is invoked.
        """
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting(
            summary="스타트업 Demo Day 참관",
            description="신규 투자 후보 스타트업 데모 참관 및 미팅",
            external_count=2,
        )

        bot = MagicMock()
        bot.send_message.return_value = True

        # Patch the aggregation so we control the raw_content
        from src.briefing.context_aggregator import (
            RawBriefingContent,
            AttendeeProfile,
        )
        now = datetime.now(UTC)
        raw = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="스타트업 Demo Day 참관",
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="신규 투자 후보 스타트업 데모 참관 및 미팅",
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="스타트업 대표",
                    is_internal=False,
                    company_domain="startup.com",
                ),
            ],
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch("src.briefing.pipeline._aggregate_meeting_context", return_value=raw):
            result = trigger_meeting_briefing(meeting, bot=bot)

        assert result is True
        bot.send_message.assert_called_once()
        # Verify blocks were passed (not just text)
        call_kwargs = bot.send_message.call_args
        assert call_kwargs is not None

    def test_no_external_attendees_suppresses_briefing(self):
        """
        When aggregated raw_content has no external attendees, briefing must
        NOT be sent (send_message for the full briefing not called with blocks).
        """
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import (
            RawBriefingContent,
            AttendeeProfile,
        )

        meeting = _make_meeting(
            summary="시리즈A 투자 논의",
            description="투자 논의 미팅",
            external_count=1,  # meeting has externals but aggregated content won't
        )
        bot = MagicMock()
        bot.send_message.return_value = True

        now = datetime.now(UTC)
        raw_no_externals = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="시리즈A 투자 논의",
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="투자 논의 미팅",
            attendee_profiles=[
                AttendeeProfile(
                    email="internal@kakaoventures.co.kr",
                    display_name="내부인",
                    is_internal=True,  # only internal → no external attendees
                ),
            ],
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=raw_no_externals,
             ), patch(
                "src.briefing.meeting_briefing_formatter.format_meeting_briefing"
             ) as mock_format:
            result = trigger_meeting_briefing(meeting, bot=bot)

        assert result is False
        # format_meeting_briefing must NOT have been called with the real briefing
        mock_format.assert_not_called()

    def test_missing_purpose_suppresses_briefing(self):
        """
        Generic title + no description → briefing suppressed, format not called.
        """
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import (
            RawBriefingContent,
            AttendeeProfile,
        )

        meeting = _make_meeting(
            summary="미팅",
            description="",
            external_count=1,
        )
        bot = MagicMock()
        bot.send_message.return_value = True

        now = datetime.now(UTC)
        raw_no_purpose = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="미팅",           # generic title
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="",         # no description
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="대표",
                    is_internal=False,
                    company_domain="startup.com",
                ),
            ],
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=raw_no_purpose,
             ), patch(
                "src.briefing.meeting_briefing_formatter.format_meeting_briefing"
             ) as mock_format:
            result = trigger_meeting_briefing(meeting, bot=bot)

        assert result is False
        mock_format.assert_not_called()

    def test_suppression_notice_sent_to_user(self):
        """
        When briefing is suppressed, a suppression notice is sent via bot.send_message
        so the user knows a briefing was skipped.
        """
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent

        meeting = _make_meeting(summary="미팅", description="", external_count=0)
        bot = MagicMock()
        bot.send_message.return_value = True

        now = datetime.now(UTC)
        raw_empty = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="미팅",
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="",
            attendee_profiles=[],  # empty
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=raw_empty,
             ):
            result = trigger_meeting_briefing(meeting, bot=bot)

        assert result is False
        # A suppression notice should be sent
        bot.send_message.assert_called_once()
        notice_text = bot.send_message.call_args[0][0]
        assert "생략" in notice_text or "suppressed" in notice_text.lower() or "브리핑" in notice_text

    def test_suppression_notice_mentions_missing_fields(self):
        """Suppression notice text includes reason for suppression."""
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent

        meeting = _make_meeting(summary="Zoom", description="", external_count=0)
        bot = MagicMock()
        bot.send_message.return_value = True

        now = datetime.now(UTC)
        raw_empty = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="Zoom",
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="",
            attendee_profiles=[],
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=raw_empty,
             ):
            trigger_meeting_briefing(meeting, bot=bot)

        call_text = bot.send_message.call_args[0][0]
        # Should mention attendees and/or purpose missing
        assert "참석자" in call_text or "목적" in call_text or "attendee" in call_text.lower()

    def test_complete_with_specific_title_only_passes(self):
        """Specific title without description should pass validation and send briefing."""
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile

        meeting = _make_meeting(
            summary="스타트업XYZ 시리즈A 투자 계약 협상",
            description="",
            external_count=1,
        )
        bot = MagicMock()
        bot.send_message.return_value = True

        now = datetime.now(UTC)
        raw = RawBriefingContent(
            meeting_id="evt-trigger-test",
            meeting_title="스타트업XYZ 시리즈A 투자 계약 협상",
            meeting_start=now + timedelta(minutes=12),
            meeting_end=now + timedelta(minutes=72),
            meeting_description="",  # no description but title is specific
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startupxyz.com",
                    display_name="XYZ 대표",
                    is_internal=False,
                    company_domain="startupxyz.com",
                ),
            ],
        )

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=True), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=raw,
             ), patch(
                "src.briefing.meeting_briefing_formatter.format_meeting_briefing",
                return_value=("fallback text", [{"type": "header"}]),
             ) as mock_format:
            result = trigger_meeting_briefing(meeting, bot=bot)

        assert result is True
        mock_format.assert_called_once()
        # First positional arg must be the raw content
        assert mock_format.call_args[0][0] == raw
