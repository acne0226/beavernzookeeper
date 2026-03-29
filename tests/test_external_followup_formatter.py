"""
Tests for AC 7 Sub-AC 4: External Follow-up Briefing Formatter.

Covers format_external_followup_briefing() with all combinations of:
  - All sources available / happy-path
  - Notion source unavailable
  - Gmail source unavailable
  - Slack history unavailable
  - Slack history not fetched (None)
  - Slack history empty (no matching messages)
  - Block count cap (≤ 50 Slack hard limit)
  - Attendee relationship history rendering
  - Fallback plain text accuracy

All tests are offline (no real API calls).

Run:
    python -m pytest tests/test_external_followup_formatter.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Fixtures / builder helpers ───────────────────────────────────────────────

def _make_raw_content(
    meeting_title: str = "AcmeCorp 후속 미팅",
    has_external: bool = True,
    has_internal: bool = True,
    past_meetings: int = 3,
    gmail_available: bool = True,
    gmail_threads: int = 2,
    notion_available: bool = True,
    notion_records: int = 2,
    slack_history_available: bool = True,
    slack_messages: int = 3,
    slack_fetched: bool = True,
    calendar_history_available: bool = True,
):
    """
    Build a RawBriefingContent mock with the given feature flags.

    Uses actual dataclasses where convenient and MagicMock for heavy
    dependencies (Gmail threads, Notion records, Slack messages) to keep
    tests fast and offline.
    """
    from src.briefing.context_aggregator import (
        RawBriefingContent,
        AttendeeProfile,
        AggregationError,
    )

    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=30)
    end = start + timedelta(minutes=60)

    # Build attendee profiles
    attendees: list[AttendeeProfile] = []
    if has_external:
        ext_profile = AttendeeProfile(
            email="ceo@acmecorp.com",
            display_name="Alice Smith",
            is_internal=False,
            company_domain="acmecorp.com",
            past_meeting_count=past_meetings,
            last_met_date=now - timedelta(days=14) if past_meetings > 0 else None,
            past_meeting_titles=(
                ["1차 미팅 — 사업 개요", "2차 미팅 — 투자 조건 논의"]
                if past_meetings > 0
                else []
            ),
        )
        attendees.append(ext_profile)
    if has_internal:
        int_profile = AttendeeProfile(
            email="invest1@kakaoventures.co.kr",
            display_name="김투자",
            is_internal=True,
            company_domain="kakaoventures.co.kr",
        )
        attendees.append(int_profile)

    # Build mock Gmail threads
    gmail_thread_list = []
    for i in range(gmail_threads):
        t = MagicMock()
        t.subject = f"[AcmeCorp] 투자 협의 {i+1}"
        t.latest_date = now - timedelta(days=i * 3)
        t.message_count = i + 2
        gmail_thread_list.append(t)

    # Build mock Notion records
    notion_record_list = []
    for i in range(notion_records):
        r = MagicMock()
        r.title = f"AcmeCorp 딜 레코드 {i+1}"
        r.company_name = "AcmeCorp"
        r.status = "검토중" if i == 0 else "보류"
        r.url = f"https://notion.so/page-{i}"
        notion_record_list.append(r)

    # Build Slack history result
    slack_history = None
    if slack_fetched:
        from src.slack.history_retriever import SlackHistoryResult, SlackMessage, SlackChannel

        messages = []
        for i in range(slack_messages):
            msg = SlackMessage(
                channel_id=f"C{i:06d}",
                channel_name="투자-딜-검토" if i % 2 == 0 else "squad-service",
                ts=str((now - timedelta(hours=i * 12)).timestamp()),
                user_id=f"U{i:06d}",
                text=f"AcmeCorp 관련 논의 — {i+1}번째 메시지. 투자 검토 진행 중.",
                message_dt=now - timedelta(hours=i * 12),
            )
            messages.append(msg)

        channels_searched = [
            SlackChannel(
                channel_id="C000001", channel_name="투자-딜-검토", is_member=True
            ),
            SlackChannel(
                channel_id="C000002", channel_name="squad-service", is_member=True
            ),
        ]
        slack_history = SlackHistoryResult(
            company_name="acmecorp",
            channels_searched=channels_searched,
            messages=messages,
            available=True,
        )

    # Build errors list
    errors = []
    if not gmail_available:
        errors.append(AggregationError(source="gmail", message="Gmail API timeout"))
    if not notion_available:
        errors.append(AggregationError(source="notion", message="Notion API error"))
    if not slack_history_available:
        errors.append(
            AggregationError(source="slack_history", message="Slack token invalid")
        )

    raw = RawBriefingContent(
        meeting_id="evt-followup-001",
        meeting_title=meeting_title,
        meeting_start=start,
        meeting_end=end,
        meeting_location="강남 KV 사무실",
        meeting_description="이전 미팅 후속 논의 및 투자 조건 확인",
        meeting_html_link="https://calendar.google.com/event/test",
        organizer_email="invest1@kakaoventures.co.kr",
        attendee_profiles=attendees,
        gmail_threads=gmail_thread_list,
        gmail_available=gmail_available,
        notion_records=notion_record_list,
        notion_available=notion_available,
        calendar_history_available=calendar_history_available,
        slack_history=slack_history,
        slack_history_available=slack_history_available,
        errors=errors,
    )
    return raw


# ── Basic smoke test ─────────────────────────────────────────────────────────

class TestFormatExternalFollowupBriefingBasic:
    """Happy-path: all sources available."""

    def test_returns_tuple(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        result = format_external_followup_briefing(raw)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fallback_is_string(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        fallback, _ = format_external_followup_briefing(raw)
        assert isinstance(fallback, str)
        assert len(fallback) > 0

    def test_blocks_is_list(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        _, blocks = format_external_followup_briefing(raw)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_blocks_within_slack_limit(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        _, blocks = format_external_followup_briefing(raw)
        assert len(blocks) <= 50, f"Slack block limit exceeded: {len(blocks)} blocks"

    def test_header_block_present(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        _, blocks = format_external_followup_briefing(raw)
        header = blocks[0]
        assert header["type"] == "header"
        assert "🔄" in header["text"]["text"]
        assert "후속 미팅 브리핑" in header["text"]["text"]
        assert raw.meeting_title in header["text"]["text"]

    def test_footer_always_present(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        _, blocks = format_external_followup_briefing(raw)
        # Last block is always footer (context type)
        assert blocks[-1]["type"] == "context"
        assert "확인 불가" in blocks[-1]["elements"][0]["text"]

    def test_meeting_title_in_fallback(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        fallback, _ = format_external_followup_briefing(raw)
        assert raw.meeting_title in fallback


# ── Attendee history rendering ────────────────────────────────────────────────

class TestAttendeeHistoryRendering:
    """Attendees with and without relationship history."""

    def test_external_attendee_with_history_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(past_meetings=5)
        _, blocks = format_external_followup_briefing(raw)
        # Find the external attendees section block
        att_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "외부 참석자" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert att_block is not None
        text = att_block["text"]["text"]
        assert "5회" in text or "과거 미팅" in text

    def test_external_attendee_no_history_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(past_meetings=0)
        _, blocks = format_external_followup_briefing(raw)
        att_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "외부 참석자" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert att_block is not None
        text = att_block["text"]["text"]
        # Should show "기록 없음" not a count
        assert "기록 없음" in text or "없음" in text

    def test_calendar_history_unavailable_annotated(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(past_meetings=0, calendar_history_available=False)
        _, blocks = format_external_followup_briefing(raw)
        att_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "외부 참석자" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert att_block is not None
        text = att_block["text"]["text"]
        assert "확인 불가" in text

    def test_no_external_attendees_shows_없음(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(has_external=False)
        _, blocks = format_external_followup_briefing(raw)
        att_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "외부 참석자" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert att_block is not None
        assert "없음" in att_block["text"]["text"]

    def test_internal_attendees_in_context_element(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(has_internal=True)
        _, blocks = format_external_followup_briefing(raw)
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        # At least one context block should mention the internal team
        internal_found = any(
            "내부" in elem.get("text", "")
            for b in context_blocks
            for elem in b.get("elements", [])
        )
        assert internal_found


# ── Notion section ────────────────────────────────────────────────────────────

class TestNotionSection:
    """Notion records available, empty, and unavailable."""

    def test_notion_records_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(notion_records=3)
        _, blocks = format_external_followup_briefing(raw)
        notion_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Notion" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert notion_block is not None
        assert "3개" in notion_block["text"]["text"]

    def test_notion_empty_shows_없음(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(notion_records=0)
        _, blocks = format_external_followup_briefing(raw)
        notion_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Notion" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert notion_block is not None
        assert "없음" in notion_block["text"]["text"]

    def test_notion_unavailable_shows_확인불가(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(notion_available=False, notion_records=0)
        _, blocks = format_external_followup_briefing(raw)
        notion_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Notion" in b.get("text", {}).get("text", "")
                and "확인 불가" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert notion_block is not None


# ── Gmail section ─────────────────────────────────────────────────────────────

class TestGmailSection:
    """Gmail exchange: available, empty, unavailable."""

    def test_gmail_threads_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(gmail_threads=3)
        _, blocks = format_external_followup_briefing(raw)
        gmail_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "이메일" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert gmail_block is not None
        assert "3개" in gmail_block["text"]["text"]

    def test_gmail_empty_shows_없음(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(gmail_threads=0)
        _, blocks = format_external_followup_briefing(raw)
        gmail_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "이메일" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert gmail_block is not None
        assert "없음" in gmail_block["text"]["text"]

    def test_gmail_unavailable_shows_확인불가(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(gmail_available=False, gmail_threads=0)
        _, blocks = format_external_followup_briefing(raw)
        gmail_err_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "이메일" in b.get("text", {}).get("text", "")
                and "확인 불가" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert gmail_err_block is not None


# ── Slack history section ─────────────────────────────────────────────────────

class TestSlackHistorySection:
    """Slack history: available with messages, empty, unavailable, not fetched."""

    def test_slack_messages_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_messages=5, slack_fetched=True)
        _, blocks = format_external_followup_briefing(raw)
        slack_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Slack" in b.get("text", {}).get("text", "")
                and "내부 논의" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert slack_block is not None
        # Should show message count
        assert "5개" in slack_block["text"]["text"]

    def test_slack_message_channel_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_messages=2, slack_fetched=True)
        _, blocks = format_external_followup_briefing(raw)
        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        # Channel name should appear in the blocks
        assert "투자-딜-검토" in all_text or "squad-service" in all_text

    def test_slack_empty_shows_없음(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_messages=0, slack_fetched=True)
        _, blocks = format_external_followup_briefing(raw)
        slack_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Slack" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert slack_block is not None
        assert "없음" in slack_block["text"]["text"]

    def test_slack_unavailable_shows_확인불가(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(
            slack_history_available=False,
            slack_fetched=False,
            slack_messages=0,
        )
        _, blocks = format_external_followup_briefing(raw)
        slack_err_block = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Slack" in b.get("text", {}).get("text", "")
                and "확인 불가" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert slack_err_block is not None

    def test_slack_not_fetched_section_omitted(self):
        """When slack_history is None and available=True, section is omitted (not an error)."""
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_fetched=False, slack_messages=0)
        # Manually ensure no error and no history
        raw.slack_history = None
        raw.slack_history_available = True
        _, blocks = format_external_followup_briefing(raw)
        slack_section = next(
            (
                b
                for b in blocks
                if b.get("type") == "section"
                and "Slack" in b.get("text", {}).get("text", "")
            ),
            None,
        )
        assert slack_section is None, "Slack section should be omitted when not fetched"

    def test_slack_search_term_context_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_messages=2, slack_fetched=True)
        _, blocks = format_external_followup_briefing(raw)
        context_blocks = [b for b in blocks if b.get("type") == "context"]
        search_term_found = any(
            "검색어" in elem.get("text", "")
            for b in context_blocks
            for elem in b.get("elements", [])
        )
        assert search_term_found

    def test_slack_overflow_notice_shown(self):
        """When more messages than _MAX_SLACK_SHOWN, overflow notice appears."""
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
            _MAX_SLACK_SHOWN,
        )
        # Create more messages than the display cap
        raw = _make_raw_content(slack_messages=_MAX_SLACK_SHOWN + 3, slack_fetched=True)
        _, blocks = format_external_followup_briefing(raw)
        overflow_found = any(
            "그 외" in elem.get("text", "") and "메시지" in elem.get("text", "")
            for b in blocks
            if b.get("type") == "context"
            for elem in b.get("elements", [])
        )
        assert overflow_found


# ── Block count cap ───────────────────────────────────────────────────────────

class TestBlockCountCap:
    """Ensure total block count never exceeds the Slack 50-block limit."""

    def test_max_blocks_not_exceeded_with_many_messages(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        # Use many messages and records to try to push over the limit
        raw = _make_raw_content(
            gmail_threads=10,
            notion_records=10,
            slack_messages=20,
            past_meetings=10,
        )
        _, blocks = format_external_followup_briefing(raw)
        assert len(blocks) <= 50, (
            f"Expected ≤50 blocks, got {len(blocks)}"
        )

    def test_footer_always_last_when_truncated(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(
            gmail_threads=10,
            notion_records=10,
            slack_messages=20,
        )
        _, blocks = format_external_followup_briefing(raw)
        # Last block must always be the footer context
        assert blocks[-1]["type"] == "context"
        footer_text = blocks[-1]["elements"][0]["text"]
        assert "Work Assistant" in footer_text


# ── Fallback text accuracy ─────────────────────────────────────────────────────

class TestFallbackText:
    """Verify the plain-text fallback contains accurate key info."""

    def test_fallback_has_meeting_title(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content()
        fallback, _ = format_external_followup_briefing(raw)
        assert raw.meeting_title in fallback

    def test_fallback_has_attendee_name(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(has_external=True)
        fallback, _ = format_external_followup_briefing(raw)
        assert "Alice Smith" in fallback

    def test_fallback_slack_line_with_history(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_messages=4, slack_fetched=True)
        fallback, _ = format_external_followup_briefing(raw)
        assert "Slack" in fallback
        assert "4개" in fallback

    def test_fallback_slack_unavailable_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(slack_history_available=False, slack_fetched=False)
        raw.slack_history = None
        fallback, _ = format_external_followup_briefing(raw)
        assert "확인 불가" in fallback

    def test_fallback_gmail_unavailable_shown(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(gmail_available=False, gmail_threads=0)
        fallback, _ = format_external_followup_briefing(raw)
        assert "확인 불가" in fallback

    def test_fallback_errors_listed(self):
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        raw = _make_raw_content(
            gmail_available=False,
            notion_available=False,
            gmail_threads=0,
            notion_records=0,
        )
        fallback, _ = format_external_followup_briefing(raw)
        # Error summary should appear
        assert "확인 불가" in fallback


# ── RawBriefingContent field ───────────────────────────────────────────────────

class TestRawBriefingContentSlackField:
    """Verify slack_history and slack_history_available are in RawBriefingContent."""

    def test_slack_history_field_exists(self):
        from src.briefing.context_aggregator import RawBriefingContent
        from datetime import date

        now = datetime.now(timezone.utc)
        raw = RawBriefingContent(
            meeting_id="x",
            meeting_title="Test",
            meeting_start=now,
            meeting_end=now + timedelta(hours=1),
        )
        assert hasattr(raw, "slack_history")
        assert raw.slack_history is None  # default

    def test_slack_history_available_field_exists(self):
        from src.briefing.context_aggregator import RawBriefingContent

        now = datetime.now(timezone.utc)
        raw = RawBriefingContent(
            meeting_id="x",
            meeting_title="Test",
            meeting_start=now,
            meeting_end=now + timedelta(hours=1),
        )
        assert hasattr(raw, "slack_history_available")
        assert raw.slack_history_available is True  # default

    def test_to_dict_includes_slack_fields(self):
        from src.briefing.context_aggregator import RawBriefingContent

        now = datetime.now(timezone.utc)
        raw = RawBriefingContent(
            meeting_id="x",
            meeting_title="Test",
            meeting_start=now,
            meeting_end=now + timedelta(hours=1),
        )
        d = raw.to_dict()
        assert "slack_history" in d
        assert "slack_history_available" in d


# ── MeetingContextAggregator: Slack fetch ─────────────────────────────────────

class TestContextAggregatorSlackFetch:
    """Verify MeetingContextAggregator._fetch_slack_history() populates correctly."""

    def _make_meeting(self) -> "Meeting":
        from src.calendar.google_calendar import Meeting, Attendee

        now = datetime.now(timezone.utc)
        return Meeting(
            event_id="evt-001",
            summary="AcmeCorp 후속 미팅",
            start=now + timedelta(minutes=30),
            end=now + timedelta(minutes=90),
            attendees=[
                Attendee(email="ceo@acmecorp.com", display_name="Alice"),
                Attendee(email="invest1@kakaoventures.co.kr", display_name="김투자"),
            ],
        )

    def test_slack_history_populated_when_retriever_succeeds(self):
        from src.briefing.context_aggregator import (
            MeetingContextAggregator,
            RawBriefingContent,
        )
        from src.slack.history_retriever import SlackHistoryResult

        mock_result = SlackHistoryResult(company_name="acmecorp", available=True)

        mock_retriever = MagicMock()
        mock_retriever.search_company_history.return_value = mock_result

        agg = MeetingContextAggregator(slack_retriever=mock_retriever)
        meeting = self._make_meeting()

        content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_slack_history(meeting, content)

        assert content.slack_history is mock_result
        assert content.slack_history_available is True
        mock_retriever.search_company_history.assert_called_once()

    def test_slack_history_available_false_on_failure(self):
        from src.briefing.context_aggregator import (
            MeetingContextAggregator,
            RawBriefingContent,
        )

        mock_retriever = MagicMock()
        mock_retriever.search_company_history.side_effect = RuntimeError("timeout")

        agg = MeetingContextAggregator(slack_retriever=mock_retriever)
        meeting = self._make_meeting()

        content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_slack_history(meeting, content)

        assert content.slack_history is None
        assert content.slack_history_available is False
        assert any(e.source == "slack_history" for e in content.errors)

    def test_no_company_name_derived_skips_slack(self):
        """When meeting has no external attendees, Slack fetch is skipped gracefully."""
        from src.briefing.context_aggregator import (
            MeetingContextAggregator,
            RawBriefingContent,
        )
        from src.calendar.google_calendar import Meeting, Attendee

        now = datetime.now(timezone.utc)
        internal_only_meeting = Meeting(
            event_id="evt-internal",
            summary="팀 스탠드업",
            start=now + timedelta(minutes=30),
            end=now + timedelta(minutes=60),
            attendees=[
                Attendee(email="invest1@kakaoventures.co.kr"),
                Attendee(email="invest2@kakaoventures.co.kr"),
            ],
        )

        mock_retriever = MagicMock()
        agg = MeetingContextAggregator(slack_retriever=mock_retriever)

        content = RawBriefingContent(
            meeting_id=internal_only_meeting.event_id,
            meeting_title=internal_only_meeting.summary,
            meeting_start=internal_only_meeting.start,
            meeting_end=internal_only_meeting.end,
        )
        agg._fetch_slack_history(internal_only_meeting, content)

        # Should not call the retriever when no company name can be derived
        mock_retriever.search_company_history.assert_not_called()
        assert content.slack_history is None
        assert content.slack_history_available is True  # not an error — just no query
