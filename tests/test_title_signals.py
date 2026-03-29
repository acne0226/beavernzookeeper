"""
Tests for src/calendar/title_signals.py (Sub-AC 4b).

Covers:
- TitleKeywordSignals extraction (label, sub-flags, confidence)
- RecurringPatternSignals (tag detection, sequence numbers, periodic keywords,
  recurring_event_id, composite is_recurring, is_likely_internal_recurring)
- MetadataSignals extraction (duration, attendee counts/composition,
  video link, location, time-of-day, recurring_event_id)
- EventSignals aggregate (inferred_is_internal, inferred_is_external,
  dominant_signal)
- extract_event_signals() entry point
- Edge cases: empty titles, missing attributes, all-day events
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar.title_signals import (
    TitleKeywordSignals,
    RecurringPatternSignals,
    MetadataSignals,
    EventSignals,
    extract_title_keyword_signals,
    extract_recurring_pattern_signals,
    extract_metadata_signals,
    extract_event_signals,
)
from src.calendar.title_classifier import MeetingLabel


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_attendee(email: str) -> MagicMock:
    att = MagicMock()
    att.email = email
    return att


def _make_meeting(
    summary: str = "Test Meeting",
    start: datetime | None = None,
    end: datetime | None = None,
    all_day: bool = False,
    attendee_emails: list[str] | None = None,
    video_link: str | None = None,
    conference_type: str | None = None,
    location: str | None = None,
    recurring_event_id: str | None = None,
    is_external: bool = False,
) -> MagicMock:
    if start is None:
        start = datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc)
    if end is None:
        end = start + timedelta(hours=1)
    if attendee_emails is None:
        attendee_emails = []

    attendees = [_make_attendee(e) for e in attendee_emails]
    ext_attendees = [a for a in attendees if not a.email.endswith("@kakaoventures.co.kr")]

    m = MagicMock()
    m.summary = summary
    m.start = start
    m.end = end
    m.all_day = all_day
    m.attendees = attendees
    m.external_attendees = ext_attendees
    m.is_external = is_external or len(ext_attendees) > 0
    m.video_link = video_link
    m.conference_type = conference_type
    m.location = location
    m.recurring_event_id = recurring_event_id
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  1. TitleKeywordSignals
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractTitleKeywordSignals:

    # ── label mirrors classify_by_title ──────────────────────────────────────

    def test_internal_label_for_internal_title(self):
        s = extract_title_keyword_signals("Weekly Team Sync")
        assert s.label == MeetingLabel.INTERNAL
        assert s.is_internal is True

    def test_external_label_for_external_title(self):
        s = extract_title_keyword_signals("IR Pitch Meeting")
        assert s.label == MeetingLabel.EXTERNAL
        assert s.is_external is True

    def test_unknown_label_for_unknown_title(self):
        s = extract_title_keyword_signals("Birthday Party")
        assert s.label == MeetingLabel.UNKNOWN
        assert s.is_unknown is True

    def test_empty_title_returns_unknown(self):
        s = extract_title_keyword_signals("")
        assert s.label == MeetingLabel.UNKNOWN

    def test_whitespace_title_returns_unknown(self):
        s = extract_title_keyword_signals("   ")
        assert s.label == MeetingLabel.UNKNOWN

    # ── Internal sub-flags ────────────────────────────────────────────────────

    def test_one_on_one_flag(self):
        s = extract_title_keyword_signals("1:1 with Alice")
        assert s.is_one_on_one is True

    @pytest.mark.parametrize("title", ["1-on-1", "1on1", "one-on-one session"])
    def test_one_on_one_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_one_on_one is True

    def test_standup_flag(self):
        s = extract_title_keyword_signals("Daily Standup")
        assert s.is_standup is True

    @pytest.mark.parametrize("title", ["Stand-up Meeting", "morning sync", "데일리", "스탠드업"])
    def test_standup_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_standup is True

    def test_all_hands_flag(self):
        s = extract_title_keyword_signals("All Hands Q1")
        assert s.is_all_hands is True

    @pytest.mark.parametrize("title", ["Town Hall", "올핸즈", "전사 회의"])
    def test_all_hands_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_all_hands is True

    def test_retro_flag(self):
        s = extract_title_keyword_signals("Sprint Retrospective")
        assert s.is_retro is True

    @pytest.mark.parametrize("title", ["Retro", "회고", "스프린트 회고"])
    def test_retro_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_retro is True

    def test_sprint_ceremony_flag(self):
        s = extract_title_keyword_signals("Sprint Planning")
        assert s.is_sprint_ceremony is True

    @pytest.mark.parametrize("title", ["Sprint Review", "Sprint Demo", "스프린트 플래닝"])
    def test_sprint_ceremony_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_sprint_ceremony is True

    def test_okr_flag(self):
        s = extract_title_keyword_signals("OKR Review")
        assert s.is_okr is True

    def test_block_flag(self):
        s = extract_title_keyword_signals("Block Time")
        assert s.is_block is True

    @pytest.mark.parametrize("title", ["blocked", "Blocked", "focus block", "블록"])
    def test_block_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_block is True

    def test_hr_flag(self):
        s = extract_title_keyword_signals("Hiring Interview")
        assert s.is_hr is True

    @pytest.mark.parametrize("title", ["면접", "Onboarding", "온보딩", "Performance Review"])
    def test_hr_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_hr is True

    # ── External sub-flags ────────────────────────────────────────────────────

    def test_ir_pitch_flag(self):
        s = extract_title_keyword_signals("IR Pitch Session")
        assert s.is_ir_pitch is True

    def test_investment_meeting_flag(self):
        s = extract_title_keyword_signals("투자 미팅")
        assert s.is_investment_meeting is True

    @pytest.mark.parametrize("title", ["투자 심사", "투자 검토", "심사"])
    def test_investment_meeting_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_investment_meeting is True

    def test_deal_review_flag(self):
        s = extract_title_keyword_signals("Deal Review")
        assert s.is_deal_review is True

    @pytest.mark.parametrize("title", ["딜 리뷰", "Deal Discussion", "Deal Call"])
    def test_deal_review_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_deal_review is True

    def test_partner_flag(self):
        s = extract_title_keyword_signals("Partner Meeting")
        assert s.is_partner is True

    @pytest.mark.parametrize("title", ["파트너 미팅", "Partnership Call"])
    def test_partner_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_partner is True

    def test_client_customer_flag(self):
        s = extract_title_keyword_signals("Client Demo")
        assert s.is_client_customer is True

    @pytest.mark.parametrize("title", ["Customer Meeting", "고객 미팅", "customer success"])
    def test_client_customer_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_client_customer is True

    def test_conference_flag(self):
        s = extract_title_keyword_signals("TechConference 2026")
        assert s.is_conference is True

    @pytest.mark.parametrize("title", ["Industry Summit", "Webinar", "Networking Event",
                                        "웨비나", "컨퍼런스", "네트워킹"])
    def test_conference_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_conference is True

    def test_advisory_board_flag(self):
        s = extract_title_keyword_signals("Advisory Board Meeting")
        assert s.is_advisory_board is True

    @pytest.mark.parametrize("title", ["Advisory Council", "자문 위원회"])
    def test_advisory_board_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_advisory_board is True

    def test_portfolio_flag(self):
        s = extract_title_keyword_signals("Portfolio Company Meeting")
        assert s.is_portfolio is True

    @pytest.mark.parametrize("title", ["포트폴리오 미팅", "portfolio check-in"])
    def test_portfolio_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_portfolio is True

    def test_legal_milestone_flag(self):
        s = extract_title_keyword_signals("MOU Signing")
        assert s.is_legal_milestone is True

    @pytest.mark.parametrize("title", ["NDA Review", "LOI Discussion"])
    def test_legal_milestone_flag_variants(self, title):
        s = extract_title_keyword_signals(title)
        assert s.is_legal_milestone is True

    # ── Matched pattern debug fields ──────────────────────────────────────────

    def test_matched_internal_pattern_populated(self):
        s = extract_title_keyword_signals("Weekly Team Sync")
        assert s.matched_internal_pattern is not None
        assert isinstance(s.matched_internal_pattern, str)

    def test_matched_internal_pattern_none_for_external(self):
        s = extract_title_keyword_signals("IR Pitch")
        assert s.matched_internal_pattern is None

    def test_matched_external_pattern_populated(self):
        s = extract_title_keyword_signals("Deal Review")
        assert s.matched_external_pattern is not None

    def test_matched_external_pattern_none_for_internal(self):
        s = extract_title_keyword_signals("Daily Standup")
        assert s.matched_external_pattern is None

    def test_summary_field_preserved(self):
        s = extract_title_keyword_signals("Quarterly OKR Review")
        assert s.summary == "Quarterly OKR Review"

    # ── Confidence tiers ──────────────────────────────────────────────────────

    def test_high_confidence_for_one_on_one(self):
        s = extract_title_keyword_signals("1:1 with Alice")
        assert s.confidence == "high"

    def test_high_confidence_for_ir_pitch(self):
        s = extract_title_keyword_signals("Series A Pitch")
        assert s.confidence == "high"

    def test_high_confidence_for_deal_review(self):
        s = extract_title_keyword_signals("Deal Review")
        assert s.confidence == "high"

    def test_medium_confidence_for_partner_meeting(self):
        s = extract_title_keyword_signals("Partner Meeting")
        assert s.confidence == "medium"

    def test_medium_confidence_for_team_sync(self):
        # "weekly" → periodic keyword → internal, but confidence depends on sub-flag
        s = extract_title_keyword_signals("Internal Team Sync")
        # "internal" is a medium-confidence internal keyword
        assert s.confidence in ("medium", "high")

    def test_none_confidence_for_unknown(self):
        s = extract_title_keyword_signals("Doctor Appointment")
        assert s.confidence == "none"


# ══════════════════════════════════════════════════════════════════════════════
#  2. RecurringPatternSignals
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractRecurringPatternSignals:

    # ── Recurring tags ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("title", [
        "[Recurring] Team Meeting",
        "[Weekly] Standup",
        "[Daily] Check-in",
        "[Monthly] OKR Review",
        "(Recurring) Partner Call",
        "(Weekly) Sync",
        "[R] all-hands",
        "(R) team review",
        "Recurring: Stand-up",
        "Weekly: Team Sync",
    ])
    def test_recurring_tag_detection(self, title: str):
        s = extract_recurring_pattern_signals(title)
        assert s.has_recurring_tag is True, f"Expected tag detection for: {title!r}"
        assert s.is_recurring is True

    @pytest.mark.parametrize("title", [
        "[반복] 팀 미팅",
        "(주간) 싱크",
        "[월간] OKR",
        "[격주] 체크인",
    ])
    def test_korean_recurring_tag_detection(self, title: str):
        s = extract_recurring_pattern_signals(title)
        assert s.has_recurring_tag is True, f"Expected Korean tag detection for: {title!r}"
        assert s.is_recurring is True

    def test_recurring_tag_text_captured(self):
        s = extract_recurring_pattern_signals("[Weekly] Team Sync")
        assert s.recurring_tag_text is not None
        assert "weekly" in s.recurring_tag_text.lower() or "Weekly" in s.recurring_tag_text

    # ── Sequence number markers ───────────────────────────────────────────────

    @pytest.mark.parametrize("title", [
        "Team Sync #5",
        "Weekly Check-in #42",
        "Stand-up #1",
        "Weekly Review - Week 12",
        "Team Sync - Week 3",
        "Sprint 14",
        "Sprint #14",
        "스프린트 14",
        "Session 5",
        "No. 7 Team Meeting",
        "Wk 8 Standup",
        "Week 3 Review",
    ])
    def test_sequence_number_detection(self, title: str):
        s = extract_recurring_pattern_signals(title)
        assert s.has_sequence_number is True, f"Expected sequence number for: {title!r}"
        assert s.is_recurring is True

    def test_sequence_marker_text_captured(self):
        s = extract_recurring_pattern_signals("Team Sync #5")
        assert s.sequence_marker_text is not None
        assert "#5" in s.sequence_marker_text or "5" in s.sequence_marker_text

    # ── Periodic keyword detection ────────────────────────────────────────────

    @pytest.mark.parametrize("title", [
        "Weekly Team Sync",
        "Bi-Weekly Check-in",
        "Biweekly Standup",
        "Daily Standup",
        "Monthly OKR Review",
        "주간 팀 싱크",
        "격주 체크인",
        "월간 보고",
        "매주 팀 미팅",
        "Every week team sync",
        "Every month review",
        "Fortnightly sync",
    ])
    def test_periodic_keyword_detection(self, title: str):
        s = extract_recurring_pattern_signals(title)
        assert s.has_periodic_keyword is True, f"Expected periodic keyword for: {title!r}"
        assert s.is_recurring is True

    def test_periodic_keyword_text_captured(self):
        s = extract_recurring_pattern_signals("Weekly Team Sync")
        assert s.periodic_keyword_text is not None
        assert "weekly" in s.periodic_keyword_text.lower() or "Weekly" in s.periodic_keyword_text

    # ── recurring_event_id ───────────────────────────────────────────────────

    def test_recurring_event_id_sets_is_recurring(self):
        s = extract_recurring_pattern_signals("Team Meeting", recurring_event_id="evt_base_123")
        assert s.is_recurring is True
        assert s.recurring_event_id == "evt_base_123"

    def test_no_recurring_event_id_does_not_set_recurring(self):
        s = extract_recurring_pattern_signals("Team Meeting", recurring_event_id=None)
        # No title pattern either → not recurring
        assert s.is_recurring is False

    # ── Non-recurring titles ──────────────────────────────────────────────────

    @pytest.mark.parametrize("title", [
        "Investor Intro Call",
        "Deal Review",
        "Birthday Party",
        "Doctor Appointment",
        "Lunch",
    ])
    def test_non_recurring_titles(self, title: str):
        s = extract_recurring_pattern_signals(title)
        assert s.is_recurring is False, f"Should NOT be recurring: {title!r}"

    # ── is_likely_internal_recurring ─────────────────────────────────────────

    def test_weekly_internal_sync_is_likely_internal(self):
        s = extract_recurring_pattern_signals("Weekly Team Sync")
        assert s.is_likely_internal_recurring is True

    def test_recurring_tag_internal_title(self):
        s = extract_recurring_pattern_signals("[Weekly] 팀 미팅")
        assert s.is_recurring is True
        assert s.is_likely_internal_recurring is True

    def test_numbered_external_meeting_not_likely_internal(self):
        # "투자 미팅 #5" — external keyword with sequence number
        s = extract_recurring_pattern_signals("투자 미팅 #5")
        # is_recurring=True (sequence), but title is EXTERNAL
        # has_periodic_keyword=False → is_likely_internal depends on title
        assert s.is_recurring is True
        # Since title_label is EXTERNAL and no periodic keyword → not likely internal
        assert s.is_likely_internal_recurring is False

    def test_recurring_id_without_title_patterns_not_likely_internal(self):
        # Generic title + recurring_event_id but no internal keyword
        s = extract_recurring_pattern_signals("Meeting", recurring_event_id="evt123")
        assert s.is_recurring is True
        # "Meeting" → UNKNOWN title → not internal (no periodic keyword either)
        # is_likely_internal = is_recurring AND (title_label==INTERNAL OR has_periodic_keyword)
        assert s.is_likely_internal_recurring is False

    def test_empty_title_not_recurring(self):
        s = extract_recurring_pattern_signals("")
        assert s.is_recurring is False
        assert s.is_likely_internal_recurring is False

    def test_combined_signals_all_detected(self):
        """Title with tag, sequence number, AND periodic keyword."""
        s = extract_recurring_pattern_signals("[Weekly] Team Sync #5")
        assert s.has_recurring_tag is True
        assert s.has_sequence_number is True
        assert s.has_periodic_keyword is True
        assert s.is_recurring is True


# ══════════════════════════════════════════════════════════════════════════════
#  3. MetadataSignals
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractMetadataSignals:

    # ── Duration signals ─────────────────────────────────────────────────────

    def test_duration_calculated_from_start_end(self):
        m = _make_meeting(
            start=datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 29, 11, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.duration_minutes == 60

    def test_short_meeting_30_min(self):
        m = _make_meeting(
            start=datetime(2026, 3, 29, 9, 30, tzinfo=timezone.utc),
            end=datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.duration_minutes == 30
        assert s.is_short_meeting is True

    def test_short_meeting_15_min(self):
        m = _make_meeting(
            start=datetime(2026, 3, 29, 9, 45, tzinfo=timezone.utc),
            end=datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_short_meeting is True

    def test_standard_meeting_not_short(self):
        m = _make_meeting(
            start=datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 29, 11, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_short_meeting is False

    def test_very_long_meeting(self):
        m = _make_meeting(
            start=datetime(2026, 3, 29, 9, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 29, 13, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.duration_minutes == 240
        assert s.is_very_long_meeting is True

    def test_all_day_event(self):
        m = _make_meeting(all_day=True)
        m.all_day = True
        s = extract_metadata_signals(m)
        assert s.is_all_day is True
        assert s.is_short_meeting is False
        assert s.is_very_long_meeting is False

    # ── Attendee composition ──────────────────────────────────────────────────

    def test_only_internal_attendees(self):
        m = _make_meeting(attendee_emails=["a@kakaoventures.co.kr", "b@kakaoventures.co.kr"])
        s = extract_metadata_signals(m)
        assert s.total_attendee_count == 2
        assert s.internal_attendee_count == 2
        assert s.external_attendee_count == 0
        assert s.has_external_attendees is False

    def test_external_attendees_detected(self):
        m = _make_meeting(
            attendee_emails=["me@kakaoventures.co.kr", "partner@acme.com"],
        )
        s = extract_metadata_signals(m)
        assert s.external_attendee_count == 1
        assert s.has_external_attendees is True
        assert "acme.com" in s.external_attendee_domains

    def test_multiple_external_domains(self):
        m = _make_meeting(
            attendee_emails=[
                "me@kakaoventures.co.kr",
                "alice@acme.com",
                "bob@beta.io",
            ],
        )
        s = extract_metadata_signals(m)
        assert s.external_attendee_count == 2
        assert "acme.com" in s.external_attendee_domains
        assert "beta.io" in s.external_attendee_domains

    def test_external_domain_deduplication(self):
        m = _make_meeting(
            attendee_emails=[
                "me@kakaoventures.co.kr",
                "alice@acme.com",
                "bob@acme.com",
            ],
        )
        s = extract_metadata_signals(m)
        assert s.external_attendee_domains.count("acme.com") == 1

    # ── Attendee count groups ─────────────────────────────────────────────────

    def test_solo_event(self):
        m = _make_meeting(attendee_emails=["me@kakaoventures.co.kr"])
        s = extract_metadata_signals(m)
        assert s.is_solo is True
        assert s.is_one_on_one is False
        assert s.is_small_group is False

    def test_one_on_one_event(self):
        m = _make_meeting(attendee_emails=["a@kakaoventures.co.kr", "b@kakaoventures.co.kr"])
        s = extract_metadata_signals(m)
        assert s.is_one_on_one is True
        assert s.is_solo is False
        assert s.is_small_group is False

    def test_small_group(self):
        m = _make_meeting(attendee_emails=[f"u{i}@kakaoventures.co.kr" for i in range(4)])
        s = extract_metadata_signals(m)
        assert s.is_small_group is True
        assert s.total_attendee_count == 4

    def test_large_group(self):
        m = _make_meeting(attendee_emails=[f"u{i}@kakaoventures.co.kr" for i in range(10)])
        s = extract_metadata_signals(m)
        assert s.is_large_group is True
        assert s.total_attendee_count == 10

    def test_attendee_ratio_external(self):
        m = _make_meeting(
            attendee_emails=[
                "a@kakaoventures.co.kr",
                "b@kakaoventures.co.kr",
                "c@external.com",
                "d@external.com",
            ],
        )
        s = extract_metadata_signals(m)
        assert abs(s.attendee_ratio_external - 0.5) < 0.01

    def test_empty_attendee_list(self):
        m = _make_meeting(attendee_emails=[])
        s = extract_metadata_signals(m)
        assert s.total_attendee_count == 0
        assert s.is_solo is False
        assert s.has_external_attendees is False
        assert s.attendee_ratio_external == 0.0

    # ── Video / location signals ──────────────────────────────────────────────

    def test_has_video_link_true(self):
        m = _make_meeting(video_link="https://meet.google.com/abc-defg-hij",
                          conference_type="Google Meet")
        s = extract_metadata_signals(m)
        assert s.has_video_link is True
        assert s.video_platform == "Google Meet"

    def test_has_video_link_false_when_none(self):
        m = _make_meeting(video_link=None)
        s = extract_metadata_signals(m)
        assert s.has_video_link is False
        assert s.video_platform is None

    def test_has_physical_location(self):
        m = _make_meeting(location="서울 강남구 카카오벤처스 사무실")
        s = extract_metadata_signals(m)
        assert s.has_physical_location is True

    def test_no_physical_location(self):
        m = _make_meeting(location=None)
        s = extract_metadata_signals(m)
        assert s.has_physical_location is False

    def test_empty_location_string_treated_as_no_location(self):
        m = _make_meeting(location="")
        s = extract_metadata_signals(m)
        assert s.has_physical_location is False

    # ── Time-of-day signals ───────────────────────────────────────────────────

    def test_early_morning_meeting(self):
        m = _make_meeting(
            start=datetime(2026, 3, 30, 7, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 7, 30, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_early_morning is True
        assert s.is_evening is False

    def test_evening_meeting(self):
        m = _make_meeting(
            start=datetime(2026, 3, 30, 19, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_evening is True
        assert s.is_early_morning is False

    def test_normal_business_hour_meeting(self):
        m = _make_meeting(
            start=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 11, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_early_morning is False
        assert s.is_evening is False

    def test_weekend_meeting(self):
        # 2026-03-28 is a Saturday
        m = _make_meeting(
            start=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 28, 11, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_weekend is True

    def test_weekday_meeting(self):
        # 2026-03-30 is a Monday
        m = _make_meeting(
            start=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 11, 0, tzinfo=timezone.utc),
        )
        s = extract_metadata_signals(m)
        assert s.is_weekend is False

    # ── Recurring event ID ────────────────────────────────────────────────────

    def test_recurring_event_id_sets_flag(self):
        m = _make_meeting(recurring_event_id="recurring_base_abc123")
        s = extract_metadata_signals(m)
        assert s.has_recurring_event_id is True

    def test_no_recurring_event_id(self):
        m = _make_meeting(recurring_event_id=None)
        s = extract_metadata_signals(m)
        assert s.has_recurring_event_id is False

    # ── Robustness ────────────────────────────────────────────────────────────

    def test_missing_video_link_attr_graceful(self):
        m = MagicMock(spec=[])  # empty spec — no attributes
        m.summary = "Meeting"
        # Should not raise
        s = extract_metadata_signals(m)
        assert isinstance(s, MetadataSignals)

    def test_all_day_skips_duration(self):
        m = _make_meeting(all_day=True)
        s = extract_metadata_signals(m)
        assert s.duration_minutes == 0
        assert s.is_short_meeting is False
        assert s.is_very_long_meeting is False


# ══════════════════════════════════════════════════════════════════════════════
#  4. EventSignals aggregate
# ══════════════════════════════════════════════════════════════════════════════

class TestEventSignals:

    def _signals(
        self,
        summary="Test",
        attendee_emails=None,
        recurring_event_id=None,
        start=None,
        end=None,
    ) -> EventSignals:
        m = _make_meeting(
            summary=summary,
            attendee_emails=attendee_emails or [],
            recurring_event_id=recurring_event_id,
            start=start,
            end=end,
        )
        return extract_event_signals(m)

    # ── inferred_is_internal ─────────────────────────────────────────────────

    def test_internal_title_inferred_internal(self):
        s = self._signals("Daily Standup", ["a@kakaoventures.co.kr"])
        assert s.inferred_is_internal is True

    def test_no_external_attendees_inferred_internal(self):
        s = self._signals("Random Meeting", ["a@kakaoventures.co.kr"])
        assert s.inferred_is_internal is True

    def test_external_attendees_not_inferred_internal(self):
        s = self._signals("Partner Discussion", ["me@kakaoventures.co.kr", "x@ext.com"])
        assert s.inferred_is_internal is False

    def test_recurring_internal_no_external_inferred_internal(self):
        s = self._signals("Weekly Sync #3", ["a@kakaoventures.co.kr"])
        # No external attendees → inferred_is_internal regardless
        assert s.inferred_is_internal is True

    # ── inferred_is_external ─────────────────────────────────────────────────

    def test_external_attendees_with_external_title(self):
        s = self._signals("Partner Meeting", ["me@kakaoventures.co.kr", "x@partner.com"])
        assert s.inferred_is_external is True

    def test_no_external_attendees_not_inferred_external(self):
        s = self._signals("Partner Meeting", ["a@kakaoventures.co.kr"])
        assert s.inferred_is_external is False

    def test_internal_title_with_external_attendees_not_external(self):
        """INTERNAL title takes priority — inferred_is_external must be False."""
        s = self._signals("All Hands", ["a@kakaoventures.co.kr", "guest@ext.com"])
        assert s.inferred_is_external is False

    def test_external_attendees_unknown_title_is_external(self):
        s = self._signals("Lunch", ["me@kakaoventures.co.kr", "x@unknown.co"])
        assert s.inferred_is_external is True

    # ── dominant_signal ───────────────────────────────────────────────────────

    def test_dominant_signal_title_internal(self):
        s = self._signals("Sprint Retro", ["a@kakaoventures.co.kr"])
        assert s.dominant_signal == "title_internal_keyword"

    def test_dominant_signal_no_external_attendees(self):
        s = self._signals("Random Event", ["a@kakaoventures.co.kr"])
        assert s.dominant_signal == "no_external_attendees"

    def test_dominant_signal_title_external(self):
        s = self._signals("IR Pitch", ["me@kakaoventures.co.kr", "founder@startup.io"])
        assert s.dominant_signal == "title_external_keyword"

    def test_dominant_signal_external_attendees_present(self):
        s = self._signals("Mystery Meeting", ["me@kakaoventures.co.kr", "x@unknown.co"])
        assert s.dominant_signal == "external_attendees_present"

    # ── Sub-object access ─────────────────────────────────────────────────────

    def test_title_keywords_accessible(self):
        s = self._signals("Weekly Team Sync")
        assert isinstance(s.title_keywords, TitleKeywordSignals)
        assert s.title_keywords.label == MeetingLabel.INTERNAL

    def test_recurring_accessible(self):
        s = self._signals("Weekly Team Sync #3")
        assert isinstance(s.recurring, RecurringPatternSignals)
        assert s.recurring.is_recurring is True

    def test_metadata_accessible(self):
        s = self._signals("Team Sync", ["a@kakaoventures.co.kr", "b@ext.com"],
                          start=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
                          end=datetime(2026, 3, 30, 11, 0, tzinfo=timezone.utc))
        assert isinstance(s.metadata, MetadataSignals)
        assert s.metadata.duration_minutes == 60
        assert s.metadata.has_external_attendees is True


# ══════════════════════════════════════════════════════════════════════════════
#  5. extract_event_signals() entry point
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractEventSignals:

    def test_returns_event_signals(self):
        m = _make_meeting(summary="Weekly Standup")
        result = extract_event_signals(m)
        assert isinstance(result, EventSignals)

    def test_recurring_event_id_override(self):
        m = _make_meeting(summary="Team Meeting")
        result = extract_event_signals(m, recurring_event_id="overridden_base_id")
        assert result.recurring.recurring_event_id == "overridden_base_id"
        assert result.recurring.is_recurring is True

    def test_recurring_event_id_from_meeting_attr(self):
        m = _make_meeting(summary="Team Meeting", recurring_event_id="native_recurring_id")
        result = extract_event_signals(m)
        assert result.recurring.recurring_event_id == "native_recurring_id"

    def test_recurring_event_id_override_wins_over_attr(self):
        m = _make_meeting(summary="Team Meeting", recurring_event_id="attr_id")
        result = extract_event_signals(m, recurring_event_id="override_id")
        # When override is passed explicitly it takes precedence
        assert result.recurring.recurring_event_id == "override_id"

    def test_full_pipeline_internal_meeting(self):
        m = _make_meeting(
            summary="Daily Standup",
            attendee_emails=["a@kakaoventures.co.kr", "b@kakaoventures.co.kr"],
            # 08:15 UTC → hour=8 → is_early_morning=True (before 9)
            start=datetime(2026, 3, 30, 8, 15, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 8, 30, tzinfo=timezone.utc),
            recurring_event_id="standup_series_abc",
        )
        s = extract_event_signals(m)
        assert s.inferred_is_internal is True
        assert s.title_keywords.is_standup is True
        assert s.recurring.is_recurring is True
        assert s.metadata.is_short_meeting is True
        assert s.metadata.is_early_morning is True
        assert s.metadata.has_recurring_event_id is True

    def test_full_pipeline_external_meeting(self):
        m = _make_meeting(
            summary="투자 미팅",
            attendee_emails=["me@kakaoventures.co.kr", "founder@startup.kr"],
            start=datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            video_link="https://zoom.us/j/123456",
            conference_type="Zoom",
        )
        s = extract_event_signals(m)
        assert s.inferred_is_external is True
        assert s.title_keywords.is_investment_meeting is True
        assert s.metadata.has_external_attendees is True
        assert s.metadata.has_video_link is True
        assert s.metadata.video_platform == "Zoom"

    def test_full_pipeline_recurring_partner_meeting(self):
        """Recurring external meeting: has sequence number but is external."""
        m = _make_meeting(
            summary="Partner Check-in #3",
            attendee_emails=["me@kakaoventures.co.kr", "cto@bigpartner.com"],
            recurring_event_id="partner_checkin_base",
        )
        s = extract_event_signals(m)
        assert s.recurring.is_recurring is True
        assert s.recurring.has_sequence_number is True
        # External partner → inferred_is_external
        assert s.inferred_is_external is True
        # Recurring but external (not likely internal recurring)
        assert s.recurring.is_likely_internal_recurring is False


# ══════════════════════════════════════════════════════════════════════════════
#  6. Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_none_summary_handled_gracefully(self):
        m = MagicMock()
        m.summary = None
        m.all_day = False
        m.attendees = []
        m.external_attendees = []
        m.is_external = False
        m.video_link = None
        m.conference_type = None
        m.location = None
        m.recurring_event_id = None
        m.start = datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc)
        m.end = datetime(2026, 3, 30, 11, 0, tzinfo=timezone.utc)
        s = extract_event_signals(m)
        assert isinstance(s, EventSignals)
        assert s.title_keywords.label == MeetingLabel.UNKNOWN

    def test_attendee_with_empty_email(self):
        m = _make_meeting()
        att = MagicMock()
        att.email = ""
        m.attendees = [att]
        m.external_attendees = []
        m.is_external = False
        s = extract_metadata_signals(m)
        # Empty email → should not be counted as external or internal domain
        assert s.total_attendee_count == 1
        assert s.external_attendee_count == 0
        assert s.internal_attendee_count == 0

    def test_recurring_pattern_title_with_no_other_signals(self):
        """Title with only a sequence number — no keyword hints."""
        s = extract_recurring_pattern_signals("Meeting #7")
        assert s.has_sequence_number is True
        assert s.is_recurring is True
        # "Meeting" → UNKNOWN label, no periodic keyword → not likely internal
        assert s.is_likely_internal_recurring is False

    def test_mixed_signals_internal_title_with_recurring_and_external_attendee(self):
        """Internal title (sprint) with an external attendee (edge case)."""
        m = _make_meeting(
            summary="Sprint Review",
            attendee_emails=["a@kakaoventures.co.kr", "observer@external.com"],
        )
        s = extract_event_signals(m)
        # Internal title takes priority even with external attendee
        assert s.inferred_is_internal is True
        assert s.inferred_is_external is False

    def test_block_event_is_internal(self):
        m = _make_meeting(summary="Block Time", attendee_emails=["me@kakaoventures.co.kr"])
        s = extract_event_signals(m)
        assert s.inferred_is_internal is True
        assert s.title_keywords.is_block is True


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
