"""
Tests for src/calendar/event_classifier.py (Sub-AC 4c).

Covers:
- EventCategory enum values and string equality
- ClassificationResult convenience properties (is_internal, is_external_first,
  is_external_followup, total_past_meetings)
- classify_event() + classify_event_full() return type contract

Internal classification:
  - Title heuristic labels INTERNAL → always EventCategory.INTERNAL
  - Unknown title + no external attendees → EventCategory.INTERNAL
  - External title label + no external attendees → EventCategory.INTERNAL
  (attendee check wins when title says EXTERNAL but attendees are all internal)

External first classification:
  - External attendees + history_cache is None → EXTERNAL_FIRST (conservative)
  - External attendees + empty cache (no past meetings) → EXTERNAL_FIRST
  - Unknown title + external attendees + no cache history → EXTERNAL_FIRST
  - External title + external attendees + no cache history → EXTERNAL_FIRST

External followup classification:
  - External attendees whose domain is in cache → EXTERNAL_FOLLOWUP
  - External attendees whose email is in cache → EXTERNAL_FOLLOWUP
  - Multiple external attendees; at least one in cache → EXTERNAL_FOLLOWUP

History metadata:
  - history_matched_emails populated when email in cache
  - history_matched_domains populated when domain in cache
  - last_meeting_with populated from cache
  - past_meeting_counts keyed by domain
  - total_past_meetings sums across domains

Debug notes:
  - debug_notes list is populated with intermediate signals
  - title_label and category reflected in debug_notes

Edge cases:
  - Meeting with empty attendee list (→ INTERNAL)
  - Attendee with empty email address (→ skipped)
  - history_available=False when cache is None
  - history_available=True when cache is provided
  - INTERNAL title overrides external attendees (title takes priority over
    attendee check for the internal signal)
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar.event_classifier import (
    EventCategory,
    ClassificationResult,
    classify_event,
    classify_event_full,
)
from src.calendar.title_classifier import MeetingLabel
from src.calendar.history_cache import CalendarHistoryCache, CachedEvent


# ── Fixtures / Factories ───────────────────────────────────────────────────────

def _make_attendee(email: str, internal: bool = False) -> MagicMock:
    """Return a mock Attendee object."""
    att = MagicMock()
    att.email = email
    att.is_internal = internal
    att.display_name = email.split("@")[0]
    return att


def _make_meeting(
    event_id: str = "evt_test",
    summary: str = "Test Meeting",
    attendee_emails: list[str] | None = None,
    external_emails: list[str] | None = None,
    is_external: bool | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> MagicMock:
    """
    Return a mock Meeting object.

    If external_emails is supplied those attendees are returned from
    external_attendees; otherwise all attendees with non-kakaoventures.co.kr
    domain are treated as external.
    """
    if start is None:
        start = datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc)
    if end is None:
        end = start + timedelta(hours=1)
    if attendee_emails is None:
        attendee_emails = []

    all_attendees = [
        _make_attendee(e, internal=e.endswith("@kakaoventures.co.kr"))
        for e in attendee_emails
    ]

    # External attendees = those not internal
    if external_emails is None:
        ext_attendees = [a for a in all_attendees if not a.is_internal]
    else:
        ext_attendees = [_make_attendee(e, internal=False) for e in external_emails]

    meeting = MagicMock()
    meeting.event_id = event_id
    meeting.summary = summary
    meeting.start = start
    meeting.end = end
    meeting.attendees = all_attendees
    meeting.external_attendees = ext_attendees
    if is_external is not None:
        meeting.is_external = is_external
    else:
        meeting.is_external = len(ext_attendees) > 0

    return meeting


def _make_cached_event(
    event_id: str = "hist1",
    title: str = "Past Meeting",
    start_iso: str = "2025-09-01T10:00:00+00:00",
    attendee_emails: list[str] | None = None,
    attendee_domains: list[str] | None = None,
    is_external: bool = True,
) -> CachedEvent:
    if attendee_emails is None:
        attendee_emails = ["me@kakaoventures.co.kr", "partner@acme.com"]
    if attendee_domains is None:
        attendee_domains = ["kakaoventures.co.kr", "acme.com"]
    return CachedEvent(
        event_id=event_id,
        title=title,
        start_iso=start_iso,
        end_iso="2025-09-01T11:00:00+00:00",
        all_day=False,
        organizer_email="me@kakaoventures.co.kr",
        is_external=is_external,
        attendee_emails=attendee_emails,
        attendee_domains=attendee_domains,
    )


def _make_cache(*cached_events: CachedEvent) -> CalendarHistoryCache:
    return CalendarHistoryCache(events=list(cached_events))


# ── EventCategory enum ────────────────────────────────────────────────────────

class TestEventCategoryEnum:

    def test_enum_values(self):
        assert EventCategory.INTERNAL.value == "internal"
        assert EventCategory.EXTERNAL_FIRST.value == "external_first"
        assert EventCategory.EXTERNAL_FOLLOWUP.value == "external_followup"

    def test_string_equality(self):
        assert EventCategory.INTERNAL == "internal"
        assert EventCategory.EXTERNAL_FIRST == "external_first"
        assert EventCategory.EXTERNAL_FOLLOWUP == "external_followup"

    def test_is_str_subclass(self):
        assert isinstance(EventCategory.INTERNAL, str)


# ── ClassificationResult convenience props ────────────────────────────────────

class TestClassificationResult:

    def _make_result(self, category: EventCategory) -> ClassificationResult:
        return ClassificationResult(
            event_id="e1",
            title="Test",
            category=category,
            title_label=MeetingLabel.UNKNOWN,
            is_external_by_attendees=False,
        )

    def test_is_internal_true(self):
        r = self._make_result(EventCategory.INTERNAL)
        assert r.is_internal is True
        assert r.is_external_first is False
        assert r.is_external_followup is False

    def test_is_external_first_true(self):
        r = self._make_result(EventCategory.EXTERNAL_FIRST)
        assert r.is_internal is False
        assert r.is_external_first is True
        assert r.is_external_followup is False

    def test_is_external_followup_true(self):
        r = self._make_result(EventCategory.EXTERNAL_FOLLOWUP)
        assert r.is_internal is False
        assert r.is_external_first is False
        assert r.is_external_followup is True

    def test_total_past_meetings_sums(self):
        r = self._make_result(EventCategory.EXTERNAL_FOLLOWUP)
        r.past_meeting_counts = {"acme.com": 3, "beta.io": 2}
        assert r.total_past_meetings == 5

    def test_total_past_meetings_empty(self):
        r = self._make_result(EventCategory.EXTERNAL_FIRST)
        assert r.total_past_meetings == 0


# ── classify_event return type ────────────────────────────────────────────────

class TestClassifyEventReturnType:

    def test_returns_event_category(self):
        meeting = _make_meeting(summary="Weekly Standup")
        result = classify_event(meeting)
        assert isinstance(result, EventCategory)

    def test_classify_event_full_returns_classification_result(self):
        meeting = _make_meeting(summary="Weekly Standup")
        result = classify_event_full(meeting)
        assert isinstance(result, ClassificationResult)

    def test_classify_event_full_category_matches_classify_event(self):
        meeting = _make_meeting(summary="Partner Meeting",
                                attendee_emails=["me@kakaoventures.co.kr",
                                                 "partner@acme.com"])
        cache = _make_cache()
        assert classify_event(meeting, cache) == classify_event_full(meeting, cache).category


# ── Internal classification ───────────────────────────────────────────────────

class TestInternalClassification:

    @pytest.mark.parametrize("title", [
        "Daily Standup",
        "Weekly Team Sync",
        "Sprint Planning",
        "1:1 with manager",
        "OKR Review",
        "팀 회의",
        "주간 싱크",
        "회고",
        "내부 미팅",
        "전사 행사",
        "면접",
    ])
    def test_internal_title_gives_internal_category(self, title: str):
        meeting = _make_meeting(summary=title,
                                attendee_emails=["a@kakaoventures.co.kr",
                                                 "b@kakaoventures.co.kr"])
        assert classify_event(meeting) == EventCategory.INTERNAL

    def test_internal_title_overrides_no_external_attendees(self):
        """Internal title + all-internal attendees → INTERNAL."""
        meeting = _make_meeting(
            summary="All Hands",
            attendee_emails=["a@kakaoventures.co.kr"],
            is_external=False,
        )
        assert classify_event(meeting) == EventCategory.INTERNAL

    def test_unknown_title_no_external_attendees_gives_internal(self):
        """Unknown title + no external attendees → INTERNAL."""
        meeting = _make_meeting(
            summary="Birthday Party",
            attendee_emails=["a@kakaoventures.co.kr"],
            is_external=False,
        )
        assert classify_event(meeting) == EventCategory.INTERNAL

    def test_external_title_but_no_external_attendees_gives_internal(self):
        """
        Title heuristic says EXTERNAL but attendee list has no outsiders.
        The attendee domain check should produce INTERNAL — we don't brief
        external meetings with zero external attendees.
        """
        meeting = _make_meeting(
            summary="Partner Meeting",  # external keyword
            attendee_emails=["a@kakaoventures.co.kr", "b@kakaoventures.co.kr"],
            external_emails=[],        # explicitly no external attendees
            is_external=False,
        )
        result = classify_event_full(meeting)
        assert result.category == EventCategory.INTERNAL

    def test_empty_attendees_gives_internal(self):
        """Meeting with no attendees → INTERNAL."""
        meeting = _make_meeting(
            summary="Unclassified Meeting",
            attendee_emails=[],
            is_external=False,
        )
        assert classify_event(meeting) == EventCategory.INTERNAL

    def test_internal_title_field_populated(self):
        meeting = _make_meeting(summary="Sprint Retro",
                                attendee_emails=["a@kakaoventures.co.kr"])
        result = classify_event_full(meeting)
        assert result.title_label == MeetingLabel.INTERNAL

    def test_internal_category_field_populated(self):
        meeting = _make_meeting(summary="Sprint Retro",
                                attendee_emails=["a@kakaoventures.co.kr"])
        result = classify_event_full(meeting)
        assert result.category == EventCategory.INTERNAL
        assert result.is_internal is True

    def test_internal_title_with_external_attendees_still_internal(self):
        """
        INTERNAL title takes priority even if there happen to be external
        attendees (e.g. a guest at an all-hands).  Internal ceremonies
        should never trigger an external briefing.
        """
        meeting = _make_meeting(
            summary="All Hands Meeting",
            attendee_emails=["a@kakaoventures.co.kr", "guest@partner.com"],
            external_emails=["guest@partner.com"],
            is_external=True,
        )
        result = classify_event_full(meeting)
        assert result.category == EventCategory.INTERNAL
        assert result.title_label == MeetingLabel.INTERNAL


# ── External first classification ─────────────────────────────────────────────

class TestExternalFirstClassification:

    def test_no_cache_external_meeting_gives_first(self):
        """history_cache=None → conservative EXTERNAL_FIRST."""
        meeting = _make_meeting(
            summary="Partner Demo",
            attendee_emails=["me@kakaoventures.co.kr", "ceo@newco.io"],
            external_emails=["ceo@newco.io"],
            is_external=True,
        )
        assert classify_event(meeting, history_cache=None) == EventCategory.EXTERNAL_FIRST

    def test_empty_cache_external_meeting_gives_first(self):
        """Cache exists but has no events → EXTERNAL_FIRST."""
        meeting = _make_meeting(
            summary="투자 미팅",
            attendee_emails=["me@kakaoventures.co.kr", "cto@startup.com"],
            external_emails=["cto@startup.com"],
            is_external=True,
        )
        empty_cache = _make_cache()
        assert classify_event(meeting, empty_cache) == EventCategory.EXTERNAL_FIRST

    def test_domain_not_in_cache_gives_first(self):
        """Domain appears in attendee list but not in history → EXTERNAL_FIRST."""
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "bob@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="외부 미팅",
            attendee_emails=["me@kakaoventures.co.kr", "alice@totally-new-co.com"],
            external_emails=["alice@totally-new-co.com"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FIRST

    def test_unknown_title_external_attendee_no_cache(self):
        meeting = _make_meeting(
            summary="Lunch",
            attendee_emails=["me@kakaoventures.co.kr", "x@unknown.org"],
            external_emails=["x@unknown.org"],
            is_external=True,
        )
        assert classify_event(meeting, None) == EventCategory.EXTERNAL_FIRST

    def test_external_title_external_attendee_no_cache(self):
        meeting = _make_meeting(
            summary="IR Pitch",
            attendee_emails=["me@kakaoventures.co.kr", "founder@seed.io"],
            external_emails=["founder@seed.io"],
            is_external=True,
        )
        empty_cache = _make_cache()
        assert classify_event(meeting, empty_cache) == EventCategory.EXTERNAL_FIRST

    def test_history_available_false_when_cache_none(self):
        meeting = _make_meeting(
            summary="Client Meeting",
            external_emails=["x@co.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, history_cache=None)
        assert result.history_available is False

    def test_history_available_true_when_cache_provided(self):
        meeting = _make_meeting(
            summary="Client Meeting",
            external_emails=["x@co.com"],
            is_external=True,
        )
        cache = _make_cache()
        result = classify_event_full(meeting, cache)
        assert result.history_available is True


# ── External followup classification ──────────────────────────────────────────

class TestExternalFollowupClassification:

    def test_domain_in_cache_gives_followup(self):
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "bob@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="Acme Follow-up Call",
            attendee_emails=["me@kakaoventures.co.kr", "cfo@acme.com"],
            external_emails=["cfo@acme.com"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FOLLOWUP

    def test_email_in_cache_gives_followup(self):
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "alice@partner.com"],
            attendee_domains=["kakaoventures.co.kr", "partner.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="Catch-up",
            attendee_emails=["me@kakaoventures.co.kr", "alice@partner.com"],
            external_emails=["alice@partner.com"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FOLLOWUP

    def test_one_of_multiple_external_in_cache_gives_followup(self):
        """If at least one external attendee/domain is in cache → FOLLOWUP."""
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "known@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="Multi-partner Review",
            attendee_emails=["me@kakaoventures.co.kr",
                             "known@acme.com",
                             "new@freshco.io"],
            external_emails=["known@acme.com", "new@freshco.io"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FOLLOWUP

    def test_history_matched_emails_populated(self):
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "alice@corp.com"],
            attendee_domains=["kakaoventures.co.kr", "corp.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="Follow-up",
            attendee_emails=["me@kakaoventures.co.kr", "alice@corp.com"],
            external_emails=["alice@corp.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, cache)
        assert "alice@corp.com" in result.history_matched_emails

    def test_history_matched_domains_populated(self):
        cached = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "x@bigdeal.com"],
            attendee_domains=["kakaoventures.co.kr", "bigdeal.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="BigDeal Check-in",
            attendee_emails=["me@kakaoventures.co.kr", "cto@bigdeal.com"],
            external_emails=["cto@bigdeal.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, cache)
        assert "bigdeal.com" in result.history_matched_domains

    def test_past_meeting_counts_keyed_by_domain(self):
        ev1 = _make_cached_event(
            event_id="h1",
            attendee_emails=["me@kakaoventures.co.kr", "a@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        ev2 = _make_cached_event(
            event_id="h2",
            attendee_emails=["me@kakaoventures.co.kr", "b@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        cache = _make_cache(ev1, ev2)

        meeting = _make_meeting(
            summary="Acme Q2 Review",
            attendee_emails=["me@kakaoventures.co.kr", "ceo@acme.com"],
            external_emails=["ceo@acme.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, cache)
        assert "acme.com" in result.past_meeting_counts
        assert result.past_meeting_counts["acme.com"] == 2

    def test_last_meeting_with_populated(self):
        cached = _make_cached_event(
            event_id="h_latest",
            title="Previous Acme Call",
            start_iso="2025-10-01T10:00:00+00:00",
            attendee_emails=["me@kakaoventures.co.kr", "x@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        cache = _make_cache(cached)

        meeting = _make_meeting(
            summary="Acme Follow-up",
            attendee_emails=["me@kakaoventures.co.kr", "cfo@acme.com"],
            external_emails=["cfo@acme.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, cache)
        assert len(result.last_meeting_with) >= 1
        titles = [ev.title for ev in result.last_meeting_with]
        assert "Previous Acme Call" in titles

    def test_total_past_meetings_sums_correctly(self):
        ev_acme = _make_cached_event(
            event_id="h_acme",
            attendee_emails=["me@kakaoventures.co.kr", "x@acme.com"],
            attendee_domains=["kakaoventures.co.kr", "acme.com"],
        )
        ev_beta = _make_cached_event(
            event_id="h_beta",
            attendee_emails=["me@kakaoventures.co.kr", "y@beta.io"],
            attendee_domains=["kakaoventures.co.kr", "beta.io"],
        )
        cache = _make_cache(ev_acme, ev_beta)

        meeting = _make_meeting(
            summary="Multi-corp review",
            attendee_emails=["me@kakaoventures.co.kr",
                             "a@acme.com",
                             "b@beta.io"],
            external_emails=["a@acme.com", "b@beta.io"],
            is_external=True,
        )
        result = classify_event_full(meeting, cache)
        assert result.total_past_meetings >= 2


# ── ClassificationResult fields ───────────────────────────────────────────────

class TestClassificationResultFields:

    def test_event_id_populated(self):
        meeting = _make_meeting(event_id="unique_123", summary="Standup")
        result = classify_event_full(meeting)
        assert result.event_id == "unique_123"

    def test_title_populated(self):
        meeting = _make_meeting(summary="Client Intro Call",
                                external_emails=["x@co.com"],
                                is_external=True)
        result = classify_event_full(meeting)
        assert result.title == "Client Intro Call"

    def test_external_attendee_emails_populated(self):
        meeting = _make_meeting(
            summary="Deal Review",
            attendee_emails=["me@kakaoventures.co.kr", "founder@startup.com"],
            external_emails=["founder@startup.com"],
            is_external=True,
        )
        result = classify_event_full(meeting)
        assert "founder@startup.com" in result.external_attendee_emails

    def test_external_attendee_emails_lowercased(self):
        """Email addresses stored in external_attendee_emails must be lowercased."""
        meeting = _make_meeting(
            summary="Deal Review",
            external_emails=["Founder@STARTUP.COM"],
            is_external=True,
        )
        result = classify_event_full(meeting)
        assert "founder@startup.com" in result.external_attendee_emails

    def test_debug_notes_nonempty(self):
        meeting = _make_meeting(summary="OKR Review",
                                attendee_emails=["a@kakaoventures.co.kr"])
        result = classify_event_full(meeting)
        assert len(result.debug_notes) > 0

    def test_debug_notes_contain_title_label(self):
        meeting = _make_meeting(summary="OKR Review",
                                attendee_emails=["a@kakaoventures.co.kr"])
        result = classify_event_full(meeting)
        notes_text = " ".join(result.debug_notes)
        assert "title_label" in notes_text

    def test_title_label_internal_for_internal_title(self):
        meeting = _make_meeting(summary="Sprint Retro",
                                attendee_emails=["a@kakaoventures.co.kr"])
        result = classify_event_full(meeting)
        assert result.title_label == MeetingLabel.INTERNAL

    def test_title_label_external_for_external_title(self):
        meeting = _make_meeting(summary="IR Pitch",
                                external_emails=["x@vc.com"],
                                is_external=True)
        result = classify_event_full(meeting)
        assert result.title_label == MeetingLabel.EXTERNAL

    def test_title_label_unknown_for_unclassified_title(self):
        meeting = _make_meeting(summary="Lunch",
                                external_emails=["x@co.com"],
                                is_external=True)
        result = classify_event_full(meeting)
        assert result.title_label == MeetingLabel.UNKNOWN

    def test_is_external_by_attendees_true(self):
        meeting = _make_meeting(
            summary="Partner Demo",
            external_emails=["x@ext.com"],
            is_external=True,
        )
        result = classify_event_full(meeting)
        assert result.is_external_by_attendees is True

    def test_is_external_by_attendees_false(self):
        meeting = _make_meeting(
            summary="Team Sync",
            attendee_emails=["a@kakaoventures.co.kr"],
            is_external=False,
        )
        result = classify_event_full(meeting)
        assert result.is_external_by_attendees is False


# ── Attendee email with empty string ──────────────────────────────────────────

class TestEdgeCases:

    def test_attendee_with_empty_email_skipped(self):
        """Attendees with empty email addresses must not cause errors."""
        att_no_email = MagicMock()
        att_no_email.email = ""
        att_no_email.is_internal = False

        meeting = MagicMock()
        meeting.event_id = "e_empty"
        meeting.summary = "Some Meeting"
        meeting.is_external = False  # no valid external attendees
        meeting.external_attendees = [att_no_email]

        # Should not raise
        result = classify_event_full(meeting)
        assert result.category == EventCategory.INTERNAL

    def test_none_cache_explicitly_passed(self):
        meeting = _make_meeting(
            summary="외부 미팅",
            external_emails=["x@new.io"],
            is_external=True,
        )
        result = classify_event_full(meeting, history_cache=None)
        assert result.category == EventCategory.EXTERNAL_FIRST
        assert result.history_available is False

    def test_duplicate_external_emails_deduplicated(self):
        """Duplicate emails in external_attendees should not cause double hits."""
        meeting = _make_meeting(
            summary="Demo",
            external_emails=["alice@partner.com", "alice@partner.com"],
            is_external=True,
        )
        empty_cache = _make_cache()
        result = classify_event_full(meeting, empty_cache)
        # deduplicated: only one unique email
        assert result.external_attendee_emails.count("alice@partner.com") == 1

    def test_multiple_domains_all_new_gives_first(self):
        """All external domains are new → EXTERNAL_FIRST."""
        ev = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "old@oldco.com"],
            attendee_domains=["kakaoventures.co.kr", "oldco.com"],
        )
        cache = _make_cache(ev)

        meeting = _make_meeting(
            summary="Intro Call",
            external_emails=["x@newco1.com", "y@newco2.com"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FIRST

    def test_history_matched_emails_empty_for_first(self):
        meeting = _make_meeting(
            summary="New Partner",
            external_emails=["z@brand-new.com"],
            is_external=True,
        )
        result = classify_event_full(meeting, _make_cache())
        assert result.history_matched_emails == []
        assert result.history_matched_domains == []


# ── Korean meeting titles integration ─────────────────────────────────────────

class TestKoreanTitles:

    def test_korean_internal_title(self):
        meeting = _make_meeting(summary="주간 팀 싱크",
                                attendee_emails=["a@kakaoventures.co.kr"])
        assert classify_event(meeting) == EventCategory.INTERNAL

    def test_korean_external_title_no_history(self):
        meeting = _make_meeting(
            summary="투자 미팅",
            external_emails=["founder@newco.kr"],
            is_external=True,
        )
        assert classify_event(meeting, _make_cache()) == EventCategory.EXTERNAL_FIRST

    def test_korean_external_title_with_history(self):
        ev = _make_cached_event(
            attendee_emails=["me@kakaoventures.co.kr", "x@portfolio.kr"],
            attendee_domains=["kakaoventures.co.kr", "portfolio.kr"],
        )
        cache = _make_cache(ev)

        meeting = _make_meeting(
            summary="포트폴리오 미팅",
            external_emails=["ceo@portfolio.kr"],
            is_external=True,
        )
        assert classify_event(meeting, cache) == EventCategory.EXTERNAL_FOLLOWUP


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
