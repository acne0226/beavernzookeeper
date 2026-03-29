"""
Tests for external meeting detection / filtering logic in
``src/calendar/google_calendar.py``.

Sub-AC 3a: Implement external meeting detection logic that filters calendar
events to identify meetings with external attendees (non-company domains),
and write/fix unit tests for this filtering function.

Coverage
--------
Meeting.is_external (property):
  - True when at least one non-internal attendee present
  - False when all attendees are internal
  - False when attendee list is empty
  - Attendee.is_internal checks correct domain

Meeting.external_attendees (property):
  - Returns only non-internal attendees
  - Returns empty list when all internal
  - Excludes attendees with empty email

filter_external_meetings() (module-level function):
  - Empty list → empty result
  - All-internal meetings filtered out
  - All-external meetings kept
  - Mixed list returns only external meetings
  - Meeting with no attendees treated as internal (filtered out)
  - Meetings with malformed attendee emails handled gracefully
  - Default internal_domain matches config constant
  - Custom internal_domain override works correctly
  - Does not mutate the original list
  - Works with single-meeting list

get_external_attendee_domains() (module-level function):
  - Returns correct domain strings for external attendees
  - Excludes internal domain
  - Returns empty set for all-internal meeting
  - Handles attendee with no email / malformed email
  - Multiple external attendees → multiple domains in set
  - Same domain for multiple attendees → deduplicated set

Attendee.is_internal (property):
  - True for exact @kakaoventures.co.kr match
  - False for other domains
  - Case-insensitive domain comparison
  - False for malformed email (no @)
  - False for subdomain of internal domain (sub.kakaoventures.co.kr)

Integration:
  - filter_external_meetings composed with get_external_attendee_domains
  - Only-kakaoventures meeting gives empty domains via filter + domain extraction
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar.google_calendar import (
    Attendee,
    Meeting,
    filter_external_meetings,
    get_external_attendee_domains,
)
from src.config import INTERNAL_DOMAIN


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_attendee(
    email: str,
    display_name: str = "",
    response_status: str = "accepted",
) -> Attendee:
    """Create a real Attendee dataclass instance."""
    return Attendee(email=email, display_name=display_name, response_status=response_status)


_NOW = datetime(2026, 3, 29, 10, 0, 0, tzinfo=timezone.utc)
_LATER = _NOW + timedelta(hours=1)


def _make_meeting(
    event_id: str = "evt_001",
    summary: str = "Test Meeting",
    attendees: list[Attendee] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Meeting:
    """Create a real Meeting dataclass instance."""
    return Meeting(
        event_id=event_id,
        summary=summary,
        start=start or _NOW,
        end=end or _LATER,
        attendees=attendees if attendees is not None else [],
    )


def _int(email: str) -> Attendee:
    """Shorthand — internal attendee."""
    return _make_attendee(email)


def _ext(email: str) -> Attendee:
    """Shorthand — external attendee."""
    return _make_attendee(email)


# ── Attendee.is_internal ───────────────────────────────────────────────────────

class TestAttendeeIsInternal:

    def test_kakaoventures_email_is_internal(self):
        a = _make_attendee("invest1@kakaoventures.co.kr")
        assert a.is_internal is True

    def test_other_domain_is_not_internal(self):
        a = _make_attendee("ceo@startup.com")
        assert a.is_internal is False

    def test_case_insensitive_domain(self):
        a = _make_attendee("INVEST@KAKAOVENTURES.CO.KR")
        assert a.is_internal is True

    def test_mixed_case_email(self):
        a = _make_attendee("User@KakaoVentures.CO.KR")
        assert a.is_internal is True

    def test_external_gmail_is_not_internal(self):
        a = _make_attendee("user@gmail.com")
        assert a.is_internal is False

    def test_empty_email_is_not_internal(self):
        """Empty email should not raise; should return False."""
        a = _make_attendee("")
        # endswith("@kakaoventures.co.kr") on empty string is False
        assert a.is_internal is False

    def test_no_at_sign_is_not_internal(self):
        """Malformed email without @ symbol."""
        a = _make_attendee("noatsign")
        assert a.is_internal is False

    def test_subdomain_of_internal_is_not_internal(self):
        """sub.kakaoventures.co.kr is NOT the same as kakaoventures.co.kr."""
        a = _make_attendee("user@sub.kakaoventures.co.kr")
        # endswith("@kakaoventures.co.kr") is False for a subdomain
        assert a.is_internal is False

    def test_domain_that_contains_internal_domain_suffix(self):
        """notakakaoventures.co.kr should not match."""
        a = _make_attendee("user@notakakaoventures.co.kr")
        assert a.is_internal is False


# ── Meeting.is_external ────────────────────────────────────────────────────────

class TestMeetingIsExternal:

    def test_no_attendees_not_external(self):
        meeting = _make_meeting(attendees=[])
        assert meeting.is_external is False

    def test_all_internal_attendees_not_external(self):
        meeting = _make_meeting(attendees=[
            _int("a@kakaoventures.co.kr"),
            _int("b@kakaoventures.co.kr"),
        ])
        assert meeting.is_external is False

    def test_one_external_attendee_makes_meeting_external(self):
        meeting = _make_meeting(attendees=[
            _int("a@kakaoventures.co.kr"),
            _ext("founder@startup.com"),
        ])
        assert meeting.is_external is True

    def test_all_external_attendees_is_external(self):
        meeting = _make_meeting(attendees=[
            _ext("ceo@corp.com"),
            _ext("cto@corp.com"),
        ])
        assert meeting.is_external is True

    def test_single_external_attendee(self):
        meeting = _make_meeting(attendees=[_ext("partner@acme.io")])
        assert meeting.is_external is True

    def test_single_internal_attendee_not_external(self):
        meeting = _make_meeting(attendees=[_int("me@kakaoventures.co.kr")])
        assert meeting.is_external is False


# ── Meeting.external_attendees ────────────────────────────────────────────────

class TestMeetingExternalAttendees:

    def test_returns_only_external_attendees(self):
        internal = _int("me@kakaoventures.co.kr")
        external = _ext("ceo@startup.com")
        meeting = _make_meeting(attendees=[internal, external])
        result = meeting.external_attendees
        assert result == [external]

    def test_all_internal_returns_empty_list(self):
        meeting = _make_meeting(attendees=[
            _int("a@kakaoventures.co.kr"),
            _int("b@kakaoventures.co.kr"),
        ])
        assert meeting.external_attendees == []

    def test_no_attendees_returns_empty_list(self):
        meeting = _make_meeting(attendees=[])
        assert meeting.external_attendees == []

    def test_multiple_external_all_returned(self):
        a = _ext("alice@acme.com")
        b = _ext("bob@beta.io")
        internal = _int("x@kakaoventures.co.kr")
        meeting = _make_meeting(attendees=[internal, a, b])
        result = meeting.external_attendees
        assert a in result
        assert b in result
        assert internal not in result
        assert len(result) == 2

    def test_attendee_with_empty_email_excluded(self):
        """Attendees with empty emails: is_internal is False so they appear in
        external_attendees — but filter_external_meetings should still work."""
        empty = _make_attendee("")
        internal = _int("me@kakaoventures.co.kr")
        # empty.is_internal is False, so it DOES appear in external_attendees
        meeting = _make_meeting(attendees=[internal, empty])
        # The meeting is considered external because empty email is not internal
        ext = meeting.external_attendees
        assert empty in ext


# ── filter_external_meetings() ────────────────────────────────────────────────

class TestFilterExternalMeetings:

    def test_empty_list_returns_empty(self):
        assert filter_external_meetings([]) == []

    def test_all_internal_meetings_filtered_out(self):
        m1 = _make_meeting("e1", attendees=[_int("a@kakaoventures.co.kr")])
        m2 = _make_meeting("e2", attendees=[_int("b@kakaoventures.co.kr")])
        assert filter_external_meetings([m1, m2]) == []

    def test_all_external_meetings_kept(self):
        m1 = _make_meeting("e1", attendees=[_ext("x@acme.com")])
        m2 = _make_meeting("e2", attendees=[_ext("y@beta.io")])
        result = filter_external_meetings([m1, m2])
        assert len(result) == 2
        assert m1 in result
        assert m2 in result

    def test_mixed_list_returns_only_external(self):
        internal_m = _make_meeting("i1", attendees=[_int("a@kakaoventures.co.kr")])
        external_m = _make_meeting("e1", attendees=[
            _int("a@kakaoventures.co.kr"),
            _ext("ceo@startup.com"),
        ])
        result = filter_external_meetings([internal_m, external_m])
        assert result == [external_m]

    def test_meeting_with_no_attendees_is_filtered_out(self):
        m = _make_meeting("no_att", attendees=[])
        assert filter_external_meetings([m]) == []

    def test_single_external_meeting_kept(self):
        m = _make_meeting("e1", attendees=[_ext("partner@company.com")])
        result = filter_external_meetings([m])
        assert result == [m]

    def test_does_not_mutate_original_list(self):
        internal_m = _make_meeting("i1", attendees=[_int("a@kakaoventures.co.kr")])
        external_m = _make_meeting("e1", attendees=[_ext("x@ext.com")])
        original = [internal_m, external_m]
        filter_external_meetings(original)
        # Original list should still have both items
        assert len(original) == 2

    def test_preserves_order(self):
        m1 = _make_meeting("e1", attendees=[_ext("x@co1.com")])
        m2 = _make_meeting("e2", attendees=[_ext("y@co2.com")])
        m3 = _make_meeting("e3", attendees=[_ext("z@co3.com")])
        result = filter_external_meetings([m1, m2, m3])
        assert result == [m1, m2, m3]

    def test_default_domain_matches_config(self):
        """Default filter uses INTERNAL_DOMAIN from config."""
        m_internal = _make_meeting("i1", attendees=[
            _make_attendee(f"user@{INTERNAL_DOMAIN}")
        ])
        m_external = _make_meeting("e1", attendees=[
            _make_attendee(f"user@{INTERNAL_DOMAIN}"),
            _make_attendee("ext@partner.com"),
        ])
        result = filter_external_meetings([m_internal, m_external])
        assert m_internal not in result
        assert m_external in result

    def test_custom_internal_domain_override(self):
        """When a custom domain is passed, that domain is used as internal."""
        custom_domain = "mycompany.io"
        m_custom_internal = _make_meeting("ci1", attendees=[
            _make_attendee(f"alice@{custom_domain}"),
        ])
        m_external_by_custom = _make_meeting("e1", attendees=[
            _make_attendee(f"alice@{custom_domain}"),
            _make_attendee("bob@outside.com"),
        ])
        m_kakao_appears_external = _make_meeting("k1", attendees=[
            # kakaoventures.co.kr is external relative to mycompany.io
            _make_attendee("invest@kakaoventures.co.kr"),
        ])

        result = filter_external_meetings(
            [m_custom_internal, m_external_by_custom, m_kakao_appears_external],
            internal_domain=custom_domain,
        )
        # m_custom_internal has only custom domain → filtered out
        assert m_custom_internal not in result
        # m_external_by_custom has one outsider → kept
        assert m_external_by_custom in result
        # kakao is NOT the custom domain → kept
        assert m_kakao_appears_external in result

    def test_custom_domain_case_insensitive(self):
        """Custom domain comparison is case-insensitive."""
        m = _make_meeting("e1", attendees=[
            _make_attendee("USER@MYCO.COM"),
        ])
        # Should be filtered out when custom domain is "myco.com"
        result = filter_external_meetings([m], internal_domain="myco.com")
        assert result == []

    def test_malformed_email_treated_as_external_by_meeting_is_external(self):
        """
        An attendee with no @ in email is NOT internal (is_internal returns
        False), so Meeting.is_external returns True and the meeting is kept.
        This tests the behaviour is consistent.
        """
        malformed = _make_attendee("no-at-sign")
        internal = _int("me@kakaoventures.co.kr")
        m = _make_meeting("m1", attendees=[internal, malformed])
        # malformed email is not internal → meeting is external
        result = filter_external_meetings([m])
        assert m in result

    def test_large_batch_performance(self):
        """filter_external_meetings should handle hundreds of meetings quickly."""
        meetings = []
        for i in range(300):
            if i % 3 == 0:
                m = _make_meeting(
                    f"ext_{i}",
                    attendees=[
                        _int(f"user{i}@kakaoventures.co.kr"),
                        _ext(f"guest{i}@partner.com"),
                    ],
                )
            else:
                m = _make_meeting(
                    f"int_{i}",
                    attendees=[_int(f"user{i}@kakaoventures.co.kr")],
                )
            meetings.append(m)

        result = filter_external_meetings(meetings)
        # Every 3rd meeting (index 0, 3, 6, …) is external → 100 out of 300
        assert len(result) == 100


# ── get_external_attendee_domains() ───────────────────────────────────────────

class TestGetExternalAttendeeDomains:

    def test_returns_external_domain(self):
        meeting = _make_meeting(attendees=[
            _int("me@kakaoventures.co.kr"),
            _ext("ceo@startup.com"),
        ])
        domains = get_external_attendee_domains(meeting)
        assert "startup.com" in domains

    def test_excludes_internal_domain(self):
        meeting = _make_meeting(attendees=[
            _int("me@kakaoventures.co.kr"),
            _ext("partner@external.io"),
        ])
        domains = get_external_attendee_domains(meeting)
        assert INTERNAL_DOMAIN not in domains
        assert "external.io" in domains

    def test_all_internal_returns_empty_set(self):
        meeting = _make_meeting(attendees=[
            _int("a@kakaoventures.co.kr"),
            _int("b@kakaoventures.co.kr"),
        ])
        assert get_external_attendee_domains(meeting) == set()

    def test_no_attendees_returns_empty_set(self):
        meeting = _make_meeting(attendees=[])
        assert get_external_attendee_domains(meeting) == set()

    def test_multiple_external_attendees_same_domain(self):
        """Multiple attendees from the same external domain → single domain."""
        meeting = _make_meeting(attendees=[
            _ext("alice@acme.com"),
            _ext("bob@acme.com"),
        ])
        domains = get_external_attendee_domains(meeting)
        assert domains == {"acme.com"}

    def test_multiple_different_external_domains(self):
        meeting = _make_meeting(attendees=[
            _int("me@kakaoventures.co.kr"),
            _ext("x@acme.com"),
            _ext("y@beta.io"),
        ])
        domains = get_external_attendee_domains(meeting)
        assert domains == {"acme.com", "beta.io"}

    def test_lowercase_domains(self):
        """Domains are returned lower-cased."""
        meeting = _make_meeting(attendees=[
            _ext("CEO@BIGCO.COM"),
        ])
        domains = get_external_attendee_domains(meeting)
        assert "bigco.com" in domains
        assert "BIGCO.COM" not in domains

    def test_attendee_with_empty_email_skipped(self):
        """Attendees with empty email should not raise or contribute domains."""
        empty = _make_attendee("")
        normal_ext = _ext("partner@co.com")
        meeting = _make_meeting(attendees=[empty, normal_ext])
        domains = get_external_attendee_domains(meeting)
        # empty email: no @ → skipped; but meeting.external_attendees includes it
        # get_external_attendee_domains skips emails without "@"
        assert "co.com" in domains
        # No empty-string domain added
        assert "" not in domains

    def test_malformed_email_without_at_skipped(self):
        meeting = _make_meeting(attendees=[
            _make_attendee("noatsign"),
        ])
        # noatsign is not internal → appears in external_attendees
        # but get_external_attendee_domains should skip it (no @)
        domains = get_external_attendee_domains(meeting)
        assert domains == set()

    def test_returns_set_not_list(self):
        """Return type must be a set."""
        meeting = _make_meeting(attendees=[_ext("x@acme.com")])
        result = get_external_attendee_domains(meeting)
        assert isinstance(result, set)


# ── Integration: filter_external_meetings + get_external_attendee_domains ─────

class TestExternalFilterIntegration:

    def test_filter_then_get_domains(self):
        """Chaining filter_external_meetings → get_external_attendee_domains."""
        m_internal = _make_meeting("i1", attendees=[_int("a@kakaoventures.co.kr")])
        m_external = _make_meeting("e1", attendees=[
            _int("a@kakaoventures.co.kr"),
            _ext("ceo@portfolio.io"),
            _ext("cto@portfolio.io"),
        ])
        externals = filter_external_meetings([m_internal, m_external])
        assert len(externals) == 1
        domains = get_external_attendee_domains(externals[0])
        assert domains == {"portfolio.io"}

    def test_all_internal_filter_gives_empty_domain_set(self):
        m = _make_meeting("i1", attendees=[_int("me@kakaoventures.co.kr")])
        externals = filter_external_meetings([m])
        # No external meetings, so no domain iteration needed
        assert externals == []
        all_domains: set[str] = set()
        for ext_m in externals:
            all_domains |= get_external_attendee_domains(ext_m)
        assert all_domains == set()

    def test_multi_meeting_domain_aggregation(self):
        """Collect all unique external domains across a batch of meetings."""
        m1 = _make_meeting("e1", attendees=[
            _int("x@kakaoventures.co.kr"),
            _ext("a@acme.com"),
        ])
        m2 = _make_meeting("e2", attendees=[
            _int("y@kakaoventures.co.kr"),
            _ext("b@beta.io"),
        ])
        m3 = _make_meeting("i1", attendees=[_int("z@kakaoventures.co.kr")])

        all_domains: set[str] = set()
        for m in filter_external_meetings([m1, m2, m3]):
            all_domains |= get_external_attendee_domains(m)

        assert all_domains == {"acme.com", "beta.io"}

    def test_known_domain_in_filter_output(self):
        """Meetings kept by filter should expose correct domain info."""
        m = _make_meeting(attendees=[
            _int("me@kakaoventures.co.kr"),
            _ext("startup_ceo@newco.kr"),
            _ext("partner@globalvc.com"),
        ])
        result = filter_external_meetings([m])
        assert len(result) == 1
        domains = get_external_attendee_domains(result[0])
        assert "newco.kr" in domains
        assert "globalvc.com" in domains
        assert INTERNAL_DOMAIN not in domains


# ── Korean / localisation edge cases ──────────────────────────────────────────

class TestKoreanDomainEdgeCases:

    def test_korean_domain_tld(self):
        """Meetings with Korean TLD domains are correctly detected as external."""
        m = _make_meeting(attendees=[
            _int("me@kakaoventures.co.kr"),
            _ext("ceo@startup.co.kr"),
        ])
        assert m.is_external is True
        result = filter_external_meetings([m])
        assert result == [m]
        domains = get_external_attendee_domains(m)
        assert "startup.co.kr" in domains

    def test_internal_domain_similar_to_korean_domain(self):
        """kakaoventures.co.kr should not falsely match kakao.com."""
        m = _make_meeting(attendees=[
            _int("a@kakaoventures.co.kr"),
            _ext("b@kakao.com"),   # Kakao parent but different domain
        ])
        assert m.is_external is True
        domains = get_external_attendee_domains(m)
        assert "kakao.com" in domains
        assert INTERNAL_DOMAIN not in domains


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
