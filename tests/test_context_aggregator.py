"""
Tests for the Meeting Context Aggregator (Sub-AC 2b).

Covers:
- RawBriefingContent construction and derived properties
- AttendeeProfile building from Meeting attendees
- Calendar history enrichment (stats per attendee)
- Gmail thread fetching (happy path, failure, no client)
- Notion records fetching (happy path, failure, no client)
- Full aggregate() orchestration
- GmailClient helpers: keyword extraction, body parsing, domain root
- NotionClient helpers: schema discovery mocking, company search
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── System under test ────────────────────────────────────────────────────────
from src.briefing.context_aggregator import (
    MeetingContextAggregator,
    RawBriefingContent,
    AttendeeProfile,
    AggregationError,
    _email_domain,
)
from src.gmail.gmail_client import (
    GmailClient,
    EmailMessage,
    EmailThread,
    _extract_keywords,
    _extract_body_text,
    _parse_email_date,
    _extract_address,
)
from src.notion.notion_client import (
    NotionClient,
    NotionRecord,
    DatabaseSchema,
    _domain_root as notion_domain_root,
    _title_keywords,
    _get_title_value,
    _get_rich_text_value,
    _get_select_value,
    _get_date_value,
)
from src.calendar.google_calendar import Meeting, Attendee


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_attendee(
    email: str,
    display_name: str = "",
    internal: bool = False,
    response_status: str = "accepted",
) -> Attendee:
    a = Attendee(email=email, display_name=display_name, response_status=response_status)
    # Patch is_internal via property — just monkey-patch the object
    type(a)  # access class
    return a


def _make_meeting(
    summary: str = "외부 미팅",
    external_emails: list[str] | None = None,
    internal_emails: list[str] | None = None,
    start_offset_minutes: int = 10,
    description: str = "",
    location: str = "",
) -> Meeting:
    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=start_offset_minutes)
    end = start + timedelta(hours=1)

    attendees: list[Attendee] = []
    for email in (external_emails or ["ceo@startup.com"]):
        attendees.append(Attendee(email=email, display_name="External Person", response_status="accepted"))
    for email in (internal_emails or ["invest1@kakaoventures.co.kr"]):
        attendees.append(Attendee(email=email, display_name="Internal Person", response_status="accepted"))

    return Meeting(
        event_id="evt-001",
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
        description=description,
        location=location,
        html_link="https://calendar.google.com/event/evt-001",
        organizer_email="invest1@kakaoventures.co.kr",
    )


def _make_email_thread(
    thread_id: str = "t1",
    subject: str = "Re: Partnership",
    days_ago: int = 5,
) -> EmailThread:
    date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    msg = EmailMessage(
        message_id="m1",
        thread_id=thread_id,
        subject=subject,
        sender="ceo@startup.com",
        recipients=["invest1@kakaoventures.co.kr"],
        date=date,
        snippet="Looking forward to the meeting",
        body_text="Hi, looking forward to our discussion.",
    )
    return EmailThread(thread_id=thread_id, subject=subject, messages=[msg])


def _make_notion_record(
    page_id: str = "page-001",
    title: str = "Startup Inc.",
    company_name: str = "Startup Inc.",
    status: str = "검토중",
) -> NotionRecord:
    return NotionRecord(
        page_id=page_id,
        url=f"https://notion.so/{page_id}",
        title=title,
        company_name=company_name,
        status=status,
        date_value="2025-01-15",
        properties={"title_field": title, "status_field": status},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RawBriefingContent
# ══════════════════════════════════════════════════════════════════════════════

class TestRawBriefingContent:

    def _make_content(self) -> RawBriefingContent:
        now = datetime.now(timezone.utc)
        return RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="외부 미팅",
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
        )

    def test_duration_minutes(self):
        c = self._make_content()
        assert c.duration_minutes == 60

    def test_external_internal_split(self):
        c = self._make_content()
        ext = AttendeeProfile(email="a@ext.com", is_internal=False)
        intern_ = AttendeeProfile(email="b@kakaoventures.co.kr", is_internal=True)
        c.attendee_profiles = [ext, intern_]

        assert len(c.external_attendees) == 1
        assert c.external_attendees[0].email == "a@ext.com"
        assert len(c.internal_attendees) == 1

    def test_has_errors_false_by_default(self):
        c = self._make_content()
        assert c.has_errors is False

    def test_has_errors_true_when_errors_present(self):
        c = self._make_content()
        c.errors.append(AggregationError(source="gmail", message="timeout"))
        assert c.has_errors is True

    def test_to_dict_keys(self):
        c = self._make_content()
        d = c.to_dict()
        for key in [
            "meeting_id", "meeting_title", "meeting_start", "meeting_end",
            "duration_minutes", "attendee_profiles", "gmail_threads",
            "gmail_available", "notion_records", "notion_available",
            "errors", "aggregated_at",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_gmail_threads_serialized(self):
        c = self._make_content()
        c.gmail_threads = [_make_email_thread()]
        d = c.to_dict()
        assert len(d["gmail_threads"]) == 1
        assert "thread_id" in d["gmail_threads"][0]

    def test_to_dict_notion_records_serialized(self):
        c = self._make_content()
        c.notion_records = [_make_notion_record()]
        d = c.to_dict()
        assert len(d["notion_records"]) == 1
        assert "page_id" in d["notion_records"][0]


# ══════════════════════════════════════════════════════════════════════════════
#  AttendeeProfile
# ══════════════════════════════════════════════════════════════════════════════

class TestAttendeeProfile:

    def test_to_dict_fields(self):
        now = datetime.now(timezone.utc)
        p = AttendeeProfile(
            email="ceo@startup.com",
            display_name="Jane CEO",
            is_internal=False,
            company_domain="startup.com",
            past_meeting_count=3,
            last_met_date=now - timedelta(days=30),
            past_meeting_titles=["Meeting A", "Meeting B", "Meeting C"],
        )
        d = p.to_dict()
        assert d["email"] == "ceo@startup.com"
        assert d["is_internal"] is False
        assert d["past_meeting_count"] == 3
        assert d["last_met_date"] is not None
        assert len(d["past_meeting_titles"]) == 3

    def test_past_meeting_titles_capped_at_5(self):
        p = AttendeeProfile(
            email="a@b.com",
            past_meeting_titles=["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
        )
        d = p.to_dict()
        assert len(d["past_meeting_titles"]) == 5

    def test_to_dict_no_last_met_date(self):
        p = AttendeeProfile(email="x@y.com")
        d = p.to_dict()
        assert d["last_met_date"] is None


# ══════════════════════════════════════════════════════════════════════════════
#  AggregationError
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregationError:

    def test_to_dict(self):
        err = AggregationError(source="gmail", message="timeout")
        d = err.to_dict()
        assert d["source"] == "gmail"
        assert d["message"] == "timeout"
        assert "timestamp" in d


# ══════════════════════════════════════════════════════════════════════════════
#  _email_domain helper
# ══════════════════════════════════════════════════════════════════════════════

class TestEmailDomainHelper:

    def test_normal_email(self):
        assert _email_domain("user@example.com") == "example.com"

    def test_empty_string(self):
        assert _email_domain("") == ""

    def test_no_at_sign(self):
        assert _email_domain("notanemail") == ""

    def test_lowercases(self):
        assert _email_domain("User@EXAMPLE.COM") == "example.com"


# ══════════════════════════════════════════════════════════════════════════════
#  MeetingContextAggregator — attendee profile building
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorAttendeeProfiles:

    def _make_aggregator(self):
        return MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
        )

    def test_profiles_created_for_all_attendees(self):
        agg = self._make_aggregator()
        meeting = _make_meeting(
            external_emails=["ceo@startup.com", "cto@startup.com"],
            internal_emails=["invest1@kakaoventures.co.kr"],
        )
        profiles = agg._build_attendee_profiles(meeting)
        assert len(profiles) == 3

    def test_external_email_not_internal(self):
        agg = self._make_aggregator()
        meeting = _make_meeting(external_emails=["ceo@startup.com"])
        profiles = agg._build_attendee_profiles(meeting)
        ext_profile = next(p for p in profiles if p.email == "ceo@startup.com")
        assert ext_profile.is_internal is False

    def test_internal_email_is_internal(self):
        agg = self._make_aggregator()
        meeting = _make_meeting(internal_emails=["invest1@kakaoventures.co.kr"])
        profiles = agg._build_attendee_profiles(meeting)
        int_profile = next(p for p in profiles if "kakaoventures" in p.email)
        assert int_profile.is_internal is True

    def test_company_domain_extracted(self):
        agg = self._make_aggregator()
        meeting = _make_meeting(external_emails=["ceo@acme.io"])
        profiles = agg._build_attendee_profiles(meeting)
        ext_profile = next(p for p in profiles if p.email == "ceo@acme.io")
        assert ext_profile.company_domain == "acme.io"

    def test_display_name_preserved(self):
        agg = self._make_aggregator()
        meeting = _make_meeting()
        # meeting fixture sets display_name="External Person"
        profiles = agg._build_attendee_profiles(meeting)
        ext = next(p for p in profiles if not p.is_internal)
        assert ext.display_name == "External Person"

    def test_history_starts_at_zero(self):
        agg = self._make_aggregator()
        meeting = _make_meeting()
        profiles = agg._build_attendee_profiles(meeting)
        for p in profiles:
            assert p.past_meeting_count == 0
            assert p.last_met_date is None


# ══════════════════════════════════════════════════════════════════════════════
#  MeetingContextAggregator — calendar history enrichment
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorCalendarHistory:

    def _make_past_meeting(self, attendee_email: str, days_ago: int, title: str) -> Meeting:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days_ago)
        return Meeting(
            event_id=f"past-{days_ago}",
            summary=title,
            start=start,
            end=start + timedelta(hours=1),
            attendees=[Attendee(email=attendee_email, display_name="Ext")],
        )

    def test_enriches_attendee_with_past_meetings(self):
        mock_calendar = MagicMock()
        past1 = self._make_past_meeting("ceo@startup.com", 10, "1차 미팅")
        past2 = self._make_past_meeting("ceo@startup.com", 20, "2차 미팅")
        mock_calendar.list_historical_external_meetings.return_value = [past1, past2]

        agg = MeetingContextAggregator(calendar_client=mock_calendar)
        meeting = _make_meeting(external_emails=["ceo@startup.com"])
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="Test",
            meeting_start=datetime.now(timezone.utc),
            meeting_end=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        content.attendee_profiles = agg._build_attendee_profiles(meeting)
        agg._enrich_with_calendar_history(meeting, content)

        ext_profile = next(
            p for p in content.attendee_profiles
            if p.email == "ceo@startup.com"
        )
        assert ext_profile.past_meeting_count == 2
        assert ext_profile.last_met_date is not None
        # Most recent meeting: 10 days ago is more recent than 20 days ago
        expected_latest = past1.start
        assert ext_profile.last_met_date == expected_latest

    def test_skips_current_meeting_in_history(self):
        mock_calendar = MagicMock()
        meeting = _make_meeting(external_emails=["ceo@startup.com"])
        # Past meeting with same event_id as current
        past_duplicate = self._make_past_meeting("ceo@startup.com", 5, "Same Meeting")
        past_duplicate = Meeting(
            event_id=meeting.event_id,  # same ID!
            summary="Same Meeting",
            start=datetime.now(timezone.utc) - timedelta(days=5),
            end=datetime.now(timezone.utc) - timedelta(days=5) + timedelta(hours=1),
            attendees=[Attendee(email="ceo@startup.com", display_name="Ext")],
        )
        past_other = self._make_past_meeting("ceo@startup.com", 10, "Other Meeting")
        mock_calendar.list_historical_external_meetings.return_value = [past_duplicate, past_other]

        agg = MeetingContextAggregator(calendar_client=mock_calendar)
        content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title="Test",
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        content.attendee_profiles = agg._build_attendee_profiles(meeting)
        agg._enrich_with_calendar_history(meeting, content)

        ext_profile = next(p for p in content.attendee_profiles if p.email == "ceo@startup.com")
        # Only 1 past meeting (the duplicate was skipped)
        assert ext_profile.past_meeting_count == 1

    def test_calendar_failure_sets_flag_and_records_error(self):
        mock_calendar = MagicMock()
        mock_calendar.list_historical_external_meetings.side_effect = RuntimeError("API failure")

        agg = MeetingContextAggregator(calendar_client=mock_calendar)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="Test",
            meeting_start=datetime.now(timezone.utc),
            meeting_end=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        content.attendee_profiles = agg._build_attendee_profiles(meeting)
        agg._enrich_with_calendar_history(meeting, content)

        assert content.calendar_history_available is False
        assert any(e.source == "calendar_history" for e in content.errors)

    def test_no_calendar_client_skips_silently(self):
        agg = MeetingContextAggregator(calendar_client=None)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="Test",
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        content.attendee_profiles = agg._build_attendee_profiles(meeting)
        # Should not raise or set any error
        agg._enrich_with_calendar_history(meeting, content)
        assert content.calendar_history_available is True
        assert len(content.errors) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  MeetingContextAggregator — Gmail context
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorGmailContext:

    def test_gmail_threads_attached_on_success(self):
        mock_gmail = MagicMock()
        thread = _make_email_thread()
        mock_gmail.get_threads_for_meeting.return_value = [thread]

        agg = MeetingContextAggregator(gmail_client=mock_gmail)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_gmail_context(meeting, content)

        assert content.gmail_available is True
        assert len(content.gmail_threads) == 1
        assert content.gmail_threads[0].thread_id == "t1"

    def test_gmail_failure_sets_flag_and_records_error(self):
        mock_gmail = MagicMock()
        mock_gmail.get_threads_for_meeting.side_effect = RuntimeError("network error")

        agg = MeetingContextAggregator(gmail_client=mock_gmail)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_gmail_context(meeting, content)

        assert content.gmail_available is False
        assert any(e.source == "gmail" for e in content.errors)
        assert len(content.gmail_threads) == 0

    def test_no_gmail_client_sets_unavailable(self):
        agg = MeetingContextAggregator(gmail_client=None)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_gmail_context(meeting, content)

        assert content.gmail_available is False
        assert any(e.source == "gmail" for e in content.errors)

    def test_gmail_called_with_external_emails(self):
        mock_gmail = MagicMock()
        mock_gmail.get_threads_for_meeting.return_value = []

        agg = MeetingContextAggregator(gmail_client=mock_gmail)
        meeting = _make_meeting(
            external_emails=["ceo@startup.com", "cto@startup.com"]
        )
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_gmail_context(meeting, content)

        call_kwargs = mock_gmail.get_threads_for_meeting.call_args
        external_emails_arg = call_kwargs[1]["external_emails"]
        assert "ceo@startup.com" in external_emails_arg
        assert "cto@startup.com" in external_emails_arg


# ══════════════════════════════════════════════════════════════════════════════
#  MeetingContextAggregator — Notion context
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorNotionContext:

    def test_notion_records_attached_on_success(self):
        mock_notion = MagicMock()
        record = _make_notion_record()
        mock_notion.get_records_for_meeting.return_value = [record]

        agg = MeetingContextAggregator(notion_client=mock_notion)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_notion_context(meeting, content)

        assert content.notion_available is True
        assert len(content.notion_records) == 1
        assert content.notion_records[0].page_id == "page-001"

    def test_notion_failure_sets_flag_and_records_error(self):
        mock_notion = MagicMock()
        mock_notion.get_records_for_meeting.side_effect = RuntimeError("network error")

        agg = MeetingContextAggregator(notion_client=mock_notion)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_notion_context(meeting, content)

        assert content.notion_available is False
        assert any(e.source == "notion" for e in content.errors)
        assert len(content.notion_records) == 0

    def test_no_notion_client_sets_unavailable(self):
        agg = MeetingContextAggregator(notion_client=None)
        meeting = _make_meeting()
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        agg._fetch_notion_context(meeting, content)

        assert content.notion_available is False
        assert any(e.source == "notion" for e in content.errors)


# ══════════════════════════════════════════════════════════════════════════════
#  MeetingContextAggregator — full aggregate() orchestration
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorOrchestration:

    def test_aggregate_returns_raw_briefing_content(self):
        mock_gmail = MagicMock()
        mock_gmail.get_threads_for_meeting.return_value = []
        mock_notion = MagicMock()
        mock_notion.get_records_for_meeting.return_value = []
        mock_calendar = MagicMock()
        mock_calendar.list_historical_external_meetings.return_value = []

        agg = MeetingContextAggregator(
            gmail_client=mock_gmail,
            notion_client=mock_notion,
            calendar_client=mock_calendar,
        )
        meeting = _make_meeting()
        result = agg.aggregate(meeting, fetch_slack_history=False)

        assert isinstance(result, RawBriefingContent)
        assert result.meeting_id == meeting.event_id
        assert result.meeting_title == meeting.summary

    def test_aggregate_populates_meeting_metadata(self):
        agg = MeetingContextAggregator()
        meeting = _make_meeting(
            summary="파트너십 미팅",
            location="서울 강남구 역삼동",
            description="파트너십 논의",
        )
        result = agg.aggregate(meeting, fetch_slack_history=False)

        assert result.meeting_title == "파트너십 미팅"
        assert result.meeting_location == "서울 강남구 역삼동"
        assert result.meeting_description == "파트너십 논의"
        assert result.meeting_html_link == "https://calendar.google.com/event/evt-001"
        assert result.organizer_email == "invest1@kakaoventures.co.kr"

    def test_aggregate_all_none_clients_no_crash(self):
        """Aggregator must not raise when all clients are None."""
        agg = MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
        )
        meeting = _make_meeting()
        result = agg.aggregate(meeting, fetch_slack_history=False)

        # Should still work, just mark sources as unavailable
        assert result.gmail_available is False
        assert result.notion_available is False

    def test_aggregate_gmail_failure_does_not_prevent_notion(self):
        mock_gmail = MagicMock()
        mock_gmail.get_threads_for_meeting.side_effect = RuntimeError("gmail down")
        mock_notion = MagicMock()
        record = _make_notion_record()
        mock_notion.get_records_for_meeting.return_value = [record]
        mock_calendar = MagicMock()
        mock_calendar.list_historical_external_meetings.return_value = []

        agg = MeetingContextAggregator(
            gmail_client=mock_gmail,
            notion_client=mock_notion,
            calendar_client=mock_calendar,
        )
        meeting = _make_meeting()
        result = agg.aggregate(meeting, fetch_slack_history=False)

        # Gmail failed but Notion should still work
        assert result.gmail_available is False
        assert result.notion_available is True
        assert len(result.notion_records) == 1

    def test_aggregate_notion_failure_does_not_prevent_gmail(self):
        mock_gmail = MagicMock()
        thread = _make_email_thread()
        mock_gmail.get_threads_for_meeting.return_value = [thread]
        mock_notion = MagicMock()
        mock_notion.get_records_for_meeting.side_effect = RuntimeError("notion down")
        mock_calendar = MagicMock()
        mock_calendar.list_historical_external_meetings.return_value = []

        agg = MeetingContextAggregator(
            gmail_client=mock_gmail,
            notion_client=mock_notion,
            calendar_client=mock_calendar,
        )
        meeting = _make_meeting()
        result = agg.aggregate(meeting, fetch_slack_history=False)

        assert result.gmail_available is True
        assert len(result.gmail_threads) == 1
        assert result.notion_available is False

    def test_aggregate_all_sources_success(self):
        thread = _make_email_thread()
        record = _make_notion_record()
        now = datetime.now(timezone.utc)
        past_meeting = Meeting(
            event_id="past-001",
            summary="이전 미팅",
            start=now - timedelta(days=30),
            end=now - timedelta(days=30) + timedelta(hours=1),
            attendees=[Attendee(email="ceo@startup.com")],
        )

        mock_gmail = MagicMock()
        mock_gmail.get_threads_for_meeting.return_value = [thread]
        mock_notion = MagicMock()
        mock_notion.get_records_for_meeting.return_value = [record]
        mock_calendar = MagicMock()
        mock_calendar.list_historical_external_meetings.return_value = [past_meeting]

        agg = MeetingContextAggregator(
            gmail_client=mock_gmail,
            notion_client=mock_notion,
            calendar_client=mock_calendar,
        )
        meeting = _make_meeting(external_emails=["ceo@startup.com"])
        result = agg.aggregate(meeting, fetch_slack_history=False)

        assert result.gmail_available is True
        assert result.notion_available is True
        assert result.calendar_history_available is True
        assert len(result.gmail_threads) == 1
        assert len(result.notion_records) == 1
        assert not result.has_errors

        # Check that the external attendee was enriched with history
        ext_profiles = result.external_attendees
        assert len(ext_profiles) == 1
        assert ext_profiles[0].past_meeting_count == 1
        assert "이전 미팅" in ext_profiles[0].past_meeting_titles

    def test_aggregate_duration_minutes(self):
        agg = MeetingContextAggregator()
        now = datetime.now(timezone.utc)
        meeting = Meeting(
            event_id="evt-002",
            summary="짧은 미팅",
            start=now + timedelta(minutes=5),
            end=now + timedelta(minutes=35),
            attendees=[Attendee(email="x@ext.com"), Attendee(email="y@kakaoventures.co.kr")],
        )
        result = agg.aggregate(meeting, fetch_slack_history=False)
        assert result.duration_minutes == 30


# ══════════════════════════════════════════════════════════════════════════════
#  Gmail helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestGmailHelpers:

    # _extract_keywords

    def test_keywords_basic(self):
        kw = _extract_keywords("Acme Corp Meeting")
        assert "acme" in kw
        assert "corp" in kw
        # "meeting" is a stop word
        assert "meeting" not in kw

    def test_keywords_max_limit(self):
        title = "Alpha Beta Gamma Delta Epsilon Zeta"
        kw = _extract_keywords(title, max_keywords=3)
        assert len(kw) == 3

    def test_keywords_empty_title(self):
        assert _extract_keywords("") == []

    def test_keywords_all_stop_words(self):
        kw = _extract_keywords("meeting call sync")
        assert kw == []

    def test_keywords_korean(self):
        kw = _extract_keywords("스타트업 투자 미팅")
        # "미팅" is a stop-word; "스타트업" and "투자" should be kept
        assert "스타트업" in kw or "투자" in kw

    # _parse_email_date

    def test_parse_rfc2822_date(self):
        dt = _parse_email_date("Mon, 01 Jan 2024 12:00:00 +0900")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_empty_date(self):
        assert _parse_email_date("") is None

    def test_parse_invalid_date(self):
        assert _parse_email_date("not a date at all") is None

    # _extract_address

    def test_extract_bare_address(self):
        assert _extract_address("user@example.com") == "user@example.com"

    def test_extract_from_display_format(self):
        assert _extract_address("John Doe <john@example.com>") == "john@example.com"

    def test_extract_lowercases(self):
        assert _extract_address("USER@EXAMPLE.COM") == "user@example.com"

    # _extract_body_text

    def _encode(self, text: str) -> str:
        return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")

    def test_extract_plain_text(self):
        payload = {
            "mimeType": "text/plain",
            "body": {"data": self._encode("Hello world")},
        }
        assert "Hello world" in _extract_body_text(payload)

    def test_extract_html_strips_tags(self):
        payload = {
            "mimeType": "text/html",
            "body": {"data": self._encode("<h1>Title</h1><p>Body text</p>")},
        }
        text = _extract_body_text(payload)
        assert "Title" in text
        assert "Body text" in text
        assert "<h1>" not in text

    def test_extract_multipart_prefers_plain(self):
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": self._encode("Plain text content")},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": self._encode("<p>HTML content</p>")},
                },
            ],
        }
        text = _extract_body_text(payload)
        assert "Plain text content" in text

    def test_extract_empty_body(self):
        payload = {"mimeType": "text/plain", "body": {}}
        assert _extract_body_text(payload) == ""

    # EmailThread properties

    def test_thread_message_count(self):
        thread = _make_email_thread()
        assert thread.message_count == 1

    def test_thread_latest_date(self):
        now = datetime.now(timezone.utc)
        early = EmailMessage("m1", "t1", "Sub", "a@b.com", date=now - timedelta(days=2), snippet="")
        late = EmailMessage("m2", "t1", "Sub", "a@b.com", date=now - timedelta(days=1), snippet="")
        thread = EmailThread("t1", "Sub", messages=[early, late])
        assert thread.latest_date == late.date

    def test_thread_latest_date_no_messages(self):
        thread = EmailThread("t1", "Sub", messages=[])
        assert thread.latest_date is None

    def test_thread_participants(self):
        msg = EmailMessage(
            "m1", "t1", "Sub",
            sender="alice@a.com",
            recipients=["bob@b.com", "charlie@c.com"],
            snippet="",
        )
        thread = EmailThread("t1", "Sub", messages=[msg])
        parts = thread.participants
        assert "alice@a.com" in parts
        assert "bob@b.com" in parts
        assert "charlie@c.com" in parts

    def test_thread_to_dict(self):
        thread = _make_email_thread()
        d = thread.to_dict()
        assert "thread_id" in d
        assert "subject" in d
        assert "message_count" in d
        assert "messages" in d
        assert isinstance(d["messages"], list)

    # _extract_address edge cases

    def test_extract_address_strips_whitespace(self):
        assert _extract_address("  user@example.com  ") == "user@example.com"

    def test_extract_address_mixed_case_display(self):
        result = _extract_address("Jane Doe <Jane@Example.COM>")
        assert result == "jane@example.com"


# ══════════════════════════════════════════════════════════════════════════════
#  Notion helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionHelpers:

    def test_get_title_value(self):
        prop = {"title": [{"plain_text": "Acme Corp"}]}
        assert _get_title_value(prop) == "Acme Corp"

    def test_get_title_value_empty(self):
        assert _get_title_value({"title": []}) == ""

    def test_get_rich_text_value(self):
        prop = {"rich_text": [{"plain_text": "Some note"}, {"plain_text": " here"}]}
        assert _get_rich_text_value(prop) == "Some note here"

    def test_get_select_value(self):
        prop = {"select": {"name": "검토중"}}
        assert _get_select_value(prop) == "검토중"

    def test_get_select_value_status_type(self):
        prop = {"status": {"name": "진행중"}}
        assert _get_select_value(prop) == "진행중"

    def test_get_select_value_none(self):
        assert _get_select_value({"select": None}) == ""

    def test_get_date_value(self):
        prop = {"date": {"start": "2025-03-15"}}
        assert _get_date_value(prop) == "2025-03-15"

    def test_get_date_value_none(self):
        assert _get_date_value({"date": None}) == ""

    def test_notion_domain_root(self):
        assert notion_domain_root("user@acme.com") == "acme"

    def test_title_keywords_basic(self):
        kw = _title_keywords("Acme Corp 미팅")
        assert "Acme" in kw or "acme" in kw.lower() or any("acme" in k.lower() for k in kw)

    def test_title_keywords_max_limit(self):
        kw = _title_keywords("Alpha Beta Gamma Delta", max_kw=2)
        assert len(kw) == 2

    def test_title_keywords_filters_stop_words(self):
        kw = _title_keywords("meeting with the company")
        assert "meeting" not in kw
        assert "the" not in kw
        assert "with" not in kw


# ══════════════════════════════════════════════════════════════════════════════
#  NotionRecord
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionRecord:

    def test_to_dict_fields(self):
        rec = _make_notion_record()
        d = rec.to_dict()
        assert d["page_id"] == "page-001"
        assert d["title"] == "Startup Inc."
        assert d["company_name"] == "Startup Inc."
        assert d["status"] == "검토중"
        assert d["date_value"] == "2025-01-15"
        assert "url" in d

    def test_to_dict_properties_included(self):
        rec = _make_notion_record()
        rec.properties = {"key1": "val1", "key2": "val2"}
        d = rec.to_dict()
        assert "properties" in d


# ══════════════════════════════════════════════════════════════════════════════
#  DatabaseSchema
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseSchema:

    def test_str_representation(self):
        schema = DatabaseSchema(
            database_id="db-001",
            database_title="Deal Pipeline",
            properties={"Company": "title", "Status": "select"},
            title_field="Company",
        )
        s = str(schema)
        assert "Deal Pipeline" in s
        assert "title_field" in s

    def test_default_fields_none(self):
        schema = DatabaseSchema(database_id="db-001")
        assert schema.title_field is None
        assert schema.company_field is None
        assert schema.status_field is None
        assert schema.date_field is None


# ══════════════════════════════════════════════════════════════════════════════
#  NotionClient — schema discovery (mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionClientSchemaDiscovery:

    def _mock_db_response(self) -> dict:
        return {
            "title": [{"plain_text": "Deal Pipeline"}],
            "properties": {
                "Company": {"type": "title"},
                "Deal Stage": {"type": "select"},
                "Investment Date": {"type": "date"},
                "Notes": {"type": "rich_text"},
                "Founders": {"type": "people"},
            },
        }

    def test_discovers_title_field(self):
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client._client.databases.retrieve.return_value = self._mock_db_response()
        client.schema = None

        with patch.object(client, "_call_with_retry", side_effect=lambda fn, *a, **kw: fn()):
            schema = client._discover_schema("db-001")

        assert schema.title_field == "Company"

    def test_discovers_status_field(self):
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client._client.databases.retrieve.return_value = self._mock_db_response()
        client.schema = None

        with patch.object(client, "_call_with_retry", side_effect=lambda fn, *a, **kw: fn()):
            schema = client._discover_schema("db-001")

        assert schema.status_field == "Deal Stage"

    def test_discovers_date_field(self):
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client._client.databases.retrieve.return_value = self._mock_db_response()
        client.schema = None

        with patch.object(client, "_call_with_retry", side_effect=lambda fn, *a, **kw: fn()):
            schema = client._discover_schema("db-001")

        assert schema.date_field == "Investment Date"

    def test_discovers_database_title(self):
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client._client.databases.retrieve.return_value = self._mock_db_response()
        client.schema = None

        with patch.object(client, "_call_with_retry", side_effect=lambda fn, *a, **kw: fn()):
            schema = client._discover_schema("db-001")

        assert schema.database_title == "Deal Pipeline"

    def test_discovery_failure_returns_empty_schema(self):
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client.schema = None

        with patch.object(
            client,
            "_call_with_retry",
            side_effect=RuntimeError("API error"),
        ):
            schema = client._discover_schema("db-001")

        assert schema.database_id == "db-001"
        assert schema.properties == {}

    def test_company_field_detection_heuristic(self):
        """Schema discovery should detect 'Company Name' as company field."""
        response = {
            "title": [{"plain_text": "Deals"}],
            "properties": {
                "Company Name": {"type": "title"},
                "Stage": {"type": "select"},
            },
        }
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client._client.databases.retrieve.return_value = response
        client.schema = None

        with patch.object(client, "_call_with_retry", side_effect=lambda fn, *a, **kw: fn()):
            schema = client._discover_schema("db-001")

        assert schema.company_field == "Company Name"


# ══════════════════════════════════════════════════════════════════════════════
#  NotionClient — page parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionClientPageParsing:

    def _make_client_with_schema(self) -> NotionClient:
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client.schema = DatabaseSchema(
            database_id="db-001",
            database_title="Deal Pipeline",
            properties={
                "Company": "title",
                "Deal Stage": "select",
                "Investment Date": "date",
            },
            title_field="Company",
            company_field="Company",
            status_field="Deal Stage",
            date_field="Investment Date",
        )
        return client

    def _make_page(self) -> dict:
        return {
            "id": "page-abc",
            "url": "https://notion.so/page-abc",
            "properties": {
                "Company": {
                    "type": "title",
                    "title": [{"plain_text": "TechStartup Inc."}],
                },
                "Deal Stage": {
                    "type": "select",
                    "select": {"name": "Term Sheet"},
                },
                "Investment Date": {
                    "type": "date",
                    "date": {"start": "2025-06-01"},
                },
            },
        }

    def test_parses_page_id_and_url(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert rec.page_id == "page-abc"
        assert rec.url == "https://notion.so/page-abc"

    def test_parses_title(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert rec.title == "TechStartup Inc."

    def test_parses_company_name(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert rec.company_name == "TechStartup Inc."

    def test_parses_status(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert rec.status == "Term Sheet"

    def test_parses_date(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert rec.date_value == "2025-06-01"

    def test_parses_all_flat_props(self):
        client = self._make_client_with_schema()
        rec = client._parse_page(self._make_page())
        assert "Company" in rec.properties
        assert "Deal Stage" in rec.properties


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline._format_raw_briefing() — Slack message format for raw briefing
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatRawBriefing:
    """Tests for pipeline._format_raw_briefing() which converts RawBriefingContent
    into Slack Block Kit messages."""

    def _make_raw_content(
        self,
        gmail_threads=None,
        notion_records=None,
        gmail_available=True,
        notion_available=True,
        calendar_history_available=True,
        description="",
        location="",
        html_link="",
    ):
        now = datetime.now(timezone.utc)
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile, AggregationError

        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="파트너 협의",
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
            meeting_description=description,
            meeting_location=location,
            meeting_html_link=html_link,
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="Jane CEO",
                    is_internal=False,
                    company_domain="startup.com",
                    past_meeting_count=2,
                    last_met_date=now - timedelta(days=30),
                    past_meeting_titles=["Intro Call", "Follow-up"],
                ),
                AttendeeProfile(
                    email="invest1@kakaoventures.co.kr",
                    display_name="투자팀원",
                    is_internal=True,
                ),
            ],
            gmail_threads=gmail_threads if gmail_threads is not None else [],
            gmail_available=gmail_available,
            notion_records=notion_records if notion_records is not None else [],
            notion_available=notion_available,
            calendar_history_available=calendar_history_available,
        )
        if not gmail_available:
            content.errors.append(AggregationError(source="gmail", message="API timeout"))
        if not notion_available:
            content.errors.append(AggregationError(source="notion", message="Notion down"))
        return content

    def test_returns_tuple_str_and_list(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        text, blocks = _format_raw_briefing(content)

        assert isinstance(text, str)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_header_block_contains_meeting_title(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        _, blocks = _format_raw_briefing(content)

        assert blocks[0]["type"] == "header"
        assert "파트너 협의" in blocks[0]["text"]["text"]

    def test_fallback_text_has_meeting_title(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        text, _ = _format_raw_briefing(content)

        assert "파트너 협의" in text

    def test_external_attendees_section_present(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "외부 참석자" in all_text

    def test_past_meeting_history_displayed(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        # Jane CEO has 2 past meetings
        assert "과거 미팅" in all_text

    def test_internal_attendees_in_context_block(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        _, blocks = _format_raw_briefing(content)

        all_context_text = " ".join(
            el.get("text", "")
            for b in blocks if b.get("type") == "context"
            for el in b.get("elements", [])
            if isinstance(el, dict)
        )
        assert "투자팀원" in all_context_text or "내부" in all_context_text

    def test_description_shown_when_present(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(description="파트너십 논의")
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "파트너십 논의" in all_text

    def test_location_shown_when_present(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(location="역삼 본사")
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "역삼 본사" in all_text

    def test_html_link_shown_when_present(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(html_link="https://calendar.google.com/test")
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "calendar.google.com" in all_text

    def test_gmail_threads_shown_when_present(self):
        from src.briefing.pipeline import _format_raw_briefing
        from src.gmail.gmail_client import EmailThread, EmailMessage

        msg = EmailMessage("m1", "t1", "Partnership Proposal", "ceo@startup.com", snippet="")
        thread = EmailThread("t1", "Partnership Proposal", messages=[msg])
        content = self._make_raw_content(gmail_threads=[thread])
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "이메일" in all_text

    def test_gmail_unavailable_shows_확인불가(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(gmail_available=False)
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_notion_records_shown_when_present(self):
        from src.briefing.pipeline import _format_raw_briefing

        record = _make_notion_record("p1", "TechStartup")
        content = self._make_raw_content(notion_records=[record])
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "딜" in all_text or "포트폴리오" in all_text

    def test_notion_unavailable_shows_확인불가(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(notion_available=False)
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "확인 불가" in all_text

    def test_blocks_end_with_context_footer(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content()
        _, blocks = _format_raw_briefing(content)

        assert blocks[-1]["type"] == "context"

    def test_fallback_text_includes_thread_count(self):
        from src.briefing.pipeline import _format_raw_briefing
        from src.gmail.gmail_client import EmailThread, EmailMessage

        msg = EmailMessage("m1", "t1", "Thread", "a@b.com", snippet="")
        thread1 = EmailThread("t1", "Thread 1", messages=[msg])
        thread2 = EmailThread("t2", "Thread 2", messages=[msg])
        content = self._make_raw_content(gmail_threads=[thread1, thread2])
        text, _ = _format_raw_briefing(content)

        assert "2" in text

    def test_fallback_text_includes_notion_count(self):
        from src.briefing.pipeline import _format_raw_briefing

        records = [_make_notion_record(f"p{i}") for i in range(3)]
        content = self._make_raw_content(notion_records=records)
        text, _ = _format_raw_briefing(content)

        assert "3" in text

    def test_no_gmail_threads_shows_empty_state(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(gmail_threads=[])
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "이메일" in all_text

    def test_no_notion_records_shows_empty_state(self):
        from src.briefing.pipeline import _format_raw_briefing

        content = self._make_raw_content(notion_records=[])
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "딜" in all_text or "포트폴리오" in all_text

    def test_calendar_history_unavailable_annotates_attendees(self):
        from src.briefing.pipeline import _format_raw_briefing
        from src.briefing.context_aggregator import (
            RawBriefingContent, AttendeeProfile, AggregationError
        )

        now = datetime.now(timezone.utc)
        content = RawBriefingContent(
            meeting_id="e",
            meeting_title="Test",
            meeting_start=now + timedelta(minutes=5),
            meeting_end=now + timedelta(minutes=35),
            attendee_profiles=[
                AttendeeProfile(
                    email="new@partner.com",
                    display_name="New Person",
                    is_internal=False,
                    past_meeting_count=0,
                ),
            ],
            calendar_history_available=False,
        )
        content.errors.append(
            AggregationError(source="calendar_history", message="Calendar down")
        )
        _, blocks = _format_raw_briefing(content)

        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if b.get("type") == "section"
        )
        assert "확인 불가" in all_text


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline.trigger_meeting_briefing() — updated to use aggregator
# ══════════════════════════════════════════════════════════════════════════════

class TestTriggerMeetingBriefingWithAggregator:
    """Tests for the updated trigger_meeting_briefing() that calls the aggregator."""

    def test_bot_none_returns_true_without_aggregation(self):
        """When bot=None, no aggregation or DM should occur."""
        from src.briefing.pipeline import trigger_meeting_briefing

        meeting = _make_meeting()
        with patch("src.briefing.pipeline._aggregate_meeting_context") as mock_agg:
            result = trigger_meeting_briefing(meeting, bot=None)

        assert result is True
        mock_agg.assert_not_called()

    def test_aggregator_called_when_bot_present(self):
        """With a real bot, the aggregator must be invoked."""
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile

        meeting = _make_meeting()
        now = datetime.now(timezone.utc)
        # AC 8: RawBriefingContent must have external attendees + non-generic title
        mock_content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="스타트업 대표",
                    is_internal=False,
                    company_domain="startup.com",
                ),
            ],
        )

        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=mock_content,
             ) as mock_agg:
            result = trigger_meeting_briefing(meeting, bot=mock_bot)

        assert result is True
        # Aggregator was called once with the meeting (keyword args may vary)
        mock_agg.assert_called_once()
        assert mock_agg.call_args[0][0] is meeting

    def test_send_message_called_with_blocks(self):
        """send_message must be called with Slack Block Kit blocks kwarg."""
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile

        meeting = _make_meeting()
        now = datetime.now(timezone.utc)
        # AC 8: must include external attendees for briefing to be sent
        mock_content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="스타트업 대표",
                    is_internal=False,
                    company_domain="startup.com",
                ),
            ],
        )

        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=mock_content,
             ):
            trigger_meeting_briefing(meeting, bot=mock_bot)

        call_kwargs = mock_bot.send_message.call_args[1]
        assert "blocks" in call_kwargs
        assert isinstance(call_kwargs["blocks"], list)

    def test_returns_false_when_send_message_fails(self):
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent

        meeting = _make_meeting()
        now = datetime.now(timezone.utc)
        mock_content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
        )

        mock_bot = MagicMock()
        mock_bot.send_message.return_value = False

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=mock_content,
             ):
            result = trigger_meeting_briefing(meeting, bot=mock_bot)

        assert result is False

    def test_returns_true_when_send_message_succeeds(self):
        from src.briefing.pipeline import trigger_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile

        meeting = _make_meeting()
        now = datetime.now(timezone.utc)
        # AC 8: must include external attendees so the briefing is not suppressed
        mock_content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
            attendee_profiles=[
                AttendeeProfile(
                    email="ceo@startup.com",
                    display_name="스타트업 대표",
                    is_internal=False,
                    company_domain="startup.com",
                ),
            ],
        )

        mock_bot = MagicMock()
        mock_bot.send_message.return_value = True

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._generate_ai_sections", return_value=None), \
             patch(
                "src.briefing.pipeline._aggregate_meeting_context",
                return_value=mock_content,
             ):
            result = trigger_meeting_briefing(meeting, bot=mock_bot)

        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline._try_init_* helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestTryInitHelpers:
    """Tests for the client initialisation helper functions."""

    def test_try_init_gmail_returns_none_on_exception(self):
        from src.briefing.pipeline import _try_init_gmail

        with patch(
            "src.gmail.gmail_client.GmailClient.connect",
            side_effect=Exception("Gmail init failed"),
        ):
            result = _try_init_gmail()

        assert result is None

    def test_try_init_notion_returns_none_on_exception(self):
        from src.briefing.pipeline import _try_init_notion

        with patch(
            "src.notion.notion_client.NotionClient.connect",
            side_effect=Exception("Notion init failed"),
        ):
            result = _try_init_notion()

        assert result is None

    def test_try_init_calendar_returns_none_on_exception(self):
        from src.briefing.pipeline import _try_init_calendar

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.connect",
            side_effect=Exception("Calendar init failed"),
        ):
            result = _try_init_calendar()

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline._aggregate_meeting_context() — integration helper
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineAggregateContext:
    """Tests for the _aggregate_meeting_context() helper in the pipeline."""

    def test_returns_raw_briefing_content_with_all_clients_none(self):
        from src.briefing.pipeline import _aggregate_meeting_context
        from src.briefing.context_aggregator import RawBriefingContent

        meeting = _make_meeting()

        with patch("src.briefing.pipeline._try_init_gmail", return_value=None), \
             patch("src.briefing.pipeline._try_init_notion", return_value=None), \
             patch("src.briefing.pipeline._try_init_calendar", return_value=None), \
             patch("src.briefing.pipeline._try_init_slack_retriever", return_value=None), \
             patch("src.briefing.pipeline._classify_is_external_first", return_value=False):

            result = _aggregate_meeting_context(meeting)

        assert isinstance(result, RawBriefingContent)
        assert result.meeting_id == meeting.event_id
        assert result.meeting_title == meeting.summary

    def test_meeting_metadata_populated_correctly(self):
        from src.briefing.pipeline import _aggregate_meeting_context
        from src.calendar.google_calendar import Meeting, Attendee
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        meeting = Meeting(
            event_id="special-evt",
            summary="투자 검토 미팅",
            start=now + timedelta(minutes=12),
            end=now + timedelta(minutes=72),
            description="A 스타트업 시리즈 A 투자 검토",
            location="본사 3층 회의실",
            html_link="https://calendar.google.com/special",
            organizer_email="invest2@kakaoventures.co.kr",
            attendees=[
                Attendee(email="founder@astartup.com", display_name="Founder"),
            ],
        )

        with patch("src.briefing.pipeline._try_init_gmail", return_value=None), \
             patch("src.briefing.pipeline._try_init_notion", return_value=None), \
             patch("src.briefing.pipeline._try_init_calendar", return_value=None), \
             patch("src.briefing.pipeline._try_init_slack_retriever", return_value=None), \
             patch("src.briefing.pipeline._classify_is_external_first", return_value=False):

            result = _aggregate_meeting_context(meeting)

        assert result.meeting_id == "special-evt"
        assert result.meeting_title == "투자 검토 미팅"
        assert result.meeting_description == "A 스타트업 시리즈 A 투자 검토"
        assert result.meeting_location == "본사 3층 회의실"
        assert result.meeting_html_link == "https://calendar.google.com/special"
        assert result.organizer_email == "invest2@kakaoventures.co.kr"

    def test_attendee_profiles_built_from_meeting(self):
        from src.briefing.pipeline import _aggregate_meeting_context
        from src.calendar.google_calendar import Meeting, Attendee

        now = datetime.now(timezone.utc)
        meeting = Meeting(
            event_id="e",
            summary="Meeting",
            start=now + timedelta(minutes=10),
            end=now + timedelta(minutes=40),
            attendees=[
                Attendee(email="ext@partner.com"),
                Attendee(email="int@kakaoventures.co.kr"),
            ],
        )

        with patch("src.briefing.pipeline._try_init_gmail", return_value=None), \
             patch("src.briefing.pipeline._try_init_notion", return_value=None), \
             patch("src.briefing.pipeline._try_init_calendar", return_value=None), \
             patch("src.briefing.pipeline._try_init_slack_retriever", return_value=None), \
             patch("src.briefing.pipeline._classify_is_external_first", return_value=False):

            result = _aggregate_meeting_context(meeting)

        assert len(result.attendee_profiles) == 2
        ext = [p for p in result.attendee_profiles if not p.is_internal]
        assert len(ext) == 1
        assert ext[0].email == "ext@partner.com"
