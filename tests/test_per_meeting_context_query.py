"""
Tests for Sub-AC 2b: Per-Meeting Prep Context Query.

Verifies that the system correctly gathers prep context for each meeting by:
1. Querying Gmail for recent email threads with external attendees
   - Searches by attendee email addresses (from/to queries)
   - Searches by meeting title keywords
   - Deduplicates results by thread_id
   - Respects max_threads cap and lookback_days window
2. Querying Notion for relevant notes/docs related to the meeting
   - Searches by company domain extracted from attendee emails
   - Searches by meeting title keywords
   - Deduplicates results by page_id
   - Respects max_records cap
3. Full aggregation integration (both sources queried independently)

All tests run entirely offline (no real API calls).

Run:
    python -m pytest tests/test_per_meeting_context_query.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, call, patch

import pytest

# ── path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixture helpers
# ══════════════════════════════════════════════════════════════════════════════

_NOW = datetime.now(timezone.utc)


def _make_email_message(
    message_id: str = "m-001",
    thread_id: str = "t-001",
    subject: str = "Partnership Discussion",
    sender: str = "ceo@acme.com",
    recipients: list[str] | None = None,
    days_ago: int = 3,
    body_text: str = "Looking forward to our meeting.",
) -> "EmailMessage":
    from src.gmail.gmail_client import EmailMessage

    return EmailMessage(
        message_id=message_id,
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        recipients=recipients or ["invest1@kakaoventures.co.kr"],
        date=_NOW - timedelta(days=days_ago),
        snippet=body_text[:100],
        body_text=body_text,
        labels=[],
    )


def _make_email_thread(
    thread_id: str = "t-001",
    subject: str = "Partnership Discussion",
    days_ago: int = 3,
    sender: str = "ceo@acme.com",
) -> "EmailThread":
    from src.gmail.gmail_client import EmailThread

    msg = _make_email_message(
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        days_ago=days_ago,
    )
    return EmailThread(thread_id=thread_id, subject=subject, messages=[msg])


def _make_notion_record(
    page_id: str = "page-001",
    company_name: str = "Acme Corp",
    status: str = "검토중",
    title: str = "Acme Corp 딜",
) -> "NotionRecord":
    from src.notion.notion_client import NotionRecord

    return NotionRecord(
        page_id=page_id,
        url=f"https://notion.so/{page_id}",
        title=title,
        company_name=company_name,
        status=status,
        date_value="2025-01-15",
    )


def _make_meeting(
    summary: str = "Acme Corp 파트너십 미팅",
    external_emails: list[str] | None = None,
    internal_emails: list[str] | None = None,
) -> "Meeting":
    from src.calendar.google_calendar import Meeting, Attendee

    attendees = []
    for email in (external_emails or ["ceo@acme.com"]):
        attendees.append(
            Attendee(email=email, display_name=email.split("@")[0])
        )
    for email in (internal_emails or ["invest1@kakaoventures.co.kr"]):
        attendees.append(
            Attendee(email=email, display_name="내부 담당자")
        )

    return Meeting(
        event_id="evt-001",
        summary=summary,
        start=_NOW + timedelta(minutes=30),
        end=_NOW + timedelta(minutes=90),
        attendees=attendees,
        organizer_email="invest1@kakaoventures.co.kr",
    )


def _make_gmail_client_mock(return_value: list | None = None) -> MagicMock:
    """Return a mock GmailClient whose search_threads returns the given threads."""
    mock = MagicMock()
    mock.search_threads.return_value = return_value or []
    return mock


def _make_notion_client_mock(
    schema_title_field: str = "이름",
    company_field: str = "이름",
    status_field: str = "Status",
) -> MagicMock:
    """Return a mock NotionClient with a pre-set schema and empty searches."""
    from src.notion.notion_client import DatabaseSchema

    mock = MagicMock()
    mock.schema = DatabaseSchema(
        database_id="db-001",
        database_title="Deals",
        properties={schema_title_field: "title", status_field: "select"},
        title_field=schema_title_field,
        company_field=company_field,
        status_field=status_field,
    )
    mock.search_by_company_name.return_value = []
    return mock


# ══════════════════════════════════════════════════════════════════════════════
# 1. GmailClient.get_threads_for_meeting — query strategy
# ══════════════════════════════════════════════════════════════════════════════

class TestGmailGetThreadsForMeeting:
    """Unit tests for GmailClient.get_threads_for_meeting (offline)."""

    def _make_client_with_search(self, threads_per_call: list | None = None):
        """Return a GmailClient instance where search_threads is replaced by a mock."""
        from src.gmail.gmail_client import GmailClient

        client = GmailClient.__new__(GmailClient)
        client._service = MagicMock()
        client._creds = MagicMock()
        client._creds.valid = True
        client._creds.expired = False
        client.search_threads = MagicMock(return_value=threads_per_call or [])
        return client

    # ── Attendee-based queries ────────────────────────────────────────────────

    def test_queries_external_attendee_email(self):
        """A from/to query is issued for each external attendee email."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Demo",
            lookback_days=30,
        )

        call_queries = [c[0][0] for c in client.search_threads.call_args_list]
        assert any("ceo@acme.com" in q for q in call_queries), (
            "Expected a query containing the attendee email address"
        )

    def test_query_uses_from_and_to_for_attendee(self):
        """The attendee email query must search BOTH from: and to: fields."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["cto@startup.io"],
            meeting_title="Startup Demo",
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        attendee_query = next(
            (q for q in queries if "cto@startup.io" in q), None
        )
        assert attendee_query is not None, "No query found with attendee email"
        assert "from:" in attendee_query, "Query should include 'from:'"
        assert "to:" in attendee_query, "Query should include 'to:'"

    def test_lookback_days_included_in_attendee_query(self):
        """The lookback_days window is part of the attendee email query."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Demo",
            lookback_days=60,
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        attendee_query = next(
            (q for q in queries if "ceo@acme.com" in q), None
        )
        assert "newer_than:60d" in attendee_query, (
            "Query should include lookback window"
        )

    def test_queries_multiple_attendees(self):
        """A separate query is issued for each external attendee (up to 3)."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["a@x.com", "b@x.com", "c@x.com"],
            meeting_title="Demo",
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        assert any("a@x.com" in q for q in queries)
        assert any("b@x.com" in q for q in queries)
        assert any("c@x.com" in q for q in queries)

    def test_caps_attendee_queries_at_three(self):
        """When more than 3 external attendees, only the first 3 are queried."""
        client = self._make_client_with_search()
        emails = [f"user{i}@company.com" for i in range(6)]
        client.get_threads_for_meeting(
            external_emails=emails,
            meeting_title="Demo",
        )

        # Each of the first 3 attendees gets a per-email query + 1 keyword query
        calls_with_email = [
            c[0][0] for c in client.search_threads.call_args_list
            if "@company.com" in c[0][0]
        ]
        assert len(calls_with_email) == 3, (
            f"Expected exactly 3 per-email queries, got {len(calls_with_email)}"
        )

    # ── Keyword-based queries ─────────────────────────────────────────────────

    def test_keyword_query_issued_for_meeting_title(self):
        """A keyword search is issued using tokens from the meeting title."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Partnership Meeting",
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        # At least one query should NOT contain the attendee's email directly
        # (that's the keyword query)
        non_email_queries = [q for q in queries if "ceo@acme.com" not in q]
        assert len(non_email_queries) >= 1, (
            "Expected at least one keyword-based query"
        )

    def test_keyword_query_contains_meaningful_title_tokens(self):
        """Title keywords (minus stop-words) appear in the keyword query."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Corp 투자 검토",
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        keyword_queries = [q for q in queries if "ceo@acme.com" not in q]
        assert keyword_queries, "Expected at least one keyword query"

        # "Acme" or "Corp" should appear (Korean stop words like 검토 should be filtered)
        kw_query = keyword_queries[0]
        assert "acme" in kw_query.lower() or "corp" in kw_query.lower(), (
            f"Expected title keywords in query: {kw_query}"
        )

    def test_no_keyword_query_when_title_only_stop_words(self):
        """When the meeting title contains only stop-words, no keyword query is issued."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="meeting call sync",  # all stop-words
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        keyword_queries = [q for q in queries if "ceo@acme.com" not in q]
        # No meaningful keywords → no keyword query issued
        assert len(keyword_queries) == 0

    # ── Result handling ───────────────────────────────────────────────────────

    def test_deduplicates_threads_by_id(self):
        """The same thread returned from multiple queries is included only once."""
        duplicate_thread = _make_email_thread(thread_id="dup-001")
        client = self._make_client_with_search(threads_per_call=[duplicate_thread])

        result = client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Meeting",
        )

        # Multiple queries returned the same thread — only 1 in result
        thread_ids = [t.thread_id for t in result]
        assert thread_ids.count("dup-001") == 1, (
            "Duplicate thread should be deduplicated"
        )

    def test_max_threads_cap_applied(self):
        """The result is capped at max_threads even when more threads are found."""
        # All queries return 3 threads each
        threads = [_make_email_thread(thread_id=f"t-{i:03d}") for i in range(3)]
        client = self._make_client_with_search(threads_per_call=threads)

        result = client.get_threads_for_meeting(
            external_emails=["a@x.com", "b@x.com"],
            meeting_title="Unique Meeting Name XYZ",
            max_threads=2,
        )

        assert len(result) <= 2, (
            f"Result should be capped at max_threads=2, got {len(result)}"
        )

    def test_returns_empty_list_when_no_threads(self):
        """Returns empty list when Gmail has no matching threads."""
        client = self._make_client_with_search(threads_per_call=[])

        result = client.get_threads_for_meeting(
            external_emails=["nobody@unknown.example.com"],
            meeting_title="No Results Meeting",
        )

        assert result == []

    def test_returns_empty_list_when_no_external_emails(self):
        """When there are no external attendees, returns empty (no queries)."""
        client = self._make_client_with_search()
        client.get_threads_for_meeting(
            external_emails=[],
            meeting_title="Internal Only Meeting",
        )

        # Without external emails, no per-email queries are issued.
        # Only a keyword query may run — but result should still be empty.
        result = client.get_threads_for_meeting(
            external_emails=[],
            meeting_title="meeting call sync",  # stop words only → no kw query
        )
        assert result == []

    def test_threads_sorted_newest_first(self):
        """Returned threads are sorted by latest_date descending."""
        older = _make_email_thread(thread_id="old", days_ago=10)
        newer = _make_email_thread(thread_id="new", days_ago=1)

        client = self._make_client_with_search(threads_per_call=[older, newer])

        result = client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="meeting sync",  # stop-words only → 1 query
        )

        if len(result) >= 2:
            assert result[0].latest_date >= result[1].latest_date, (
                "Threads should be sorted newest-first"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. NotionClient.get_records_for_meeting — query strategy
# ══════════════════════════════════════════════════════════════════════════════

class TestNotionGetRecordsForMeeting:
    """Unit tests for NotionClient.get_records_for_meeting (offline)."""

    def _make_client(self, search_results: dict[str, list] | None = None):
        """
        Return a NotionClient instance where search_by_company_name is mocked.

        ``search_results`` maps a keyword (case-insensitive substring) to
        the NotionRecord list to return when that keyword is part of the
        search argument.  Defaults to returning empty for all queries.
        """
        from src.notion.notion_client import NotionClient, DatabaseSchema

        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client.schema = DatabaseSchema(
            database_id="db-001",
            database_title="Deals",
            properties={"이름": "title", "Status": "select"},
            title_field="이름",
            company_field="이름",
            status_field="Status",
        )

        _results = search_results or {}

        def _mock_search(name, max_results=10):
            name_lower = name.lower()
            for kw, records in _results.items():
                if kw.lower() in name_lower:
                    return records
            return []

        client.search_by_company_name = MagicMock(side_effect=_mock_search)
        return client

    # ── Domain-based queries ──────────────────────────────────────────────────

    def test_queries_company_domain_from_email(self):
        """A Notion search uses the root domain label from the attendee email."""
        rec = _make_notion_record(company_name="Acme Corp")
        client = self._make_client(search_results={"acme": [rec]})

        result = client.get_records_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Corp 미팅",
        )

        # Check search_by_company_name was called with domain-based term
        calls = [c[0][0] for c in client.search_by_company_name.call_args_list]
        assert any("acme" in c.lower() for c in calls), (
            f"Expected domain 'acme' in search calls: {calls}"
        )
        assert len(result) >= 1
        assert result[0].company_name == "Acme Corp"

    def test_queries_co_kr_domain_root(self):
        """Domain root is extracted correctly from .co.kr addresses."""
        rec = _make_notion_record(company_name="Startup Co")
        client = self._make_client(search_results={"startup": [rec]})

        result = client.get_records_for_meeting(
            external_emails=["sales@startup.co.kr"],
            meeting_title="Startup Demo",
        )

        calls = [c[0][0] for c in client.search_by_company_name.call_args_list]
        assert any("startup" in c.lower() for c in calls), (
            f"Expected 'startup' in search queries: {calls}"
        )

    def test_queries_multiple_external_email_domains(self):
        """One search per distinct external attendee domain."""
        rec_a = _make_notion_record(page_id="p-a", company_name="Alpha Inc")
        rec_b = _make_notion_record(page_id="p-b", company_name="Beta Corp")

        client = self._make_client(
            search_results={"alpha": [rec_a], "beta": [rec_b]}
        )

        result = client.get_records_for_meeting(
            external_emails=["ceo@alpha.com", "cto@beta.io"],
            meeting_title="Combined Meeting",
        )

        calls = [c[0][0] for c in client.search_by_company_name.call_args_list]
        assert any("alpha" in c.lower() for c in calls)
        assert any("beta" in c.lower() for c in calls)

    # ── Keyword-based queries ─────────────────────────────────────────────────

    def test_queries_meeting_title_keywords(self):
        """Keywords from meeting title are used as additional Notion queries."""
        rec = _make_notion_record(company_name="GreenTech Solutions")
        client = self._make_client(search_results={"greentech": [rec]})

        result = client.get_records_for_meeting(
            external_emails=["contact@unrelated.com"],
            meeting_title="GreenTech Solutions 투자 검토",
        )

        calls = [c[0][0] for c in client.search_by_company_name.call_args_list]
        assert any("greentech" in c.lower() or "GreenTech" in c for c in calls), (
            f"Expected title keyword in queries: {calls}"
        )

    def test_stop_words_not_queried(self):
        """Korean and English stop-words from the title are not used as queries."""
        client = self._make_client()

        client.get_records_for_meeting(
            external_emails=["ceo@corp.com"],
            meeting_title="미팅 논의 검토 meeting call",
        )

        calls = [c[0][0].lower() for c in client.search_by_company_name.call_args_list]
        for stop in ["미팅", "논의", "검토", "meeting", "call"]:
            assert stop not in calls, f"Stop word '{stop}' should not be queried"

    def test_keyword_queries_capped_at_three(self):
        """Only up to 3 keyword queries are issued to limit API calls."""
        client = self._make_client()

        client.get_records_for_meeting(
            external_emails=[],
            meeting_title="Alpha Beta Gamma Delta Epsilon Zeta",  # 6 words
        )

        # All calls come from keywords (no email to search)
        assert client.search_by_company_name.call_count <= 3, (
            "At most 3 keyword queries should be issued"
        )

    # ── Deduplication ─────────────────────────────────────────────────────────

    def test_deduplicates_records_by_page_id(self):
        """The same Notion page returned from multiple queries appears only once."""
        dup_record = _make_notion_record(page_id="dup-page")
        client = self._make_client(
            search_results={"acme": [dup_record], "corp": [dup_record]}
        )

        result = client.get_records_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Corp Partnership",
        )

        page_ids = [r.page_id for r in result]
        assert page_ids.count("dup-page") == 1, (
            "Duplicate page_id should be deduplicated"
        )

    # ── Max records cap ───────────────────────────────────────────────────────

    def test_max_records_cap_applied(self):
        """Result is capped at max_records even if more records are found."""
        many_records = [
            _make_notion_record(page_id=f"p-{i:03d}", company_name=f"Co {i}")
            for i in range(10)
        ]
        # Return all 10 records for any query
        client = self._make_client(search_results={"acme": many_records})

        result = client.get_records_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Acme Corp Meeting",
            max_records=3,
        )

        assert len(result) <= 3, (
            f"Result should be capped at max_records=3, got {len(result)}"
        )

    def test_returns_empty_when_no_matches(self):
        """Returns empty list when Notion has no matching records."""
        client = self._make_client(search_results={})

        result = client.get_records_for_meeting(
            external_emails=["nobody@unknowncorp.example"],
            meeting_title="Unknown Corp Meeting",
        )

        assert result == []

    # ── Short domain labels skipped ───────────────────────────────────────────

    def test_very_short_domain_label_skipped(self):
        """Domain labels shorter than 3 characters are not queried."""
        client = self._make_client()

        client.get_records_for_meeting(
            external_emails=["user@ab.com"],  # domain root 'ab' is 2 chars
            meeting_title="meeting",
        )

        # With only a 2-char domain root and stop-word title, no search should
        # have been issued (the 2-char label is below the min-length threshold)
        calls = [c[0][0] for c in client.search_by_company_name.call_args_list]
        for c in calls:
            assert len(c) >= 3, f"Unexpectedly short query term: {c!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. MeetingContextAggregator — combined Gmail + Notion context gathering
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregatorPerMeetingContext:
    """
    Integration tests: verifies MeetingContextAggregator correctly feeds
    meeting data into Gmail and Notion queries and assembles RawBriefingContent.
    """

    def _run_aggregate(
        self,
        meeting,
        gmail_mock,
        notion_mock,
        fetch_slack_history: bool = False,
    ):
        from src.briefing.context_aggregator import MeetingContextAggregator

        agg = MeetingContextAggregator(
            gmail_client=gmail_mock,
            notion_client=notion_mock,
            calendar_client=None,
        )
        return agg.aggregate(meeting, fetch_slack_history=fetch_slack_history)

    # ── Gmail integration ─────────────────────────────────────────────────────

    def test_gmail_receives_external_attendee_emails(self):
        """Aggregator passes all external attendee emails to GmailClient."""
        meeting = _make_meeting(
            external_emails=["ceo@acme.com", "cfo@acme.com"]
        )
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        self._run_aggregate(meeting, gmail_mock, notion_mock)

        kwargs = gmail_mock.get_threads_for_meeting.call_args[1]
        external_emails = kwargs["external_emails"]
        assert "ceo@acme.com" in external_emails
        assert "cfo@acme.com" in external_emails

    def test_gmail_receives_meeting_title(self):
        """Meeting title is passed to GmailClient for keyword search."""
        meeting = _make_meeting(summary="Acme Corp 투자 검토 미팅")
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        self._run_aggregate(meeting, gmail_mock, notion_mock)

        kwargs = gmail_mock.get_threads_for_meeting.call_args[1]
        assert kwargs["meeting_title"] == "Acme Corp 투자 검토 미팅"

    def test_gmail_threads_stored_in_content(self):
        """EmailThread results are stored in RawBriefingContent.gmail_threads."""
        meeting = _make_meeting()
        thread = _make_email_thread()

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = [thread]
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.gmail_available is True
        assert len(content.gmail_threads) == 1
        assert content.gmail_threads[0].thread_id == thread.thread_id

    def test_gmail_only_queries_external_not_internal_attendees(self):
        """Internal attendee emails are NOT included in Gmail search."""
        meeting = _make_meeting(
            external_emails=["ceo@acme.com"],
            internal_emails=["invest1@kakaoventures.co.kr", "pm@kakaoventures.co.kr"],
        )
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        self._run_aggregate(meeting, gmail_mock, notion_mock)

        kwargs = gmail_mock.get_threads_for_meeting.call_args[1]
        external_emails = kwargs["external_emails"]
        assert "invest1@kakaoventures.co.kr" not in external_emails
        assert "pm@kakaoventures.co.kr" not in external_emails
        assert "ceo@acme.com" in external_emails

    # ── Notion integration ────────────────────────────────────────────────────

    def test_notion_receives_external_attendee_emails(self):
        """Aggregator passes all external attendee emails to NotionClient."""
        meeting = _make_meeting(
            external_emails=["ceo@beta.io", "cto@beta.io"]
        )
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        self._run_aggregate(meeting, gmail_mock, notion_mock)

        kwargs = notion_mock.get_records_for_meeting.call_args[1]
        external_emails = kwargs["external_emails"]
        assert "ceo@beta.io" in external_emails
        assert "cto@beta.io" in external_emails

    def test_notion_receives_meeting_title(self):
        """Meeting title is passed to NotionClient for keyword search."""
        meeting = _make_meeting(summary="Beta Labs 딜 미팅")
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        self._run_aggregate(meeting, gmail_mock, notion_mock)

        kwargs = notion_mock.get_records_for_meeting.call_args[1]
        assert kwargs["meeting_title"] == "Beta Labs 딜 미팅"

    def test_notion_records_stored_in_content(self):
        """NotionRecord results are stored in RawBriefingContent.notion_records."""
        meeting = _make_meeting()
        record = _make_notion_record()

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = [record]

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.notion_available is True
        assert len(content.notion_records) == 1
        assert content.notion_records[0].page_id == record.page_id

    # ── Combined data assembly ────────────────────────────────────────────────

    def test_both_sources_fetched_independently(self):
        """Gmail and Notion are always queried in the same aggregate() call."""
        meeting = _make_meeting()
        thread = _make_email_thread()
        record = _make_notion_record()

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = [thread]
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = [record]

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.gmail_available is True
        assert content.notion_available is True
        assert len(content.gmail_threads) == 1
        assert len(content.notion_records) == 1

    def test_notion_fetch_runs_even_when_gmail_fails(self):
        """Notion context is still fetched when Gmail raises an exception."""
        meeting = _make_meeting()
        record = _make_notion_record()

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.side_effect = RuntimeError("Gmail down")
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = [record]

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.gmail_available is False
        assert content.notion_available is True
        assert len(content.notion_records) == 1

    def test_gmail_fetch_runs_even_when_notion_fails(self):
        """Gmail context is still fetched when Notion raises an exception."""
        meeting = _make_meeting()
        thread = _make_email_thread()

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = [thread]
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.side_effect = RuntimeError("Notion down")

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.gmail_available is True
        assert content.notion_available is False
        assert len(content.gmail_threads) == 1

    def test_raw_content_has_correct_attendee_count(self):
        """AttendeeProfiles reflect both external and internal attendees."""
        meeting = _make_meeting(
            external_emails=["ceo@acme.com", "cfo@acme.com"],
            internal_emails=["invest1@kakaoventures.co.kr"],
        )
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert len(content.external_attendees) == 2
        assert len(content.internal_attendees) == 1
        assert len(content.attendee_profiles) == 3

    def test_raw_content_to_dict_includes_context_data(self):
        """to_dict() properly serialises the gathered Gmail and Notion context."""
        meeting = _make_meeting()
        thread = _make_email_thread(thread_id="t-abc", subject="Partnership call")
        record = _make_notion_record(page_id="page-xyz", company_name="Acme Corp")

        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = [thread]
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = [record]

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)
        d = content.to_dict()

        assert d["gmail_available"] is True
        assert d["gmail_threads_count"] == 1
        assert d["gmail_threads"][0]["thread_id"] == "t-abc"

        assert d["notion_available"] is True
        assert d["notion_records_count"] == 1
        assert d["notion_records"][0]["page_id"] == "page-xyz"

    # ── Error recording ───────────────────────────────────────────────────────

    def test_gmail_failure_recorded_as_aggregation_error(self):
        """When Gmail fails, an AggregationError with source='gmail' is recorded."""
        meeting = _make_meeting()
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.side_effect = Exception("timeout")
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.has_errors is True
        gmail_errors = [e for e in content.errors if e.source == "gmail"]
        assert len(gmail_errors) == 1
        assert "timeout" in gmail_errors[0].message

    def test_notion_failure_recorded_as_aggregation_error(self):
        """When Notion fails, an AggregationError with source='notion' is recorded."""
        meeting = _make_meeting()
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.side_effect = Exception("rate limited")

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        assert content.has_errors is True
        notion_errors = [e for e in content.errors if e.source == "notion"]
        assert len(notion_errors) == 1
        assert "rate limited" in notion_errors[0].message

    def test_no_errors_when_both_sources_succeed(self):
        """No errors are recorded when both Gmail and Notion succeed."""
        meeting = _make_meeting()
        gmail_mock = MagicMock()
        gmail_mock.get_threads_for_meeting.return_value = []
        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        content = self._run_aggregate(meeting, gmail_mock, notion_mock)

        gmail_errors = [e for e in content.errors if e.source in ("gmail", "notion")]
        assert gmail_errors == []


# ══════════════════════════════════════════════════════════════════════════════
# 4. GmailClient.get_threads_for_meeting — lookback window
# ══════════════════════════════════════════════════════════════════════════════

class TestGmailLookbackWindow:
    """Tests specifically for the lookback_days parameter."""

    def _make_client(self):
        from src.gmail.gmail_client import GmailClient

        client = GmailClient.__new__(GmailClient)
        client._service = MagicMock()
        client._creds = MagicMock()
        client._creds.valid = True
        client._creds.expired = False
        client.search_threads = MagicMock(return_value=[])
        return client

    def test_default_lookback_is_30_days(self):
        """Default lookback_days=30 is applied when not specified."""
        client = self._make_client()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Test",
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        assert any("newer_than:30d" in q for q in queries)

    def test_custom_lookback_applied(self):
        """Custom lookback_days overrides the default."""
        client = self._make_client()
        client.get_threads_for_meeting(
            external_emails=["ceo@acme.com"],
            meeting_title="Test",
            lookback_days=14,
        )

        queries = [c[0][0] for c in client.search_threads.call_args_list]
        assert any("newer_than:14d" in q for q in queries)
        assert not any("newer_than:30d" in q for q in queries)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Package-level imports for Sub-AC 2b components
# ══════════════════════════════════════════════════════════════════════════════

class TestSubAC2bImports:
    """Verify all Sub-AC 2b components are importable from their packages."""

    def test_gmail_client_importable(self):
        from src.gmail.gmail_client import GmailClient
        assert callable(GmailClient)

    def test_gmail_get_threads_for_meeting_importable(self):
        from src.gmail.gmail_client import GmailClient
        assert hasattr(GmailClient, "get_threads_for_meeting")
        assert callable(GmailClient.get_threads_for_meeting)

    def test_notion_client_importable(self):
        from src.notion.notion_client import NotionClient
        assert callable(NotionClient)

    def test_notion_get_records_for_meeting_importable(self):
        from src.notion.notion_client import NotionClient
        assert hasattr(NotionClient, "get_records_for_meeting")
        assert callable(NotionClient.get_records_for_meeting)

    def test_context_aggregator_importable(self):
        from src.briefing.context_aggregator import MeetingContextAggregator
        assert callable(MeetingContextAggregator)

    def test_raw_briefing_content_importable(self):
        from src.briefing.context_aggregator import RawBriefingContent
        assert callable(RawBriefingContent)

    def test_aggregation_error_importable(self):
        from src.briefing.context_aggregator import AggregationError
        assert callable(AggregationError)

    def test_attendee_profile_importable(self):
        from src.briefing.context_aggregator import AttendeeProfile
        assert callable(AttendeeProfile)

    def test_email_thread_has_to_dict(self):
        from src.gmail.gmail_client import EmailThread, EmailMessage
        msg = EmailMessage(
            message_id="m1",
            thread_id="t1",
            subject="Sub",
            sender="a@b.com",
        )
        thread = EmailThread(thread_id="t1", subject="Sub", messages=[msg])
        d = thread.to_dict()
        assert "thread_id" in d
        assert "messages" in d

    def test_notion_record_has_to_dict(self):
        from src.notion.notion_client import NotionRecord
        rec = NotionRecord(
            page_id="p1",
            title="Test",
            company_name="Test Co",
        )
        d = rec.to_dict()
        assert "page_id" in d
        assert "company_name" in d
