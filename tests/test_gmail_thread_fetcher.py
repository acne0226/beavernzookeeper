"""
Tests for Sub-AC 7.2: Gmail Thread Fetcher.

Covers:
  1. Helper functions: _extract_domain, _extract_address, _is_internal_email,
     _parse_display_name, _latest_message
  2. _determine_direction() — inbound / outbound / unknown
  3. _determine_status()    — active / waiting_reply / stale / unknown
  4. get_threads_for_company() — domain query, contact-email query, empty result
  5. build_thread_summary()    — full data, minimal data, no messages
  6. get_latest_thread_summary() — success, no threads, API failure with retry
  7. get_all_thread_summaries()  — success, empty, retry exhaustion
  8. ThreadSummary.to_dict()     — all fields serialised correctly

All tests run offline (no real Gmail API calls).

Run:
    python -m pytest tests/test_gmail_thread_fetcher.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ── path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# Test fixtures / helpers
# ══════════════════════════════════════════════════════════════════════════════

_NOW = datetime.now(timezone.utc)
_SENTINEL = object()  # Distinguishes "no date given" from "date=None"


def _make_email_message(
    message_id: str = "msg-001",
    thread_id: str = "thread-001",
    subject: str = "Test Subject",
    sender: str = "Alice <alice@startup.com>",
    recipients: list[str] | None = None,
    date=_SENTINEL,           # Use _SENTINEL so callers can pass date=None explicitly
    snippet: str = "Hello there",
    body_text: str = "Hello, this is the body of the email.",
    labels: list[str] | None = None,
) -> "EmailMessage":
    """
    Build a real EmailMessage for testing (no mocks needed).

    Pass ``date=None`` explicitly to create a message with no date.
    Omit *date* (or pass the sentinel) to use a sensible default.
    """
    from src.gmail.gmail_client import EmailMessage

    actual_date = (_NOW - timedelta(hours=2)) if date is _SENTINEL else date

    return EmailMessage(
        message_id=message_id,
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        recipients=recipients or ["invest1@kakaoventures.co.kr"],
        date=actual_date,
        snippet=snippet,
        body_text=body_text,
        labels=labels or [],
    )


def _make_email_thread(
    thread_id: str = "thread-001",
    subject: str = "Test Subject",
    messages: list | None = None,
) -> "EmailThread":
    """Build a real EmailThread for testing."""
    from src.gmail.gmail_client import EmailThread

    return EmailThread(
        thread_id=thread_id,
        subject=subject,
        messages=messages or [],
    )


def _make_mock_gmail_client(threads: list | None = None) -> MagicMock:
    """Return a MagicMock GmailClient whose search_threads returns *threads*."""
    client = MagicMock()
    client.search_threads.return_value = threads or []
    return client


# ══════════════════════════════════════════════════════════════════════════════
# 1. Helper functions
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractDomain:
    def test_standard_email(self):
        from src.gmail.thread_fetcher import _extract_domain
        assert _extract_domain("alice@startup.com") == "startup.com"

    def test_upper_case_normalised(self):
        from src.gmail.thread_fetcher import _extract_domain
        assert _extract_domain("Bob@STARTUP.COM") == "startup.com"

    def test_no_at_sign(self):
        from src.gmail.thread_fetcher import _extract_domain
        assert _extract_domain("not-an-email") == ""

    def test_empty_string(self):
        from src.gmail.thread_fetcher import _extract_domain
        assert _extract_domain("") == ""

    def test_subdomain(self):
        from src.gmail.thread_fetcher import _extract_domain
        assert _extract_domain("x@mail.startup.com") == "mail.startup.com"


class TestExtractAddress:
    def test_display_name_format(self):
        from src.gmail.thread_fetcher import _extract_address
        assert _extract_address("Alice <alice@startup.com>") == "alice@startup.com"

    def test_bare_address(self):
        from src.gmail.thread_fetcher import _extract_address
        assert _extract_address("alice@startup.com") == "alice@startup.com"

    def test_upper_case_normalised(self):
        from src.gmail.thread_fetcher import _extract_address
        assert _extract_address("ALICE@STARTUP.COM") == "alice@startup.com"

    def test_extra_whitespace(self):
        from src.gmail.thread_fetcher import _extract_address
        assert _extract_address("  alice@startup.com  ") == "alice@startup.com"

    def test_angle_brackets_with_spaces(self):
        from src.gmail.thread_fetcher import _extract_address
        assert _extract_address("Display Name < alice@startup.com >") == "alice@startup.com"


class TestIsInternalEmail:
    def test_internal_domain(self):
        from src.gmail.thread_fetcher import _is_internal_email
        assert _is_internal_email("invest1@kakaoventures.co.kr") is True

    def test_external_domain(self):
        from src.gmail.thread_fetcher import _is_internal_email
        assert _is_internal_email("ceo@startup.com") is False

    def test_empty_string(self):
        from src.gmail.thread_fetcher import _is_internal_email
        assert _is_internal_email("") is False

    def test_case_insensitive(self):
        from src.gmail.thread_fetcher import _is_internal_email
        assert _is_internal_email("INVEST1@KAKAOVENTURES.CO.KR") is True


class TestParseDisplayName:
    def test_standard_format(self):
        from src.gmail.thread_fetcher import _parse_display_name
        assert _parse_display_name("Alice Smith <alice@startup.com>") == "Alice Smith"

    def test_quoted_name(self):
        from src.gmail.thread_fetcher import _parse_display_name
        # Quoted display names are stripped of their quotes
        result = _parse_display_name('"Alice Smith" <alice@startup.com>')
        assert result == "Alice Smith"

    def test_bare_address(self):
        from src.gmail.thread_fetcher import _parse_display_name
        assert _parse_display_name("alice@startup.com") == "alice@startup.com"

    def test_empty_name_falls_back_to_address(self):
        from src.gmail.thread_fetcher import _parse_display_name
        result = _parse_display_name("<alice@startup.com>")
        assert result == "alice@startup.com"


class TestLatestMessage:
    def test_returns_most_recent(self):
        from src.gmail.thread_fetcher import _latest_message

        msg1 = _make_email_message(message_id="m1", date=_NOW - timedelta(hours=5))
        msg2 = _make_email_message(message_id="m2", date=_NOW - timedelta(hours=1))
        msg3 = _make_email_message(message_id="m3", date=_NOW - timedelta(hours=3))

        thread = _make_email_thread(messages=[msg1, msg2, msg3])
        assert _latest_message(thread).message_id == "m2"

    def test_empty_thread_returns_none(self):
        from src.gmail.thread_fetcher import _latest_message
        thread = _make_email_thread(messages=[])
        assert _latest_message(thread) is None

    def test_no_dates_falls_back_to_last_element(self):
        from src.gmail.thread_fetcher import _latest_message

        msg1 = _make_email_message(message_id="m1", date=None)
        msg2 = _make_email_message(message_id="m2", date=None)
        thread = _make_email_thread(messages=[msg1, msg2])
        assert _latest_message(thread).message_id == "m2"


# ══════════════════════════════════════════════════════════════════════════════
# 2. _determine_direction
# ══════════════════════════════════════════════════════════════════════════════

class TestDetermineDirection:
    def test_inbound_external_sender(self):
        from src.gmail.thread_fetcher import _determine_direction
        msg = _make_email_message(sender="CEO <ceo@startup.com>")
        thread = _make_email_thread(messages=[msg])
        assert _determine_direction(thread) == "inbound"

    def test_outbound_internal_sender(self):
        from src.gmail.thread_fetcher import _determine_direction
        msg = _make_email_message(sender="Invest1 <invest1@kakaoventures.co.kr>")
        thread = _make_email_thread(messages=[msg])
        assert _determine_direction(thread) == "outbound"

    def test_unknown_no_messages(self):
        from src.gmail.thread_fetcher import _determine_direction
        thread = _make_email_thread(messages=[])
        assert _determine_direction(thread) == "unknown"

    def test_unknown_empty_sender(self):
        from src.gmail.thread_fetcher import _determine_direction
        msg = _make_email_message(sender="")
        thread = _make_email_thread(messages=[msg])
        assert _determine_direction(thread) == "unknown"

    def test_uses_latest_message(self):
        """When messages have mixed senders, direction is based on newest message."""
        from src.gmail.thread_fetcher import _determine_direction

        old_msg = _make_email_message(
            message_id="m1",
            sender="CEO <ceo@startup.com>",
            date=_NOW - timedelta(hours=5),
        )
        new_msg = _make_email_message(
            message_id="m2",
            sender="Invest1 <invest1@kakaoventures.co.kr>",
            date=_NOW - timedelta(hours=1),
        )
        thread = _make_email_thread(messages=[old_msg, new_msg])
        assert _determine_direction(thread) == "outbound"


# ══════════════════════════════════════════════════════════════════════════════
# 3. _determine_status
# ══════════════════════════════════════════════════════════════════════════════

class TestDetermineStatus:
    def test_active_recent_inbound(self):
        from src.gmail.thread_fetcher import _determine_status

        msg = _make_email_message(date=_NOW - timedelta(hours=2))
        thread = _make_email_thread(messages=[msg])
        thread_with_date = MagicMock()
        thread_with_date.latest_date = _NOW - timedelta(hours=2)
        thread_with_date.messages = [msg]
        assert _determine_status(thread_with_date, "inbound") == "active"

    def test_stale_old_thread(self):
        from src.gmail.thread_fetcher import _determine_status

        mock_thread = MagicMock()
        mock_thread.latest_date = _NOW - timedelta(days=30)
        assert _determine_status(mock_thread, "inbound") == "stale"

    def test_waiting_reply_outbound_recent(self):
        from src.gmail.thread_fetcher import _determine_status

        mock_thread = MagicMock()
        mock_thread.latest_date = _NOW - timedelta(hours=12)  # within 3 days
        assert _determine_status(mock_thread, "outbound") == "waiting_reply"

    def test_active_outbound_but_older_than_waiting_threshold(self):
        """Outbound but older than _WAITING_REPLY_DAYS → 'active' not 'waiting_reply'."""
        from src.gmail.thread_fetcher import _determine_status

        mock_thread = MagicMock()
        mock_thread.latest_date = _NOW - timedelta(days=5)  # >3 days but ≤7 days
        assert _determine_status(mock_thread, "outbound") == "active"

    def test_unknown_no_date(self):
        from src.gmail.thread_fetcher import _determine_status

        mock_thread = MagicMock()
        mock_thread.latest_date = None
        assert _determine_status(mock_thread, "inbound") == "unknown"

    def test_naive_datetime_handled(self):
        """Naive datetimes (no tzinfo) should not cause errors."""
        from src.gmail.thread_fetcher import _determine_status

        mock_thread = MagicMock()
        mock_thread.latest_date = datetime.utcnow()  # naive
        # Should not raise
        result = _determine_status(mock_thread, "inbound")
        assert result in ("active", "stale", "waiting_reply", "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# 4. get_threads_for_company
# ══════════════════════════════════════════════════════════════════════════════

class TestGetThreadsForCompany:
    def test_domain_query_uses_at_syntax(self):
        """Domain-only search must use @domain.com in the Gmail query."""
        from src.gmail.thread_fetcher import get_threads_for_company

        mock_client = _make_mock_gmail_client(threads=[])
        get_threads_for_company(mock_client, domain="startup.com")

        call_args = mock_client.search_threads.call_args
        query_str = call_args[0][0]
        assert "@startup.com" in query_str

    def test_contact_email_query_uses_address(self):
        """When contact_email is given the query must use that address, not @domain."""
        from src.gmail.thread_fetcher import get_threads_for_company

        mock_client = _make_mock_gmail_client(threads=[])
        get_threads_for_company(
            mock_client,
            domain="startup.com",
            contact_email="ceo@startup.com",
        )

        call_args = mock_client.search_threads.call_args
        query_str = call_args[0][0]
        assert "ceo@startup.com" in query_str
        # Should NOT use the @domain syntax when contact_email is provided
        assert "@startup.com" not in query_str.replace("ceo@startup.com", "")

    def test_lookback_days_in_query(self):
        from src.gmail.thread_fetcher import get_threads_for_company

        mock_client = _make_mock_gmail_client()
        get_threads_for_company(mock_client, domain="startup.com", lookback_days=60)

        query_str = mock_client.search_threads.call_args[0][0]
        assert "newer_than:60d" in query_str

    def test_max_results_passed_to_client(self):
        from src.gmail.thread_fetcher import get_threads_for_company

        mock_client = _make_mock_gmail_client()
        get_threads_for_company(mock_client, domain="startup.com", max_threads=3)

        call_args = mock_client.search_threads.call_args
        assert call_args[1].get("max_results") == 3 or call_args[0][1] == 3

    def test_returns_threads_from_client(self):
        from src.gmail.thread_fetcher import get_threads_for_company

        thread = _make_email_thread()
        mock_client = _make_mock_gmail_client(threads=[thread])
        result = get_threads_for_company(mock_client, domain="startup.com")
        assert len(result) == 1
        assert result[0].thread_id == "thread-001"

    def test_empty_result(self):
        from src.gmail.thread_fetcher import get_threads_for_company

        mock_client = _make_mock_gmail_client(threads=[])
        result = get_threads_for_company(mock_client, domain="unknown.com")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 5. build_thread_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildThreadSummary:
    def test_basic_inbound_thread(self):
        from src.gmail.thread_fetcher import build_thread_summary

        msg = _make_email_message(
            sender="CEO <ceo@startup.com>",
            subject="Investment Proposal",
            snippet="Hi, we'd like to discuss...",
            body_text="Full body text here.",
            date=_NOW - timedelta(hours=1),
        )
        thread = _make_email_thread(
            thread_id="t-100",
            subject="Investment Proposal",
            messages=[msg],
        )

        summary = build_thread_summary(thread, domain="startup.com")

        assert summary.thread_id == "t-100"
        assert summary.subject == "Investment Proposal"
        assert summary.company_domain == "startup.com"
        assert summary.contact_email is None
        assert summary.message_count == 1
        assert summary.last_sender == "CEO"
        assert summary.last_sender_email == "ceo@startup.com"
        assert summary.last_snippet == "Hi, we'd like to discuss..."
        assert summary.last_body_preview == "Full body text here."
        assert summary.direction == "inbound"
        assert summary.status in ("active", "waiting_reply")

    def test_outbound_thread(self):
        from src.gmail.thread_fetcher import build_thread_summary

        msg = _make_email_message(
            sender="Invest1 <invest1@kakaoventures.co.kr>",
            date=_NOW - timedelta(hours=1),
        )
        thread = _make_email_thread(messages=[msg])

        summary = build_thread_summary(thread, domain="startup.com")
        assert summary.direction == "outbound"
        assert summary.last_sender_email == "invest1@kakaoventures.co.kr"

    def test_contact_email_stored(self):
        from src.gmail.thread_fetcher import build_thread_summary

        msg = _make_email_message()
        thread = _make_email_thread(messages=[msg])

        summary = build_thread_summary(
            thread, domain="startup.com", contact_email="ceo@startup.com"
        )
        assert summary.contact_email == "ceo@startup.com"

    def test_no_messages(self):
        """Thread with no messages should return empty strings, not crash."""
        from src.gmail.thread_fetcher import build_thread_summary

        thread = _make_email_thread(messages=[])
        summary = build_thread_summary(thread, domain="startup.com")

        assert summary.last_sender == ""
        assert summary.last_sender_email == ""
        assert summary.last_snippet == ""
        assert summary.last_body_preview == ""
        assert summary.direction == "unknown"
        assert summary.status == "unknown"
        assert summary.messages == []

    def test_body_preview_truncated_to_500(self):
        from src.gmail.thread_fetcher import build_thread_summary

        long_body = "x" * 1000
        msg = _make_email_message(body_text=long_body, date=_NOW)
        thread = _make_email_thread(messages=[msg])

        summary = build_thread_summary(thread, domain="startup.com")
        assert len(summary.last_body_preview) == 500

    def test_messages_serialised(self):
        from src.gmail.thread_fetcher import build_thread_summary

        msg = _make_email_message(message_id="m-xyz")
        thread = _make_email_thread(messages=[msg])
        summary = build_thread_summary(thread, domain="startup.com")

        assert isinstance(summary.messages, list)
        assert len(summary.messages) == 1
        assert summary.messages[0]["message_id"] == "m-xyz"

    def test_participants_included(self):
        from src.gmail.thread_fetcher import build_thread_summary

        msg = _make_email_message(
            sender="CEO <ceo@startup.com>",
            recipients=["invest1@kakaoventures.co.kr"],
        )
        thread = _make_email_thread(messages=[msg])
        summary = build_thread_summary(thread, domain="startup.com")

        assert "ceo@startup.com" in summary.participants
        assert "invest1@kakaoventures.co.kr" in summary.participants

    def test_multi_message_direction_uses_latest(self):
        """Direction should reflect the LATEST message, not any earlier ones."""
        from src.gmail.thread_fetcher import build_thread_summary

        first_msg = _make_email_message(
            message_id="m1",
            sender="CEO <ceo@startup.com>",
            date=_NOW - timedelta(hours=10),
        )
        last_msg = _make_email_message(
            message_id="m2",
            sender="Invest1 <invest1@kakaoventures.co.kr>",
            date=_NOW - timedelta(hours=1),
        )
        thread = _make_email_thread(messages=[first_msg, last_msg])
        summary = build_thread_summary(thread, domain="startup.com")

        assert summary.direction == "outbound"
        assert summary.last_sender_email == "invest1@kakaoventures.co.kr"


# ══════════════════════════════════════════════════════════════════════════════
# 6. ThreadSummary.to_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSummaryToDict:
    def test_all_fields_present(self):
        from src.gmail.thread_fetcher import ThreadSummary

        ts = ThreadSummary(
            thread_id="t-1",
            subject="Hello",
            company_domain="startup.com",
            contact_email="ceo@startup.com",
            latest_date=_NOW,
            message_count=2,
            participants=["ceo@startup.com", "invest1@kakaoventures.co.kr"],
            last_sender="CEO",
            last_sender_email="ceo@startup.com",
            last_snippet="snippet",
            last_body_preview="preview",
            direction="inbound",
            status="active",
            messages=[],
        )
        d = ts.to_dict()

        assert d["thread_id"] == "t-1"
        assert d["subject"] == "Hello"
        assert d["company_domain"] == "startup.com"
        assert d["contact_email"] == "ceo@startup.com"
        assert d["latest_date"] == _NOW.isoformat()
        assert d["message_count"] == 2
        assert d["participants"] == ["ceo@startup.com", "invest1@kakaoventures.co.kr"]
        assert d["last_sender"] == "CEO"
        assert d["last_sender_email"] == "ceo@startup.com"
        assert d["last_snippet"] == "snippet"
        assert d["last_body_preview"] == "preview"
        assert d["direction"] == "inbound"
        assert d["status"] == "active"
        assert d["messages"] == []

    def test_none_latest_date(self):
        from src.gmail.thread_fetcher import ThreadSummary

        ts = ThreadSummary(
            thread_id="t-2", subject="", company_domain="x.com",
            contact_email=None, latest_date=None, message_count=0,
            participants=[], last_sender="", last_sender_email="",
            last_snippet="", last_body_preview="",
            direction="unknown", status="unknown",
        )
        assert ts.to_dict()["latest_date"] is None
        assert ts.to_dict()["contact_email"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 7. get_latest_thread_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGetLatestThreadSummary:
    def test_returns_summary_on_success(self):
        from src.gmail.thread_fetcher import get_latest_thread_summary

        msg = _make_email_message(sender="CEO <ceo@startup.com>")
        thread = _make_email_thread(messages=[msg])
        mock_client = _make_mock_gmail_client(threads=[thread])

        result = get_latest_thread_summary(
            domain="startup.com",
            client=mock_client,
        )

        assert result is not None
        assert result.thread_id == "thread-001"
        assert result.company_domain == "startup.com"

    def test_returns_none_when_no_threads(self):
        from src.gmail.thread_fetcher import get_latest_thread_summary

        mock_client = _make_mock_gmail_client(threads=[])
        result = get_latest_thread_summary(domain="unknown.com", client=mock_client)
        assert result is None

    def test_returns_none_after_all_retries_exhausted(self):
        """When the client raises on every attempt, return None (not raise)."""
        from src.gmail.thread_fetcher import get_latest_thread_summary

        mock_client = MagicMock()
        mock_client.search_threads.side_effect = RuntimeError("API failure")

        with patch("src.gmail.thread_fetcher.time.sleep"):  # skip real sleep
            result = get_latest_thread_summary(domain="startup.com", client=mock_client)

        assert result is None

    def test_retries_on_failure_then_succeeds(self):
        """Simulate one failure then a success — should return the summary."""
        from src.gmail.thread_fetcher import get_latest_thread_summary

        msg = _make_email_message()
        thread = _make_email_thread(messages=[msg])

        mock_client = MagicMock()
        mock_client.search_threads.side_effect = [
            RuntimeError("transient error"),
            [thread],
        ]

        with patch("src.gmail.thread_fetcher.time.sleep"):
            result = get_latest_thread_summary(domain="startup.com", client=mock_client)

        assert result is not None
        assert result.thread_id == "thread-001"

    def test_contact_email_passed_to_query(self):
        from src.gmail.thread_fetcher import get_latest_thread_summary

        mock_client = _make_mock_gmail_client(threads=[])
        get_latest_thread_summary(
            domain="startup.com",
            contact_email="ceo@startup.com",
            client=mock_client,
        )

        query_str = mock_client.search_threads.call_args[0][0]
        assert "ceo@startup.com" in query_str

    def test_creates_client_if_none_provided(self):
        """When client=None, a GmailClient should be created internally."""
        from src.gmail.thread_fetcher import get_latest_thread_summary

        msg = _make_email_message()
        thread = _make_email_thread(messages=[msg])
        mock_instance = _make_mock_gmail_client(threads=[thread])

        with patch(
            "src.gmail.thread_fetcher.GmailClient",
            return_value=mock_instance,
        ):
            result = get_latest_thread_summary(domain="startup.com")

        mock_instance.connect.assert_called_once()
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# 8. get_all_thread_summaries
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAllThreadSummaries:
    def test_returns_all_summaries(self):
        from src.gmail.thread_fetcher import get_all_thread_summaries

        msg1 = _make_email_message(message_id="m1", date=_NOW - timedelta(hours=1))
        msg2 = _make_email_message(message_id="m2", date=_NOW - timedelta(hours=2))
        thread1 = _make_email_thread(thread_id="t-1", messages=[msg1])
        thread2 = _make_email_thread(thread_id="t-2", messages=[msg2])

        mock_client = _make_mock_gmail_client(threads=[thread1, thread2])
        results = get_all_thread_summaries(domain="startup.com", client=mock_client)

        assert len(results) == 2
        assert results[0].thread_id == "t-1"
        assert results[1].thread_id == "t-2"

    def test_empty_result(self):
        from src.gmail.thread_fetcher import get_all_thread_summaries

        mock_client = _make_mock_gmail_client(threads=[])
        results = get_all_thread_summaries(domain="unknown.com", client=mock_client)
        assert results == []

    def test_returns_empty_after_all_retries_fail(self):
        from src.gmail.thread_fetcher import get_all_thread_summaries

        mock_client = MagicMock()
        mock_client.search_threads.side_effect = RuntimeError("API down")

        with patch("src.gmail.thread_fetcher.time.sleep"):
            results = get_all_thread_summaries(domain="startup.com", client=mock_client)

        assert results == []

    def test_retry_count(self):
        """search_threads should be called exactly API_RETRY_ATTEMPTS times on failure."""
        from src.gmail.thread_fetcher import get_all_thread_summaries
        from src.config import API_RETRY_ATTEMPTS

        mock_client = MagicMock()
        mock_client.search_threads.side_effect = RuntimeError("fail")

        with patch("src.gmail.thread_fetcher.time.sleep"):
            get_all_thread_summaries(domain="startup.com", client=mock_client)

        assert mock_client.search_threads.call_count == API_RETRY_ATTEMPTS

    def test_max_threads_parameter_honoured(self):
        from src.gmail.thread_fetcher import get_all_thread_summaries

        mock_client = _make_mock_gmail_client(threads=[])
        get_all_thread_summaries(domain="startup.com", max_threads=3, client=mock_client)

        call_args = mock_client.search_threads.call_args
        max_results = (
            call_args[1].get("max_results")
            if call_args[1]
            else (call_args[0][1] if len(call_args[0]) > 1 else None)
        )
        assert max_results == 3

    def test_contact_email_stored_in_summaries(self):
        from src.gmail.thread_fetcher import get_all_thread_summaries

        msg = _make_email_message()
        thread = _make_email_thread(messages=[msg])
        mock_client = _make_mock_gmail_client(threads=[thread])

        results = get_all_thread_summaries(
            domain="startup.com",
            contact_email="ceo@startup.com",
            client=mock_client,
        )

        assert results[0].contact_email == "ceo@startup.com"


# ══════════════════════════════════════════════════════════════════════════════
# 9. Integration: __init__.py re-exports
# ══════════════════════════════════════════════════════════════════════════════

class TestPackageExports:
    def test_thread_summary_importable_from_package(self):
        from src.gmail import ThreadSummary
        assert ThreadSummary is not None

    def test_get_latest_thread_summary_importable(self):
        from src.gmail import get_latest_thread_summary
        assert callable(get_latest_thread_summary)

    def test_get_all_thread_summaries_importable(self):
        from src.gmail import get_all_thread_summaries
        assert callable(get_all_thread_summaries)

    def test_build_thread_summary_importable(self):
        from src.gmail import build_thread_summary
        assert callable(build_thread_summary)

    def test_get_threads_for_company_importable(self):
        from src.gmail import get_threads_for_company
        assert callable(get_threads_for_company)
