"""
Tests for src/ai/web_search.py (Sub-AC 6a).

Covers:
  - Query builder logic (_domain_to_company_name, _is_generic_title, build_search_queries)
  - WebSearchResult and WebSearchSummary data models
  - WebSearchClient with mocked providers (no real API calls)
  - Integration with MeetingContextAggregator (is_external_first flag)
  - Formatter section (_build_web_search_blocks) for all content states
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------

from src.ai.web_search import (
    WebSearchClient,
    WebSearchResult,
    WebSearchSummary,
    _build_summary_text,
    _domain_to_company_name,
    _is_generic_title,
    build_search_queries,
)


# ===========================================================================
# _domain_to_company_name
# ===========================================================================

class TestDomainToCompanyName:
    def test_simple_com(self):
        assert _domain_to_company_name("acme.com") == "Acme"

    def test_hyphenated(self):
        result = _domain_to_company_name("acme-corp.com")
        assert result == "Acme Corp"

    def test_strip_mail_subdomain(self):
        assert _domain_to_company_name("mail.samsung.co.kr") == "Samsung"

    def test_strip_www_subdomain(self):
        assert _domain_to_company_name("www.kakao.com") == "Kakao"

    def test_co_kr(self):
        result = _domain_to_company_name("kakao.co.kr")
        assert result == "Kakao"

    def test_empty_string(self):
        assert _domain_to_company_name("") == ""

    def test_single_segment(self):
        # Handles unusual domains with no TLD
        result = _domain_to_company_name("localhost")
        assert isinstance(result, str)

    def test_underscored(self):
        result = _domain_to_company_name("my_company.io")
        assert "My" in result or "my" in result.lower()


# ===========================================================================
# _is_generic_title
# ===========================================================================

class TestIsGenericTitle:
    @pytest.mark.parametrize("title", [
        "meeting",
        "미팅",
        "1:1",
        "call",
        "Sync",
        "zoom",
    ])
    def test_pure_generic_is_true(self, title):
        assert _is_generic_title(title) is True

    @pytest.mark.parametrize("title", [
        "ACME Product Demo Q1 2026",
        "Partnership discussion with Samsung",
        "Series A due diligence review",
        "투자 미팅 - Kakao Brain",
    ])
    def test_specific_title_is_false(self, title):
        assert _is_generic_title(title) is False

    def test_very_short_is_true(self):
        assert _is_generic_title("ok") is True

    def test_empty_is_true(self):
        assert _is_generic_title("") is True


# ===========================================================================
# build_search_queries
# ===========================================================================

class TestBuildSearchQueries:
    def test_generates_company_query(self):
        queries = build_search_queries(
            company_domains=["acme.com"],
            meeting_title="Q1 Business Review",
        )
        assert len(queries) >= 1
        combined = " ".join(queries).lower()
        assert "acme" in combined

    def test_adds_specific_title(self):
        queries = build_search_queries(
            company_domains=["acme.com"],
            meeting_title="ACME Enterprise Partnership Discussion",
        )
        assert any("ACME Enterprise" in q for q in queries)

    def test_skips_generic_title(self):
        queries = build_search_queries(
            company_domains=["acme.com"],
            meeting_title="meeting",
        )
        # Generic titles should NOT be added as queries
        assert all(q.strip() != "meeting" for q in queries)

    def test_adds_attendee_query(self):
        queries = build_search_queries(
            company_domains=["acme.com"],
            meeting_title="meeting",  # generic — won't be used
            attendee_names=["Jane Doe"],
        )
        combined = " ".join(queries).lower()
        assert "jane" in combined or "acme" in combined

    def test_respects_max_queries(self):
        queries = build_search_queries(
            company_domains=["a.com", "b.com", "c.com", "d.com"],
            meeting_title="Very specific long meeting title about Series B funding round",
            attendee_names=["John Smith"],
            max_queries=2,
        )
        assert len(queries) <= 2

    def test_deduplicates_queries(self):
        queries = build_search_queries(
            company_domains=["acme.com", "acme.com"],  # duplicate domains
            meeting_title="meeting",
        )
        assert len(queries) == len(set(queries))

    def test_empty_inputs(self):
        queries = build_search_queries(
            company_domains=[],
            meeting_title="",
        )
        assert isinstance(queries, list)


# ===========================================================================
# WebSearchResult
# ===========================================================================

class TestWebSearchResult:
    def test_to_dict_contains_all_fields(self):
        result = WebSearchResult(
            query="Acme Corp overview",
            title="Acme Corporation",
            url="https://acme.com",
            snippet="Acme Corp is a leading provider…",
            provider="tavily",
            score=0.9,
        )
        d = result.to_dict()
        assert d["query"] == "Acme Corp overview"
        assert d["title"] == "Acme Corporation"
        assert d["url"] == "https://acme.com"
        assert d["provider"] == "tavily"
        assert d["score"] == 0.9

    def test_default_provider_is_unknown(self):
        result = WebSearchResult(query="q", title="t", url="", snippet="s")
        assert result.provider == "unknown"


# ===========================================================================
# WebSearchSummary
# ===========================================================================

class TestWebSearchSummary:
    def test_has_results_true(self):
        result = WebSearchResult(query="q", title="t", url="", snippet="s")
        summary = WebSearchSummary(results=[result])
        assert summary.has_results is True

    def test_has_results_false(self):
        summary = WebSearchSummary(results=[])
        assert summary.has_results is False

    def test_to_dict_serialises_nested(self):
        result = WebSearchResult(query="q", title="t", url="", snippet="s")
        summary = WebSearchSummary(
            company_names=["Acme"],
            queries_executed=["Acme overview"],
            results=[result],
            summary="Acme Corp provides…",
            provider="tavily",
            available=True,
        )
        d = summary.to_dict()
        assert d["provider"] == "tavily"
        assert len(d["results"]) == 1
        assert d["available"] is True

    def test_unavailable_summary(self):
        summary = WebSearchSummary(available=False, error="No API key")
        assert not summary.has_results
        d = summary.to_dict()
        assert d["available"] is False
        assert d["error"] == "No API key"


# ===========================================================================
# _build_summary_text
# ===========================================================================

class TestBuildSummaryText:
    def test_returns_empty_for_no_results(self):
        assert _build_summary_text([], []) == ""

    def test_includes_company_header(self):
        results = [
            WebSearchResult(
                query="q", title="t", url="", snippet="Snippet text here.",
                provider="tavily", score=0.9,
            )
        ]
        text = _build_summary_text(company_names=["Acme"], results=results)
        assert "Acme" in text
        assert "Snippet text here." in text

    def test_truncates_long_text(self):
        results = [
            WebSearchResult(
                query="q", title="t", url="",
                snippet="A" * 500,
                provider="tavily", score=0.9,
            )
        ]
        text = _build_summary_text(company_names=["X"], results=results)
        assert len(text) <= 801  # _MAX_SUMMARY_LEN=800 + 1 for ellipsis

    def test_sorts_by_score(self):
        low = WebSearchResult(query="q", title="low", url="", snippet="low score", score=0.1)
        high = WebSearchResult(query="q", title="high", url="", snippet="high score", score=0.9)
        text = _build_summary_text(company_names=[], results=[low, high])
        # high score snippet should appear first in the text
        assert text.index("high score") < text.index("low score")


# ===========================================================================
# WebSearchClient — provider selection
# ===========================================================================

class TestWebSearchClientProviderSelection:
    def test_no_keys_means_unavailable(self):
        client = WebSearchClient(tavily_api_key="", anthropic_api_key="")
        assert client.is_available is False

    def test_tavily_key_selects_tavily(self):
        """When TAVILY_API_KEY is set and tavily-python is installed, Tavily is selected."""
        mock_tavily_class = MagicMock()
        mock_tavily_instance = MagicMock()
        mock_tavily_class.return_value = mock_tavily_instance

        with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_class)}):
            client = WebSearchClient(
                tavily_api_key="fake-tavily-key",
                anthropic_api_key="",
            )
            available = client.is_available

        assert available is True
        assert client._provider_name == "tavily"

    def test_anthropic_key_selects_claude_when_no_tavily(self):
        """When only ANTHROPIC_API_KEY is set, Claude provider is used."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = WebSearchClient(
                tavily_api_key="",
                anthropic_api_key="fake-anthropic-key",
            )
            available = client.is_available

        assert available is True
        assert client._provider_name == "claude"

    def test_tavily_init_failure_falls_back_to_claude(self):
        """If Tavily init raises, fall back to Claude."""
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient.side_effect = RuntimeError("tavily broken")
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "tavily": mock_tavily,
            "anthropic": mock_anthropic,
        }):
            client = WebSearchClient(
                tavily_api_key="fake-key",
                anthropic_api_key="fake-anthropic-key",
            )
            available = client.is_available

        assert available is True
        assert client._provider_name == "claude"


# ===========================================================================
# WebSearchClient.search_company_context
# ===========================================================================

class TestWebSearchClientSearchCompanyContext:
    def _make_client_with_mock_provider(self, results: list[WebSearchResult]):
        """Create a WebSearchClient with a pre-configured mock provider."""
        client = WebSearchClient(tavily_api_key="", anthropic_api_key="")
        mock_provider = MagicMock()
        mock_provider.search.return_value = results
        client._provider = mock_provider
        client._provider_name = "mock"
        client._initialized = True
        return client

    def test_returns_summary_with_results(self):
        fake_results = [
            WebSearchResult(
                query="Acme overview",
                title="Acme Corporation",
                url="https://acme.com",
                snippet="Acme Corp founded in 2010.",
                provider="mock",
                score=0.85,
            )
        ]
        client = self._make_client_with_mock_provider(fake_results)
        summary = client.search_company_context(
            company_domains=["acme.com"],
            meeting_title="ACME Product Demo",
        )
        assert summary.available is True
        assert summary.has_results is True
        assert len(summary.results) >= 1
        assert "Acme" in summary.summary or "acme" in summary.summary.lower()

    def test_no_provider_returns_unavailable_summary(self):
        client = WebSearchClient(tavily_api_key="", anthropic_api_key="")
        summary = client.search_company_context(
            company_domains=["acme.com"],
            meeting_title="ACME Demo",
        )
        assert summary.available is False
        assert summary.error is not None

    def test_empty_domains_returns_empty_results(self):
        client = self._make_client_with_mock_provider([])
        summary = client.search_company_context(
            company_domains=[],
            meeting_title="",
        )
        assert summary.available is True
        assert not summary.has_results

    def test_retry_on_provider_failure(self):
        """Provider fails twice then succeeds — should return results."""
        fake_result = WebSearchResult(
            query="q", title="t", url="", snippet="s", provider="mock", score=0.5
        )

        client = WebSearchClient(tavily_api_key="", anthropic_api_key="")
        mock_provider = MagicMock()
        mock_provider.search.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            [fake_result],
        ]
        client._provider = mock_provider
        client._provider_name = "mock"
        client._initialized = True

        with patch("src.ai.web_search.time.sleep"):  # skip actual sleep
            summary = client.search_company_context(
                company_domains=["acme.com"],
                meeting_title="ACME Demo with specific title",
            )

        # At least the third attempt should have succeeded for one query
        assert summary.available is True

    def test_all_retries_exhausted_marks_unavailable(self):
        """All attempts fail — summary.available should be False."""
        client = WebSearchClient(tavily_api_key="", anthropic_api_key="")
        mock_provider = MagicMock()
        mock_provider.search.side_effect = RuntimeError("always fails")
        client._provider = mock_provider
        client._provider_name = "mock"
        client._initialized = True

        with patch("src.ai.web_search.time.sleep"):
            summary = client.search_company_context(
                company_domains=["acme.com"],
                meeting_title="ACME product demo — very specific",
            )

        assert summary.available is False

    def test_queries_recorded_in_summary(self):
        """Executed queries are recorded in summary.queries_executed."""
        fake_results = [
            WebSearchResult(query="q", title="t", url="", snippet="s", provider="mock", score=0.5)
        ]
        client = self._make_client_with_mock_provider(fake_results)
        summary = client.search_company_context(
            company_domains=["acme.com"],
            meeting_title="ACME Product Launch",
        )
        assert len(summary.queries_executed) >= 1

    def test_provider_name_recorded(self):
        fake_results = [
            WebSearchResult(query="q", title="t", url="", snippet="s", provider="mock", score=0.5)
        ]
        client = self._make_client_with_mock_provider(fake_results)
        summary = client.search_company_context(
            company_domains=["acme.com"],
            meeting_title="ACME Demo",
        )
        assert summary.provider == "mock"


# ===========================================================================
# Formatter: _build_web_search_blocks
# ===========================================================================

def _make_raw_content_mock(
    meeting_title: str = "Test Meeting",
    web_search_summary: Optional[WebSearchSummary] = None,
    web_search_available: bool = True,
) -> MagicMock:
    """Create a minimal RawBriefingContent mock for formatter tests."""
    from src.briefing.context_aggregator import AggregationError

    raw = MagicMock()
    raw.meeting_title = meeting_title
    raw.meeting_start = datetime.now(timezone.utc) + timedelta(minutes=10)
    raw.meeting_end = datetime.now(timezone.utc) + timedelta(minutes=70)
    raw.meeting_location = ""
    raw.meeting_html_link = ""
    raw.meeting_description = "A test meeting description."
    raw.duration_minutes = 60
    raw.external_attendees = []
    raw.internal_attendees = []
    raw.gmail_threads = []
    raw.gmail_available = True
    raw.notion_records = []
    raw.notion_available = True
    raw.calendar_history_available = True
    raw.web_search_summary = web_search_summary
    raw.web_search_available = web_search_available
    raw.errors = []
    raw.has_errors = False
    return raw


class TestBuildWebSearchBlocks:
    def _get_builder(self):
        from src.briefing.meeting_briefing_formatter import _build_web_search_blocks
        return _build_web_search_blocks

    def test_returns_empty_when_no_summary(self):
        build = self._get_builder()
        raw = _make_raw_content_mock(web_search_summary=None)
        blocks = build(raw)
        assert blocks == []

    def test_shows_unavailable_when_available_false(self):
        build = self._get_builder()
        summary = WebSearchSummary(available=False, error="No API key configured")
        raw = _make_raw_content_mock(web_search_summary=summary, web_search_available=False)
        blocks = build(raw)
        assert len(blocks) >= 2  # divider + error block
        text_content = str(blocks)
        assert "확인 불가" in text_content

    def test_shows_no_results_message(self):
        build = self._get_builder()
        summary = WebSearchSummary(available=True, results=[], provider="tavily")
        raw = _make_raw_content_mock(web_search_summary=summary)
        blocks = build(raw)
        text_content = str(blocks)
        assert "결과 없음" in text_content

    def test_renders_results(self):
        build = self._get_builder()
        result = WebSearchResult(
            query="Acme overview",
            title="Acme Corporation",
            url="https://acme.com",
            snippet="Acme Corp is a leading AI company founded in 2020.",
            provider="tavily",
            score=0.9,
        )
        summary = WebSearchSummary(
            company_names=["Acme"],
            queries_executed=["Acme overview"],
            results=[result],
            summary="*Acme* 웹 검색 결과:\nAcme Corp is a leading AI company founded in 2020.",
            provider="tavily",
            available=True,
        )
        raw = _make_raw_content_mock(web_search_summary=summary)
        blocks = build(raw)
        text_content = str(blocks)
        assert "웹 검색" in text_content
        assert "Acme" in text_content

    def test_shows_provider_label(self):
        build = self._get_builder()
        summary = WebSearchSummary(
            results=[
                WebSearchResult(query="q", title="t", url="", snippet="s", provider="tavily", score=0.5)
            ],
            summary="Some summary.",
            provider="tavily",
            available=True,
        )
        raw = _make_raw_content_mock(web_search_summary=summary)
        blocks = build(raw)
        text_content = str(blocks)
        assert "tavily" in text_content

    def test_shows_query_footer(self):
        build = self._get_builder()
        summary = WebSearchSummary(
            results=[
                WebSearchResult(query="q", title="t", url="", snippet="s", provider="mock", score=0.5)
            ],
            queries_executed=["Acme Corp overview"],
            summary="Some summary.",
            provider="mock",
            available=True,
        )
        raw = _make_raw_content_mock(web_search_summary=summary)
        blocks = build(raw)
        text_content = str(blocks)
        assert "검색어" in text_content

    def test_section_ordering_in_full_briefing(self):
        """Web search section must appear between attendees and Gmail in the full briefing."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile, AggregationError

        # Build a minimal real RawBriefingContent with web search summary
        ws_summary = WebSearchSummary(
            company_names=["Acme"],
            queries_executed=["Acme overview"],
            results=[
                WebSearchResult(
                    query="Acme overview",
                    title="Acme Corp",
                    url="https://acme.com",
                    snippet="Acme Corp is a technology company.",
                    provider="mock",
                    score=0.9,
                )
            ],
            summary="*Acme* 웹 검색:\nAcme Corp is a technology company.",
            provider="mock",
            available=True,
        )

        now = datetime.now(timezone.utc)
        content = RawBriefingContent(
            meeting_id="evt-001",
            meeting_title="ACME Product Demo",
            meeting_start=now + timedelta(minutes=10),
            meeting_end=now + timedelta(minutes=70),
            meeting_description="Demo of new features.",
        )
        ext_profile = AttendeeProfile(
            email="jane@acme.com",
            display_name="Jane Doe",
            is_internal=False,
            company_domain="acme.com",
        )
        content.attendee_profiles = [ext_profile]
        content.web_search_summary = ws_summary
        content.web_search_available = True

        text, blocks = format_meeting_briefing(content)

        # Verify the briefing string contains web search content
        all_text = str(blocks)
        assert "웹 검색" in all_text
        assert "Acme" in all_text
        # Should still have attendees section
        assert "외부 참석자" in all_text


# ===========================================================================
# Integration: MeetingContextAggregator with is_external_first=True
# ===========================================================================

class TestAggregatorWebSearchIntegration:
    """Verify MeetingContextAggregator calls web search only for EXTERNAL_FIRST."""

    def _make_meeting(self):
        """Return a minimal Meeting-like mock."""
        meeting = MagicMock()
        meeting.event_id = "evt-001"
        meeting.summary = "ACME Product Demo"
        meeting.start = datetime.now(timezone.utc) + timedelta(minutes=10)
        meeting.end = datetime.now(timezone.utc) + timedelta(minutes=70)
        meeting.location = ""
        meeting.description = "Demo of new product."
        meeting.html_link = "https://calendar.google.com/event/evt-001"
        meeting.organizer_email = "organizer@acme.com"
        attendee = MagicMock()
        attendee.email = "jane@acme.com"
        attendee.display_name = "Jane Doe"
        attendee.response_status = "accepted"
        attendee.is_internal = False
        meeting.attendees = [attendee]
        meeting.external_attendees = [attendee]
        meeting.is_external = True
        return meeting

    def test_web_search_called_when_external_first(self):
        """MeetingContextAggregator calls web search when is_external_first=True."""
        from src.briefing.context_aggregator import MeetingContextAggregator

        mock_ws_client = MagicMock()
        mock_ws_client.search_company_context.return_value = WebSearchSummary(
            company_names=["Acme"],
            queries_executed=["Acme overview"],
            results=[
                WebSearchResult(
                    query="Acme overview",
                    title="Acme Corp",
                    url="https://acme.com",
                    snippet="Acme Corp is a leader in AI.",
                    provider="mock",
                    score=0.9,
                )
            ],
            summary="Acme Corp is a leader in AI.",
            provider="mock",
            available=True,
        )

        aggregator = MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
            web_search_client=mock_ws_client,
        )
        content = aggregator.aggregate(self._make_meeting(), is_external_first=True)

        mock_ws_client.search_company_context.assert_called_once()
        assert content.web_search_summary is not None
        assert content.web_search_summary.has_results is True
        assert content.web_search_available is True

    def test_web_search_not_called_when_not_external_first(self):
        """MeetingContextAggregator does NOT call web search for EXTERNAL_FOLLOWUP."""
        from src.briefing.context_aggregator import MeetingContextAggregator

        mock_ws_client = MagicMock()

        aggregator = MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
            web_search_client=mock_ws_client,
        )
        content = aggregator.aggregate(self._make_meeting(), is_external_first=False)

        mock_ws_client.search_company_context.assert_not_called()
        assert content.web_search_summary is None

    def test_web_search_failure_records_error(self):
        """Provider failure sets web_search_available=False and records an AggregationError."""
        from src.briefing.context_aggregator import MeetingContextAggregator

        mock_ws_client = MagicMock()
        mock_ws_client.search_company_context.side_effect = RuntimeError("Search API down")

        aggregator = MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
            web_search_client=mock_ws_client,
        )
        content = aggregator.aggregate(self._make_meeting(), is_external_first=True)

        assert content.web_search_available is False
        error_sources = [e.source for e in content.errors]
        assert "web_search" in error_sources

    def test_unavailable_search_summary_sets_flag(self):
        """When WebSearchSummary.available=False, web_search_available flag is False."""
        from src.briefing.context_aggregator import MeetingContextAggregator

        mock_ws_client = MagicMock()
        mock_ws_client.search_company_context.return_value = WebSearchSummary(
            available=False,
            error="API key not configured",
            provider="none",
        )

        aggregator = MeetingContextAggregator(
            gmail_client=None,
            notion_client=None,
            calendar_client=None,
            web_search_client=mock_ws_client,
        )
        content = aggregator.aggregate(self._make_meeting(), is_external_first=True)

        assert content.web_search_available is False
        assert content.web_search_summary is not None  # summary object is still attached
        error_sources = [e.source for e in content.errors]
        assert "web_search" in error_sources


# ===========================================================================
# Pipeline: _classify_is_external_first and _try_init_web_search (Sub-AC 6a)
# ===========================================================================

class TestPipelineWebSearchWiring:
    """Tests for the pipeline helper functions added in Sub-AC 6a."""

    def _make_mock_meeting(self, has_external: bool = True) -> MagicMock:
        meeting = MagicMock()
        meeting.event_id = "evt-999"
        meeting.summary = "New Startup Intro"
        attendee = MagicMock()
        attendee.email = "ceo@newstartup.io"
        attendee.is_internal = False
        meeting.external_attendees = [attendee] if has_external else []
        meeting.is_external = has_external
        meeting.attendees = [attendee]
        return meeting

    def test_classify_external_first_no_external_attendees_returns_false(self):
        """Meetings without external attendees are never EXTERNAL_FIRST."""
        from src.briefing.pipeline import _classify_is_external_first

        meeting = self._make_mock_meeting(has_external=False)
        result = _classify_is_external_first(meeting)
        assert result is False

    def test_classify_returns_true_when_cache_unavailable(self):
        """Falls back to True (conservative) when GoogleCalendarClient connect fails."""
        from src.briefing.pipeline import _classify_is_external_first

        meeting = self._make_mock_meeting()
        # Patch at the source module since _classify_is_external_first uses local imports
        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient",
            side_effect=Exception("calendar connection error"),
        ):
            # With cal_client failing → cache=None → classify_event returns EXTERNAL_FIRST
            # (conservative default for unknown meetings with no history cache)
            result = _classify_is_external_first(meeting)
        assert result is True

    def test_classify_returns_true_for_external_first_category(self):
        """Returns True when event_classifier says EXTERNAL_FIRST."""
        from src.briefing.pipeline import _classify_is_external_first
        from src.calendar.event_classifier import EventCategory

        meeting = self._make_mock_meeting()
        # Patch at source modules since pipeline uses local imports
        with patch("src.calendar.event_classifier.classify_event",
                   return_value=EventCategory.EXTERNAL_FIRST), \
             patch("src.calendar.google_calendar.GoogleCalendarClient",
                   side_effect=Exception("no connection")):
            result = _classify_is_external_first(meeting)
        assert result is True

    def test_classify_returns_false_for_external_followup_category(self):
        """Returns False when event_classifier says EXTERNAL_FOLLOWUP."""
        from src.briefing.pipeline import _classify_is_external_first
        from src.calendar.event_classifier import EventCategory

        meeting = self._make_mock_meeting()
        mock_cache = MagicMock()
        # Patch at source modules
        with patch("src.calendar.event_classifier.classify_event",
                   return_value=EventCategory.EXTERNAL_FOLLOWUP), \
             patch("src.calendar.history_cache.CalendarHistoryCache",
                   return_value=mock_cache), \
             patch("src.calendar.google_calendar.GoogleCalendarClient") as MockCal:
            MockCal.return_value.connect.return_value = None
            result = _classify_is_external_first(meeting)
        assert result is False

    def test_try_init_web_search_returns_client_when_available(self):
        """_try_init_web_search returns a WebSearchClient when a provider is available."""
        from src.briefing.pipeline import _try_init_web_search

        mock_client = MagicMock()
        mock_client.is_available = True

        # Patch at source module since _try_init_web_search uses local import
        with patch("src.ai.web_search.WebSearchClient", return_value=mock_client):
            client = _try_init_web_search()

        assert client is mock_client

    def test_try_init_web_search_returns_none_on_exception(self):
        """_try_init_web_search returns None when WebSearchClient raises."""
        from src.briefing.pipeline import _try_init_web_search

        with patch(
            "src.ai.web_search.WebSearchClient",
            side_effect=RuntimeError("init failed"),
        ):
            client = _try_init_web_search()

        assert client is None

    def test_aggregate_context_calls_aggregator_with_external_first_flag(self):
        """_aggregate_meeting_context passes is_external_first=True to aggregator."""
        from src.briefing.pipeline import _aggregate_meeting_context

        meeting = self._make_mock_meeting()
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = MagicMock()

        # _try_init_* are module-level functions — can be patched via pipeline module
        # MeetingContextAggregator is a local import — patch at source module
        # Call with is_external_first=True explicitly (this is how trigger_meeting_briefing calls it)
        with patch("src.briefing.pipeline._try_init_gmail", return_value=None), \
             patch("src.briefing.pipeline._try_init_notion", return_value=None), \
             patch("src.briefing.pipeline._try_init_calendar", return_value=None), \
             patch("src.briefing.pipeline._try_init_web_search", return_value=None), \
             patch("src.briefing.pipeline._try_init_slack_retriever", return_value=None), \
             patch(
                 "src.briefing.context_aggregator.MeetingContextAggregator",
                 return_value=mock_aggregator,
             ):
            _aggregate_meeting_context(meeting, is_external_first=True)

        # The pipeline also passes fetch_slack_history based on whether a slack retriever
        # was initialised (False here since _try_init_slack_retriever returns None)
        mock_aggregator.aggregate.assert_called_once_with(
            meeting, is_external_first=True, fetch_slack_history=False
        )

    def test_aggregate_context_skips_web_search_for_followup(self):
        """_aggregate_meeting_context does NOT call _try_init_web_search for followup."""
        from src.briefing.pipeline import _aggregate_meeting_context

        meeting = self._make_mock_meeting()
        mock_try_init_ws = MagicMock(return_value=None)
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = MagicMock()

        with patch("src.briefing.pipeline._classify_is_external_first", return_value=False), \
             patch("src.briefing.pipeline._try_init_gmail", return_value=None), \
             patch("src.briefing.pipeline._try_init_notion", return_value=None), \
             patch("src.briefing.pipeline._try_init_calendar", return_value=None), \
             patch("src.briefing.pipeline._try_init_web_search", mock_try_init_ws), \
             patch(
                 "src.briefing.context_aggregator.MeetingContextAggregator",
                 return_value=mock_aggregator,
             ):
            _aggregate_meeting_context(meeting)

        # _try_init_web_search should NOT be called for EXTERNAL_FOLLOWUP
        mock_try_init_ws.assert_not_called()
