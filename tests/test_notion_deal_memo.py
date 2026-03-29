"""
Tests for Sub-AC 6b: Notion Deal Memo Retrieval for external_first briefings.

Covers:
- RawBriefingContent.notion_deal_memo / notion_deal_memo_available fields
- MeetingContextAggregator._fetch_notion_deal_memo()
  * happy path (memo found via email domain candidate)
  * happy path (memo found via meeting title keyword)
  * no match (legitimate — available=True, memo=None)
  * notion client unavailable (available=False, error recorded)
  * api error during get_company_page_content (available=False)
  * multiple candidates tried in order
  * blocks_fetched=False propagated correctly
  * only called for is_external_first=True (not EXTERNAL_FOLLOWUP)
- Helper functions _domain_root_label, _title_keywords_for_notion
- meeting_briefing_formatter._build_notion_deal_memo_blocks()
  * not shown when memo=None and available=True (non-EXTERNAL_FIRST)
  * shows error annotation when available=False
  * happy path renders company, status, body
  * blocks_fetched=False shows '확인 불가' body
  * url linking works
  * to_briefing_summary truncated at 600 chars
- format_meeting_briefing() integration (deal memo section appears for EXTERNAL_FIRST)

All tests run entirely offline (no real Notion API calls).

Run:
    python -m pytest tests/test_notion_deal_memo.py -v
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
# Shared fixture helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_meeting(
    title: str = "AcmeCorp 제품 데모",
    external_emails: list[str] | None = None,
    event_id: str = "evt-001",
) -> MagicMock:
    """Return a Meeting-like mock with controlled external attendees."""
    from src.calendar.google_calendar import Meeting, Attendee

    if external_emails is None:
        external_emails = ["ceo@acme.com", "cto@acme.com"]

    now = datetime.now(timezone.utc)
    attendees = [Attendee(email=e, display_name=e.split("@")[0]) for e in external_emails]
    attendees.append(Attendee(email="invest1@kakaoventures.co.kr", display_name="내부인"))

    return Meeting(
        event_id=event_id,
        summary=title,
        start=now + timedelta(minutes=10),
        end=now + timedelta(minutes=70),
        attendees=attendees,
    )


def _make_notion_page_content(
    page_id: str = "page-001",
    company_name: str = "AcmeCorp",
    status: str = "심사 중",
    date_value: str = "2024-03-15",
    blocks_fetched: bool = True,
    body_text: str = "투자 포인트\n혁신적인 B2B SaaS\n\n리스크\n규제 불확실성",
) -> "NotionPageContent":  # type: ignore[name-defined]
    from src.notion.notion_client import NotionPageContent, NotionPageSection

    sections = []
    if blocks_fetched and body_text:
        sections = [
            NotionPageSection(heading="투자 포인트", content="혁신적인 B2B SaaS"),
            NotionPageSection(heading="리스크", content="규제 불확실성"),
        ]

    return NotionPageContent(
        page_id=page_id,
        url=f"https://notion.so/{page_id}",
        title=f"{company_name} 딜",
        company_name=company_name,
        status=status,
        date_value=date_value,
        body_text=body_text if blocks_fetched else "",
        sections=sections,
        blocks_fetched=blocks_fetched,
    )


def _make_raw_content(**overrides) -> "RawBriefingContent":  # type: ignore[name-defined]
    """Convenience builder for RawBriefingContent."""
    from src.briefing.context_aggregator import RawBriefingContent, AttendeeProfile

    now = datetime.now(timezone.utc)
    defaults = dict(
        meeting_id="evt-001",
        meeting_title="AcmeCorp 제품 데모",
        meeting_start=now + timedelta(minutes=10),
        meeting_end=now + timedelta(minutes=70),
        attendee_profiles=[
            AttendeeProfile(
                email="ceo@acme.com",
                display_name="Acme CEO",
                is_internal=False,
                company_domain="acme.com",
            ),
            AttendeeProfile(
                email="invest1@kakaoventures.co.kr",
                display_name="내부인",
                is_internal=True,
                company_domain="kakaoventures.co.kr",
            ),
        ],
    )
    defaults.update(overrides)
    return RawBriefingContent(**defaults)


def _make_aggregator(notion_client=None):
    """Return a MeetingContextAggregator with the given notion client."""
    from src.briefing.context_aggregator import MeetingContextAggregator

    return MeetingContextAggregator(
        gmail_client=None,
        notion_client=notion_client,
        calendar_client=None,
        web_search_client=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. RawBriefingContent fields
# ══════════════════════════════════════════════════════════════════════════════

class TestRawBriefingContentDealMemoFields:
    """notion_deal_memo and notion_deal_memo_available fields on RawBriefingContent."""

    def test_defaults(self):
        rc = _make_raw_content()
        assert rc.notion_deal_memo is None
        assert rc.notion_deal_memo_available is True

    def test_to_dict_includes_deal_memo_none(self):
        rc = _make_raw_content()
        d = rc.to_dict()
        assert "notion_deal_memo" in d
        assert d["notion_deal_memo"] is None
        assert d["notion_deal_memo_available"] is True

    def test_to_dict_includes_deal_memo_when_set(self):
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content()
        d = rc.to_dict()
        assert d["notion_deal_memo"] is not None
        assert d["notion_deal_memo"]["company_name"] == "AcmeCorp"
        assert d["notion_deal_memo"]["blocks_fetched"] is True

    def test_to_dict_deal_memo_unavailable(self):
        rc = _make_raw_content()
        rc.notion_deal_memo_available = False
        d = rc.to_dict()
        assert d["notion_deal_memo_available"] is False
        assert d["notion_deal_memo"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Helper functions
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainRootLabel:
    def _fn(self, email):
        from src.briefing.context_aggregator import _domain_root_label
        return _domain_root_label(email)

    def test_simple_domain(self):
        assert self._fn("ceo@acme.com") == "acme"

    def test_co_kr_domain(self):
        # 'startup.co.kr' → meaningful parts: ['startup'], last = 'startup'
        assert self._fn("user@startup.co.kr") == "startup"

    def test_io_domain(self):
        assert self._fn("john@cooltech.io") == "cooltech"

    def test_subdomain(self):
        # 'sub.bigcorp.com' → meaningful: ['sub', 'bigcorp'], last = 'bigcorp'
        assert self._fn("user@sub.bigcorp.com") == "bigcorp"

    def test_invalid_email(self):
        from src.briefing.context_aggregator import _domain_root_label
        assert _domain_root_label("notanemail") == ""

    def test_empty_string(self):
        from src.briefing.context_aggregator import _domain_root_label
        assert _domain_root_label("") == ""


class TestTitleKeywordsForNotion:
    def _fn(self, title, max_kw=4):
        from src.briefing.context_aggregator import _title_keywords_for_notion
        return _title_keywords_for_notion(title, max_kw)

    def test_simple_company_name(self):
        kws = self._fn("AcmeCorp 미팅")
        assert "AcmeCorp" in kws

    def test_strips_stop_words(self):
        kws = self._fn("meeting with Corp")
        # 'meeting' and 'with' are stop words; 'Corp' also in stop words
        assert "meeting" not in kws
        assert "with" not in kws

    def test_korean_stop_words(self):
        kws = self._fn("스타트업 미팅 논의")
        # 미팅 and 논의 are stop words
        assert "미팅" not in kws
        assert "논의" not in kws

    def test_max_kw_respected(self):
        kws = self._fn("Alpha Beta Gamma Delta Epsilon", max_kw=3)
        assert len(kws) <= 3

    def test_short_tokens_skipped(self):
        kws = self._fn("A B C BigCo")
        # Single-char tokens should be skipped
        assert "A" not in kws
        assert "B" not in kws
        assert "BigCo" in kws

    def test_empty_title(self):
        assert self._fn("") == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. MeetingContextAggregator._fetch_notion_deal_memo
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchNotionDealMemo:

    def _run_fetch(self, meeting, notion_client):
        """
        Call _fetch_notion_deal_memo directly and return the mutated content.
        """
        from src.briefing.context_aggregator import (
            MeetingContextAggregator,
            RawBriefingContent,
        )
        now = datetime.now(timezone.utc)
        content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
        )
        aggregator = MeetingContextAggregator(notion_client=notion_client)
        aggregator._fetch_notion_deal_memo(meeting, content)
        return content

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_happy_path_memo_found_via_domain(self):
        """When the first email domain candidate yields a match, memo is stored."""
        meeting = _make_meeting(
            title="AcmeCorp 제품 미팅",
            external_emails=["ceo@acme.com"],
        )
        page = _make_notion_page_content(company_name="AcmeCorp")

        notion_mock = MagicMock()
        # 'acme' (domain root) → found
        notion_mock.get_company_page_content.return_value = page

        content = self._run_fetch(meeting, notion_mock)

        assert content.notion_deal_memo is not None
        assert content.notion_deal_memo.company_name == "AcmeCorp"
        assert content.notion_deal_memo_available is True
        # At least one call was made
        notion_mock.get_company_page_content.assert_called()

    def test_happy_path_blocks_fetched_false(self):
        """When page found but blocks unavailable, still stores memo."""
        meeting = _make_meeting(external_emails=["sales@beta.io"])
        page = _make_notion_page_content(
            company_name="Beta Inc",
            blocks_fetched=False,
        )

        notion_mock = MagicMock()
        notion_mock.get_company_page_content.return_value = page

        content = self._run_fetch(meeting, notion_mock)

        assert content.notion_deal_memo is not None
        assert content.notion_deal_memo.blocks_fetched is False
        assert content.notion_deal_memo_available is True

    def test_memo_found_via_title_keyword(self):
        """Falls back to title keyword when domain search returns None first."""
        meeting = _make_meeting(
            title="GreatStartup 투자 검토",
            external_emails=["user@unrelated.com"],
        )
        page = _make_notion_page_content(company_name="GreatStartup")

        notion_mock = MagicMock()
        # Return None for domain-based searches, page for keyword search
        def _side_effect(company_name, **kwargs):
            if "unrelated" in company_name.lower():
                return None
            if "GreatStartup" in company_name or "greatstartup" in company_name.lower():
                return page
            return None

        notion_mock.get_company_page_content.side_effect = _side_effect

        content = self._run_fetch(meeting, notion_mock)

        assert content.notion_deal_memo is not None
        assert content.notion_deal_memo.company_name == "GreatStartup"

    # ── No match ────────────────────────────────────────────────────────────

    def test_no_match_sets_memo_none_and_available_true(self):
        """When no candidate yields a result, memo is None but available stays True."""
        meeting = _make_meeting(
            title="Unknown Company Meeting",
            external_emails=["john@nobody.com"],
        )

        notion_mock = MagicMock()
        notion_mock.get_company_page_content.return_value = None

        content = self._run_fetch(meeting, notion_mock)

        assert content.notion_deal_memo is None
        assert content.notion_deal_memo_available is True
        # No error recorded for "not found" — it is a normal outcome
        deal_memo_errors = [e for e in content.errors if e.source == "notion_deal_memo"]
        assert deal_memo_errors == []

    # ── Client unavailable ──────────────────────────────────────────────────

    def test_notion_client_none_sets_unavailable(self):
        """When no Notion client is provided, marks unavailable with error."""
        meeting = _make_meeting()
        content = self._run_fetch(meeting, notion_client=None)

        assert content.notion_deal_memo is None
        assert content.notion_deal_memo_available is False
        error_sources = [e.source for e in content.errors]
        assert "notion_deal_memo" in error_sources

    # ── API error ───────────────────────────────────────────────────────────

    def test_api_exception_sets_unavailable(self):
        """When get_company_page_content raises on all candidates, marks unavailable."""
        meeting = _make_meeting(
            title="TechCo 미팅",
            external_emails=["ceo@techco.com"],
        )

        notion_mock = MagicMock()
        notion_mock.get_company_page_content.side_effect = Exception("Notion API down")

        content = self._run_fetch(meeting, notion_mock)

        assert content.notion_deal_memo is None
        assert content.notion_deal_memo_available is False
        deal_memo_errors = [e for e in content.errors if e.source == "notion_deal_memo"]
        assert len(deal_memo_errors) == 1
        assert "Notion API down" in deal_memo_errors[0].message

    def test_partial_failure_recovers_on_later_candidate(self):
        """If first candidate raises but second returns result, memo is set."""
        meeting = _make_meeting(
            title="GoodCo 투자 미팅",
            external_emails=["ceo@bad-domain.com"],
        )
        page = _make_notion_page_content(company_name="GoodCo")

        calls = []
        def _side_effect(company_name, **kwargs):
            calls.append(company_name)
            if "bad" in company_name.lower():
                raise Exception("First candidate fails")
            if "GoodCo" in company_name or "goodco" in company_name.lower():
                return page
            return None

        notion_mock = MagicMock()
        notion_mock.get_company_page_content.side_effect = _side_effect

        content = self._run_fetch(meeting, notion_mock)

        # The memo should be found via a later candidate
        assert content.notion_deal_memo is not None
        assert content.notion_deal_memo.company_name == "GoodCo"
        # Available should be True since a result was found
        assert content.notion_deal_memo_available is True

    # ── Integration with aggregate() ────────────────────────────────────────

    def test_aggregate_calls_deal_memo_for_external_first(self):
        """aggregate(is_external_first=True) triggers deal memo fetch."""
        meeting = _make_meeting()
        page = _make_notion_page_content(company_name="AcmeCorp")

        notion_mock = MagicMock()
        notion_mock.get_company_page_content.return_value = page
        # Stub out records_for_meeting used in step 4
        notion_mock.get_records_for_meeting.return_value = []

        from src.briefing.context_aggregator import MeetingContextAggregator
        aggregator = MeetingContextAggregator(notion_client=notion_mock)
        content = aggregator.aggregate(meeting, is_external_first=True)

        assert content.notion_deal_memo is not None
        assert content.notion_deal_memo.company_name == "AcmeCorp"

    def test_aggregate_skips_deal_memo_for_external_followup(self):
        """aggregate(is_external_first=False) does NOT fetch deal memo."""
        meeting = _make_meeting()

        notion_mock = MagicMock()
        notion_mock.get_records_for_meeting.return_value = []

        from src.briefing.context_aggregator import MeetingContextAggregator
        aggregator = MeetingContextAggregator(notion_client=notion_mock)
        content = aggregator.aggregate(meeting, is_external_first=False)

        assert content.notion_deal_memo is None
        # get_company_page_content should NOT have been called
        notion_mock.get_company_page_content.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Formatter: _build_notion_deal_memo_blocks
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildNotionDealMemoBlocks:
    """Unit tests for the deal memo Block Kit section builder."""

    def _build(self, raw_content):
        from src.briefing.meeting_briefing_formatter import (
            _build_notion_deal_memo_blocks,
        )
        return _build_notion_deal_memo_blocks(raw_content)

    def test_returns_empty_list_when_memo_none_and_available(self):
        """Non-EXTERNAL_FIRST: no section rendered."""
        rc = _make_raw_content()
        assert rc.notion_deal_memo is None
        assert rc.notion_deal_memo_available is True
        blocks = self._build(rc)
        assert blocks == []

    def test_error_annotation_when_unavailable(self):
        """Client error: section shown with ⚠️ 확인 불가."""
        from src.briefing.context_aggregator import AggregationError

        rc = _make_raw_content()
        rc.notion_deal_memo_available = False
        rc.errors.append(
            AggregationError(source="notion_deal_memo", message="Timeout error")
        )
        blocks = self._build(rc)

        assert len(blocks) >= 2  # divider + error block
        text_blocks = [b for b in blocks if b.get("type") == "section"]
        assert text_blocks
        assert "확인 불가" in text_blocks[0]["text"]["text"]
        assert "Timeout error" in text_blocks[0]["text"]["text"]

    def test_happy_path_renders_company_and_status(self):
        """Found memo: company name and status appear in output."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(
            company_name="AcmeCorp",
            status="심사 중",
        )
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        assert "AcmeCorp" in full_text
        assert "심사 중" in full_text

    def test_happy_path_includes_notion_url_link(self):
        """Found memo with URL: renders linked company name."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(
            company_name="LinkedCo",
            page_id="page-link",
        )
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        assert "https://notion.so/page-link" in full_text
        assert "LinkedCo" in full_text

    def test_happy_path_includes_body_sections(self):
        """Body content (investment thesis) appears in the blocks."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(
            body_text="투자 포인트\n혁신적인 B2B SaaS",
        )
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        assert "투자 포인트" in full_text

    def test_blocks_fetched_false_shows_unavailable(self):
        """When blocks_fetched=False, body shows '확인 불가' annotation."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(
            blocks_fetched=False,
        )
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        assert "확인 불가" in full_text

    def test_has_divider_before_content(self):
        """Section is preceded by a divider block."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content()
        blocks = self._build(rc)

        assert blocks[0]["type"] == "divider"

    def test_body_text_truncated_at_600_chars(self):
        """Very long body is truncated so the block stays compact."""
        from src.notion.notion_client import NotionPageContent, NotionPageSection

        long_body = "A" * 1000
        rc = _make_raw_content()
        rc.notion_deal_memo = NotionPageContent(
            page_id="p1",
            company_name="VeryLongCo",
            status="심사",
            blocks_fetched=True,
            body_text=long_body,
            sections=[NotionPageSection(heading="", content=long_body)],
        )
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        # The truncated text must end with "…"
        assert "…" in full_text

    def test_date_value_appears(self):
        """date_value is shown as a badge in the section."""
        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(date_value="2024-06-01")
        blocks = self._build(rc)

        full_text = " ".join(
            b["text"]["text"] for b in blocks if b.get("type") == "section"
        )
        assert "2024-06-01" in full_text


# ══════════════════════════════════════════════════════════════════════════════
# 5. format_meeting_briefing() integration
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatMeetingBriefingWithDealMemo:
    """Integration tests for deal memo section in the full briefing formatter."""

    def test_deal_memo_section_appears_for_external_first(self):
        """When deal memo is present, its content appears in the full briefing."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(
            company_name="AcmeCorp",
            status="심사 중",
        )
        _text, blocks = format_meeting_briefing(rc)

        # Find all section text blocks
        all_text = " ".join(
            b["text"]["text"]
            for b in blocks
            if b.get("type") == "section" and "text" in b
        )
        assert "딜 메모" in all_text or "AcmeCorp" in all_text

    def test_deal_memo_absent_for_non_external_first(self):
        """When memo is None and available, the section is not rendered."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        rc = _make_raw_content()
        # Explicitly leave notion_deal_memo=None, available=True
        assert rc.notion_deal_memo is None
        assert rc.notion_deal_memo_available is True

        _text, blocks = format_meeting_briefing(rc)

        # '딜 메모' heading should NOT appear in blocks
        all_text = " ".join(
            b["text"]["text"]
            for b in blocks
            if b.get("type") == "section" and "text" in b
        )
        assert "딜 메모" not in all_text

    def test_deal_memo_error_annotation_in_briefing(self):
        """When fetch failed, error annotation appears in the full briefing."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        rc = _make_raw_content()
        rc.notion_deal_memo_available = False
        rc.errors.append(
            AggregationError(source="notion_deal_memo", message="Notion is down")
        )
        _text, blocks = format_meeting_briefing(rc)

        all_text = " ".join(
            b["text"]["text"]
            for b in blocks
            if b.get("type") == "section" and "text" in b
        )
        assert "확인 불가" in all_text

    def test_fallback_text_includes_deal_memo(self):
        """Fallback plain text reflects deal memo when present."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content(company_name="FallbackCo")
        fallback, _blocks = format_meeting_briefing(rc)

        assert "딜 메모" in fallback
        assert "FallbackCo" in fallback

    def test_fallback_text_unavailable_annotation(self):
        """Fallback plain text shows '확인 불가' when memo fetch failed."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        from src.briefing.context_aggregator import AggregationError

        rc = _make_raw_content()
        rc.notion_deal_memo_available = False
        rc.errors.append(
            AggregationError(source="notion_deal_memo", message="err")
        )
        fallback, _blocks = format_meeting_briefing(rc)

        assert "확인 불가" in fallback

    def test_total_blocks_within_slack_limit(self):
        """Full briefing with deal memo stays within 50-block Slack limit."""
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing

        rc = _make_raw_content()
        rc.notion_deal_memo = _make_notion_page_content()
        _text, blocks = format_meeting_briefing(rc)

        assert len(blocks) <= 50
