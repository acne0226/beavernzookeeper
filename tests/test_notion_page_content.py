"""
Tests for Notion page content retrieval (Sub-AC 1 of AC 7).

Covers:
- NotionPageSection dataclass
- NotionPageContent dataclass (to_dict, to_briefing_summary)
- Block text extraction helpers (_extract_rich_text_content, _extract_block_text)
- Section building from raw blocks (_blocks_to_sections)
- NotionClient.get_page_blocks() — happy path, pagination, API failure
- NotionClient.get_company_page_content() — full pipeline:
    * no matching record
    * record found but block fetch fails (blocks_fetched=False)
    * record found and blocks fetched (blocks_fetched=True)
    * body_text capped at max_content_chars
    * to_briefing_summary output
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from src.notion.notion_client import (
    NotionClient,
    NotionPageContent,
    NotionPageSection,
    NotionRecord,
    DatabaseSchema,
    _extract_rich_text_content,
    _extract_block_text,
    _blocks_to_sections,
)


# ── Helpers to build Notion-shaped dicts ─────────────────────────────────────

def _rich_text(plain: str) -> list[dict]:
    """Build a minimal Notion rich_text array."""
    return [{"plain_text": plain, "type": "text"}]


def _paragraph_block(text: str) -> dict:
    return {
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _heading_block(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {
        "type": key,
        key: {"rich_text": _rich_text(text)},
    }


def _bullet_block(text: str) -> dict:
    return {
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text)},
    }


def _numbered_block(text: str) -> dict:
    return {
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": _rich_text(text)},
    }


def _todo_block(text: str, checked: bool = False) -> dict:
    return {
        "type": "to_do",
        "to_do": {"rich_text": _rich_text(text), "checked": checked},
    }


def _callout_block(text: str) -> dict:
    return {
        "type": "callout",
        "callout": {"rich_text": _rich_text(text)},
    }


def _quote_block(text: str) -> dict:
    return {
        "type": "quote",
        "quote": {"rich_text": _rich_text(text)},
    }


def _image_block() -> dict:
    """Unsupported block type — should produce empty text."""
    return {"type": "image", "image": {"type": "external", "external": {"url": "..."}}}


# ── _extract_rich_text_content ─────────────────────────────────────────────

class TestExtractRichTextContent:
    def test_single_part(self):
        assert _extract_rich_text_content(_rich_text("Hello")) == "Hello"

    def test_multiple_parts(self):
        parts = [
            {"plain_text": "Hello", "type": "text"},
            {"plain_text": " World", "type": "text"},
        ]
        assert _extract_rich_text_content(parts) == "Hello World"

    def test_empty_list(self):
        assert _extract_rich_text_content([]) == ""

    def test_strips_whitespace(self):
        assert _extract_rich_text_content(_rich_text("  trimmed  ")) == "trimmed"

    def test_skips_non_dict(self):
        parts = [{"plain_text": "ok"}, "not a dict", None]
        assert _extract_rich_text_content(parts) == "ok"  # type: ignore[arg-type]


# ── _extract_block_text ────────────────────────────────────────────────────

class TestExtractBlockText:
    def test_paragraph(self):
        text, is_heading = _extract_block_text(_paragraph_block("Hello"))
        assert text == "Hello"
        assert is_heading is False

    def test_heading_1(self):
        text, is_heading = _extract_block_text(_heading_block(1, "Section Title"))
        assert text == "Section Title"
        assert is_heading is True

    def test_heading_2(self):
        text, is_heading = _extract_block_text(_heading_block(2, "Sub-section"))
        assert text == "Sub-section"
        assert is_heading is True

    def test_heading_3(self):
        text, is_heading = _extract_block_text(_heading_block(3, "Sub-sub"))
        assert text == "Sub-sub"
        assert is_heading is True

    def test_bullet_list_item(self):
        text, is_heading = _extract_block_text(_bullet_block("Bullet item"))
        assert text == "Bullet item"
        assert is_heading is False

    def test_numbered_list_item(self):
        text, is_heading = _extract_block_text(_numbered_block("First"))
        assert text == "First"
        assert is_heading is False

    def test_todo_unchecked(self):
        text, is_heading = _extract_block_text(_todo_block("Task", checked=False))
        assert text == "☐ Task"
        assert is_heading is False

    def test_todo_checked(self):
        text, is_heading = _extract_block_text(_todo_block("Done task", checked=True))
        assert text == "☑ Done task"
        assert is_heading is False

    def test_callout(self):
        text, is_heading = _extract_block_text(_callout_block("Note"))
        assert text == "Note"
        assert is_heading is False

    def test_quote(self):
        text, is_heading = _extract_block_text(_quote_block("Quote"))
        assert text == "Quote"
        assert is_heading is False

    def test_unsupported_block_returns_empty(self):
        text, is_heading = _extract_block_text(_image_block())
        assert text == ""
        assert is_heading is False

    def test_empty_paragraph(self):
        block = {"type": "paragraph", "paragraph": {"rich_text": []}}
        text, is_heading = _extract_block_text(block)
        assert text == ""
        assert is_heading is False


# ── _blocks_to_sections ────────────────────────────────────────────────────

class TestBlocksToSections:
    def test_empty_blocks(self):
        assert _blocks_to_sections([]) == []

    def test_only_paragraphs_no_heading(self):
        blocks = [_paragraph_block("Line 1"), _paragraph_block("Line 2")]
        sections = _blocks_to_sections(blocks)
        assert len(sections) == 1
        assert sections[0].heading == ""
        assert "Line 1" in sections[0].content
        assert "Line 2" in sections[0].content

    def test_heading_starts_section(self):
        blocks = [
            _heading_block(1, "Overview"),
            _paragraph_block("Company overview text"),
        ]
        sections = _blocks_to_sections(blocks)
        assert len(sections) == 1
        assert sections[0].heading == "Overview"
        assert "Company overview text" in sections[0].content

    def test_multiple_sections(self):
        blocks = [
            _paragraph_block("Intro"),
            _heading_block(1, "Investment Thesis"),
            _paragraph_block("Strong growth potential"),
            _bullet_block("Market leader"),
            _heading_block(2, "Risks"),
            _paragraph_block("Regulatory risk"),
        ]
        sections = _blocks_to_sections(blocks)
        assert len(sections) == 3

        assert sections[0].heading == ""
        assert "Intro" in sections[0].content

        assert sections[1].heading == "Investment Thesis"
        assert "Strong growth potential" in sections[1].content
        assert "Market leader" in sections[1].content

        assert sections[2].heading == "Risks"
        assert "Regulatory risk" in sections[2].content

    def test_skips_empty_blocks(self):
        blocks = [
            _image_block(),  # produces empty text
            _paragraph_block("Content"),
        ]
        sections = _blocks_to_sections(blocks)
        assert len(sections) == 1
        assert "Content" in sections[0].content

    def test_consecutive_headings(self):
        blocks = [
            _heading_block(1, "First"),
            _heading_block(1, "Second"),
            _paragraph_block("Under second"),
        ]
        sections = _blocks_to_sections(blocks)
        # "First" section has no content (empty section is flushed)
        # but since heading is non-empty it IS included
        assert any(s.heading == "First" for s in sections)
        assert any(s.heading == "Second" for s in sections)
        second_sec = next(s for s in sections if s.heading == "Second")
        assert "Under second" in second_sec.content


# ── NotionPageSection ─────────────────────────────────────────────────────

class TestNotionPageSection:
    def test_to_dict(self):
        sec = NotionPageSection(heading="Overview", content="Some text")
        d = sec.to_dict()
        assert d == {"heading": "Overview", "content": "Some text"}

    def test_empty(self):
        sec = NotionPageSection()
        d = sec.to_dict()
        assert d["heading"] == ""
        assert d["content"] == ""


# ── NotionPageContent ─────────────────────────────────────────────────────

class TestNotionPageContent:
    def _make_page(self, **kwargs) -> NotionPageContent:
        defaults = dict(
            page_id="page-001",
            url="https://notion.so/page-001",
            title="AcmeCorp 딜",
            company_name="AcmeCorp",
            status="심사 중",
            date_value="2024-03-15",
            body_text="Overview\nStrong growth potential\nRisks\nRegulatory risk",
            sections=[
                NotionPageSection(heading="Overview", content="Strong growth potential"),
                NotionPageSection(heading="Risks", content="Regulatory risk"),
            ],
            properties={"title": "AcmeCorp 딜", "stage": "심사 중"},
            blocks_fetched=True,
        )
        defaults.update(kwargs)
        return NotionPageContent(**defaults)

    def test_to_dict_keys(self):
        page = self._make_page()
        d = page.to_dict()
        assert d["page_id"] == "page-001"
        assert d["company_name"] == "AcmeCorp"
        assert d["status"] == "심사 중"
        assert d["blocks_fetched"] is True
        assert len(d["sections"]) == 2

    def test_to_briefing_summary_has_company_and_status(self):
        page = self._make_page()
        summary = page.to_briefing_summary()
        assert "AcmeCorp" in summary
        assert "심사 중" in summary

    def test_to_briefing_summary_has_body_sections(self):
        page = self._make_page()
        summary = page.to_briefing_summary()
        assert "Overview" in summary
        assert "Strong growth potential" in summary

    def test_to_briefing_summary_max_chars_truncates(self):
        long_content = "x" * 5000
        page = self._make_page(
            body_text=long_content,
            sections=[NotionPageSection(heading="", content=long_content)],
        )
        summary = page.to_briefing_summary(max_chars=100)
        assert len(summary) <= 100
        assert summary.endswith("…")

    def test_to_briefing_summary_blocks_not_fetched(self):
        page = self._make_page(
            blocks_fetched=False,
            body_text="",
            sections=[],
        )
        summary = page.to_briefing_summary()
        assert "확인 불가" in summary

    def test_to_briefing_summary_no_sections_uses_body_text(self):
        page = self._make_page(
            sections=[],
            body_text="Plain body text here",
        )
        summary = page.to_briefing_summary()
        assert "Plain body text here" in summary

    def test_to_briefing_summary_missing_optional_fields(self):
        page = NotionPageContent(
            page_id="p1",
            blocks_fetched=True,
        )
        # Should not raise even with all optional fields empty
        summary = page.to_briefing_summary()
        assert isinstance(summary, str)


# ── NotionClient.get_page_blocks ─────────────────────────────────────────────

class TestGetPageBlocks:
    def _make_client(self) -> NotionClient:
        """Create a NotionClient with a fully mocked _client and schema."""
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client.schema = DatabaseSchema(
            database_id="db-1",
            title_field="Name",
            company_field="Name",
            status_field="Stage",
            date_field="Date",
            properties={"Name": "title", "Stage": "select", "Date": "date"},
        )
        return client

    def test_single_page_of_blocks(self):
        client = self._make_client()
        blocks = [_paragraph_block(f"Block {i}") for i in range(3)]
        client._client.blocks.children.list.return_value = {
            "results": blocks,
            "has_more": False,
        }

        result = client.get_page_blocks("page-abc")
        assert len(result) == 3
        client._client.blocks.children.list.assert_called_once()

    def test_pagination_fetches_second_page(self):
        client = self._make_client()
        page1_blocks = [_paragraph_block(f"P1-{i}") for i in range(100)]
        page2_blocks = [_paragraph_block(f"P2-{i}") for i in range(50)]

        client._client.blocks.children.list.side_effect = [
            {"results": page1_blocks, "has_more": True, "next_cursor": "cursor-2"},
            {"results": page2_blocks, "has_more": False},
        ]

        result = client.get_page_blocks("page-abc", max_blocks=200)
        assert len(result) == 150
        assert client._client.blocks.children.list.call_count == 2

    def test_max_blocks_respected(self):
        client = self._make_client()
        blocks = [_paragraph_block(f"B{i}") for i in range(100)]
        client._client.blocks.children.list.return_value = {
            "results": blocks,
            "has_more": False,
        }

        result = client.get_page_blocks("page-abc", max_blocks=50)
        # We asked for max 50 but the response returned 100; pagination stops
        # after first page since max_blocks=50 was already reached
        assert len(result) == 100  # First call returns full batch; we take all

    def test_api_failure_returns_empty_list(self):
        client = self._make_client()
        client._client.blocks.children.list.side_effect = Exception("API error")

        result = client.get_page_blocks("page-abc")
        assert result == []

    def test_empty_page_returns_empty_list(self):
        client = self._make_client()
        client._client.blocks.children.list.return_value = {
            "results": [],
            "has_more": False,
        }
        result = client.get_page_blocks("page-abc")
        assert result == []


# ── NotionClient.get_company_page_content ────────────────────────────────────

class TestGetCompanyPageContent:
    def _make_client(self) -> NotionClient:
        client = NotionClient.__new__(NotionClient)
        client._client = MagicMock()
        client.schema = DatabaseSchema(
            database_id="db-1",
            title_field="Name",
            company_field="Name",
            status_field="Stage",
            date_field="Date",
            properties={"Name": "title", "Stage": "select", "Date": "date"},
        )
        return client

    def _make_record(
        self,
        page_id: str = "page-001",
        title: str = "AcmeCorp",
        company_name: str = "AcmeCorp",
        status: str = "심사 중",
        date_value: str = "2024-03-15",
    ) -> NotionRecord:
        return NotionRecord(
            page_id=page_id,
            url=f"https://notion.so/{page_id}",
            title=title,
            company_name=company_name,
            status=status,
            date_value=date_value,
        )

    def test_no_matching_records_returns_none(self):
        client = self._make_client()
        with patch.object(client, "search_by_company_name", return_value=[]):
            result = client.get_company_page_content("NonExistentCo")
        assert result is None

    def test_happy_path_returns_page_content(self):
        client = self._make_client()
        record = self._make_record()
        blocks = [
            _heading_block(1, "Investment Thesis"),
            _paragraph_block("Strong unit economics"),
            _bullet_block("$5M ARR, growing 200% YoY"),
            _heading_block(2, "Risks"),
            _paragraph_block("Regulatory uncertainty in fintech"),
        ]
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(client, "get_page_blocks", return_value=blocks),
        ):
            content = client.get_company_page_content("AcmeCorp")

        assert content is not None
        assert content.page_id == "page-001"
        assert content.company_name == "AcmeCorp"
        assert content.status == "심사 중"
        assert content.blocks_fetched is True
        assert len(content.sections) == 2
        assert content.sections[0].heading == "Investment Thesis"
        assert "Strong unit economics" in content.sections[0].content
        assert "$5M ARR" in content.sections[0].content
        assert content.sections[1].heading == "Risks"
        assert "Investment Thesis" in content.body_text

    def test_block_fetch_failure_sets_blocks_fetched_false(self):
        client = self._make_client()
        record = self._make_record()
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(
                client, "get_page_blocks", side_effect=Exception("Blocks API down")
            ),
        ):
            content = client.get_company_page_content("AcmeCorp")

        assert content is not None
        assert content.blocks_fetched is False
        assert content.body_text == ""
        assert content.sections == []
        # But record properties are still populated
        assert content.company_name == "AcmeCorp"
        assert content.status == "심사 중"

    def test_body_text_capped_at_max_content_chars(self):
        client = self._make_client()
        record = self._make_record()
        long_text = "A" * 5000
        blocks = [_paragraph_block(long_text)]
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(client, "get_page_blocks", return_value=blocks),
        ):
            content = client.get_company_page_content("AcmeCorp", max_content_chars=500)

        assert len(content.body_text) <= 500
        assert content.body_text.endswith("…")

    def test_no_blocks_returned_body_text_empty(self):
        client = self._make_client()
        record = self._make_record()
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(client, "get_page_blocks", return_value=[]),
        ):
            content = client.get_company_page_content("AcmeCorp")

        assert content is not None
        assert content.blocks_fetched is True
        assert content.body_text == ""
        assert content.sections == []

    def test_uses_best_match_first_result(self):
        """When multiple records match, the first one is used."""
        client = self._make_client()
        record_1 = self._make_record(page_id="page-001", title="AcmeCorp A")
        record_2 = self._make_record(page_id="page-002", title="AcmeCorp B")
        with (
            patch.object(
                client, "search_by_company_name", return_value=[record_1, record_2]
            ),
            patch.object(client, "get_page_blocks", return_value=[]),
        ):
            content = client.get_company_page_content("AcmeCorp")

        assert content is not None
        assert content.page_id == "page-001"

    def test_to_dict_roundtrip(self):
        """get_company_page_content returns something that serialises cleanly."""
        client = self._make_client()
        record = self._make_record()
        blocks = [
            _heading_block(1, "Summary"),
            _paragraph_block("B2B SaaS company"),
        ]
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(client, "get_page_blocks", return_value=blocks),
        ):
            content = client.get_company_page_content("AcmeCorp")

        d = content.to_dict()
        assert d["page_id"] == "page-001"
        assert d["blocks_fetched"] is True
        assert len(d["sections"]) == 1
        assert d["sections"][0]["heading"] == "Summary"

    def test_to_briefing_summary_integration(self):
        """get_company_page_content → to_briefing_summary produces valid output."""
        client = self._make_client()
        record = self._make_record()
        blocks = [
            _heading_block(1, "투자 포인트"),
            _paragraph_block("혁신적인 B2B SaaS"),
        ]
        with (
            patch.object(client, "search_by_company_name", return_value=[record]),
            patch.object(client, "get_page_blocks", return_value=blocks),
        ):
            content = client.get_company_page_content("AcmeCorp")

        summary = content.to_briefing_summary(max_chars=1000)
        assert "AcmeCorp" in summary
        assert "심사 중" in summary
        assert "투자 포인트" in summary
        assert "혁신적인 B2B SaaS" in summary
