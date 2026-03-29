"""
Notion client for querying the investment deal database.

Features:
- Dynamic schema discovery at startup (no hardcoded property names)
- Search records by company name or freetext query
- get_records_for_meeting(): high-level helper for the briefing aggregator
- get_company_page_content(): retrieve full page body for external_followup briefings
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from notion_client import Client
from notion_client.errors import APIResponseError

from src.config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
    NOTION_DB_ID,
    NOTION_TOKEN,
)

logger = logging.getLogger(__name__)


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class DatabaseSchema:
    """
    Discovered schema for a Notion database.
    Maps property names to their Notion type (e.g. 'title', 'select', 'date').
    Populated once at startup by NotionClient.connect().
    """
    database_id: str
    database_title: str = ""
    # property_name → property_type
    properties: dict[str, str] = field(default_factory=dict)

    # Inferred semantic fields (set during schema discovery)
    title_field: Optional[str] = None        # 'title' type property
    company_field: Optional[str] = None      # best-guess company name field
    status_field: Optional[str] = None       # select/status field for deal stage
    date_field: Optional[str] = None         # first 'date' type property

    def __str__(self) -> str:
        return (
            f"DatabaseSchema({self.database_title!r}, "
            f"{len(self.properties)} properties, "
            f"title_field={self.title_field!r})"
        )


@dataclass
class NotionRecord:
    """A single Notion database page (row) with key property values."""

    page_id: str
    url: str = ""
    title: str = ""
    company_name: str = ""
    status: str = ""
    # ISO-format date string or empty
    date_value: str = ""
    # All raw property values as a flat dict for additional context
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "url": self.url,
            "title": self.title,
            "company_name": self.company_name,
            "status": self.status,
            "date_value": self.date_value,
            "properties": {
                k: v for k, v in self.properties.items()
                if not isinstance(v, (dict, list)) or len(str(v)) < 500
            },
        }


@dataclass
class NotionPageSection:
    """A single content section extracted from a Notion page's body blocks."""

    heading: str = ""       # Heading text (empty for top-level paragraphs)
    content: str = ""       # Concatenated text content of this section

    def to_dict(self) -> dict:
        return {"heading": self.heading, "content": self.content}


@dataclass
class NotionPageContent:
    """
    Full content of a Notion page, combining database properties and body
    blocks.

    Used in external_followup briefings to provide investment context for a
    company. Returned by ``NotionClient.get_company_page_content()``.

    Attributes
    ----------
    page_id:
        Notion page (database row) UUID.
    url:
        Canonical Notion page URL.
    title:
        Value of the page's title property.
    company_name:
        Best-effort company name (title field or dedicated company field).
    status:
        Deal stage / status value (empty if not discovered).
    date_value:
        Primary date property value (ISO string, empty if not discovered).
    body_text:
        Full plain-text concatenation of all block content (capped at
        ``max_content_chars`` passed to ``get_company_page_content``).
    sections:
        Structured body: a list of :class:`NotionPageSection` objects,
        each grouping content under a heading.  Top-level paragraphs before
        the first heading are in a section with ``heading=""``.
    properties:
        Flat dict of all extracted property values (for additional context).
    blocks_fetched:
        True if the block content API call succeeded; False on failure.
    """

    page_id: str
    url: str = ""
    title: str = ""
    company_name: str = ""
    status: str = ""
    date_value: str = ""
    body_text: str = ""
    sections: list[NotionPageSection] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    blocks_fetched: bool = False

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "url": self.url,
            "title": self.title,
            "company_name": self.company_name,
            "status": self.status,
            "date_value": self.date_value,
            "body_text": self.body_text,
            "sections": [s.to_dict() for s in self.sections],
            "blocks_fetched": self.blocks_fetched,
            "properties": {
                k: v for k, v in self.properties.items()
                if not isinstance(v, (dict, list)) or len(str(v)) < 500
            },
        }

    def to_briefing_summary(self, max_chars: int = 800) -> str:
        """
        Return a concise text summary of this page suitable for inclusion in
        a Slack briefing message.

        Format::

            [Company] ACME Corp  |  [Status] 심사 중  |  [Date] 2024-01-15
            ---
            Section A heading
            Section A content...

            Section B heading
            ...

        The output is capped at *max_chars* characters.  If sections are
        present they are used; otherwise ``body_text`` is used directly.
        Incomplete content is marked with ``…`` when truncated.
        """
        parts: list[str] = []

        # Header line with key properties
        header_parts = []
        if self.company_name:
            header_parts.append(f"[회사] {self.company_name}")
        if self.status:
            header_parts.append(f"[상태] {self.status}")
        if self.date_value:
            header_parts.append(f"[날짜] {self.date_value}")
        if header_parts:
            parts.append("  |  ".join(header_parts))

        if not self.blocks_fetched:
            parts.append("(페이지 본문 확인 불가)")
            return "\n".join(parts)[:max_chars]

        # Body content from sections or raw body_text
        body = ""
        if self.sections:
            section_lines: list[str] = []
            for sec in self.sections:
                if sec.heading:
                    section_lines.append(f"▶ {sec.heading}")
                if sec.content:
                    # Indent content lines for readability
                    for line in sec.content.splitlines():
                        if line.strip():
                            section_lines.append(f"  {line.strip()}")
            body = "\n".join(section_lines)
        elif self.body_text:
            body = self.body_text

        if body:
            parts.append(body)

        combined = "\n".join(parts)
        if len(combined) > max_chars:
            combined = combined[: max_chars - 1] + "…"
        return combined


# ── Property value extraction helpers ───────────────────────────────────────

def _get_title_value(prop: dict) -> str:
    """Extract plain-text from a Notion 'title' property."""
    parts = prop.get("title", [])
    return "".join(p.get("plain_text", "") for p in parts).strip()


def _get_rich_text_value(prop: dict) -> str:
    """Extract plain-text from a Notion 'rich_text' property."""
    parts = prop.get("rich_text", [])
    return "".join(p.get("plain_text", "") for p in parts).strip()


def _get_select_value(prop: dict) -> str:
    """Extract label from a Notion 'select' or 'status' property."""
    sel = prop.get("select") or prop.get("status")
    if sel:
        return sel.get("name", "")
    return ""


def _get_multi_select_values(prop: dict) -> list[str]:
    items = prop.get("multi_select", [])
    return [i.get("name", "") for i in items]


def _get_date_value(prop: dict) -> str:
    """Extract start date string from a Notion 'date' property."""
    date = prop.get("date")
    if date:
        return date.get("start", "") or ""
    return ""


def _get_number_value(prop: dict) -> Optional[float]:
    return prop.get("number")


def _get_url_value(prop: dict) -> str:
    return prop.get("url", "") or ""


def _get_email_value(prop: dict) -> str:
    return prop.get("email", "") or ""


def _get_people_names(prop: dict) -> list[str]:
    people = prop.get("people", [])
    names: list[str] = []
    for p in people:
        name = p.get("name", "") or p.get("id", "")
        if name:
            names.append(name)
    return names


def _extract_prop_value(prop_type: str, prop_data: dict) -> Any:
    """Dispatch to the correct extractor based on property type."""
    dispatch = {
        "title": _get_title_value,
        "rich_text": _get_rich_text_value,
        "select": _get_select_value,
        "status": _get_select_value,
        "multi_select": _get_multi_select_values,
        "date": _get_date_value,
        "number": _get_number_value,
        "url": _get_url_value,
        "email": _get_email_value,
        "people": _get_people_names,
    }
    extractor = dispatch.get(prop_type)
    if extractor:
        return extractor(prop_data)
    # Fallback: return raw value for unknown types
    return str(prop_data.get(prop_type, ""))


# ── Block content extraction helpers ────────────────────────────────────────

# Notion block types that carry rich-text content
_RICH_TEXT_BLOCK_TYPES = {
    "paragraph",
    "bulleted_list_item",
    "numbered_list_item",
    "to_do",
    "toggle",
    "quote",
    "callout",
    "code",
}

# Notion block types that represent headings
_HEADING_BLOCK_TYPES = {"heading_1", "heading_2", "heading_3"}


def _extract_rich_text_content(rich_text_list: list[dict]) -> str:
    """
    Concatenate plain_text fragments from a Notion rich_text array.

    This is the lowest-level extractor used for paragraph, list items, etc.
    """
    return "".join(
        part.get("plain_text", "")
        for part in rich_text_list
        if isinstance(part, dict)
    ).strip()


def _extract_block_text(block: dict) -> tuple[str, bool]:
    """
    Extract the visible text from a single Notion block.

    Returns
    -------
    (text, is_heading):
        *text*       – extracted plain text (empty string if none).
        *is_heading* – True when the block is a heading type.
    """
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})

    if block_type in _HEADING_BLOCK_TYPES:
        rich_text = block_data.get("rich_text", [])
        return _extract_rich_text_content(rich_text), True

    if block_type in _RICH_TEXT_BLOCK_TYPES:
        rich_text = block_data.get("rich_text", [])
        # For to_do items, prefix with checkbox state
        if block_type == "to_do":
            checked = block_data.get("checked", False)
            prefix = "☑ " if checked else "☐ "
            return prefix + _extract_rich_text_content(rich_text), False
        return _extract_rich_text_content(rich_text), False

    # Unsupported block type (image, embed, table, etc.)
    return "", False


def _blocks_to_sections(blocks: list[dict]) -> list[NotionPageSection]:
    """
    Convert a list of raw Notion block dicts into a list of
    :class:`NotionPageSection` objects.

    Algorithm:
    - Walk blocks sequentially.
    - When a heading block is encountered, start a new section.
    - All content before the first heading goes into a section with
      ``heading=""``.
    - Non-heading blocks append their text to the current section's content.
    - Empty sections (no heading and no content) are omitted.
    """
    sections: list[NotionPageSection] = []
    current_heading = ""
    current_lines: list[str] = []

    def _flush():
        content = "\n".join(line for line in current_lines if line)
        if current_heading or content:
            sections.append(
                NotionPageSection(heading=current_heading, content=content)
            )

    for block in blocks:
        text, is_heading = _extract_block_text(block)
        if not text:
            continue
        if is_heading:
            _flush()
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    _flush()
    return sections


# ── Schema discovery helpers ─────────────────────────────────────────────────

# Heuristic name patterns for company-name field detection
_COMPANY_FIELD_PATTERNS = [
    r"company", r"회사", r"기업", r"스타트업", r"startup",
    r"portfolio", r"포트폴리오", r"brand", r"name",
]


def _looks_like_company_field(name: str) -> bool:
    name_lower = name.lower()
    return any(re.search(pat, name_lower) for pat in _COMPANY_FIELD_PATTERNS)


_STATUS_FIELD_PATTERNS = [
    r"status", r"stage", r"단계", r"상태", r"진행",
    r"phase", r"deal.?stage",
]


def _looks_like_status_field(name: str) -> bool:
    name_lower = name.lower()
    return any(re.search(pat, name_lower) for pat in _STATUS_FIELD_PATTERNS)


# ── Client ───────────────────────────────────────────────────────────────────

class NotionClient:
    """
    Wrapper around the notion-client library for deal database queries.

    Schema is discovered once at connect() time and cached on self.schema.
    All subsequent queries use the discovered field names.
    """

    def __init__(self) -> None:
        self._client: Optional[Client] = None
        self.schema: Optional[DatabaseSchema] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate and discover the database schema."""
        self._client = Client(auth=NOTION_TOKEN)
        self.schema = self._discover_schema(NOTION_DB_ID)
        logger.info("NotionClient connected. Schema: %s", self.schema)

    def _ensure_connected(self) -> None:
        if self._client is None or self.schema is None:
            self.connect()

    # ── retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute *fn* with up to API_RETRY_ATTEMPTS retries."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_connected()
                return fn(*args, **kwargs)
            except APIResponseError as exc:
                logger.warning(
                    "Notion API error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "Notion error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"Notion API failed after {API_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    # ── schema discovery ──────────────────────────────────────────────────────

    def _discover_schema(self, database_id: str) -> DatabaseSchema:
        """
        Retrieve database metadata from Notion and build a DatabaseSchema.
        Property names and types are discovered dynamically — nothing is
        hardcoded.

        NOTE: uses its own retry loop instead of _call_with_retry to avoid
        infinite recursion (_call_with_retry → _ensure_connected → connect
        → _discover_schema → _call_with_retry …).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                db_meta = self._client.databases.retrieve(database_id=database_id)
                last_exc = None
                break
            except Exception as exc:
                logger.warning(
                    "Notion schema discovery (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
                if attempt < API_RETRY_ATTEMPTS:
                    time.sleep(API_RETRY_DELAY_SECONDS)

        try:
            if last_exc is not None:
                raise RuntimeError(
                    f"Notion API failed after {API_RETRY_ATTEMPTS} attempts"
                ) from last_exc
        except RuntimeError as exc:
            logger.error("Could not retrieve Notion database schema: %s", exc)
            return DatabaseSchema(database_id=database_id)

        # Extract database title
        title_parts = db_meta.get("title", [])
        db_title = "".join(p.get("plain_text", "") for p in title_parts).strip()

        # Build property map: name → type
        raw_props = db_meta.get("properties", {})
        prop_map: dict[str, str] = {}
        for prop_name, prop_meta in raw_props.items():
            prop_map[prop_name] = prop_meta.get("type", "unknown")

        schema = DatabaseSchema(
            database_id=database_id,
            database_title=db_title,
            properties=prop_map,
        )

        # ── Semantic field inference ──────────────────────────────────────────
        for name, ptype in prop_map.items():
            if ptype == "title" and schema.title_field is None:
                schema.title_field = name

            if ptype in ("select", "status") and schema.status_field is None:
                if _looks_like_status_field(name):
                    schema.status_field = name

            if ptype == "date" and schema.date_field is None:
                schema.date_field = name

        # Company field: prefer title field, but check for a dedicated field
        company_candidates = [
            name for name in prop_map
            if _looks_like_company_field(name)
        ]
        if company_candidates:
            schema.company_field = company_candidates[0]
        elif schema.title_field:
            schema.company_field = schema.title_field

        # If no status yet, pick first select/status field
        if schema.status_field is None:
            for name, ptype in prop_map.items():
                if ptype in ("select", "status"):
                    schema.status_field = name
                    break

        logger.info(
            "Notion schema discovered: title=%r, company=%r, status=%r, date=%r | %d properties",
            schema.title_field,
            schema.company_field,
            schema.status_field,
            schema.date_field,
            len(prop_map),
        )
        return schema

    # ── page parsing ──────────────────────────────────────────────────────────

    def _parse_page(self, page: dict) -> NotionRecord:
        """Convert a raw Notion page object into a NotionRecord."""
        page_id = page.get("id", "")
        url = page.get("url", "")
        raw_props = page.get("properties", {})

        # Extract flat values for all properties
        flat_props: dict[str, Any] = {}
        schema = self.schema

        for prop_name, prop_data in raw_props.items():
            prop_type = prop_data.get("type", "")
            value = _extract_prop_value(prop_type, prop_data)
            flat_props[prop_name] = value

        # Semantic field extraction
        title = ""
        if schema and schema.title_field:
            title = flat_props.get(schema.title_field, "") or ""
            if isinstance(title, list):
                title = ", ".join(str(x) for x in title)

        company_name = ""
        if schema and schema.company_field:
            company_name = flat_props.get(schema.company_field, "") or ""
            if isinstance(company_name, list):
                company_name = ", ".join(str(x) for x in company_name)
        if not company_name:
            company_name = title

        status = ""
        if schema and schema.status_field:
            status = flat_props.get(schema.status_field, "") or ""
            if isinstance(status, list):
                status = ", ".join(str(x) for x in status)

        date_value = ""
        if schema and schema.date_field:
            date_value = flat_props.get(schema.date_field, "") or ""

        return NotionRecord(
            page_id=page_id,
            url=url,
            title=title,
            company_name=str(company_name),
            status=str(status),
            date_value=str(date_value),
            properties=flat_props,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def query_database(
        self,
        filter_payload: Optional[dict] = None,
        sorts: Optional[list[dict]] = None,
        max_results: int = 20,
    ) -> list[NotionRecord]:
        """
        Run a database query with an optional Notion filter and sort.
        Returns parsed NotionRecord objects.
        """
        def _query():
            params: dict[str, Any] = {
                "database_id": NOTION_DB_ID,
                "page_size": min(max_results, 100),
            }
            if filter_payload:
                params["filter"] = filter_payload
            if sorts:
                params["sorts"] = sorts
            return self._client.databases.query(**params)

        try:
            result = self._call_with_retry(_query)
        except RuntimeError:
            logger.error("Notion database query failed")
            return []

        pages = result.get("results", [])
        return [self._parse_page(p) for p in pages]

    def search_by_company_name(
        self,
        company_name: str,
        max_results: int = 10,
    ) -> list[NotionRecord]:
        """
        Search Notion deal records whose company/title field contains
        *company_name*.  Uses a 'contains' filter on the discovered
        title/company field.
        """
        if not self.schema:
            self._ensure_connected()

        search_field = (self.schema.company_field or self.schema.title_field) if self.schema else None
        if not search_field:
            logger.warning("No title/company field found in Notion schema; returning empty")
            return []

        field_type = (self.schema.properties.get(search_field, "title")) if self.schema else "title"
        # Notion filter for title vs rich_text
        if field_type == "title":
            filter_payload = {
                "property": search_field,
                "title": {"contains": company_name},
            }
        else:
            filter_payload = {
                "property": search_field,
                "rich_text": {"contains": company_name},
            }

        logger.debug(
            "Notion search: company_name=%r on field=%r (type=%r)",
            company_name, search_field, field_type,
        )
        return self.query_database(
            filter_payload=filter_payload,
            max_results=max_results,
        )

    def get_records_for_meeting(
        self,
        external_emails: list[str],
        meeting_title: str,
        max_records: int = 10,
    ) -> list[NotionRecord]:
        """
        Find Notion deal records relevant to an upcoming meeting.

        Strategy:
        1. Extract company domains from external attendee emails.
        2. For each domain root (e.g. "acme" from "acme.com"), search deals.
        3. Also search by keywords from the meeting title.
        4. Deduplicate by page_id.

        Returns at most *max_records* results.
        """
        self._ensure_connected()

        seen: dict[str, NotionRecord] = {}

        # 1. Domain-based search from external attendee emails
        for email_addr in external_emails:
            domain = _domain_root(email_addr)
            if domain and len(domain) >= 3:
                records = self.search_by_company_name(domain, max_results=5)
                for r in records:
                    seen[r.page_id] = r

        # 2. Keyword search from meeting title
        keywords = _title_keywords(meeting_title)
        for kw in keywords[:3]:  # limit to 3 keyword queries
            records = self.search_by_company_name(kw, max_results=5)
            for r in records:
                seen[r.page_id] = r

        result = list(seen.values())[:max_records]
        logger.info(
            "Notion context: found %d unique records for meeting '%s'",
            len(result),
            meeting_title,
        )
        return result

    # ── Page block content retrieval ──────────────────────────────────────────

    def get_page_blocks(self, page_id: str, max_blocks: int = 200) -> list[dict]:
        """
        Fetch and return the block children for a Notion page.

        Uses the Notion Blocks API (``blocks.children.list``) to retrieve
        the top-level blocks of *page_id*.  Does **not** recurse into nested
        blocks (e.g., toggle children) to keep the call count manageable.

        Parameters
        ----------
        page_id:
            The UUID of the Notion page whose blocks to retrieve.
        max_blocks:
            Maximum number of blocks to return (default: 200).  Notion
            returns blocks in pages of up to 100; we issue at most two
            requests (200 blocks).

        Returns
        -------
        List of raw Notion block dicts.  Returns an empty list on API
        failure (the caller decides how to handle the unavailability).
        """
        self._ensure_connected()
        blocks: list[dict] = []
        start_cursor: Optional[str] = None

        while len(blocks) < max_blocks:
            page_size = min(100, max_blocks - len(blocks))

            def _fetch_page(cursor=start_cursor, size=page_size):
                kwargs: dict[str, Any] = {
                    "block_id": page_id,
                    "page_size": size,
                }
                if cursor:
                    kwargs["start_cursor"] = cursor
                return self._client.blocks.children.list(**kwargs)

            try:
                response = self._call_with_retry(_fetch_page)
            except RuntimeError as exc:
                logger.warning(
                    "get_page_blocks: failed to fetch blocks for page %s: %s",
                    page_id,
                    exc,
                )
                break

            batch = response.get("results", [])
            blocks.extend(batch)

            if response.get("has_more") and len(blocks) < max_blocks:
                start_cursor = response.get("next_cursor")
            else:
                break

        logger.debug(
            "get_page_blocks: page_id=%s fetched %d blocks", page_id, len(blocks)
        )
        return blocks

    def get_company_page_content(
        self,
        company_name: str,
        max_content_chars: int = 2000,
        max_blocks: int = 200,
    ) -> Optional["NotionPageContent"]:
        """
        Retrieve full Notion page content for a company name.

        This is the primary entry point for Sub-AC 1.  It:

        1. Searches the deal database for records matching *company_name*
           (using :meth:`search_by_company_name`).
        2. Takes the best match (first result).
        3. Fetches the page's block children via :meth:`get_page_blocks`.
        4. Extracts structured sections and concatenated body text.
        5. Returns a :class:`NotionPageContent` with both properties and body.

        Parameters
        ----------
        company_name:
            Company name (or partial name) to search for.
        max_content_chars:
            Cap on ``body_text`` length.  The full ``sections`` list is
            always returned untruncated; only the flat ``body_text`` string
            is trimmed.
        max_blocks:
            Maximum blocks to fetch from the page body (default: 200).

        Returns
        -------
        :class:`NotionPageContent` for the best-matching page, or ``None``
        if no matching record is found.

        Notes
        -----
        * If the block-content fetch fails, ``blocks_fetched`` is ``False``
          and ``body_text`` / ``sections`` are empty — the caller can
          annotate the briefing with ``확인 불가``.
        * Property data (title, status, date) is always included from the
          database record regardless of block fetch success.
        """
        self._ensure_connected()

        # Step 1: find matching database records
        records = self.search_by_company_name(company_name, max_results=3)
        if not records:
            logger.info(
                "get_company_page_content: no records found for company=%r",
                company_name,
            )
            return None

        # Step 2: use the best match (first result from contains filter)
        best_record = records[0]

        # Build base NotionPageContent from the record's properties
        page_content = NotionPageContent(
            page_id=best_record.page_id,
            url=best_record.url,
            title=best_record.title,
            company_name=best_record.company_name,
            status=best_record.status,
            date_value=best_record.date_value,
            properties=best_record.properties,
        )

        # Step 3 & 4: fetch and parse block content
        try:
            raw_blocks = self.get_page_blocks(
                best_record.page_id, max_blocks=max_blocks
            )
            if raw_blocks:
                page_content.sections = _blocks_to_sections(raw_blocks)
                # Produce flat body_text for quick access / AI prompts
                all_lines: list[str] = []
                for sec in page_content.sections:
                    if sec.heading:
                        all_lines.append(sec.heading)
                    if sec.content:
                        all_lines.extend(sec.content.splitlines())
                full_text = "\n".join(line for line in all_lines if line)
                if len(full_text) > max_content_chars:
                    full_text = full_text[:max_content_chars - 1] + "…"
                page_content.body_text = full_text
            page_content.blocks_fetched = True
        except Exception as exc:
            logger.warning(
                "get_company_page_content: block fetch failed for page %s: %s",
                best_record.page_id,
                exc,
            )
            page_content.blocks_fetched = False

        logger.info(
            "get_company_page_content: company=%r → page_id=%s "
            "blocks_fetched=%s sections=%d body_chars=%d",
            company_name,
            page_content.page_id,
            page_content.blocks_fetched,
            len(page_content.sections),
            len(page_content.body_text),
        )
        return page_content


# ── Helpers ──────────────────────────────────────────────────────────────────

def _domain_root(email_addr: str) -> str:
    """
    Extract the primary domain label from an email address.
    'john@acme-corp.com' → 'acme-corp'
    'user@sub.example.co.kr' → 'example'
    """
    try:
        domain = email_addr.split("@")[1].lower()
        parts = domain.split(".")
        # Drop common TLD suffixes: com, co, kr, net, org, io, ai …
        _TLDS = {"com", "co", "kr", "net", "org", "io", "ai", "vc", "biz"}
        meaningful = [p for p in parts if p not in _TLDS and len(p) > 1]
        if meaningful:
            return meaningful[-1]
    except (IndexError, AttributeError):
        pass
    return ""


_KO_STOP = {
    "주식회사", "유한", "합자", "재단", "협회", "협동조합",
    "미팅", "회의", "논의", "검토", "협의", "관련", "건",
}
_EN_STOP = {
    "meeting", "call", "sync", "discussion", "inc", "corp", "ltd",
    "the", "and", "for", "with", "of", "about", "regarding",
}


def _title_keywords(title: str, max_kw: int = 3) -> list[str]:
    """
    Extract meaningful keywords from a meeting title for Notion search.
    Returns short list of candidate company/topic names.
    """
    tokens = re.split(r"[\s/\-_\[\]()]+", title)
    results: list[str] = []
    for tok in tokens:
        tok_clean = tok.strip(".,!?;:\"'")
        tok_lower = tok_clean.lower()
        if (
            len(tok_clean) >= 2
            and tok_lower not in _KO_STOP
            and tok_lower not in _EN_STOP
        ):
            results.append(tok_clean)
        if len(results) >= max_kw:
            break
    return results
