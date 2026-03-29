"""
Notion client for fetching portfolio company deadline items.

Discovers the Notion database schema dynamically at startup (no hardcoded
property names). Identifies date properties that likely represent deadlines
and queries for records with upcoming or overdue dates.

Provides:
- NotionClient          – authenticated wrapper around the Notion API
- NotionDeadlineItem    – structured deadline data model
- fetch_notion_deadlines() – convenience top-level function
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from notion_client import Client
from notion_client.errors import APIResponseError

from src.config import (
    NOTION_TOKEN,
    NOTION_DB_ID,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")

# ── Property type constants ────────────────────────────────────────────────────

_DATE_PROP_TYPES = {"date", "created_time", "last_edited_time"}

# Keywords suggesting a property is a deadline/due-date
_DEADLINE_KEYWORDS = [
    "deadline", "due", "마감", "기한", "예정", "일정", "date",
    "duedate", "due_date", "deadline_date", "close", "invest",
]

# Keywords suggesting a property is a company/deal name
_NAME_KEYWORDS = [
    "name", "title", "company", "deal", "startup", "portfolio",
    "회사", "기업", "딜", "포트폴리오", "이름",
]

# Keywords suggesting a stage/status property
_STATUS_KEYWORDS = [
    "stage", "status", "단계", "상태", "progress", "단계", "type",
]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class NotionDeadlineItem:
    """A single Notion record with a deadline date."""

    page_id: str
    name: str                    # Company / deal name
    deadline: date               # The deadline date
    deadline_prop: str           # Which property name the date came from
    status: str = ""             # Stage / status value (if found)
    url: str = ""                # Notion page URL
    is_overdue: bool = False     # True if deadline is in the past
    days_until: int = 0          # Negative = past, 0 = today, positive = future
    extra: dict = field(default_factory=dict)  # Additional raw properties

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "name": self.name,
            "deadline": self.deadline.isoformat(),
            "deadline_prop": self.deadline_prop,
            "status": self.status,
            "url": self.url,
            "is_overdue": self.is_overdue,
            "days_until": self.days_until,
        }


# ── Schema discovery ───────────────────────────────────────────────────────────

@dataclass
class _DbSchema:
    """Discovered schema for a Notion database."""
    name_prop: Optional[str]           # Title/name property
    deadline_props: list[str]          # All date-like properties
    primary_deadline_prop: Optional[str]  # Best candidate for deadline
    status_prop: Optional[str]         # Stage/status property
    all_props: dict[str, str]          # prop_name -> prop_type


def _score_deadline_prop(name: str) -> int:
    """
    Score a property name as a deadline candidate (higher = better).
    Returns 0 for non-matches.
    """
    name_lower = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    score = 0
    for kw in _DEADLINE_KEYWORDS:
        if kw.replace(" ", "").replace("_", "") in name_lower:
            score += 10
    return score


def _discover_schema(db_meta: dict) -> _DbSchema:
    """
    Inspect Notion database metadata to discover property roles.

    Parameters
    ----------
    db_meta:
        Raw response from notion_client.databases.retrieve()

    Returns
    -------
    _DbSchema with best-guess property assignments.
    """
    properties: dict[str, dict] = db_meta.get("properties", {})
    all_props: dict[str, str] = {
        name: prop.get("type", "unknown")
        for name, prop in properties.items()
    }

    # Find title property (type == "title")
    name_prop: Optional[str] = next(
        (n for n, t in all_props.items() if t == "title"), None
    )

    # Find all date-type properties
    deadline_props = [n for n, t in all_props.items() if t in _DATE_PROP_TYPES]

    # Score each and pick the best candidate
    if deadline_props:
        scored = sorted(
            deadline_props,
            key=lambda n: _score_deadline_prop(n),
            reverse=True,
        )
        primary_deadline_prop = scored[0] if scored else None
    else:
        primary_deadline_prop = None

    # Find status / stage property
    status_prop: Optional[str] = None
    best_status_score = 0
    for name in all_props:
        name_lower = name.lower()
        score = sum(1 for kw in _STATUS_KEYWORDS if kw in name_lower)
        if score > best_status_score:
            best_status_score = score
            status_prop = name

    schema = _DbSchema(
        name_prop=name_prop,
        deadline_props=deadline_props,
        primary_deadline_prop=primary_deadline_prop,
        status_prop=status_prop,
        all_props=all_props,
    )

    logger.info(
        "Notion schema discovered: name=%r deadline=%r status=%r  all_date_props=%s",
        schema.name_prop,
        schema.primary_deadline_prop,
        schema.status_prop,
        schema.deadline_props,
    )
    return schema


# ── Value extractors ───────────────────────────────────────────────────────────

def _extract_title(prop: dict) -> str:
    """Extract plain text from a Notion title property value."""
    parts = prop.get("title", [])
    return "".join(
        t.get("plain_text", "")
        for t in parts
        if isinstance(t, dict)
    ).strip() or "(이름 없음)"


def _extract_date(prop: dict) -> Optional[date]:
    """
    Extract a date from a Notion date property value.

    Handles:
    - type "date"          → prop["date"]["start"]
    - type "created_time"  → prop["created_time"]
    - type "last_edited_time" → prop["last_edited_time"]
    """
    prop_type = prop.get("type", "")

    if prop_type == "date":
        date_obj = prop.get("date")
        if not date_obj:
            return None
        start = date_obj.get("start", "")
        if not start:
            return None
        # Handles both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS..." formats
        return _parse_notion_date(start)

    if prop_type in ("created_time", "last_edited_time"):
        raw = prop.get(prop_type, "")
        return _parse_notion_date(raw) if raw else None

    # For select / rich_text / etc — not a date
    return None


def _parse_notion_date(raw: str) -> Optional[date]:
    """Parse a Notion date string to a Python date object."""
    if not raw:
        return None
    try:
        if "T" in raw:
            # ISO datetime string; parse and convert to KST date
            dt = datetime.fromisoformat(raw.rstrip("Z").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(KST).date()
        else:
            # Plain date string "YYYY-MM-DD"
            return date.fromisoformat(raw)
    except Exception as exc:
        logger.debug("_parse_notion_date failed for %r: %s", raw, exc)
        return None


def _extract_text(prop: dict) -> str:
    """Extract plain text from select, multi_select, rich_text, or status prop."""
    prop_type = prop.get("type", "")
    if prop_type == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    if prop_type == "multi_select":
        items = prop.get("multi_select", [])
        return ", ".join(i.get("name", "") for i in items if isinstance(i, dict))
    if prop_type == "status":
        sel = prop.get("status")
        return sel.get("name", "") if sel else ""
    if prop_type == "rich_text":
        parts = prop.get("rich_text", [])
        return "".join(
            t.get("plain_text", "")
            for t in parts
            if isinstance(t, dict)
        ).strip()
    return ""


# ── Client ─────────────────────────────────────────────────────────────────────

class NotionClient:
    """Thin wrapper around the Notion API for deadline retrieval."""

    def __init__(self, db_id: str = NOTION_DB_ID) -> None:
        self._db_id = db_id
        self._client: Optional[Client] = None
        self._schema: Optional[_DbSchema] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate and discover database schema."""
        self._client = Client(auth=NOTION_TOKEN)
        self._schema = self._discover_schema()
        logger.info("NotionClient connected; DB=%s", self._db_id)

    def _ensure_connected(self) -> None:
        if self._client is None:
            self.connect()

    def _discover_schema(self) -> _DbSchema:
        """Retrieve DB metadata and discover property schema."""
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                db_meta = self._client.databases.retrieve(database_id=self._db_id)
                return _discover_schema(db_meta)
            except APIResponseError as exc:
                logger.warning(
                    "Notion schema discovery failed (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
            except Exception as exc:
                logger.warning(
                    "Notion unexpected error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"Notion schema discovery failed after {API_RETRY_ATTEMPTS} attempts"
        )

    # ── retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute *fn* with up to API_RETRY_ATTEMPTS retries on failure."""
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
                    "Notion unexpected error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"Notion API call failed after {API_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_page(
        self,
        page: dict,
        deadline_prop: str,
        today: date,
    ) -> Optional[NotionDeadlineItem]:
        """
        Parse a raw Notion page dict into a NotionDeadlineItem.

        Returns None if the deadline property is missing or unparseable.
        """
        props = page.get("properties", {})

        # Extract name
        name = "(이름 없음)"
        if self._schema and self._schema.name_prop:
            name_prop_val = props.get(self._schema.name_prop, {})
            name = _extract_title(name_prop_val)

        # Extract deadline date
        deadline_prop_val = props.get(deadline_prop, {})
        deadline = _extract_date(deadline_prop_val)
        if deadline is None:
            return None

        # Extract status
        status = ""
        if self._schema and self._schema.status_prop:
            status_prop_val = props.get(self._schema.status_prop, {})
            status = _extract_text(status_prop_val)

        # Compute days until deadline
        delta = (deadline - today).days
        is_overdue = delta < 0

        return NotionDeadlineItem(
            page_id=page.get("id", ""),
            name=name,
            deadline=deadline,
            deadline_prop=deadline_prop,
            status=status,
            url=page.get("url", ""),
            is_overdue=is_overdue,
            days_until=delta,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_deadline_items(
        self,
        lookahead_days: int = 30,
        include_overdue: bool = True,
        max_results: int = 100,
    ) -> list[NotionDeadlineItem]:
        """
        Fetch Notion records with deadline dates within the next *lookahead_days*.

        Parameters
        ----------
        lookahead_days:
            How far ahead to look for upcoming deadlines (default: 30 days).
        include_overdue:
            If True, also include records whose deadline has already passed
            (within the last 30 days). These represent missed/unresolved items.
        max_results:
            Maximum number of records to return.

        Returns
        -------
        List of NotionDeadlineItem objects sorted by deadline ascending
        (overdue items first).
        """
        self._ensure_connected()

        if not self._schema or not self._schema.primary_deadline_prop:
            logger.warning(
                "No deadline property discovered in Notion DB %s — "
                "returning all records without date filter",
                self._db_id,
            )
            return self._fetch_all_pages(max_results)

        deadline_prop = self._schema.primary_deadline_prop
        today = datetime.now(KST).date()
        future_date = today + timedelta(days=lookahead_days)

        # Build Notion filter: deadline within [past_cutoff, future_date]
        filter_conditions: list[dict] = [
            {
                "property": deadline_prop,
                "date": {"on_or_before": future_date.isoformat()},
            },
        ]
        if not include_overdue:
            filter_conditions.append({
                "property": deadline_prop,
                "date": {"on_or_after": today.isoformat()},
            })

        if len(filter_conditions) == 1:
            query_filter = filter_conditions[0]
        else:
            query_filter = {"and": filter_conditions}

        def _query():
            return self._client.databases.query(
                database_id=self._db_id,
                filter=query_filter,
                sorts=[{"property": deadline_prop, "direction": "ascending"}],
                page_size=min(max_results, 100),
            )

        result = self._call_with_retry(_query)
        pages = result.get("results", [])

        items: list[NotionDeadlineItem] = []
        for page in pages:
            item = self._parse_page(page, deadline_prop, today)
            if item is not None:
                items.append(item)

        # Sort: overdue first, then by days_until ascending
        items.sort(key=lambda x: x.days_until)

        logger.info(
            "NotionClient.fetch_deadline_items: found %d items "
            "(lookahead=%d days, include_overdue=%s)",
            len(items),
            lookahead_days,
            include_overdue,
        )
        return items

    def _fetch_all_pages(self, max_results: int = 100) -> list[NotionDeadlineItem]:
        """
        Fallback: fetch records without a date filter (used when no deadline
        property is discovered). Returns basic items without deadline info.
        """
        def _query():
            return self._client.databases.query(
                database_id=self._db_id,
                page_size=min(max_results, 100),
            )

        result = self._call_with_retry(_query)
        pages = result.get("results", [])
        today = datetime.now(KST).date()
        items = []
        for page in pages:
            if self._schema and self._schema.name_prop:
                props = page.get("properties", {})
                name_prop_val = props.get(self._schema.name_prop, {})
                name = _extract_title(name_prop_val)
            else:
                name = "(이름 없음)"
            items.append(NotionDeadlineItem(
                page_id=page.get("id", ""),
                name=name,
                deadline=today,
                deadline_prop="확인 불가",
                status="확인 불가",
                url=page.get("url", ""),
                is_overdue=False,
                days_until=0,
            ))
        return items

    def get_schema(self) -> Optional[_DbSchema]:
        """Return the discovered schema (None if not yet connected)."""
        return self._schema


# ── Module-level convenience function ─────────────────────────────────────────

def fetch_notion_deadlines(
    lookahead_days: int = 30,
    include_overdue: bool = True,
) -> list[NotionDeadlineItem]:
    """
    Top-level convenience function used by the briefing aggregator.

    Creates a NotionClient, fetches deadline items, and returns them.
    Returns an empty list (not an exception) on API failure.
    """
    try:
        client = NotionClient()
        return client.fetch_deadline_items(
            lookahead_days=lookahead_days,
            include_overdue=include_overdue,
        )
    except Exception as exc:
        logger.error("fetch_notion_deadlines failed: %s", exc)
        return []
