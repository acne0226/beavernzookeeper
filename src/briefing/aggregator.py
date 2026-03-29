"""
Briefing Data Aggregator.

Orchestrates fetching data from all three sources when /brief is invoked:
  1. Google Calendar  — upcoming events for the target date
  2. Gmail            — pending/recent inbox emails
  3. Notion           — portfolio company deadline items

Each source is fetched independently with 3-retry / 10s-delay error handling.
If a source fails after all retries, its slot in BriefingData is left empty
and the `source_errors` dict records the failure — the formatter will annotate
affected sections with '확인 불가' per project accuracy requirements.

Usage::

    from src.briefing.aggregator import BriefingData, aggregate_briefing_data

    data = aggregate_briefing_data(target_date=date.today())
    # data.calendar_events  – list[Meeting]
    # data.emails           – list[EmailMessage]
    # data.notion_deadlines – list[NotionDeadlineItem]
    # data.source_errors    – dict[str, str]  (source_name -> error message)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.config import API_RETRY_ATTEMPTS, API_RETRY_DELAY_SECONDS

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting
    from src.gmail.client import EmailMessage
    from src.notion.client import NotionDeadlineItem

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class BriefingData:
    """
    Aggregated data from all sources, ready for the briefing formatter.

    Attributes
    ----------
    target_date:
        The date this briefing covers.
    calendar_events:
        All calendar events (Meeting objects) for target_date, sorted by start.
    emails:
        Recent inbox emails (EmailMessage objects), newest first.
    notion_deadlines:
        Portfolio company deadline items (NotionDeadlineItem objects),
        sorted by deadline ascending (overdue first).
    source_errors:
        Maps source name to error message when a source failed to load.
        Possible keys: "calendar", "gmail", "notion".
        Absence of a key means the source loaded successfully.
    fetched_at:
        UTC timestamp when aggregation completed.
    """

    target_date: date
    calendar_events: list = field(default_factory=list)       # list[Meeting]
    emails: list = field(default_factory=list)                 # list[EmailMessage]
    notion_deadlines: list = field(default_factory=list)       # list[NotionDeadlineItem]
    source_errors: dict[str, str] = field(default_factory=dict)
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def has_calendar(self) -> bool:
        return "calendar" not in self.source_errors

    @property
    def has_gmail(self) -> bool:
        return "gmail" not in self.source_errors

    @property
    def has_notion(self) -> bool:
        return "notion" not in self.source_errors

    @property
    def all_sources_ok(self) -> bool:
        return len(self.source_errors) == 0

    @property
    def external_meetings(self) -> list:
        """Calendar events that have at least one external attendee."""
        return [m for m in self.calendar_events if getattr(m, "is_external", False)]

    @property
    def unread_emails(self) -> list:
        """Emails that are marked unread."""
        return [e for e in self.emails if getattr(e, "is_unread", False)]

    @property
    def urgent_emails(self) -> list:
        """
        Emails requiring immediate attention, surfaced for the briefing.

        An email is considered urgent when **any** of the following is true:

        1. **Unread + external** — not yet read and from outside the internal
           domain.  These are the most actionable items: an external party is
           waiting for engagement.
        2. **Gmail-important** — explicitly flagged with the ``IMPORTANT`` label
           by Gmail's own importance heuristics (or manually starred /
           filtered).  This catches read emails that still require action.

        Internal-only unread messages (e.g. Slack notifications forwarded to
        Gmail, internal announcements) are intentionally excluded to keep the
        briefing focused on external communication.

        Sorted newest-first so the most recent items appear at the top.
        """
        candidates = [
            e for e in self.emails
            if (
                (getattr(e, "is_unread", False) and getattr(e, "is_external", False))
                or getattr(e, "is_important", False)
            )
        ]
        # Preserve input order (already newest-first from GmailClient)
        return candidates

    @property
    def overdue_deadlines(self) -> list:
        """Notion items whose deadline has already passed."""
        return [d for d in self.notion_deadlines if getattr(d, "is_overdue", False)]

    @property
    def upcoming_deadlines(self) -> list:
        """Notion items with deadlines today or in the future."""
        return [d for d in self.notion_deadlines if not getattr(d, "is_overdue", False)]

    def summary(self) -> str:
        """One-line summary of the aggregated data for logging."""
        parts = [
            f"calendar={len(self.calendar_events)}",
            f"emails={len(self.emails)}",
            f"urgent_emails={len(self.urgent_emails)}",
            f"notion_deadlines={len(self.notion_deadlines)}",
        ]
        if self.source_errors:
            parts.append(f"errors={list(self.source_errors.keys())}")
        return f"BriefingData[{', '.join(parts)}]"


# ── Per-source fetch helpers ───────────────────────────────────────────────────

def _fetch_calendar(target_date: date) -> tuple[list, Optional[str]]:
    """
    Fetch all calendar events for *target_date* (KST day boundaries).

    Returns (events, error_message).  error_message is None on success.
    """
    try:
        from src.calendar.google_calendar import GoogleCalendarClient

        # Compute KST day boundaries in UTC
        kst_start = datetime(
            target_date.year, target_date.month, target_date.day,
            0, 0, 0, tzinfo=KST,
        )
        kst_end = kst_start + timedelta(days=1)

        client = GoogleCalendarClient()
        events = client.list_upcoming_events(
            time_min=kst_start.astimezone(timezone.utc),
            time_max=kst_end.astimezone(timezone.utc),
            max_results=50,
        )
        logger.info(
            "_fetch_calendar: %d events for %s", len(events), target_date
        )
        return events, None
    except Exception as exc:
        logger.error("_fetch_calendar failed: %s", exc)
        return [], str(exc)


def _fetch_gmail(target_date: date) -> tuple[list, Optional[str]]:
    """
    Fetch inbox emails received on *target_date* (or last 24 hours for today).

    Returns (emails, error_message).  error_message is None on success.
    """
    try:
        from src.gmail.client import GmailClient

        today = datetime.now(KST).date()
        days = max(1, (today - target_date).days + 1)

        client = GmailClient()
        emails = client.fetch_inbox_emails(days=days, max_results=50)
        logger.info(
            "_fetch_gmail: %d emails for past %d days (target=%s)",
            len(emails),
            days,
            target_date,
        )
        return emails, None
    except Exception as exc:
        logger.error("_fetch_gmail failed: %s", exc)
        return [], str(exc)


def _fetch_notion(_target_date: date) -> tuple[list, Optional[str]]:
    """
    Fetch Notion deadline items (target_date is not used for filtering —
    we always return items within the next 30 days + overdue).

    Returns (deadlines, error_message).  error_message is None on success.
    """
    try:
        from src.notion.client import NotionClient

        client = NotionClient()
        deadlines = client.fetch_deadline_items(
            lookahead_days=30,
            include_overdue=True,
        )
        logger.info(
            "_fetch_notion: %d deadline items", len(deadlines)
        )
        return deadlines, None
    except Exception as exc:
        logger.error("_fetch_notion failed: %s", exc)
        return [], str(exc)


# ── Retry wrapper ──────────────────────────────────────────────────────────────

def _fetch_with_retry(
    fetch_fn,
    source_name: str,
    *args,
    **kwargs,
) -> tuple[list, Optional[str]]:
    """
    Call *fetch_fn(*args, **kwargs)* with up to API_RETRY_ATTEMPTS attempts,
    sleeping API_RETRY_DELAY_SECONDS between retries.

    Returns (data, error_message).
    On permanent failure: ([], error_message).
    """
    last_error: Optional[str] = None
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        data, error = fetch_fn(*args, **kwargs)
        if error is None:
            return data, None
        last_error = error
        logger.warning(
            "[%s] fetch attempt %d/%d failed: %s",
            source_name, attempt, API_RETRY_ATTEMPTS, error,
        )
        if attempt < API_RETRY_ATTEMPTS:
            logger.info(
                "[%s] retrying in %d seconds…",
                source_name, API_RETRY_DELAY_SECONDS,
            )
            time.sleep(API_RETRY_DELAY_SECONDS)

    logger.error(
        "[%s] all %d fetch attempts failed. Last error: %s",
        source_name, API_RETRY_ATTEMPTS, last_error,
    )
    return [], last_error


# ── Public API ─────────────────────────────────────────────────────────────────

def aggregate_briefing_data(
    target_date: Optional[date] = None,
    fetch_calendar: bool = True,
    fetch_gmail: bool = True,
    fetch_notion: bool = True,
) -> BriefingData:
    """
    Aggregate briefing data from all configured sources.

    Each source is fetched independently.  Failures are captured in
    ``BriefingData.source_errors`` rather than raising — callers should
    check ``data.has_calendar``, ``data.has_gmail``, ``data.has_notion``.

    Parameters
    ----------
    target_date:
        The date to brief.  Defaults to today (KST).
    fetch_calendar:
        Set False to skip calendar (useful for testing).
    fetch_gmail:
        Set False to skip Gmail.
    fetch_notion:
        Set False to skip Notion.

    Returns
    -------
    BriefingData populated with whatever data was successfully retrieved.
    """
    if target_date is None:
        target_date = datetime.now(KST).date()

    logger.info(
        "aggregate_briefing_data: target_date=%s  "
        "calendar=%s gmail=%s notion=%s",
        target_date, fetch_calendar, fetch_gmail, fetch_notion,
    )

    data = BriefingData(target_date=target_date)

    # ── 1. Google Calendar ────────────────────────────────────────────────────
    if fetch_calendar:
        events, err = _fetch_with_retry(_fetch_calendar, "calendar", target_date)
        data.calendar_events = events
        if err:
            data.source_errors["calendar"] = err
    else:
        logger.debug("Calendar fetch skipped (fetch_calendar=False)")

    # ── 2. Gmail ──────────────────────────────────────────────────────────────
    if fetch_gmail:
        emails, err = _fetch_with_retry(_fetch_gmail, "gmail", target_date)
        data.emails = emails
        if err:
            data.source_errors["gmail"] = err
    else:
        logger.debug("Gmail fetch skipped (fetch_gmail=False)")

    # ── 3. Notion ─────────────────────────────────────────────────────────────
    if fetch_notion:
        deadlines, err = _fetch_with_retry(_fetch_notion, "notion", target_date)
        data.notion_deadlines = deadlines
        if err:
            data.source_errors["notion"] = err
    else:
        logger.debug("Notion fetch skipped (fetch_notion=False)")

    logger.info("aggregate_briefing_data: done → %s", data.summary())
    return data
