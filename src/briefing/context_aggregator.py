"""
Meeting Context Aggregator (Sub-AC 2b).

Compiles raw briefing content for a single meeting by aggregating:
  - Attendee profiles (including historical meeting frequency)
  - Related Gmail email threads (from/to external attendees, keyword search)
  - Related Notion deal records (company domain + title keyword search)

The aggregated ``RawBriefingContent`` is the input to the AI briefing
generator (Sub-AC 2c).  This module is intentionally free of any
Claude/AI calls — it only fetches and structures raw data.

Usage::

    from src.briefing.context_aggregator import MeetingContextAggregator
    from src.gmail.gmail_client import GmailClient
    from src.notion.notion_client import NotionClient
    from src.calendar.google_calendar import GoogleCalendarClient

    aggregator = MeetingContextAggregator(
        gmail_client=GmailClient(),
        notion_client=NotionClient(),
        calendar_client=GoogleCalendarClient(),
    )
    content = aggregator.aggregate(meeting)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting, Attendee
    from src.gmail.gmail_client import GmailClient, EmailThread
    from src.notion.notion_client import NotionClient, NotionRecord, NotionPageContent
    from src.ai.web_search import WebSearchClient, WebSearchSummary
    from src.slack.history_retriever import SlackHistoryRetriever, SlackHistoryResult

logger = logging.getLogger(__name__)

# ── How far back to look for historical calendar meetings with attendees ────
_HISTORY_LOOKBACK_DAYS: int = 365
# ── Max Gmail threads to include in raw context ──────────────────────────────
_MAX_GMAIL_THREADS: int = 10
# ── Max Notion records to include in raw context ─────────────────────────────
_MAX_NOTION_RECORDS: int = 10
# ── Lookback window for Gmail thread search ──────────────────────────────────
_GMAIL_LOOKBACK_DAYS: int = 30
# ── Lookback window for Slack history search ─────────────────────────────────
_SLACK_LOOKBACK_DAYS: int = 90
# ── Max Slack messages per channel ───────────────────────────────────────────
_SLACK_MAX_MESSAGES: int = 20


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class AttendeeProfile:
    """
    Enriched profile for a single meeting attendee.

    Combines calendar invite fields with historical meeting statistics
    derived from scanning past calendar events.
    """
    email: str
    display_name: str = ""
    response_status: str = "needsAction"
    is_internal: bool = False
    company_domain: str = ""

    # Historical stats (populated by _build_attendee_profiles)
    past_meeting_count: int = 0
    last_met_date: Optional[datetime] = None
    past_meeting_titles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "display_name": self.display_name,
            "response_status": self.response_status,
            "is_internal": self.is_internal,
            "company_domain": self.company_domain,
            "past_meeting_count": self.past_meeting_count,
            "last_met_date": self.last_met_date.isoformat() if self.last_met_date else None,
            "past_meeting_titles": self.past_meeting_titles[:5],  # cap for brevity
        }


@dataclass
class AggregationError:
    """Records a non-fatal error that occurred during context aggregation."""
    source: str          # e.g. "gmail", "notion", "calendar_history"
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RawBriefingContent:
    """
    All raw data needed to generate a meeting briefing.

    Produced by MeetingContextAggregator.aggregate() and consumed by the
    AI briefing generator (Sub-AC 2c).

    Fields marked '확인 불가' in the briefing when their source returned
    no data or raised an error (accuracy constraint: never send incorrect
    info, but incomplete is acceptable with annotation).
    """
    # ── Core meeting info ────────────────────────────────────────────────────
    meeting_id: str
    meeting_title: str
    meeting_start: datetime
    meeting_end: datetime
    meeting_location: str = ""
    meeting_description: str = ""
    meeting_html_link: str = ""
    organizer_email: str = ""

    # ── Enriched attendee profiles ───────────────────────────────────────────
    attendee_profiles: list[AttendeeProfile] = field(default_factory=list)

    # ── Gmail context ────────────────────────────────────────────────────────
    gmail_threads: list["EmailThread"] = field(default_factory=list)
    gmail_available: bool = True   # False when Gmail fetch failed entirely

    # ── Notion context ───────────────────────────────────────────────────────
    notion_records: list["NotionRecord"] = field(default_factory=list)
    notion_available: bool = True  # False when Notion fetch failed entirely

    # ── Calendar history ─────────────────────────────────────────────────────
    # Was historical calendar data fetched for attendee enrichment?
    calendar_history_available: bool = True

    # ── Web search context (Sub-AC 6a) ────────────────────────────────────────
    # Populated only for EXTERNAL_FIRST meetings (first-time external encounter).
    # None means web search was not attempted (e.g. EXTERNAL_FOLLOWUP or INTERNAL).
    web_search_summary: Optional["WebSearchSummary"] = None
    web_search_available: bool = True  # False when search entirely failed

    # ── Notion deal memo (Sub-AC 6b) ─────────────────────────────────────────
    # Full deal memo page content (NotionPageContent) for EXTERNAL_FIRST meetings.
    # None when: (a) not an EXTERNAL_FIRST meeting, (b) no matching page found,
    # or (c) the fetch failed entirely (see notion_deal_memo_available).
    notion_deal_memo: Optional["NotionPageContent"] = None
    # True while Notion client was available and a search was attempted.
    # False when the client itself was unavailable or the fetch raised.
    notion_deal_memo_available: bool = True

    # ── Slack message history context (AC 7 / Sub-AC 3) ──────────────────────
    # Populated for EXTERNAL_FOLLOWUP meetings (and optionally EXTERNAL_FIRST).
    # Contains messages from priority channels mentioning the company name.
    # None means Slack history was not yet fetched (not attempted).
    slack_history: Optional["SlackHistoryResult"] = None
    # True while the Slack API was reachable and history search was attempted.
    # False when the SlackHistoryRetriever itself failed (API down / token error).
    slack_history_available: bool = True

    # ── Non-fatal errors ─────────────────────────────────────────────────────
    errors: list[AggregationError] = field(default_factory=list)

    # ── Timestamps ───────────────────────────────────────────────────────────
    aggregated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def external_attendees(self) -> list[AttendeeProfile]:
        return [p for p in self.attendee_profiles if not p.is_internal]

    @property
    def internal_attendees(self) -> list[AttendeeProfile]:
        return [p for p in self.attendee_profiles if p.is_internal]

    @property
    def duration_minutes(self) -> int:
        return int((self.meeting_end - self.meeting_start).total_seconds() // 60)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "meeting_title": self.meeting_title,
            "meeting_start": self.meeting_start.isoformat(),
            "meeting_end": self.meeting_end.isoformat(),
            "duration_minutes": self.duration_minutes,
            "meeting_location": self.meeting_location,
            "meeting_description": self.meeting_description,
            "meeting_html_link": self.meeting_html_link,
            "organizer_email": self.organizer_email,
            "attendee_profiles": [p.to_dict() for p in self.attendee_profiles],
            "external_attendee_count": len(self.external_attendees),
            "internal_attendee_count": len(self.internal_attendees),
            "gmail_threads_count": len(self.gmail_threads),
            "gmail_available": self.gmail_available,
            "gmail_threads": [t.to_dict() for t in self.gmail_threads],
            "notion_records_count": len(self.notion_records),
            "notion_available": self.notion_available,
            "notion_records": [r.to_dict() for r in self.notion_records],
            "calendar_history_available": self.calendar_history_available,
            "web_search_summary": (
                self.web_search_summary.to_dict()
                if self.web_search_summary is not None
                else None
            ),
            "web_search_available": self.web_search_available,
            "notion_deal_memo": (
                self.notion_deal_memo.to_dict()
                if self.notion_deal_memo is not None
                else None
            ),
            "notion_deal_memo_available": self.notion_deal_memo_available,
            "slack_history": (
                self.slack_history.to_dict()
                if self.slack_history is not None
                else None
            ),
            "slack_history_available": self.slack_history_available,
            "errors": [e.to_dict() for e in self.errors],
            "aggregated_at": self.aggregated_at.isoformat(),
        }


# ── Aggregator ───────────────────────────────────────────────────────────────

class MeetingContextAggregator:
    """
    Fetches and compiles raw context data for a meeting briefing.

    All external API calls are wrapped so that a failure in one source
    (Gmail, Notion, Calendar history) does not prevent the others from
    running.  Errors are captured in ``RawBriefingContent.errors`` and
    the corresponding ``*_available`` flag is set to False.

    Args:
        gmail_client:    Authenticated GmailClient (or None to skip Gmail).
        notion_client:   Connected NotionClient (or None to skip Notion).
        calendar_client: Connected GoogleCalendarClient (or None to skip
                         historical attendee enrichment).
    """

    def __init__(
        self,
        gmail_client: Optional["GmailClient"] = None,
        notion_client: Optional["NotionClient"] = None,
        calendar_client=None,
        web_search_client: Optional["WebSearchClient"] = None,
        slack_retriever: Optional["SlackHistoryRetriever"] = None,
    ) -> None:
        self._gmail = gmail_client
        self._notion = notion_client
        self._calendar = calendar_client
        self._web_search = web_search_client
        self._slack_retriever = slack_retriever

    # ── public API ────────────────────────────────────────────────────────────

    def aggregate(
        self,
        meeting: "Meeting",
        is_external_first: bool = False,
        fetch_slack_history: bool = True,
    ) -> RawBriefingContent:
        """
        Compile ``RawBriefingContent`` for *meeting*.

        Steps (all run even if one fails):
        1. Build base attendee profiles from the meeting's attendee list.
        2. Optionally enrich profiles with calendar history.
        3. Fetch related Gmail threads.
        4. Fetch related Notion deal records.
        5. (EXTERNAL_FIRST only) Perform web search for company context.
        6. (EXTERNAL_FIRST only) Fetch Notion deal memo page.
        7. (When fetch_slack_history=True) Fetch Slack message history for
           the company across priority channels (AC 7 Sub-AC 3).

        Args:
            meeting:          The ``Meeting`` object to brief.
            is_external_first: When True, steps 5-6 (web search + deal memo)
                               are executed.  Only set this for EXTERNAL_FIRST
                               meetings to avoid unnecessary API costs.
            fetch_slack_history: When True (default), fetch Slack message
                               history from priority channels for the company.
                               Set False to skip (e.g. in unit tests).

        Returns a fully populated RawBriefingContent; never raises.
        """
        logger.info(
            "[Aggregator] Starting context aggregation for '%s' (id=%s, "
            "external_first=%s, slack=%s)",
            meeting.summary,
            meeting.event_id,
            is_external_first,
            fetch_slack_history,
        )

        content = RawBriefingContent(
            meeting_id=meeting.event_id,
            meeting_title=meeting.summary,
            meeting_start=meeting.start,
            meeting_end=meeting.end,
            meeting_location=meeting.location or "",
            meeting_description=meeting.description or "",
            meeting_html_link=meeting.html_link or "",
            organizer_email=meeting.organizer_email or "",
        )

        # 1. Base attendee profiles
        content.attendee_profiles = self._build_attendee_profiles(meeting)

        # 2. Enrich with historical calendar data
        self._enrich_with_calendar_history(meeting, content)

        # 3. Gmail context
        self._fetch_gmail_context(meeting, content)

        # 4. Notion context
        self._fetch_notion_context(meeting, content)

        # 5. Web search context (only for EXTERNAL_FIRST meetings, Sub-AC 6a)
        if is_external_first:
            self._fetch_web_search_context(meeting, content)

        # 6. Notion deal memo page content (only for EXTERNAL_FIRST, Sub-AC 6b)
        #    Fetches the full Notion page body (deal memo) for the company being
        #    met for the first time, providing investment thesis / deal context.
        if is_external_first:
            self._fetch_notion_deal_memo(meeting, content)

        # 7. Slack message history (AC 7 Sub-AC 3):
        #    For EXTERNAL_FOLLOWUP: company relationship context from Slack.
        #    For EXTERNAL_FIRST: prior Slack mentions if any exist.
        if fetch_slack_history:
            self._fetch_slack_history(meeting, content)

        logger.info(
            "[Aggregator] Done: %d attendees, %d gmail threads, %d notion records, "
            "%d web results, deal_memo=%s, slack_msgs=%s, %d errors",
            len(content.attendee_profiles),
            len(content.gmail_threads),
            len(content.notion_records),
            len(content.web_search_summary.results) if content.web_search_summary else 0,
            "yes" if content.notion_deal_memo is not None else "no",
            len(content.slack_history.messages) if content.slack_history else "n/a",
            len(content.errors),
        )
        return content

    # ── Step 1: attendee profiles ─────────────────────────────────────────────

    def _build_attendee_profiles(
        self,
        meeting: "Meeting",
    ) -> list[AttendeeProfile]:
        """
        Build a base AttendeeProfile for each meeting attendee.
        Historical stats start at zero and are filled in by step 2.
        """
        profiles: list[AttendeeProfile] = []
        for attendee in meeting.attendees:
            domain = _email_domain(attendee.email)
            profiles.append(
                AttendeeProfile(
                    email=attendee.email,
                    display_name=attendee.display_name or "",
                    response_status=attendee.response_status or "needsAction",
                    is_internal=attendee.is_internal,
                    company_domain=domain,
                )
            )
        return profiles

    # ── Step 2: calendar history enrichment ───────────────────────────────────

    def _enrich_with_calendar_history(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Scan past external meetings (up to 1 year back) to build
        per-attendee meeting history.  Mutates attendee_profiles in-place.
        """
        if self._calendar is None:
            logger.debug("[Aggregator] No calendar client; skipping history enrichment")
            return

        try:
            past_meetings = self._calendar.list_historical_external_meetings(
                lookback_days=_HISTORY_LOOKBACK_DAYS,
            )
        except Exception as exc:
            msg = f"Calendar history fetch failed: {exc}"
            logger.warning("[Aggregator] %s", msg)
            content.calendar_history_available = False
            content.errors.append(
                AggregationError(source="calendar_history", message=str(exc))
            )
            return

        # Build index: email → list of (meeting_start, meeting_title)
        history_index: dict[str, list[tuple[datetime, str]]] = {}
        for past in past_meetings:
            if past.event_id == meeting.event_id:
                # Skip the meeting we're briefing
                continue
            for att in past.attendees:
                key = att.email.lower()
                if key not in history_index:
                    history_index[key] = []
                history_index[key].append((past.start, past.summary))

        # Enrich profiles
        for profile in content.attendee_profiles:
            history = history_index.get(profile.email.lower(), [])
            if history:
                history_sorted = sorted(history, key=lambda x: x[0], reverse=True)
                profile.past_meeting_count = len(history_sorted)
                profile.last_met_date = history_sorted[0][0]
                profile.past_meeting_titles = [title for _, title in history_sorted[:5]]

        logger.debug(
            "[Aggregator] Calendar history: %d past external meetings indexed",
            len(past_meetings),
        )

    # ── Step 3: Gmail context ─────────────────────────────────────────────────

    def _fetch_gmail_context(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Fetch relevant Gmail threads and attach to *content*.
        Sets gmail_available=False and records an error on failure.
        """
        if self._gmail is None:
            logger.debug("[Aggregator] No Gmail client; skipping Gmail context")
            content.gmail_available = False
            content.errors.append(
                AggregationError(
                    source="gmail",
                    message="GmailClient not provided to aggregator",
                )
            )
            return

        external_emails = [a.email for a in meeting.external_attendees]

        try:
            threads = self._gmail.get_threads_for_meeting(
                external_emails=external_emails,
                meeting_title=meeting.summary,
                lookback_days=_GMAIL_LOOKBACK_DAYS,
                max_threads=_MAX_GMAIL_THREADS,
            )
            content.gmail_threads = threads
        except Exception as exc:
            msg = f"Gmail thread fetch failed: {exc}"
            logger.warning("[Aggregator] %s", msg)
            content.gmail_available = False
            content.errors.append(AggregationError(source="gmail", message=str(exc)))

    # ── Step 4: Notion context ────────────────────────────────────────────────

    def _fetch_notion_context(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Fetch relevant Notion deal records and attach to *content*.
        Sets notion_available=False and records an error on failure.
        """
        if self._notion is None:
            logger.debug("[Aggregator] No Notion client; skipping Notion context")
            content.notion_available = False
            content.errors.append(
                AggregationError(
                    source="notion",
                    message="NotionClient not provided to aggregator",
                )
            )
            return

        external_emails = [a.email for a in meeting.external_attendees]

        try:
            records = self._notion.get_records_for_meeting(
                external_emails=external_emails,
                meeting_title=meeting.summary,
                max_records=_MAX_NOTION_RECORDS,
            )
            content.notion_records = records
        except Exception as exc:
            msg = f"Notion records fetch failed: {exc}"
            logger.warning("[Aggregator] %s", msg)
            content.notion_available = False
            content.errors.append(AggregationError(source="notion", message=str(exc)))

    # ── Step 7: Slack history (AC 7 Sub-AC 3) ────────────────────────────────

    def _fetch_slack_history(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Fetch Slack message history mentioning this meeting's company.

        Searches priority channels (names containing "투자" or "squad-service")
        for messages that mention the company name derived from external attendee
        email domains or meeting title keywords.

        Populates ``content.slack_history`` with a ``SlackHistoryResult``.
        Sets ``content.slack_history_available = False`` and records an
        ``AggregationError`` on any API failure so the formatter can annotate
        the section as '확인 불가' without suppressing the briefing entirely.
        """
        # Derive company name for Slack search
        company_name = _derive_company_name(meeting)
        if not company_name:
            logger.info(
                "[Aggregator] No company name derived for Slack history search "
                "(meeting='%s')",
                meeting.summary,
            )
            # Leave slack_history=None and available=True (no error — just no query)
            return

        # Auto-create retriever if one was not injected
        retriever = self._slack_retriever
        if retriever is None:
            try:
                from src.slack.history_retriever import SlackHistoryRetriever
                retriever = SlackHistoryRetriever()
            except Exception as exc:
                msg = f"SlackHistoryRetriever init failed: {exc}"
                logger.warning("[Aggregator] %s", msg)
                content.slack_history_available = False
                content.errors.append(
                    AggregationError(source="slack_history", message=str(exc))
                )
                return

        try:
            result = retriever.search_company_history(
                company_name=company_name,
                lookback_days=_SLACK_LOOKBACK_DAYS,
                max_messages_per_channel=_SLACK_MAX_MESSAGES,
            )
            content.slack_history = result

            if not result.available:
                content.slack_history_available = False
                content.errors.append(
                    AggregationError(
                        source="slack_history",
                        message=result.error or "Slack API unavailable",
                    )
                )
                logger.warning(
                    "[Aggregator] Slack history unavailable for '%s': %s",
                    company_name,
                    result.error,
                )
            else:
                logger.info(
                    "[Aggregator] Slack history: %d messages for '%s' "
                    "across %d channels",
                    len(result.messages),
                    company_name,
                    len(result.channels_searched),
                )
        except Exception as exc:
            msg = f"Slack history search failed for '{company_name}': {exc}"
            logger.warning("[Aggregator] %s", msg)
            content.slack_history_available = False
            content.errors.append(
                AggregationError(source="slack_history", message=str(exc))
            )

    # ── Step 5: Web search context (Sub-AC 6a) ────────────────────────────────

    def _fetch_web_search_context(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Run web search for external company/meeting context.

        Only called for EXTERNAL_FIRST meetings (first-time external contact).
        Uses WebSearchClient which auto-selects Tavily (primary) or Claude
        web_search (fallback).

        Results are attached to ``content.web_search_summary``.  Failures set
        ``content.web_search_available = False`` and record an AggregationError
        so the formatter can annotate the section as '확인 불가'.
        """
        # Auto-create client if not injected
        web_search_client = self._web_search
        if web_search_client is None:
            try:
                from src.ai.web_search import WebSearchClient
                web_search_client = WebSearchClient()
            except Exception as exc:
                msg = f"WebSearchClient init failed: {exc}"
                logger.warning("[Aggregator] %s", msg)
                content.web_search_available = False
                content.errors.append(
                    AggregationError(source="web_search", message=str(exc))
                )
                return

        # Collect external domains and attendee names
        external_attendees = meeting.external_attendees
        company_domains = list({
            _email_domain(a.email)
            for a in external_attendees
            if _email_domain(a.email)
        })
        attendee_names = [
            a.display_name for a in external_attendees if a.display_name
        ]

        try:
            summary = web_search_client.search_company_context(
                company_domains=company_domains,
                meeting_title=meeting.summary,
                attendee_names=attendee_names or None,
            )
            content.web_search_summary = summary

            if not summary.available:
                content.web_search_available = False
                content.errors.append(
                    AggregationError(
                        source="web_search",
                        message=summary.error or "Web search unavailable",
                    )
                )
                logger.warning(
                    "[Aggregator] Web search unavailable for '%s': %s",
                    meeting.summary,
                    summary.error,
                )
            else:
                logger.info(
                    "[Aggregator] Web search: %d results for '%s' (provider=%s)",
                    len(summary.results),
                    meeting.summary,
                    summary.provider,
                )

        except Exception as exc:
            msg = f"Web search failed: {exc}"
            logger.warning("[Aggregator] %s", msg)
            content.web_search_available = False
            content.errors.append(AggregationError(source="web_search", message=str(exc)))


    # ── Step 6: Notion deal memo (Sub-AC 6b) ─────────────────────────────────

    def _fetch_notion_deal_memo(
        self,
        meeting: "Meeting",
        content: RawBriefingContent,
    ) -> None:
        """
        Fetch the full Notion deal memo page for an EXTERNAL_FIRST meeting.

        Called only when ``is_external_first=True`` in :meth:`aggregate`.

        Strategy
        --------
        1. Derive candidate company names from:
           a. External attendee email domains (root label, e.g. "acme" from
              "ceo@acme.com").
           b. Keywords extracted from the meeting title.
        2. Try each candidate in order via
           :meth:`NotionClient.get_company_page_content` until a match is
           found.
        3. Store the first successful result in
           ``content.notion_deal_memo``.

        Failures set ``content.notion_deal_memo_available=False`` and
        append an :class:`AggregationError` so the formatter can annotate
        the section as ``확인 불가`` without suppressing the entire briefing.

        Args:
            meeting: The ``Meeting`` being briefed.
            content: The ``RawBriefingContent`` to mutate in-place.
        """
        if self._notion is None:
            logger.debug(
                "[Aggregator] No Notion client; skipping deal memo fetch"
            )
            # Mark as unavailable — the formatter will annotate accordingly
            content.notion_deal_memo_available = False
            content.errors.append(
                AggregationError(
                    source="notion_deal_memo",
                    message="NotionClient not provided to aggregator",
                )
            )
            return

        # ── Build candidate company names ─────────────────────────────────────
        external_attendees = meeting.external_attendees
        candidates: list[str] = []

        # 1a. From email domains (root label only, e.g. "acme" from acme.com)
        seen_domains: set[str] = set()
        for att in external_attendees:
            domain_root = _domain_root_label(att.email)
            if domain_root and domain_root not in seen_domains:
                candidates.append(domain_root)
                seen_domains.add(domain_root)
            # Also try full domain minus TLD for better fuzzy matches
            full_domain = _email_domain(att.email)
            if full_domain and full_domain not in seen_domains:
                candidates.append(full_domain.split(".")[0])  # leftmost label
                seen_domains.add(full_domain)

        # 1b. From meeting title keywords
        for kw in _title_keywords_for_notion(meeting.summary):
            if kw.lower() not in seen_domains:
                candidates.append(kw)
                seen_domains.add(kw.lower())

        logger.debug(
            "[Aggregator] Deal memo candidates for '%s': %s",
            meeting.summary,
            candidates,
        )

        # ── Try each candidate until a match is found ─────────────────────────
        page_content = None
        last_exc: Optional[Exception] = None

        for candidate in candidates:
            if not candidate or len(candidate) < 2:
                continue
            try:
                result = self._notion.get_company_page_content(
                    company_name=candidate,
                    max_content_chars=2000,
                )
                if result is not None:
                    page_content = result
                    logger.info(
                        "[Aggregator] Deal memo found for '%s' via candidate=%r "
                        "page_id=%s blocks_fetched=%s",
                        meeting.summary,
                        candidate,
                        result.page_id,
                        result.blocks_fetched,
                    )
                    break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[Aggregator] Deal memo search failed for candidate=%r: %s",
                    candidate,
                    exc,
                )

        if page_content is not None:
            content.notion_deal_memo = page_content
        else:
            # No match found — this is acceptable (not every meeting has a deal
            # memo).  Only mark unavailable if an exception was raised.
            if last_exc is not None:
                content.notion_deal_memo_available = False
                content.errors.append(
                    AggregationError(
                        source="notion_deal_memo",
                        message=str(last_exc),
                    )
                )
                logger.warning(
                    "[Aggregator] Deal memo fetch failed for '%s': %s",
                    meeting.summary,
                    last_exc,
                )
            else:
                logger.info(
                    "[Aggregator] No deal memo found in Notion for meeting '%s'",
                    meeting.summary,
                )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _email_domain(email_addr: str) -> str:
    """Return the domain portion of an email address, lower-cased."""
    try:
        return email_addr.split("@")[1].lower()
    except (IndexError, AttributeError):
        return ""


# ── TLD set for domain root extraction ───────────────────────────────────────
_TLDS: frozenset[str] = frozenset({
    "com", "co", "kr", "net", "org", "io", "ai", "vc", "biz",
    "app", "dev", "tech", "inc", "ltd", "llc",
})

# Korean / English stop-words used in meeting titles (for keyword extraction)
_KO_STOP: frozenset[str] = frozenset({
    "주식회사", "유한", "합자", "재단", "협회", "협동조합",
    "미팅", "회의", "논의", "검토", "협의", "관련", "건",
})
_EN_STOP: frozenset[str] = frozenset({
    "meeting", "call", "sync", "discussion", "inc", "corp", "ltd",
    "the", "and", "for", "with", "of", "about", "regarding", "re",
})


def _domain_root_label(email_addr: str) -> str:
    """
    Extract the most meaningful label from an email domain for company search.

    Examples:
        'ceo@acme-corp.com'          → 'acme-corp'
        'user@sub.startup.co.kr'     → 'startup'
        'john@bigtech.io'            → 'bigtech'
    """
    try:
        domain = email_addr.split("@")[1].lower()
        parts = domain.split(".")
        meaningful = [p for p in parts if p not in _TLDS and len(p) > 1]
        if meaningful:
            # Prefer the second-to-last label (most specific company part)
            return meaningful[-1]
    except (IndexError, AttributeError):
        pass
    return ""


def _derive_company_name(meeting: "Meeting") -> str:
    """
    Derive a company name for Slack history search from a meeting.

    Only returns a name when the meeting has at least one external attendee —
    internal-only meetings (no external attendees) always return "" so the
    Slack history search is skipped entirely.

    Priority order (applied only when external attendees are present):
    1. Root domain label from the first external attendee's email
       (e.g. "acme" from "ceo@acme.com").
    2. Display name of the first external attendee (may be their company name).
    3. First keyword extracted from the meeting title.

    Returns an empty string when no suitable name can be derived.
    """
    # Guard: no external attendees → not an external meeting, skip Slack search
    if not meeting.external_attendees:
        return ""

    # Priority 1: Email domain root label
    for att in meeting.external_attendees:
        name = _domain_root_label(att.email)
        if name and len(name) >= 2:
            return name

    # Priority 2: Display name (may be a person name, but better than nothing)
    for att in meeting.external_attendees:
        if att.display_name and len(att.display_name) >= 2:
            return att.display_name

    # Priority 3: Meeting title keywords
    keywords = _title_keywords_for_notion(meeting.summary, max_kw=1)
    if keywords:
        return keywords[0]

    return ""


def _title_keywords_for_notion(title: str, max_kw: int = 4) -> list[str]:
    """
    Extract candidate company / brand names from a meeting title.

    Strips common meeting-related stop-words and returns up to *max_kw*
    tokens suitable for a Notion ``contains`` search.
    """
    import re
    tokens = re.split(r"[\s/\-_\[\]()]+", title)
    results: list[str] = []
    for tok in tokens:
        tok_clean = tok.strip(".,!?;:\"'·~")
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
