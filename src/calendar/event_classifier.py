"""
event_classifier.py — Final meeting category assignment.

Purpose
-------
This module combines two independent classification signals to produce a
definitive ``EventCategory`` for each calendar event:

1. **Title heuristic** (``classify_by_title`` from ``title_classifier.py``)
   Keyword-based pass that returns ``INTERNAL`` / ``EXTERNAL`` / ``UNKNOWN``
   without any API calls.

2. **Attendee-domain history** (``CalendarHistoryCache``)
   One-year lookback cache that records every external domain and email seen
   in past meetings.

Decision logic
--------------
The rules below are evaluated in priority order:

+---------------------------+------------------+-------------------+
| title_label               | meeting context  | → EventCategory   |
+===========================+==================+===================+
| INTERNAL                  | (any)            | internal          |
+---------------------------+------------------+-------------------+
| EXTERNAL or UNKNOWN       | no ext. attendees| internal          |
+---------------------------+------------------+-------------------+
| EXTERNAL or UNKNOWN       | has ext. attend. | external_first    |
|                           | not in cache     |                   |
+---------------------------+------------------+-------------------+
| EXTERNAL or UNKNOWN       | has ext. attend. | external_followup |
|                           | found in cache   |                   |
+---------------------------+------------------+-------------------+

The function also populates a ``ClassificationResult`` dataclass that
exposes the intermediate signals for debug / logging purposes.

Usage::

    from src.calendar.event_classifier import classify_event, EventCategory

    category = classify_event(meeting, history_cache)

    if category == EventCategory.INTERNAL:
        pass  # skip – internal meeting
    elif category == EventCategory.EXTERNAL_FIRST:
        ...   # first-time external meeting → richer briefing
    else:
        ...   # EventCategory.EXTERNAL_FOLLOWUP → relationship context
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from src.calendar.title_classifier import classify_by_title, MeetingLabel

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting
    from src.calendar.history_cache import CalendarHistoryCache, CachedEvent

logger = logging.getLogger(__name__)


# ── Category enum ──────────────────────────────────────────────────────────────

class EventCategory(str, Enum):
    """
    Final meeting category assigned by ``classify_event()``.

    Attributes
    ----------
    INTERNAL:
        Meeting involves only internal (kakaoventures.co.kr) attendees or the
        title heuristic strongly identifies it as an internal ceremony.
        The briefing pipeline should **skip** these events.
    EXTERNAL_FIRST:
        External meeting where none of the external attendees / domains appear
        in the 1-year history cache.  This is the first interaction — the
        briefing should provide extra introductory context.
    EXTERNAL_FOLLOWUP:
        External meeting where at least one external attendee or domain has
        appeared in a past meeting from the history cache.  The briefing should
        highlight relationship history and prior interaction context.
    """

    INTERNAL = "internal"
    EXTERNAL_FIRST = "external_first"
    EXTERNAL_FOLLOWUP = "external_followup"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Full classification output including intermediate signals.

    Attributes
    ----------
    event_id:
        Calendar event ID.
    title:
        Event summary / title string.
    category:
        Final assigned ``EventCategory``.
    title_label:
        Raw ``MeetingLabel`` from the title heuristic (INTERNAL / EXTERNAL /
        UNKNOWN).
    is_external_by_attendees:
        True if at least one non-internal attendee is present in the event
        (from ``Meeting.is_external``).
    external_attendee_emails:
        List of non-internal attendee email addresses.
    history_matched_emails:
        External attendee emails that were found in the history cache.
    history_matched_domains:
        External attendee domains that were found in the history cache.
    history_available:
        False when ``history_cache`` was ``None`` (degraded mode).
    first_meeting_with:
        List of ``CachedEvent`` objects representing the earliest prior meeting
        with each matched domain (useful for the briefing formatter).
    last_meeting_with:
        List of ``CachedEvent`` objects representing the most recent prior
        meeting with each matched domain.
    past_meeting_counts:
        Dict mapping each external domain to the number of historical meetings.
    debug_notes:
        Free-form strings added during classification (useful for tests /
        logging).
    """

    event_id: str
    title: str
    category: EventCategory
    title_label: MeetingLabel
    is_external_by_attendees: bool
    external_attendee_emails: list[str] = field(default_factory=list)
    history_matched_emails: list[str] = field(default_factory=list)
    history_matched_domains: list[str] = field(default_factory=list)
    history_available: bool = True
    last_meeting_with: list["CachedEvent"] = field(default_factory=list)
    past_meeting_counts: dict[str, int] = field(default_factory=dict)
    debug_notes: list[str] = field(default_factory=list)

    # ── convenience props ────────────────────────────────────────────────────

    @property
    def is_internal(self) -> bool:
        return self.category == EventCategory.INTERNAL

    @property
    def is_external_first(self) -> bool:
        return self.category == EventCategory.EXTERNAL_FIRST

    @property
    def is_external_followup(self) -> bool:
        return self.category == EventCategory.EXTERNAL_FOLLOWUP

    @property
    def total_past_meetings(self) -> int:
        """Sum of past meeting counts across all external domains."""
        return sum(self.past_meeting_counts.values())


# ── Internal helpers ───────────────────────────────────────────────────────────

def _email_domain(email: str) -> str:
    """Return the lower-cased domain part of an email address."""
    try:
        return email.split("@")[1].lower()
    except (IndexError, AttributeError):
        return ""


def _dedupe_ordered(seq: list[str]) -> list[str]:
    """Return a deduplicated list preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_event(
    meeting: "Meeting",
    history_cache: Optional["CalendarHistoryCache"] = None,
) -> EventCategory:
    """
    Assign a final ``EventCategory`` to *meeting*.

    This is a thin convenience wrapper around ``classify_event_full()`` that
    returns only the category enum value.

    Args:
        meeting:       A ``Meeting`` object from ``google_calendar.py``.
        history_cache: Optional ``CalendarHistoryCache`` for domain/email
                       lookback.  When ``None``, external meetings default to
                       ``EXTERNAL_FIRST`` (no history available).

    Returns:
        ``EventCategory.INTERNAL``          – skip (internal meeting)
        ``EventCategory.EXTERNAL_FIRST``    – first interaction with attendees
        ``EventCategory.EXTERNAL_FOLLOWUP`` – prior meeting history found
    """
    return classify_event_full(meeting, history_cache).category


def classify_event_full(
    meeting: "Meeting",
    history_cache: Optional["CalendarHistoryCache"] = None,
) -> ClassificationResult:
    """
    Full classification of *meeting* returning a ``ClassificationResult``.

    Classification is performed in four steps:

    1.  Run the title heuristic (``classify_by_title``) to obtain
        ``title_label``.
    2.  If ``title_label == INTERNAL`` → assign ``EventCategory.INTERNAL``
        immediately (title takes priority).
    3.  Otherwise check ``Meeting.is_external`` (attendee-domain logic).
        If no external attendees → assign ``EventCategory.INTERNAL``
        regardless of title label.
    4.  For external meetings consult the history cache:
        - If cache is unavailable → ``EXTERNAL_FIRST`` (conservative default)
        - If any external attendee email or domain appears in the cache
          → ``EXTERNAL_FOLLOWUP``
        - Otherwise → ``EXTERNAL_FIRST``

    Args:
        meeting:       A ``Meeting`` object from ``google_calendar.py``.
        history_cache: Optional ``CalendarHistoryCache``.

    Returns:
        A fully populated ``ClassificationResult``.
    """
    debug_notes: list[str] = []

    # ── Step 1: Title heuristic ────────────────────────────────────────────────
    title_label = classify_by_title(meeting.summary)
    debug_notes.append(f"title_label={title_label.value}")

    # ── Step 2: Attendee-domain external check ────────────────────────────────
    is_external_by_attendees = meeting.is_external
    ext_attendees = meeting.external_attendees  # list[Attendee]
    ext_emails = _dedupe_ordered([a.email.lower() for a in ext_attendees if a.email])
    ext_domains = _dedupe_ordered([_email_domain(e) for e in ext_emails if _email_domain(e)])

    debug_notes.append(
        f"is_external_by_attendees={is_external_by_attendees}  "
        f"ext_emails={ext_emails}  ext_domains={ext_domains}"
    )

    # ── Step 3: Early-exit for internal events ────────────────────────────────
    if title_label == MeetingLabel.INTERNAL:
        debug_notes.append("→ INTERNAL (title heuristic)")
        return ClassificationResult(
            event_id=meeting.event_id,
            title=meeting.summary,
            category=EventCategory.INTERNAL,
            title_label=title_label,
            is_external_by_attendees=is_external_by_attendees,
            external_attendee_emails=ext_emails,
            history_available=history_cache is not None,
            debug_notes=debug_notes,
        )

    if not is_external_by_attendees:
        debug_notes.append("→ INTERNAL (no external attendees)")
        return ClassificationResult(
            event_id=meeting.event_id,
            title=meeting.summary,
            category=EventCategory.INTERNAL,
            title_label=title_label,
            is_external_by_attendees=is_external_by_attendees,
            external_attendee_emails=ext_emails,
            history_available=history_cache is not None,
            debug_notes=debug_notes,
        )

    # ── Step 4: History cache lookup ──────────────────────────────────────────
    if history_cache is None:
        debug_notes.append("→ EXTERNAL_FIRST (no history cache available)")
        return ClassificationResult(
            event_id=meeting.event_id,
            title=meeting.summary,
            category=EventCategory.EXTERNAL_FIRST,
            title_label=title_label,
            is_external_by_attendees=is_external_by_attendees,
            external_attendee_emails=ext_emails,
            history_available=False,
            debug_notes=debug_notes,
        )

    # Lookup emails first (more specific), then domains
    matched_emails: list[str] = []
    matched_domains: list[str] = []
    last_meetings: list["CachedEvent"] = []
    past_counts: dict[str, int] = {}

    for email in ext_emails:
        count = history_cache.past_meeting_count_for_email(email)
        if count > 0:
            matched_emails.append(email)
            last_ev = history_cache.last_meeting_with_email(email)
            if last_ev and last_ev not in last_meetings:
                last_meetings.append(last_ev)
            debug_notes.append(f"  email match: {email} → {count} past meetings")

    for domain in ext_domains:
        count = history_cache.past_meeting_count_for_domain(domain)
        past_counts[domain] = count
        if count > 0:
            matched_domains.append(domain)
            last_ev = history_cache.last_meeting_with_domain(domain)
            if last_ev and last_ev not in last_meetings:
                last_meetings.append(last_ev)
            debug_notes.append(f"  domain match: {domain} → {count} past meetings")

    history_hit = bool(matched_emails or matched_domains)
    category = EventCategory.EXTERNAL_FOLLOWUP if history_hit else EventCategory.EXTERNAL_FIRST
    debug_notes.append(
        f"→ {category.value}  "
        f"(matched_emails={matched_emails}, matched_domains={matched_domains})"
    )

    logger.debug(
        "classify_event '%s' [%s]: %s  ext_domains=%s  history_hit=%s",
        meeting.summary,
        meeting.event_id,
        category.value,
        ext_domains,
        history_hit,
    )

    return ClassificationResult(
        event_id=meeting.event_id,
        title=meeting.summary,
        category=category,
        title_label=title_label,
        is_external_by_attendees=is_external_by_attendees,
        external_attendee_emails=ext_emails,
        history_matched_emails=matched_emails,
        history_matched_domains=matched_domains,
        history_available=True,
        last_meeting_with=last_meetings,
        past_meeting_counts=past_counts,
        debug_notes=debug_notes,
    )
