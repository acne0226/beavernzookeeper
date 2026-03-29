"""
title_signals.py — Structured signal extraction from event titles and metadata.

Purpose
-------
Extend the keyword-based ``classify_by_title`` heuristic with a richer
**rules engine** that extracts multiple named signals from an event's title
and raw metadata, enabling downstream logic to combine signals flexibly.

Three signal groups are extracted:

1. **TitleKeywordSignals** — keyword / regex matches from the summary string.
   Wraps :func:`classify_by_title` and exposes the matched pattern, confidence
   tier, and individual flag groups (is_one_on_one, is_standup, …).

2. **RecurringPatternSignals** — patterns in the title that indicate the event
   is a *recurring* series (numbered suffix, periodic keywords, explicit
   ``[Recurring]`` tag, etc.) plus the raw ``recurring_event_id`` from the
   Google Calendar API.  Recurring events are almost always internal.

3. **MetadataSignals** — event-level metadata signals derived from attendee
   composition, duration, time-of-day, and video conferencing info.

All three are assembled into a top-level :class:`EventSignals` dataclass
via :func:`extract_event_signals`.

Typical usage::

    from src.calendar.title_signals import extract_event_signals, EventSignals

    signals: EventSignals = extract_event_signals(meeting)

    if signals.inferred_is_internal:
        ...  # skip — internal meeting
    print(signals.title_keywords.matched_internal_pattern)
    print(signals.recurring.is_recurring)
    print(signals.metadata.external_attendee_count)

The module is intentionally pure-Python / no I/O so that it can be called
synchronously from the briefing pipeline without async overhead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from src.calendar.title_classifier import (
    classify_by_title,
    matched_internal_pattern,
    matched_external_pattern,
    MeetingLabel,
    _INTERNAL_PATTERNS,
    _EXTERNAL_PATTERNS,
    _COMPILED_INTERNAL,
    _COMPILED_EXTERNAL,
)
from src.config import INTERNAL_DOMAIN

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting


# ── Recurring-pattern regex catalogue ─────────────────────────────────────────
#
# These patterns look for signals IN THE TITLE STRING that suggest the event
# is part of a recurring meeting series.  The Google Calendar ``recurringEventId``
# field is a definitive server-side signal, but many manually-created recurring
# events don't use the Google recurring feature — instead the organiser encodes
# recurrence in the title itself.
#
# Categories:
#
# A. Explicit recurrence tags (usually added by the organiser or tools)
#    - [Recurring], [R], (Recurring), (R)
#    - [Weekly], [Daily], [Monthly], (Weekly), …
#    - "Recurring:" / "Weekly:" / "Daily:" prefix
#
# B. Numbered sequence markers (N-th occurrence of a series)
#    - #5, #42, No.5, Week 12, Wk 12, Episode 7, Ep 7
#    - Sprint 14, Sprint #14, Iteration 3
#    - "- Week 5" / "— Week 5" suffix
#
# C. Periodic-frequency keywords in the title
#    - Weekly / Bi-Weekly / Biweekly / Every week
#    - Daily / Every day
#    - Monthly / Every month
#    - 주간 / 격주 / 월간 / 매주 / 매일 (Korean)
#
# D. Recurring ceremony names without frequency words
#    (these are captured by the existing INTERNAL patterns in title_classifier.py
#    so they're not duplicated here — but we flag them as "likely recurring")

_RECURRING_TAG_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[\s*(recurring|weekly|daily|monthly|bi[\s\-]?weekly|fortnightly)\s*\]", re.IGNORECASE),
    re.compile(r"\(\s*(recurring|weekly|daily|monthly|bi[\s\-]?weekly|fortnightly)\s*\)", re.IGNORECASE),
    re.compile(r"^(recurring|weekly|daily|monthly|bi[\s\-]?weekly)\s*:", re.IGNORECASE),
    re.compile(r"\[\s*R\s*\]", re.IGNORECASE),
    re.compile(r"\(\s*R\s*\)", re.IGNORECASE),
    # Korean recurrence tags
    re.compile(r"\[\s*(반복|주간|일간|월간|격주)\s*\]", re.UNICODE),
    re.compile(r"\(\s*(반복|주간|일간|월간|격주)\s*\)", re.UNICODE),
]

_NUMBERED_SEQUENCE_PATTERNS: list[re.Pattern] = [
    # #N suffix: "Team Sync #5", "Weekly Check-in #42"
    re.compile(r"#\s*\d+\b"),
    # "No. 5" / "No 5" suffix
    re.compile(r"\bNo\.?\s*\d+\b", re.IGNORECASE),
    # "Week 12" / "Wk 12" / "Week 3 ..."
    re.compile(r"\bWeek\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bWk\.?\s*\d+\b", re.IGNORECASE),
    # "- Week 12" / "— Week 12" suffix
    re.compile(r"[\-–—]\s*Week\s*\d+\b", re.IGNORECASE),
    # "Episode 7" / "Ep 7" (sometimes used in knowledge-sharing series)
    re.compile(r"\b(Episode|Ep\.?)\s*\d+\b", re.IGNORECASE),
    # "Sprint 14" or "Sprint #14"
    re.compile(r"\bSprint\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b스프린트\s*#?\d+\b", re.UNICODE),
    # "Iteration 3"
    re.compile(r"\bIteration\s*\d+\b", re.IGNORECASE),
    # "Session 5" when preceded by an internal ceremony word
    re.compile(r"\bSession\s*\d+\b", re.IGNORECASE),
]

_PERIODIC_KEYWORD_PATTERNS: list[re.Pattern] = [
    # English frequency words — standalone
    re.compile(r"\b(weekly|bi[\s\-]?weekly|biweekly|fortnightly)\b", re.IGNORECASE),
    re.compile(r"\b(daily|every[\s\-]day)\b", re.IGNORECASE),
    re.compile(r"\b(monthly|every[\s\-]month)\b", re.IGNORECASE),
    re.compile(r"\bevery\s+(week|two\s+weeks|month)\b", re.IGNORECASE),
    # Korean frequency words
    re.compile(r"\b(주간|격주|월간|매주|매일|매월)\b", re.UNICODE),
]


def _first_match(patterns: list[re.Pattern], text: str) -> Optional[str]:
    """Return the matching group (or full match) for the first pattern that fires."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


# ── 1. TitleKeywordSignals ─────────────────────────────────────────────────────

@dataclass
class TitleKeywordSignals:
    """
    Keyword-level signals extracted purely from the event title string.

    Attributes
    ----------
    summary : str
        The original event title.
    label : MeetingLabel
        Overall INTERNAL / EXTERNAL / UNKNOWN label from :func:`classify_by_title`.
    matched_internal_pattern : str | None
        The first internal regex pattern string that fired, or None.
    matched_external_pattern : str | None
        The first external regex pattern string that fired, or None.
    is_one_on_one : bool
        True if the title looks like a 1:1 / one-on-one meeting.
    is_standup : bool
        True if the title suggests a standup / daily ceremony.
    is_all_hands : bool
        True if the title suggests an all-hands / town-hall.
    is_retro : bool
        True if the title suggests a retrospective.
    is_sprint_ceremony : bool
        True if the title suggests a sprint ceremony (planning/review/demo).
    is_okr : bool
        True if the title contains an OKR-related keyword.
    is_block : bool
        True if the title is a calendar time-block (blocked / block time).
    is_ir_pitch : bool
        True if the title matches an IR / pitch pattern.
    is_investment_meeting : bool
        True if the title matches a Korean investment meeting pattern.
    is_deal_review : bool
        True if the title matches a deal review pattern.
    is_conference : bool
        True if the title matches a conference / summit / webinar pattern.
    confidence : str
        "high"   – a domain-specific pattern matched (investment, IR, sprint, …)
        "medium" – a general pattern matched (weekly, internal, external, …)
        "low"    – only a generic / ambiguous pattern matched
        "none"   – no pattern matched (label is UNKNOWN)
    """

    summary: str
    label: MeetingLabel

    # Pattern debug info
    matched_internal_pattern: Optional[str] = None
    matched_external_pattern: Optional[str] = None

    # Internal sub-flags
    is_one_on_one: bool = False
    is_standup: bool = False
    is_all_hands: bool = False
    is_retro: bool = False
    is_sprint_ceremony: bool = False
    is_okr: bool = False
    is_block: bool = False
    is_hr: bool = False

    # External sub-flags
    is_ir_pitch: bool = False
    is_investment_meeting: bool = False
    is_deal_review: bool = False
    is_partner: bool = False
    is_client_customer: bool = False
    is_conference: bool = False
    is_advisory_board: bool = False
    is_portfolio: bool = False
    is_legal_milestone: bool = False

    confidence: str = "none"

    @property
    def is_internal(self) -> bool:
        return self.label == MeetingLabel.INTERNAL

    @property
    def is_external(self) -> bool:
        return self.label == MeetingLabel.EXTERNAL

    @property
    def is_unknown(self) -> bool:
        return self.label == MeetingLabel.UNKNOWN


# ── Sub-flag detection helpers ────────────────────────────────────────────────

_ONE_ON_ONE = re.compile(
    r"\b(1\s*[:\-]\s*1|1[\s\-]?on[\s\-]?1|one[\s\-]on[\s\-]one|1on1)\b",
    re.IGNORECASE,
)
_STANDUP = re.compile(
    r"\b(stand[\s\-]?up|standup|daily|스탠드[\s\-]?업|데일리|morning\s+sync)\b",
    re.IGNORECASE | re.UNICODE,
)
_ALL_HANDS = re.compile(
    r"\b(all[\s\-]?hands|allhands|town[\s\-]?hall|올핸즈|전사)\b",
    re.IGNORECASE | re.UNICODE,
)
_RETRO = re.compile(
    r"\b(retro(spective)?|회고)\b",
    re.IGNORECASE | re.UNICODE,
)
_SPRINT = re.compile(
    r"\b(sprint[\s\-]?(planning|review|demo|grooming|ceremony)?|스프린트)\b",
    re.IGNORECASE | re.UNICODE,
)
_OKR = re.compile(r"\bokr\b", re.IGNORECASE)
_BLOCK = re.compile(r"\bblock(ed|ing)?\b|\b블록\b", re.IGNORECASE | re.UNICODE)
_HR = re.compile(
    r"\b(hiring|onboarding|면접|채용|온보딩|perf(ormance)?\s*(review|check)?|인사\s*(평가|면담))\b",
    re.IGNORECASE | re.UNICODE,
)

_IR_PITCH = re.compile(r"\b(ir|pitch)\b", re.IGNORECASE)
_INVESTMENT = re.compile(r"\b투자\s*(미팅|심사|검토|상담|협의|논의)\b|\b심사\b", re.UNICODE)
_DEAL = re.compile(r"\b(deal[\s\-]?(review|discussion|call|meeting)|딜[\s\-]?(리뷰|미팅|협의))\b", re.IGNORECASE | re.UNICODE)
_PARTNER = re.compile(r"\b(partner(ship)?|파트너)\b", re.IGNORECASE | re.UNICODE)
_CLIENT = re.compile(r"\b(client|customer|고객)\b", re.IGNORECASE | re.UNICODE)
_CONFERENCE = re.compile(
    # conference\b (no leading \b) so it also matches "TechConference"
    r"(conference\b|summit\b|webinar\b|컨퍼런스|웨비나|network(ing)?\b|네트워킹)",
    re.IGNORECASE | re.UNICODE,
)
_ADVISORY = re.compile(r"\b(advisory[\s\-]?(board|council|committee)|자문)\b", re.IGNORECASE | re.UNICODE)
_PORTFOLIO = re.compile(r"\b(portfolio|포트폴리오)\b", re.IGNORECASE | re.UNICODE)
_LEGAL = re.compile(r"\b(mou|nda|loi)\b", re.IGNORECASE)

# Confidence tiers — ordered from most specific (high) to least specific (low)
_HIGH_CONFIDENCE_INTERNAL = [_ONE_ON_ONE, _STANDUP, _ALL_HANDS, _RETRO, _SPRINT, _OKR]
_HIGH_CONFIDENCE_EXTERNAL = [_IR_PITCH, _INVESTMENT, _DEAL, _PORTFOLIO, _LEGAL]
_MEDIUM_CONFIDENCE_EXTERNAL = [_PARTNER, _CLIENT, _CONFERENCE, _ADVISORY]


def extract_title_keyword_signals(summary: str) -> TitleKeywordSignals:
    """
    Extract structured keyword signals from an event *summary* string.

    Returns a :class:`TitleKeywordSignals` dataclass populated with individual
    sub-flags (is_one_on_one, is_standup, …) and a ``confidence`` tier.

    The ``label`` field mirrors :func:`classify_by_title`.

    Args:
        summary: The calendar event title / summary string.

    Returns:
        :class:`TitleKeywordSignals`

    Examples::

        >>> s = extract_title_keyword_signals("Weekly 1:1 with Alice")
        >>> s.label
        <MeetingLabel.INTERNAL: 'internal'>
        >>> s.is_one_on_one
        True
        >>> s.confidence
        'high'
    """
    if not summary or not summary.strip():
        return TitleKeywordSignals(
            summary=summary or "",
            label=MeetingLabel.UNKNOWN,
            confidence="none",
        )

    label = classify_by_title(summary)
    matched_int = matched_internal_pattern(summary)
    matched_ext = matched_external_pattern(summary)

    # Sub-flags (checked regardless of final label — useful for debug)
    is_one_on_one = bool(_ONE_ON_ONE.search(summary))
    is_standup    = bool(_STANDUP.search(summary))
    is_all_hands  = bool(_ALL_HANDS.search(summary))
    is_retro      = bool(_RETRO.search(summary))
    is_sprint     = bool(_SPRINT.search(summary))
    is_okr        = bool(_OKR.search(summary))
    is_block      = bool(_BLOCK.search(summary))
    is_hr         = bool(_HR.search(summary))

    is_ir_pitch   = bool(_IR_PITCH.search(summary))
    is_investment = bool(_INVESTMENT.search(summary))
    is_deal       = bool(_DEAL.search(summary))
    is_partner    = bool(_PARTNER.search(summary))
    is_client     = bool(_CLIENT.search(summary))
    is_conference = bool(_CONFERENCE.search(summary))
    is_advisory   = bool(_ADVISORY.search(summary))
    is_portfolio  = bool(_PORTFOLIO.search(summary))
    is_legal      = bool(_LEGAL.search(summary))

    # Determine confidence tier
    if label == MeetingLabel.UNKNOWN:
        confidence = "none"
    elif label == MeetingLabel.INTERNAL:
        if any(p.search(summary) for p in _HIGH_CONFIDENCE_INTERNAL):
            confidence = "high"
        else:
            confidence = "medium"
    else:  # EXTERNAL
        if any(p.search(summary) for p in _HIGH_CONFIDENCE_EXTERNAL):
            confidence = "high"
        elif any(p.search(summary) for p in _MEDIUM_CONFIDENCE_EXTERNAL):
            confidence = "medium"
        else:
            confidence = "low"

    return TitleKeywordSignals(
        summary=summary,
        label=label,
        matched_internal_pattern=matched_int,
        matched_external_pattern=matched_ext,
        is_one_on_one=is_one_on_one,
        is_standup=is_standup,
        is_all_hands=is_all_hands,
        is_retro=is_retro,
        is_sprint_ceremony=is_sprint,
        is_okr=is_okr,
        is_block=is_block,
        is_hr=is_hr,
        is_ir_pitch=is_ir_pitch,
        is_investment_meeting=is_investment,
        is_deal_review=is_deal,
        is_partner=is_partner,
        is_client_customer=is_client,
        is_conference=is_conference,
        is_advisory_board=is_advisory,
        is_portfolio=is_portfolio,
        is_legal_milestone=is_legal,
        confidence=confidence,
    )


# ── 2. RecurringPatternSignals ─────────────────────────────────────────────────

@dataclass
class RecurringPatternSignals:
    """
    Signals from the title string and event metadata that indicate whether
    an event belongs to a recurring meeting series.

    Attributes
    ----------
    summary : str
        The original event title.
    recurring_event_id : str | None
        Google Calendar ``recurringEventId`` field — non-None means the event
        is a concrete occurrence of a recurring series defined in the API.
    has_recurring_tag : bool
        True if the title contains an explicit recurrence tag like
        ``[Recurring]``, ``[Weekly]``, ``(R)`` etc.
    has_sequence_number : bool
        True if the title contains a numbered occurrence marker like
        ``#5``, ``Week 12``, ``Sprint 14``.
    has_periodic_keyword : bool
        True if the title contains a frequency word like ``weekly``,
        ``monthly``, ``bi-weekly``, ``주간``, etc.
    recurring_tag_text : str | None
        The actual text that matched the recurring tag pattern.
    sequence_marker_text : str | None
        The actual text that matched the sequence number pattern.
    periodic_keyword_text : str | None
        The actual text that matched the periodic keyword pattern.
    is_recurring : bool
        Composite flag — True if ANY of the three title signals fired OR
        ``recurring_event_id`` is set.
    is_likely_internal_recurring : bool
        True when is_recurring is True AND the title keyword signals suggest
        this is an internal ceremony (standup, sync, retro, etc.) OR the title
        label is INTERNAL.  Purely title-based — does not consult attendees.
    """

    summary: str
    recurring_event_id: Optional[str] = None

    has_recurring_tag: bool = False
    has_sequence_number: bool = False
    has_periodic_keyword: bool = False

    recurring_tag_text: Optional[str] = None
    sequence_marker_text: Optional[str] = None
    periodic_keyword_text: Optional[str] = None

    is_recurring: bool = False
    is_likely_internal_recurring: bool = False


def extract_recurring_pattern_signals(
    summary: str,
    recurring_event_id: Optional[str] = None,
) -> RecurringPatternSignals:
    """
    Extract recurring-pattern signals from an event title and optional
    ``recurring_event_id`` metadata field.

    Args:
        summary:            The calendar event title / summary string.
        recurring_event_id: Google Calendar ``recurringEventId`` field from the
                            raw event dict; ``None`` for non-recurring events.

    Returns:
        :class:`RecurringPatternSignals`

    Examples::

        >>> s = extract_recurring_pattern_signals("Weekly Team Sync #12")
        >>> s.has_sequence_number
        True
        >>> s.has_periodic_keyword
        True
        >>> s.is_recurring
        True
    """
    text = (summary or "").strip()

    # A. Recurring tag in title
    tag_text = _first_match(_RECURRING_TAG_PATTERNS, text)
    has_tag = tag_text is not None

    # B. Sequence number in title
    seq_text = _first_match(_NUMBERED_SEQUENCE_PATTERNS, text)
    has_seq = seq_text is not None

    # C. Periodic keyword in title
    period_text = _first_match(_PERIODIC_KEYWORD_PATTERNS, text)
    has_period = period_text is not None

    # Composite is_recurring
    is_recurring = has_tag or has_seq or has_period or bool(recurring_event_id)

    # is_likely_internal_recurring: recurring + title says internal
    title_label = classify_by_title(text) if text else MeetingLabel.UNKNOWN
    is_likely_internal = (
        is_recurring and (
            title_label == MeetingLabel.INTERNAL
            or has_period  # periodic keyword alone strongly suggests internal rhythm
        )
    )

    return RecurringPatternSignals(
        summary=text,
        recurring_event_id=recurring_event_id,
        has_recurring_tag=has_tag,
        has_sequence_number=has_seq,
        has_periodic_keyword=has_period,
        recurring_tag_text=tag_text,
        sequence_marker_text=seq_text,
        periodic_keyword_text=period_text,
        is_recurring=is_recurring,
        is_likely_internal_recurring=is_likely_internal,
    )


# ── 3. MetadataSignals ─────────────────────────────────────────────────────────

@dataclass
class MetadataSignals:
    """
    Event-metadata signals derived from the meeting object (attendees,
    start/end time, video link, etc.).

    Attributes
    ----------
    duration_minutes : int
        Duration of the meeting in minutes.
    is_short_meeting : bool
        True if duration ≤ 30 minutes.  Short meetings (stand-ups, quick syncs)
        skew heavily internal.
    is_very_long_meeting : bool
        True if duration > 180 minutes (3 hours).  Long blocks lean toward
        conferences, off-sites, or all-day workshops.
    is_all_day : bool
        True if this is an all-day event.
    total_attendee_count : int
        Total number of attendees in the meeting invite.
    internal_attendee_count : int
        Number of attendees from the internal domain
        (``kakaoventures.co.kr`` by default).
    external_attendee_count : int
        Number of attendees from external domains.
    external_attendee_domains : list[str]
        Unique external attendee domains.
    is_solo : bool
        True if total_attendee_count == 1 (self-blocked time).
    is_one_on_one : bool
        True if total_attendee_count == 2.
    is_small_group : bool
        True if 3 ≤ total_attendee_count ≤ 6.
    is_large_group : bool
        True if total_attendee_count > 6.
    has_external_attendees : bool
        True if external_attendee_count > 0.
    has_video_link : bool
        True if the event has a video conference link (Meet, Zoom, Teams).
    video_platform : str | None
        The video platform name (``"Google Meet"``, ``"Zoom"``, ``"Microsoft Teams"``).
    has_physical_location : bool
        True if the event has a physical location string.
    is_early_morning : bool
        True if meeting starts before 09:00 (local time).  Indicative of
        pre-work stand-ups or cross-timezone calls.
    is_evening : bool
        True if meeting starts at 18:00 or later.
    is_weekend : bool
        True if the meeting falls on Saturday or Sunday.
    has_recurring_event_id : bool
        True if the ``recurring_event_id`` metadata field is non-empty.
        Mirror of RecurringPatternSignals to avoid circular imports.
    """

    duration_minutes: int = 0
    is_short_meeting: bool = False
    is_very_long_meeting: bool = False
    is_all_day: bool = False

    total_attendee_count: int = 0
    internal_attendee_count: int = 0
    external_attendee_count: int = 0
    external_attendee_domains: list[str] = field(default_factory=list)

    is_solo: bool = False
    is_one_on_one: bool = False
    is_small_group: bool = False
    is_large_group: bool = False
    has_external_attendees: bool = False

    has_video_link: bool = False
    video_platform: Optional[str] = None
    has_physical_location: bool = False

    is_early_morning: bool = False
    is_evening: bool = False
    is_weekend: bool = False

    has_recurring_event_id: bool = False

    @property
    def attendee_ratio_external(self) -> float:
        """Fraction of attendees that are external (0.0 – 1.0)."""
        if self.total_attendee_count == 0:
            return 0.0
        return self.external_attendee_count / self.total_attendee_count


def _extract_domain(email: str) -> str:
    """Return lower-cased domain part of *email*, or empty string."""
    try:
        return email.split("@")[1].lower()
    except (IndexError, AttributeError):
        return ""


def extract_metadata_signals(meeting: "Meeting") -> MetadataSignals:
    """
    Extract metadata signals from a ``Meeting`` object.

    This function is non-failing: any attribute that is unavailable on the
    mock or real ``Meeting`` object is silently skipped and the corresponding
    signal defaults to its zero/False value.

    Args:
        meeting: A ``Meeting`` object (or duck-typed equivalent with the
                 standard attribute set: ``start``, ``end``, ``all_day``,
                 ``attendees``, ``is_external``, ``external_attendees``,
                 ``video_link``, ``conference_type``, ``location``,
                 ``recurring_event_id``).

    Returns:
        :class:`MetadataSignals`
    """
    # ── Duration ──────────────────────────────────────────────────────────────
    duration_minutes = 0
    is_all_day = False
    try:
        is_all_day = bool(getattr(meeting, "all_day", False))
        if not is_all_day:
            start: datetime = meeting.start
            end: datetime = meeting.end
            delta: timedelta = end - start
            duration_minutes = max(0, int(delta.total_seconds() / 60))
    except Exception:
        pass

    is_short = (not is_all_day) and 0 < duration_minutes <= 30
    is_very_long = (not is_all_day) and duration_minutes > 180

    # ── Attendees ─────────────────────────────────────────────────────────────
    total_count = 0
    internal_count = 0
    external_count = 0
    ext_domains: list[str] = []

    try:
        attendees = getattr(meeting, "attendees", []) or []
        total_count = len(attendees)
        for att in attendees:
            email = getattr(att, "email", "") or ""
            domain = _extract_domain(email)
            if domain == INTERNAL_DOMAIN.lower():
                internal_count += 1
            elif domain:
                external_count += 1
                if domain not in ext_domains:
                    ext_domains.append(domain)
    except Exception:
        pass

    has_external = external_count > 0

    # ── Video / location ──────────────────────────────────────────────────────
    has_video = False
    video_platform: Optional[str] = None
    has_location = False

    try:
        video_link = getattr(meeting, "video_link", None)
        has_video = bool(video_link)
        video_platform = getattr(meeting, "conference_type", None) or None
    except Exception:
        pass

    try:
        location = getattr(meeting, "location", None)
        has_location = bool(location and str(location).strip())
    except Exception:
        pass

    # ── Time-of-day ───────────────────────────────────────────────────────────
    is_early_morning = False
    is_evening = False
    is_weekend = False

    try:
        if not is_all_day:
            start_dt: datetime = meeting.start
            hour = start_dt.hour
            weekday = start_dt.weekday()  # 0=Mon, 6=Sun
            is_early_morning = hour < 9
            is_evening = hour >= 18
            is_weekend = weekday >= 5
    except Exception:
        pass

    # ── Recurring event id ────────────────────────────────────────────────────
    has_recurring_id = False
    try:
        rid = getattr(meeting, "recurring_event_id", None)
        has_recurring_id = bool(rid)
    except Exception:
        pass

    return MetadataSignals(
        duration_minutes=duration_minutes,
        is_short_meeting=is_short,
        is_very_long_meeting=is_very_long,
        is_all_day=is_all_day,
        total_attendee_count=total_count,
        internal_attendee_count=internal_count,
        external_attendee_count=external_count,
        external_attendee_domains=ext_domains,
        is_solo=total_count == 1,
        is_one_on_one=total_count == 2,
        is_small_group=3 <= total_count <= 6,
        is_large_group=total_count > 6,
        has_external_attendees=has_external,
        has_video_link=has_video,
        video_platform=video_platform,
        has_physical_location=has_location,
        is_early_morning=is_early_morning,
        is_evening=is_evening,
        is_weekend=is_weekend,
        has_recurring_event_id=has_recurring_id,
    )


# ── 4. EventSignals — top-level aggregate ─────────────────────────────────────

@dataclass
class EventSignals:
    """
    Aggregated signal bundle for a single calendar event.

    Combines :class:`TitleKeywordSignals`, :class:`RecurringPatternSignals`,
    and :class:`MetadataSignals` into one object and provides high-level
    composite properties for use by the briefing pipeline.

    Attributes
    ----------
    title_keywords : TitleKeywordSignals
        Keyword / pattern signals extracted from the event title.
    recurring : RecurringPatternSignals
        Recurring-series signals from title patterns + metadata.
    metadata : MetadataSignals
        Attendee composition, duration, and video signals.
    inferred_is_internal : bool
        Composite heuristic — True when the combined signals strongly
        suggest an internal-only meeting:

        * Title label is INTERNAL, **or**
        * No external attendees present, **or**
        * Event is a known recurring internal ceremony.

    inferred_is_external : bool
        Composite heuristic — True when the combined signals strongly
        suggest an external-facing meeting:

        * Title label is EXTERNAL **and** external attendees are present, **or**
        * External attendees are present regardless of title.
    """

    title_keywords: TitleKeywordSignals
    recurring: RecurringPatternSignals
    metadata: MetadataSignals

    @property
    def inferred_is_internal(self) -> bool:
        # Rule 1: Title heuristic says internal — always internal
        if self.title_keywords.label == MeetingLabel.INTERNAL:
            return True
        # Rule 2: No external attendees — internal by default
        if not self.metadata.has_external_attendees:
            return True
        # Rule 3: Recurring internal ceremony
        if (
            self.recurring.is_recurring
            and self.recurring.is_likely_internal_recurring
            and not self.metadata.has_external_attendees
        ):
            return True
        return False

    @property
    def inferred_is_external(self) -> bool:
        # Must have external attendees to be external
        if not self.metadata.has_external_attendees:
            return False
        # Title says internal → still treat as internal
        if self.title_keywords.label == MeetingLabel.INTERNAL:
            return False
        return True

    @property
    def dominant_signal(self) -> str:
        """
        Human-readable description of the signal that most strongly
        determined the classification.

        Returns one of:
        ``"title_internal_keyword"``, ``"title_external_keyword"``,
        ``"recurring_internal_pattern"``, ``"no_external_attendees"``,
        ``"external_attendees_present"``, ``"unknown"``.
        """
        if self.title_keywords.label == MeetingLabel.INTERNAL:
            return "title_internal_keyword"
        if self.recurring.is_likely_internal_recurring:
            return "recurring_internal_pattern"
        if not self.metadata.has_external_attendees:
            return "no_external_attendees"
        if self.title_keywords.label == MeetingLabel.EXTERNAL:
            return "title_external_keyword"
        if self.metadata.has_external_attendees:
            return "external_attendees_present"
        return "unknown"


def extract_event_signals(
    meeting: "Meeting",
    *,
    recurring_event_id: Optional[str] = None,
) -> EventSignals:
    """
    Extract all classification signals from *meeting* and assemble an
    :class:`EventSignals` bundle.

    This is the primary entry point for the title-parsing rules engine.
    It is intentionally lightweight — no API calls are made.

    Args:
        meeting:            A ``Meeting`` object (or duck-typed mock).
        recurring_event_id: Override for the ``recurringEventId`` field.
                            When ``None``, the value is read from
                            ``meeting.recurring_event_id`` if it exists.

    Returns:
        :class:`EventSignals`

    Examples::

        >>> from unittest.mock import MagicMock
        >>> m = MagicMock()
        >>> m.summary = "Weekly Team Sync #5"
        >>> m.all_day = False
        >>> m.attendees = []
        >>> m.is_external = False
        >>> m.external_attendees = []
        >>> signals = extract_event_signals(m)
        >>> signals.recurring.is_recurring
        True
        >>> signals.inferred_is_internal
        True
    """
    summary = getattr(meeting, "summary", "") or ""

    # Resolve recurring_event_id
    rid = recurring_event_id
    if rid is None:
        rid = getattr(meeting, "recurring_event_id", None)

    title_sigs = extract_title_keyword_signals(summary)
    recurring_sigs = extract_recurring_pattern_signals(summary, recurring_event_id=rid)
    meta_sigs = extract_metadata_signals(meeting)

    return EventSignals(
        title_keywords=title_sigs,
        recurring=recurring_sigs,
        metadata=meta_sigs,
    )
