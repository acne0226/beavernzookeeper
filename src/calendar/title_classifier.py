"""
title_classifier.py — Title-parsing heuristics for internal/external meeting detection.

Purpose
-------
Complement attendee-domain-based classification (Meeting.is_external) with a
purely title-based heuristic signal.  The function ``classify_by_title`` accepts
an event summary string and returns a raw ``"internal"`` / ``"external"`` /
``"unknown"`` label without making any API calls or inspecting attendees.

Design
------
The classifier works in two passes:

1. **Internal-keyword pass** — if the title matches any pattern in
   ``INTERNAL_PATTERNS``, the event is immediately labelled ``"internal"``.
   These patterns catch recurring internal ceremonies (standups, 1:1s,
   all-hands, retrospectives, OKR reviews, …) in both Korean and English.

2. **External-keyword pass** — if no internal pattern matched, the title is
   checked against ``EXTERNAL_PATTERNS``.  These patterns catch partner demos,
   IR pitches, customer meetings, deal reviews, etc.

3. If neither set matches, ``"unknown"`` is returned so that the caller can
   fall back to attendee-domain logic.

Usage::

    from src.calendar.title_classifier import classify_by_title, MeetingLabel

    label = classify_by_title("주간 팀 싱크")   # → MeetingLabel.INTERNAL
    label = classify_by_title("ABC Corp 투자 미팅")  # → MeetingLabel.EXTERNAL
    label = classify_by_title("Birthday party")   # → MeetingLabel.UNKNOWN
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


# ── Label enum ────────────────────────────────────────────────────────────────

class MeetingLabel(str, Enum):
    """Raw classification label produced by the title classifier."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    UNKNOWN  = "unknown"


# ── Internal-event patterns ───────────────────────────────────────────────────
#
# Ordered from most-specific to least-specific so that the first match wins
# in any future priority-based extension.  All patterns are compiled
# case-insensitively; Korean characters are already case-neutral.
#
# Coverage targets (from project constraints / typical investment-team calendar):
#
# English:
#   1:1 / 1-on-1 / one-on-one / 1on1
#   standup / stand-up / stand up / daily / morning sync
#   all-hands / allhands / all hands / townhall / town hall
#   retrospective / retro
#   sprint planning / sprint review / sprint demo
#   OKR / OKR review / OKR check-in
#   team sync / weekly sync / bi-weekly / squad sync
#   internal / company-internal
#   hiring / interview (usually internal process)
#   onboarding
#   lunch & learn / brown bag
#   performance review / perf review
#
# Korean:
#   1:1 (숫자 형태 그대로)
#   스탠드업 / 스탠드-업
#   올핸즈 / 전사 회의 / 전사 미팅
#   팀 회의 / 팀 싱크 / 주간 회의 / 주간 싱크 / 격주 싱크
#   회고 / 스프린트 회고 / 스프린트 플래닝 / 스프린트 리뷰
#   내부 회의 / 내부 미팅
#   OKR 리뷰 / OKR 체크인
#   면접 (internal HR, not external deal)
#   온보딩
#   squad 관련 (squad + 회의/싱크/스탠드업)

_INTERNAL_PATTERNS: list[str] = [
    # ── 1:1 variants ────────────────────────────────────────────────────────
    r"\b1\s*[:\-]\s*1\b",                       # 1:1, 1-1, 1 : 1
    r"\b1[\s\-]?on[\s\-]?1\b",                  # 1-on-1, 1on1, 1 on 1
    r"\bone[\s\-]on[\s\-]one\b",
    r"\b1on1\b",

    # ── Standup / daily ─────────────────────────────────────────────────────
    r"\bstand[\s\-]?up\b",
    r"\bstandup\b",
    r"\b데일리\b",
    r"\b스탠드[\s\-]?업\b",
    r"\bdaily\s*(standup|sync|check[\s\-]?in|scrum)?\b",
    r"\bmorning\s+sync\b",

    # ── All-hands / townhall ─────────────────────────────────────────────────
    r"\ball[\s\-]?hands\b",
    r"\ballhands\b",
    r"\btown[\s\-]?hall\b",
    r"\b올핸즈\b",
    r"\b전사\s*(회의|미팅|싱크|행사)?\b",

    # ── Retrospective ────────────────────────────────────────────────────────
    r"\bretro(spective)?\b",
    r"\b회고\b",
    r"\b스프린트\s*회고\b",

    # ── Sprint ceremonies ────────────────────────────────────────────────────
    r"\bsprint\s*(planning|review|demo|grooming|ceremony)\b",
    r"\bsprint\b",                                # bare "sprint" rarely external
    r"\b스프린트\s*(플래닝|리뷰|데모|그루밍)?\b",

    # ── OKR ─────────────────────────────────────────────────────────────────
    r"\bokr[\s\-]?(review|check[\s\-]?in|kickoff|planning|세션|리뷰|체크인)?\b",

    # ── Team / squad syncs ───────────────────────────────────────────────────
    r"\bteam\s*(sync|meeting|mtg|standup|call)\b",
    r"\bsquad\s*(sync|standup|meeting|mtg|call|회의|싱크)?\b",
    r"\bweekly\s*(sync|team|meeting|check[\s\-]?in|standup)?\b",
    r"\bbi[\s\-]?weekly\b",
    r"\b격주\s*(싱크|회의|미팅|체크인)?\b",
    r"\b주간\s*(회의|싱크|미팅|업무보고)?\b",
    r"\b월간\s*(회의|싱크|미팅|업무보고)?\b",
    r"\b팀\s*(회의|싱크|미팅|스탠드업)\b",
    r"\b내부\s*(회의|미팅|싱크)\b",

    # ── Internal / company-internal labels ──────────────────────────────────
    r"\binternal\s*(meeting|sync|call|mtg)?\b",
    r"\bcompany[\s\-]?internal\b",

    # ── HR / hiring ─────────────────────────────────────────────────────────
    r"\b(hiring|onboarding|채용|면접|온보딩)\b",
    r"\bperf(ormance)?\s*(review|check[\s\-]?in)?\b",
    r"\b인사\s*(평가|면담)\b",

    # ── Calendar time blocks ─────────────────────────────────────────────────
    # Events titled "block", "blocked", "block time", "focus block", etc. are
    # personal or team calendar reservations that should never appear in
    # external-meeting briefings.
    r"\bblock(ed|ing)?\b",
    r"\b블록\b",                                # Korean "block"

    # ── Knowledge sharing ────────────────────────────────────────────────────
    r"\blunch[\s&and]+learn\b",
    r"\bbrown[\s\-]?bag\b",
    r"\b지식\s*공유\b",

    # ── Recurring internal ceremonies (Korean) ──────────────────────────────
    r"\b사업\s*(계획|보고|검토)\b",              # business planning / review
    r"\b전략\s*(회의|세션)\b",                   # strategy session
    r"\b조직\s*(문화|행사)\b",                   # culture events
]

# ── External-event patterns ───────────────────────────────────────────────────
#
# These patterns fire only when no internal pattern matched.
#
# Coverage:
#   IR / pitch (investor relations)
#   투자 미팅 / 투자심사 / deal review
#   partner / 파트너
#   customer / client / 고객
#   demo (when not a sprint demo → already caught above)
#   conference / 컨퍼런스
#   외부 미팅 / 외부 방문
#   네트워킹 / networking event
#   포트폴리오 / portfolio company meetings
#   VC / 투자사 명칭

_EXTERNAL_PATTERNS: list[str] = [
    # ── Investment-specific ──────────────────────────────────────────────────
    r"\bir\b",                                   # investor relations
    r"\bpitch\b",
    r"\b투자\s*(미팅|심사|검토|상담|협의|논의)\b",
    r"\bdeal\s*(review|discussion|call|meeting)\b",
    r"\b딜\s*(리뷰|미팅|협의)\b",
    r"\b심사\b",                                 # (investment) screening
    r"\b포트폴리오\s*(미팅|방문|상담)\b",
    r"\bportfolio\s*(company\s*)?(meeting|call|check[\s\-]?in)\b",

    # ── Partner / client ─────────────────────────────────────────────────────
    r"\bpartner(ship)?\s*(meeting|call|mtg|demo|sync|협의)?\b",
    r"\b파트너\s*(미팅|협의|상담|방문)?\b",
    r"\bclient\s*(meeting|call|sync|demo)?\b",
    r"\bcustomer\s*(meeting|call|sync|success|demo)?\b",
    r"\b고객\s*(미팅|상담|방문|발표)?\b",

    # ── External visit / meeting explicit label ──────────────────────────────
    r"\b외부\s*(미팅|방문|회의|발표)\b",
    r"\bexternal\s*(meeting|call|visit|demo)?\b",

    # ── Demo to external party ───────────────────────────────────────────────
    r"\bproduct\s*demo\b",
    r"\bsales\s*(meeting|call|demo|deck)?\b",
    r"\b세일즈\s*(미팅|콜|데모)?\b",

    # ── Conference / event ───────────────────────────────────────────────────
    r"conference\b",                             # TechConference, conference
    r"\b컨퍼런스\b",
    r"\bsummit\b",
    r"\bwebinar\b",
    r"\b웨비나\b",
    r"\bnetwork(ing)?\s*(event|session|lunch|dinner)?\b",
    r"\b네트워킹\b",

    # ── MOU / NDA / legal milestone ─────────────────────────────────────────
    r"\b(mou|nda|loi)\b",

    # ── Advisory board ───────────────────────────────────────────────────────
    r"\badvisory\s*(board|council|committee)\b",
    r"\b자문\s*(위원회|회의|미팅)?\b",
]


# ── Compiled regex sets ───────────────────────────────────────────────────────

_COMPILED_INTERNAL: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in _INTERNAL_PATTERNS
]

_COMPILED_EXTERNAL: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in _EXTERNAL_PATTERNS
]


# ── Public API ────────────────────────────────────────────────────────────────

def classify_by_title(
    summary: str,
    *,
    internal_patterns: Optional[list[re.Pattern]] = None,
    external_patterns: Optional[list[re.Pattern]] = None,
) -> MeetingLabel:
    """
    Return a raw ``MeetingLabel`` for *summary* based on keyword heuristics.

    The classification is performed in priority order:

    1. If *summary* matches any internal pattern → ``MeetingLabel.INTERNAL``
    2. Else if it matches any external pattern   → ``MeetingLabel.EXTERNAL``
    3. Otherwise                                 → ``MeetingLabel.UNKNOWN``

    Args:
        summary:           The calendar event title / summary string.
        internal_patterns: Override the default compiled internal patterns
                           (used in tests to inject custom rules).
        external_patterns: Override the default compiled external patterns.

    Returns:
        A ``MeetingLabel`` enum value.

    Examples::

        >>> classify_by_title("Weekly Team Sync")
        <MeetingLabel.INTERNAL: 'internal'>
        >>> classify_by_title("ABC Corp IR Pitch")
        <MeetingLabel.EXTERNAL: 'external'>
        >>> classify_by_title("Birthday Cake 🎂")
        <MeetingLabel.UNKNOWN: 'unknown'>
    """
    if not summary or not summary.strip():
        return MeetingLabel.UNKNOWN

    text = summary.strip()

    _internal = internal_patterns if internal_patterns is not None else _COMPILED_INTERNAL
    _external = external_patterns if external_patterns is not None else _COMPILED_EXTERNAL

    # Pass 1 — internal
    for pattern in _internal:
        if pattern.search(text):
            return MeetingLabel.INTERNAL

    # Pass 2 — external
    for pattern in _external:
        if pattern.search(text):
            return MeetingLabel.EXTERNAL

    return MeetingLabel.UNKNOWN


def is_title_internal(summary: str) -> bool:
    """
    Convenience wrapper — return ``True`` iff the title heuristic labels the
    event as internal.  Does **not** consult attendee domains.
    """
    return classify_by_title(summary) == MeetingLabel.INTERNAL


def is_title_external(summary: str) -> bool:
    """
    Convenience wrapper — return ``True`` iff the title heuristic labels the
    event as external.  Does **not** consult attendee domains.
    """
    return classify_by_title(summary) == MeetingLabel.EXTERNAL


def matched_internal_pattern(summary: str) -> Optional[str]:
    """
    Return the first internal pattern string that matched *summary*, or
    ``None``.  Useful for debug logging.
    """
    for pat, src in zip(_COMPILED_INTERNAL, _INTERNAL_PATTERNS):
        if pat.search(summary):
            return src
    return None


def matched_external_pattern(summary: str) -> Optional[str]:
    """
    Return the first external pattern string that matched *summary*, or
    ``None``.  Useful for debug logging.
    """
    for pat, src in zip(_COMPILED_EXTERNAL, _EXTERNAL_PATTERNS):
        if pat.search(summary):
            return src
    return None
