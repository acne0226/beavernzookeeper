"""
Daily Schedule Briefing Formatter.

Transforms a list of calendar events (Meeting dataclass OR calendar_fetcher dicts)
into a human-readable Slack Block Kit message summarising the full day.

Usage::

    from src.briefing.formatter import format_daily_briefing

    text, blocks = format_daily_briefing(events, target_date=date.today())
    bot.send_message(text, blocks=blocks)

Input formats accepted
----------------------
* ``Meeting`` dataclass from ``src.calendar.google_calendar``
* ``dict`` event from ``src.calendar_fetcher`` (fields: title, start, end,
  all_day, attendees, location, video_link, conference_type, html_link)

Output
------
``(fallback_text: str, blocks: list[dict])``
  – ``fallback_text`` is shown by clients that can't render Block Kit.
  – ``blocks`` is a Slack Block Kit payload (≤ 50 blocks; safe for
    ``chat_postMessage(blocks=...)``.

Design notes
------------
* External meetings (any non-kakaoventures attendee) are marked 🌐 외부.
* Internal-only meetings are marked 🏢 내부.
* All-day events are listed in a separate section at the top.
* If information cannot be confirmed it is annotated '확인 불가' as per
  project accuracy requirements.
* Attendee lists are capped at 5 external / 4 internal to keep blocks compact.
* Overlapping events are flagged with ⚠️ 시간 충돌.
* Back-to-back events (gap ≤ BACK_TO_BACK_THRESHOLD_MINUTES) are flagged 🔔.
* A time-aware greeting is prepended to every briefing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.config import INTERNAL_DOMAIN

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")

# Korean weekday abbreviations (Monday = index 0)
_KR_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# Slack Block Kit hard limit
_MAX_BLOCKS = 50

# Max attendees shown per event to keep blocks concise
_MAX_EXTERNAL_SHOWN = 5
_MAX_INTERNAL_SHOWN = 4

# Max characters for event title before truncation
_MAX_TITLE_LEN = 60

# Back-to-back threshold: meetings with gap ≤ this many minutes are flagged
BACK_TO_BACK_THRESHOLD_MINUTES: int = 5


# ── Internal normalised event ──────────────────────────────────────────────────

@dataclass
class _NEvent:
    """Unified internal representation of a single calendar event."""
    title: str
    start: datetime          # always a datetime (tz-aware or naive-KST for all-day)
    end: datetime
    all_day: bool
    is_external: bool
    external_attendees: list[str]   # display name or email
    internal_attendees: list[str]   # display name or email
    location: Optional[str]
    video_link: Optional[str]
    conference_type: Optional[str]  # e.g. "Zoom", "Google Meet"
    html_link: str
    duration_minutes: int
    # Flags set by conflict/back-to-back analysis (post-normalisation)
    has_conflict: bool = False          # overlaps with another timed event
    is_back_to_back: bool = False       # gap to previous/next event ≤ threshold
    conflict_with: list[str] = field(default_factory=list)    # titles of conflicting events
    back_to_back_with: list[str] = field(default_factory=list)  # titles of adjacent events


# ── Conflict & back-to-back detection ─────────────────────────────────────────

def _detect_conflicts_and_back_to_backs(
    timed_events: list[_NEvent],
    threshold_minutes: int = BACK_TO_BACK_THRESHOLD_MINUTES,
) -> None:
    """
    Analyse a list of timed (non-all-day) events and mutate their
    ``has_conflict``, ``is_back_to_back``, ``conflict_with``, and
    ``back_to_back_with`` fields in-place.

    Two events **conflict** when their time ranges overlap:
        event_a.start < event_b.end  AND  event_b.start < event_a.end

    Two events are **back-to-back** when, after sorting by start time,
    the gap between consecutive events is ≤ ``threshold_minutes`` minutes
    AND they do not overlap (i.e. they are adjacent but tight).

    Parameters
    ----------
    timed_events:
        List of non-all-day _NEvent objects, already sorted by start time.
    threshold_minutes:
        Maximum gap (in minutes) between two consecutive events to be
        considered "back-to-back".  Default: BACK_TO_BACK_THRESHOLD_MINUTES (5).
    """
    n = len(timed_events)
    if n < 2:
        return

    # ── Pass 1: pairwise conflict detection (O(n²) — acceptable for ≤50 events)
    for i in range(n):
        for j in range(i + 1, n):
            a = timed_events[i]
            b = timed_events[j]
            # Skip all-day events in overlap calculation
            if a.all_day or b.all_day:
                continue
            # Normalise to UTC for comparison
            try:
                a_start = a.start.astimezone(timezone.utc)
                a_end = a.end.astimezone(timezone.utc)
                b_start = b.start.astimezone(timezone.utc)
                b_end = b.end.astimezone(timezone.utc)
            except (AttributeError, TypeError):
                continue  # skip if datetime conversion fails

            if a_start < b_end and b_start < a_end:
                # Genuine overlap
                a.has_conflict = True
                b.has_conflict = True
                if b.title not in a.conflict_with:
                    a.conflict_with.append(b.title)
                if a.title not in b.conflict_with:
                    b.conflict_with.append(a.title)

    # ── Pass 2: back-to-back detection on sorted consecutive pairs
    for i in range(n - 1):
        a = timed_events[i]
        b = timed_events[i + 1]
        if a.all_day or b.all_day:
            continue
        try:
            a_end = a.end.astimezone(timezone.utc)
            b_start = b.start.astimezone(timezone.utc)
        except (AttributeError, TypeError):
            continue

        gap_seconds = (b_start - a_end).total_seconds()
        # Only flag as back-to-back when events don't overlap (gap ≥ 0)
        # and the gap is within the threshold
        if 0 <= gap_seconds <= threshold_minutes * 60:
            a.is_back_to_back = True
            b.is_back_to_back = True
            if b.title not in a.back_to_back_with:
                a.back_to_back_with.append(b.title)
            if a.title not in b.back_to_back_with:
                b.back_to_back_with.append(a.title)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_external_email(email: str) -> bool:
    """Return True when *email* does NOT belong to the internal domain."""
    return bool(email) and not email.lower().endswith(f"@{INTERNAL_DOMAIN}")


def _extract_video_link(description: str = "", location: str = "") -> tuple[Optional[str], Optional[str]]:
    """
    Scan *description* and *location* for video-conference URLs.

    Returns ``(url, conference_type)`` or ``(None, None)`` if none found.
    Supported: Zoom, Microsoft Teams, Google Meet.
    """
    patterns = [
        (r"https?://[^\s<>\"']+zoom\.us/[^\s<>\"',;)]+", "Zoom"),
        (r"https?://[^\s<>\"']*teams\.microsoft\.com[^\s<>\"',;)]+", "Microsoft Teams"),
        (r"https?://meet\.google\.com/[^\s<>\"',;)]+", "Google Meet"),
    ]
    for text in [description, location]:
        if not text:
            continue
        for pattern, name in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(0).rstrip(".,;)"), name
    return None, None


def _fmt_time(dt: datetime) -> str:
    """Return KST HH:MM string from a (possibly UTC) datetime."""
    return dt.astimezone(KST).strftime("%H:%M")


def _fmt_date_kr(d: date) -> str:
    """Format date as Korean-style string, e.g. '2026년 3월 29일 (일)'."""
    wd = _KR_WEEKDAYS[d.weekday()]
    return f"{d.year}년 {d.month}월 {d.day}일 ({wd})"


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending '…' if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _greeting(now_kst: Optional[datetime] = None) -> str:
    """
    Return a time-aware Korean greeting string.

    Time ranges (KST):
        05:00 – 11:59  → morning greeting ☀️
        12:00 – 17:59  → afternoon greeting 👋
        18:00 – 21:59  → evening greeting 🌙
        22:00 – 04:59  → late-night greeting 🌙

    Parameters
    ----------
    now_kst:
        Current KST datetime for greeting selection.  Defaults to
        ``datetime.now(KST)`` when not provided.  Injecting this parameter
        makes the function fully deterministic in tests.

    Returns
    -------
    str
        A single-line greeting, e.g. ``"좋은 아침이에요! ☀️ 오늘 하루도 잘 부탁드립니다."``
    """
    if now_kst is None:
        now_kst = datetime.now(KST)

    hour = now_kst.hour
    if 5 <= hour < 12:
        return "좋은 아침이에요! ☀️ 오늘 하루도 잘 부탁드립니다."
    elif 12 <= hour < 18:
        return "안녕하세요! 👋 오후 일정을 확인해 드릴게요."
    elif 18 <= hour < 22:
        return "좋은 저녁이에요! 🌙 오늘 남은 일정입니다."
    else:
        return "안녕하세요! 🌙 늦은 시간에도 수고 많으십니다."


# ── Normalisers ────────────────────────────────────────────────────────────────

def _normalise(event: Any) -> _NEvent:
    """
    Dispatch to the correct normaliser based on event type.

    Accepts either a ``Meeting`` dataclass or a ``dict`` from calendar_fetcher.
    Raises ``TypeError`` for unknown types.
    """
    # Late import avoids circular-dependency issues
    try:
        from src.calendar.google_calendar import Meeting
        if isinstance(event, Meeting):
            return _normalise_meeting(event)
    except ImportError:
        pass

    if isinstance(event, dict):
        return _normalise_dict(event)

    raise TypeError(
        f"format_daily_briefing: unsupported event type {type(event).__name__!r}. "
        "Expected Meeting dataclass or calendar_fetcher dict."
    )


def _normalise_meeting(m: Any) -> _NEvent:
    """Normalise a ``src.calendar.google_calendar.Meeting`` dataclass."""
    external = [a.display_name or a.email for a in m.external_attendees]
    internal = [a.display_name or a.email for a in m.attendees if a.is_internal]

    # Prefer the Meeting.all_day flag (set since google_calendar.py was updated);
    # fall back to heuristic (midnight UTC + whole-day duration) for older objects.
    if hasattr(m, "all_day"):
        all_day = m.all_day
    else:
        all_day = (
            m.start.hour == 0 and m.start.minute == 0
            and m.end.hour == 0 and m.end.minute == 0
            and m.duration_minutes > 0
            and m.duration_minutes % (60 * 24) == 0
        )

    video_link, conf_type = _extract_video_link(
        m.description or "", m.location or ""
    )

    return _NEvent(
        title=m.summary,
        start=m.start,
        end=m.end,
        all_day=all_day,
        is_external=m.is_external,
        external_attendees=external,
        internal_attendees=internal,
        location=m.location or None,
        video_link=video_link,
        conference_type=conf_type,
        html_link=m.html_link,
        duration_minutes=m.duration_minutes,
    )


def _normalise_dict(ev: dict) -> _NEvent:
    """Normalise a ``calendar_fetcher`` dict event."""
    attendees: list[dict] = ev.get("attendees") or []

    external = [
        a.get("name") or a.get("email", "")
        for a in attendees
        if _is_external_email(a.get("email", ""))
    ]
    internal = [
        a.get("name") or a.get("email", "")
        for a in attendees
        if not _is_external_email(a.get("email", ""))
    ]
    is_external = bool(external)

    start = ev["start"]
    end = ev["end"]
    all_day: bool = ev.get("all_day", False)

    # Convert bare date → datetime for all-day events so the type is uniform
    if all_day:
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime(start.year, start.month, start.day, tzinfo=KST)
        if isinstance(end, date) and not isinstance(end, datetime):
            end = datetime(end.year, end.month, end.day, tzinfo=KST)

    if isinstance(start, datetime) and isinstance(end, datetime):
        duration = int((end - start).total_seconds() // 60)
    else:
        duration = 0

    return _NEvent(
        title=ev.get("title", "(제목 없음)"),
        start=start,
        end=end,
        all_day=all_day,
        is_external=is_external,
        external_attendees=external,
        internal_attendees=internal,
        location=ev.get("location"),
        video_link=ev.get("video_link"),
        conference_type=ev.get("conference_type"),
        html_link=ev.get("html_link", ""),
        duration_minutes=duration,
    )


# ── Block builders ─────────────────────────────────────────────────────────────

def _header_block(date_str: str) -> dict:
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"📅 {date_str} 일정 브리핑",
            "emoji": True,
        },
    }


def _greeting_block(greeting_text: str) -> dict:
    """
    Context block showing the time-aware greeting at the top of the briefing.

    Using a ``context`` block (rather than ``section``) keeps the greeting
    visually subtle — important metadata but not the primary content.
    """
    return {
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": greeting_text}
        ],
    }


def _warnings_block(
    timed: list[_NEvent],
) -> Optional[dict]:
    """
    Build a single section block listing all scheduling warnings.

    Returns ``None`` when there are no conflicts or back-to-back events.

    Warnings produced:
    * ⚠️ *시간 충돌* — event A overlaps event B (both events are flagged)
    * 🔔 *연속 미팅* — event A ends immediately before event B
      (gap ≤ BACK_TO_BACK_THRESHOLD_MINUTES)

    De-duplication: each conflict pair is reported only once (A → B),
    not twice (A → B and B → A).

    Parameters
    ----------
    timed:
        List of non-all-day _NEvent objects, **already** processed by
        ``_detect_conflicts_and_back_to_backs()``.

    Returns
    -------
    dict | None
        A Slack ``section`` block with ``mrkdwn`` text, or ``None``.
    """
    lines: list[str] = []
    seen_conflict_pairs: set[frozenset] = set()
    seen_b2b_pairs: set[frozenset] = set()

    for ev in timed:
        if ev.has_conflict:
            for other_title in ev.conflict_with:
                pair = frozenset([ev.title, other_title])
                if pair not in seen_conflict_pairs:
                    seen_conflict_pairs.add(pair)
                    a_name = _truncate(ev.title, 30)
                    b_name = _truncate(other_title, 30)
                    lines.append(
                        f"⚠️ *시간 충돌*: _{a_name}_ 과 _{b_name}_ 이 겹칩니다."
                    )

        if ev.is_back_to_back:
            for other_title in ev.back_to_back_with:
                pair = frozenset([ev.title, other_title])
                if pair not in seen_b2b_pairs:
                    seen_b2b_pairs.add(pair)
                    a_name = _truncate(ev.title, 30)
                    b_name = _truncate(other_title, 30)
                    lines.append(
                        f"🔔 *연속 미팅*: _{a_name}_ 직후 _{b_name}_ 이 시작됩니다 "
                        f"(휴식 없음 또는 {BACK_TO_BACK_THRESHOLD_MINUTES}분 이하)."
                    )

    if not lines:
        return None

    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "\n".join(lines),
        },
    }


def _summary_block(timed: list[_NEvent], all_day: list[_NEvent]) -> dict:
    """One-line summary with event counts."""
    if not timed and not all_day:
        text = "오늘은 예정된 일정이 없습니다. 여유로운 하루 되세요! 😊"
    else:
        ext_cnt = sum(1 for e in timed if e.is_external)
        int_cnt = len(timed) - ext_cnt
        parts: list[str] = [f"총 *{len(timed)}개* 미팅"]
        if ext_cnt:
            parts.append(f"🌐 외부 *{ext_cnt}개*")
        if int_cnt:
            parts.append(f"🏢 내부 *{int_cnt}개*")
        if all_day:
            parts.append(f"📌 종일 *{len(all_day)}개*")
        text = "  |  ".join(parts)

    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _all_day_block(events: list[_NEvent]) -> dict:
    """Compact block listing all-day events."""
    lines = ["*📌 종일 이벤트*"]
    for ev in events:
        link = f"  <{ev.html_link}|🔗>" if ev.html_link else ""
        lines.append(f"• {ev.title}{link}")
    return {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}


def _event_block(ev: _NEvent) -> dict:
    """Rich section block for a single timed meeting."""
    # ── Time / duration label ────────────────────────────────────────────────
    if ev.all_day:
        time_label = "종일"
    else:
        time_label = f"{_fmt_time(ev.start)} ~ {_fmt_time(ev.end)}"
        if ev.duration_minutes:
            time_label += f"  ({ev.duration_minutes}분)"

    # ── Type badge ───────────────────────────────────────────────────────────
    badge = "🌐 *외부*" if ev.is_external else "🏢 내부"

    # ── Title (linked if we have a URL) ──────────────────────────────────────
    title = _truncate(ev.title, _MAX_TITLE_LEN)
    title_md = f"*<{ev.html_link}|{title}>*" if ev.html_link else f"*{title}*"

    # ── Conflict / back-to-back inline indicators ────────────────────────────
    inline_flags: list[str] = []
    if ev.has_conflict:
        inline_flags.append("⚠️ *충돌*")
    if ev.is_back_to_back:
        inline_flags.append("🔔 *연속*")
    flag_str = "  ".join(inline_flags)

    lines = [
        f"{badge}  {time_label}" + (f"  {flag_str}" if flag_str else ""),
        title_md,
    ]

    # ── Attendees ────────────────────────────────────────────────────────────
    if ev.is_external and ev.external_attendees:
        shown = ev.external_attendees[:_MAX_EXTERNAL_SHOWN]
        extras = len(ev.external_attendees) - len(shown)
        ext_str = ", ".join(_truncate(a, 30) for a in shown)
        if extras:
            ext_str += f" 외 {extras}명"
        lines.append(f"👥 외부: {ext_str}")

    if ev.internal_attendees:
        shown = ev.internal_attendees[:_MAX_INTERNAL_SHOWN]
        extras = len(ev.internal_attendees) - len(shown)
        int_str = ", ".join(_truncate(a, 25) for a in shown)
        if extras:
            int_str += f" 외 {extras}명"
        lines.append(f"🏢 내부: {int_str}")

    # ── Location / video ─────────────────────────────────────────────────────
    loc_parts: list[str] = []
    if ev.video_link:
        label = ev.conference_type or "참가 링크"
        loc_parts.append(f"🎥 <{ev.video_link}|{label}>")
    if ev.location:
        # Don't repeat the URL if it's the same as video_link
        if ev.location != ev.video_link:
            loc_parts.append(f"📍 {_truncate(ev.location, 50)}")
    if loc_parts:
        lines.append("  ".join(loc_parts))

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    }


def _footer_block() -> dict:
    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    "🤖 Work Assistant  |  "
                    "정보 미확인 항목은 *확인 불가* 로 표시됩니다"
                ),
            }
        ],
    }


_DIVIDER: dict = {"type": "divider"}


# ── Fallback text ──────────────────────────────────────────────────────────────

def _build_fallback(
    target_date: date,
    timed: list[_NEvent],
    all_day: list[_NEvent],
    greeting_text: str = "",
) -> str:
    """Plain-text version shown in notifications and non-Block-Kit clients."""
    lines = []
    if greeting_text:
        lines.append(greeting_text)
    lines.append(f"📅 {_fmt_date_kr(target_date)} 일정 브리핑")

    cnt = len(timed)
    ext = sum(1 for e in timed if e.is_external)
    suffix = f" + {len(all_day)}개 종일 이벤트" if all_day else ""
    lines.append(f"총 {cnt}개 미팅 (외부 {ext}개, 내부 {cnt - ext}개){suffix}")

    # ── Scheduling warnings ───────────────────────────────────────────────────
    seen_conflict_pairs: set[frozenset] = set()
    seen_b2b_pairs: set[frozenset] = set()
    warnings: list[str] = []
    for ev in timed:
        if ev.has_conflict:
            for other in ev.conflict_with:
                pair = frozenset([ev.title, other])
                if pair not in seen_conflict_pairs:
                    seen_conflict_pairs.add(pair)
                    warnings.append(
                        f"  ⚠️ 시간 충돌: '{_truncate(ev.title, 25)}' "
                        f"↔ '{_truncate(other, 25)}'"
                    )
        if ev.is_back_to_back:
            for other in ev.back_to_back_with:
                pair = frozenset([ev.title, other])
                if pair not in seen_b2b_pairs:
                    seen_b2b_pairs.add(pair)
                    warnings.append(
                        f"  🔔 연속 미팅: '{_truncate(ev.title, 25)}' "
                        f"→ '{_truncate(other, 25)}'"
                    )
    if warnings:
        lines.append("\n── 일정 주의 ──")
        lines.extend(warnings)

    if all_day:
        lines.append("\n── 종일 이벤트 ──")
        for ev in all_day:
            lines.append(f"  • {ev.title}")

    if timed:
        lines.append("\n── 미팅 ──")
        for ev in timed:
            tag = "[외부]" if ev.is_external else "[내부]"
            time_str = (
                f"{_fmt_time(ev.start)}~{_fmt_time(ev.end)}"
                if not ev.all_day
                else "종일"
            )
            flags = ""
            if ev.has_conflict:
                flags += " [충돌]"
            if ev.is_back_to_back:
                flags += " [연속]"
            lines.append(f"  {time_str} {tag} {ev.title}{flags}")
    elif not all_day:
        lines.append("예정된 일정 없음")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def format_daily_briefing(
    events: list[Any],
    target_date: Optional[date] = None,
    now_kst: Optional[datetime] = None,
) -> tuple[str, list[dict]]:
    """
    Format a list of calendar events into a Slack daily briefing message.

    Parameters
    ----------
    events:
        Ordered list of calendar events for the day.  Each item may be either:
        * ``Meeting`` dataclass (``src.calendar.google_calendar``)
        * ``dict`` from ``src.calendar_fetcher``
        Empty list is valid and results in an "no events" message.
    target_date:
        The calendar date being briefed.  If ``None``, inferred from the first
        event's start time (KST); falls back to today.
    now_kst:
        Current KST datetime used for the time-aware greeting.  Defaults to
        ``datetime.now(KST)``.  Inject a fixed value in tests to get a
        deterministic greeting.

    Returns
    -------
    ``(fallback_text, blocks)``
        Ready for ``WorkAssistantBot.send_message(fallback_text, blocks=blocks)``.

    New in Sub-AC 1.2
    -----------------
    * A time-aware greeting context block is inserted after the header.
    * Overlapping events are flagged with ⚠️ 시간 충돌 in a warnings block and
      inline on each affected event block.
    * Back-to-back events (gap ≤ BACK_TO_BACK_THRESHOLD_MINUTES) are flagged
      with 🔔 연속 미팅.
    """
    # ── Resolve target date ───────────────────────────────────────────────────
    if target_date is None and events:
        first = events[0]
        raw_start = first.get("start") if isinstance(first, dict) else getattr(first, "start", None)
        if isinstance(raw_start, datetime):
            target_date = raw_start.astimezone(KST).date()
        elif isinstance(raw_start, date):
            target_date = raw_start

    if target_date is None:
        from datetime import date as date_cls
        target_date = date_cls.today()

    # ── Normalise ─────────────────────────────────────────────────────────────
    normalised: list[_NEvent] = []
    for ev in events:
        try:
            normalised.append(_normalise(ev))
        except Exception as exc:  # pylint: disable=broad-except
            title = ev.get("title") if isinstance(ev, dict) else getattr(ev, "summary", "?")
            logger.warning("Skipping event '%s' — normalisation failed: %s", title, exc)

    all_day_evts = [e for e in normalised if e.all_day]
    timed_evts = [e for e in normalised if not e.all_day]

    # ── Detect conflicts and back-to-backs ────────────────────────────────────
    _detect_conflicts_and_back_to_backs(timed_evts)

    # ── Greeting ──────────────────────────────────────────────────────────────
    greeting_text = _greeting(now_kst)

    # ── Build blocks ──────────────────────────────────────────────────────────
    date_str = _fmt_date_kr(target_date)
    blocks: list[dict] = []

    blocks.append(_header_block(date_str))
    blocks.append(_greeting_block(greeting_text))
    blocks.append(_summary_block(timed_evts, all_day_evts))
    blocks.append(_DIVIDER)

    # ── Warnings block (conflicts + back-to-backs) ────────────────────────────
    warnings = _warnings_block(timed_evts)
    if warnings is not None:
        blocks.append(warnings)
        blocks.append(_DIVIDER)

    if all_day_evts:
        blocks.append(_all_day_block(all_day_evts))
        blocks.append(_DIVIDER)

    if timed_evts:
        # After the loop we always add: footer-divider (1) + footer-context (1) = 2.
        # When truncation fires we also add: truncation-notice (1) = 1 more.
        # So the guard must leave ≥ 3 spare slots: trigger when len >= MAX - 3.
        for i, ev in enumerate(timed_evts):
            if len(blocks) >= _MAX_BLOCKS - 3:
                remaining = len(timed_evts) - i
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_… 그 외 {remaining}개 일정 (블록 한도 초과)_",
                        }
                    ],
                })
                break
            blocks.append(_event_block(ev))
            # Divider between events (skip after last); also guard against overflow
            if i < len(timed_evts) - 1 and len(blocks) < _MAX_BLOCKS - 3:
                blocks.append(_DIVIDER)
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_예정된 미팅이 없습니다._"},
        })

    blocks.append(_DIVIDER)
    blocks.append(_footer_block())

    # ── Fallback text ─────────────────────────────────────────────────────────
    fallback = _build_fallback(target_date, timed_evts, all_day_evts, greeting_text)

    logger.debug(
        "format_daily_briefing: date=%s  timed=%d  all_day=%d  blocks=%d",
        target_date,
        len(timed_evts),
        len(all_day_evts),
        len(blocks),
    )
    return fallback, blocks
