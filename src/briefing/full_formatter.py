"""
Full Briefing Formatter.

Formats a BriefingData object (calendar events + Gmail emails + Notion deadlines)
into a comprehensive Slack Block Kit message.

This is the formatter used by the /brief slash command when the full
aggregation pipeline (Sub-AC 3.2) is active.

Sections produced
-----------------
1. Header          — date and fetched_at timestamp
2. Calendar        — today's meetings (external/internal badge, times, attendees)
3. Gmail           — pending inbox emails (unread highlighted, sender, subject)
4. Notion          — portfolio company deadline items (overdue first)
5. Source errors   — '확인 불가' annotations for any failed data sources
6. Footer          — accuracy reminder

Design notes
------------
* Blocks are capped at 50 (Slack hard limit) with a truncation notice.
* Missing/failed sources are annotated '확인 불가' per accuracy requirements.
* All times are shown in KST (Asia/Seoul).
* No hallucinated data — if a source failed, only the failure is reported.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional, Any
from zoneinfo import ZoneInfo

from src.config import INTERNAL_DOMAIN

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

# Korean weekday abbreviations (Monday = index 0)
_KR_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# Slack Block Kit hard limit
_MAX_BLOCKS = 50

# Attendee display caps
_MAX_EXTERNAL_SHOWN = 5
_MAX_INTERNAL_SHOWN = 4

# Email display cap
_MAX_EMAILS_SHOWN = 10

# Notion deadline display cap
_MAX_NOTION_SHOWN = 15


# ── Formatting helpers ──────────────────────────────────────────────────────────

def _fmt_time(dt: datetime) -> str:
    """Format a UTC/tz-aware datetime as KST HH:MM."""
    return dt.astimezone(KST).strftime("%H:%M")


def _fmt_date_kr(d: date) -> str:
    """Format date as '2026년 3월 29일 (일)'."""
    wd = _KR_WEEKDAYS[d.weekday()]
    return f"{d.year}년 {d.month}월 {d.day}일 ({wd})"


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending '…' if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_add_block(blocks: list[dict], block: dict) -> bool:
    """
    Add *block* to *blocks* if within the 50-block Slack limit.
    Returns True if block was added, False if limit reached.
    """
    # Reserve 2 slots for divider + footer
    if len(blocks) >= _MAX_BLOCKS - 2:
        return False
    blocks.append(block)
    return True


# ── Block builders ──────────────────────────────────────────────────────────────

def _header_block(target_date: date, fetched_at: Optional[datetime] = None) -> dict:
    date_str = _fmt_date_kr(target_date)
    text = f"📋 {date_str} 브리핑"
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text, "emoji": True},
    }


def _fetched_at_block(fetched_at: Optional[datetime]) -> Optional[dict]:
    """
    Return a context block showing when the briefing data was fetched.

    Returns None when *fetched_at* is not available so callers can skip it
    cleanly without an extra None-check.
    """
    if fetched_at is None:
        return None
    time_str = fetched_at.astimezone(KST).strftime("%H:%M")
    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"브리핑 생성 시각: {time_str} KST",
            }
        ],
    }


def _section_header(title: str) -> dict:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*{title}*"},
    }


def _unavailable_block(source: str, error: str) -> dict:
    """Block shown when a source failed — annotated '확인 불가'."""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"⚠️ *{source}* — 확인 불가 (`{_truncate(error, 80)}`)",
        },
    }


_DIVIDER: dict = {"type": "divider"}


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


# ── Calendar section ───────────────────────────────────────────────────────────

def _build_calendar_blocks(briefing_data: Any) -> list[dict]:
    """
    Build Slack blocks for the calendar section.

    Uses the existing formatter (format_daily_briefing) to render the
    events, but strips its header/footer and keeps only the event blocks.
    Falls back to a simple error block if the calendar source failed.
    """
    if not briefing_data.has_calendar:
        return [_unavailable_block("📅 캘린더", briefing_data.source_errors.get("calendar", "오류"))]

    events = briefing_data.calendar_events
    if not events:
        return [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_예정된 일정이 없습니다._"},
            }
        ]

    # Use the existing formatter for calendar events
    from src.briefing.formatter import format_daily_briefing
    _, blocks = format_daily_briefing(events, target_date=briefing_data.target_date)

    # Strip the header (first block) and footer (last 2 blocks: divider + context)
    # to avoid duplicating them in the full briefing
    inner_blocks = blocks
    if inner_blocks and inner_blocks[0].get("type") == "header":
        inner_blocks = inner_blocks[1:]
    # Remove trailing divider + footer if present
    while inner_blocks and inner_blocks[-1].get("type") in ("context", "divider"):
        inner_blocks = inner_blocks[:-1]
    # Also strip the summary block (first section after header) — we'll add our own
    if inner_blocks and inner_blocks[0].get("type") == "section":
        # Keep the summary block — it has useful event counts
        pass

    return inner_blocks if inner_blocks else [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_일정 정보를 가져올 수 없습니다._"},
        }
    ]


# ── Gmail section ──────────────────────────────────────────────────────────────

def _email_line(email: Any, *, urgent: bool = False) -> str:
    """
    Render a single EmailMessage as a Slack mrkdwn line.

    Parameters
    ----------
    email:
        An EmailMessage object.
    urgent:
        When True prepend a 🔴 badge to draw the user's attention.
    """
    subject = getattr(email, "subject", "(제목 없음)") or "(제목 없음)"
    sender_name = getattr(email, "sender", "") or ""
    sender_email_addr = getattr(email, "sender_email", "") or ""
    sender_display = sender_name or sender_email_addr or "알 수 없음"
    is_unread = getattr(email, "is_unread", False)
    is_external = getattr(email, "is_external", False)
    is_important = getattr(email, "is_important", False)
    received_at = getattr(email, "received_at", None)

    unread_icon = "📬" if is_unread else "📭"
    external_badge = "🌐" if is_external else "🏢"
    important_badge = "⭐" if is_important else ""

    time_str = ""
    if received_at and isinstance(received_at, datetime):
        time_str = f"  `{received_at.astimezone(KST).strftime('%m/%d %H:%M')}`"

    subject_display = _truncate(subject, 50)
    sender_trunc = _truncate(sender_display, 30)

    urgency_prefix = "🔴 " if urgent else ""
    line = (
        f"{urgency_prefix}{unread_icon} {external_badge}{important_badge} "
        f"*{subject_display}*  — {sender_trunc}{time_str}"
    )
    return line


def _build_gmail_blocks(briefing_data: Any) -> list[dict]:
    """
    Build Slack blocks for the Gmail section.

    Structure:
    1. Summary line — total / unread / external / urgent counts
    2. Urgent sub-section (if any) — unread-external and/or Gmail-important emails
    3. All remaining inbox emails (capped at _MAX_EMAILS_SHOWN)
    4. Overflow notice (if more than cap)

    Urgent emails are displayed first with a 🔴 marker so the user can
    immediately see what requires action without scrolling through all messages.
    """
    if not briefing_data.has_gmail:
        return [_unavailable_block("📧 이메일", briefing_data.source_errors.get("gmail", "오류"))]

    emails = briefing_data.emails
    if not emails:
        return [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_새 이메일이 없습니다._"},
            }
        ]

    blocks: list[dict] = []

    # Count summary
    total = len(emails)
    unread = sum(1 for e in emails if getattr(e, "is_unread", False))
    external = sum(1 for e in emails if getattr(e, "is_external", False))
    # urgent_emails is a derived property on BriefingData
    urgent_list = getattr(briefing_data, "urgent_emails", [])
    urgent_count = len(urgent_list)

    parts = [f"총 *{total}개*"]
    if urgent_count:
        parts.append(f"🔴 즉시 확인 *{urgent_count}개*")
    if unread:
        parts.append(f"📬 미읽음 *{unread}개*")
    if external:
        parts.append(f"🌐 외부 *{external}개*")
    summary_text = "  |  ".join(parts)

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary_text},
    })

    # ── Urgent sub-section ─────────────────────────────────────────────────────
    if urgent_list:
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "*🔴 즉시 확인 필요*"}
            ],
        })
        for email in urgent_list[:5]:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": _email_line(email, urgent=True)},
            })
        if len(urgent_list) > 5:
            remaining_urgent = len(urgent_list) - 5
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_… 즉시 확인 필요 이메일 {remaining_urgent}개 더_"}
                ],
            })

    # ── All inbox emails (deduped from urgent, capped) ─────────────────────────
    urgent_ids = {getattr(e, "message_id", None) for e in urgent_list}
    non_urgent = [
        e for e in emails if getattr(e, "message_id", None) not in urgent_ids
    ]

    if non_urgent:
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "*📬 받은 편지함*"}
            ],
        })
        shown_emails = non_urgent[:_MAX_EMAILS_SHOWN]
        for email in shown_emails:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": _email_line(email, urgent=False)},
            })

        if len(non_urgent) > _MAX_EMAILS_SHOWN:
            remaining = len(non_urgent) - _MAX_EMAILS_SHOWN
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_… 그 외 {remaining}개 이메일_",
                    }
                ],
            })

    return blocks


# ── Notion section ─────────────────────────────────────────────────────────────

def _build_notion_blocks(briefing_data: Any) -> list[dict]:
    """Build Slack blocks for the Notion deadlines section."""
    if not briefing_data.has_notion:
        return [_unavailable_block("📑 Notion 마감일", briefing_data.source_errors.get("notion", "오류"))]

    deadlines = briefing_data.notion_deadlines
    if not deadlines:
        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_향후 30일 내 마감 예정 항목이 없습니다._",
                },
            }
        ]

    blocks: list[dict] = []

    # Summary
    total = len(deadlines)
    overdue = sum(1 for d in deadlines if getattr(d, "is_overdue", False))
    today_due = sum(
        1 for d in deadlines
        if getattr(d, "days_until", 1) == 0 and not getattr(d, "is_overdue", False)
    )
    this_week = sum(
        1 for d in deadlines
        if 0 < getattr(d, "days_until", 999) <= 7
    )

    parts = [f"총 *{total}개*"]
    if overdue:
        parts.append(f"🚨 기한 초과 *{overdue}개*")
    if today_due:
        parts.append(f"🔴 오늘 마감 *{today_due}개*")
    if this_week:
        parts.append(f"🟡 이번 주 *{this_week}개*")

    summary_text = "  |  ".join(parts)
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary_text},
    })

    # List deadlines (capped)
    shown_deadlines = deadlines[:_MAX_NOTION_SHOWN]
    for item in shown_deadlines:
        name = getattr(item, "name", "(이름 없음)") or "(이름 없음)"
        deadline_date = getattr(item, "deadline", None)
        status = getattr(item, "status", "") or ""
        is_overdue = getattr(item, "is_overdue", False)
        days_until = getattr(item, "days_until", 0)
        url = getattr(item, "url", "") or ""

        # Build date display
        if deadline_date:
            date_str = deadline_date.strftime("%m/%d")
        else:
            date_str = "확인 불가"

        # Urgency icon
        if is_overdue:
            urgency = "🚨"
            days_label = f"(`{abs(days_until)}일 초과`)"
        elif days_until == 0:
            urgency = "🔴"
            days_label = "(`오늘`)"
        elif days_until <= 3:
            urgency = "🟠"
            days_label = f"(`D-{days_until}`)"
        elif days_until <= 7:
            urgency = "🟡"
            days_label = f"(`D-{days_until}`)"
        else:
            urgency = "🟢"
            days_label = f"(`D-{days_until}`)"

        # Build line
        name_display = _truncate(name, 40)
        if url:
            name_md = f"<{url}|{name_display}>"
        else:
            name_md = name_display

        status_part = f"  [{_truncate(status, 20)}]" if status else ""
        line = f"{urgency} *{name_md}*  `{date_str}` {days_label}{status_part}"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": line},
        })

    # Overflow notice
    if len(deadlines) > _MAX_NOTION_SHOWN:
        remaining = len(deadlines) - _MAX_NOTION_SHOWN
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_… 그 외 {remaining}개 항목_",
                }
            ],
        })

    return blocks


# ── Fallback text ──────────────────────────────────────────────────────────────

def _build_fallback(briefing_data: Any) -> str:
    """Plain-text fallback for notifications / non-Block-Kit clients."""
    lines = [f"📋 {_fmt_date_kr(briefing_data.target_date)} 브리핑"]

    # Calendar summary
    if briefing_data.has_calendar:
        events = briefing_data.calendar_events
        ext = sum(1 for e in events if getattr(e, "is_external", False))
        lines.append(f"📅 캘린더: {len(events)}개 미팅 (외부 {ext}개)")
    else:
        lines.append("📅 캘린더: 확인 불가")

    # Gmail summary
    if briefing_data.has_gmail:
        emails = briefing_data.emails
        unread = sum(1 for e in emails if getattr(e, "is_unread", False))
        urgent = len(getattr(briefing_data, "urgent_emails", []))
        urgent_part = f", 즉시 확인 {urgent}개" if urgent else ""
        lines.append(f"📧 이메일: {len(emails)}개 (미읽음 {unread}개{urgent_part})")
    else:
        lines.append("📧 이메일: 확인 불가")

    # Notion summary
    if briefing_data.has_notion:
        deadlines = briefing_data.notion_deadlines
        overdue = sum(1 for d in deadlines if getattr(d, "is_overdue", False))
        lines.append(f"📑 Notion: {len(deadlines)}개 마감 항목 (기한 초과 {overdue}개)")
    else:
        lines.append("📑 Notion: 확인 불가")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def format_full_briefing(briefing_data: Any) -> tuple[str, list[dict]]:
    """
    Format a BriefingData object into a comprehensive Slack message.

    Parameters
    ----------
    briefing_data:
        A BriefingData instance from src.briefing.aggregator.

    Returns
    -------
    (fallback_text, blocks)
        Ready for WorkAssistantBot.send_message(fallback_text, blocks=blocks).

    Notes
    -----
    * Sources that failed are annotated '확인 불가'.
    * Total block count is capped at 50 (Slack limit).
    * Calendar events are rendered using the existing format_daily_briefing()
      formatter for consistency.
    """
    blocks: list[dict] = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    blocks.append(_header_block(briefing_data.target_date, briefing_data.fetched_at))
    fetched_block = _fetched_at_block(getattr(briefing_data, "fetched_at", None))
    if fetched_block is not None:
        blocks.append(fetched_block)

    # ── 2. Calendar section ────────────────────────────────────────────────────
    blocks.append(_DIVIDER)
    blocks.append(_section_header("📅 오늘의 일정"))
    for block in _build_calendar_blocks(briefing_data):
        if not _safe_add_block(blocks, block):
            break

    # ── 3. Gmail section ───────────────────────────────────────────────────────
    blocks.append(_DIVIDER)
    blocks.append(_section_header("📧 받은 편지함"))
    for block in _build_gmail_blocks(briefing_data):
        if not _safe_add_block(blocks, block):
            break

    # ── 4. Notion deadlines section ────────────────────────────────────────────
    blocks.append(_DIVIDER)
    blocks.append(_section_header("📑 포트폴리오 마감 일정 (30일 이내)"))
    for block in _build_notion_blocks(briefing_data):
        if not _safe_add_block(blocks, block):
            break

    # ── 5. Footer ──────────────────────────────────────────────────────────────
    # Ensure there is always a divider + footer at the end
    # (trim back to make room if necessary)
    while len(blocks) > _MAX_BLOCKS - 2:
        blocks.pop()
    blocks.append(_DIVIDER)
    blocks.append(_footer_block())

    # ── Fallback text ─────────────────────────────────────────────────────────
    fallback = _build_fallback(briefing_data)

    logger.debug(
        "format_full_briefing: date=%s  blocks=%d  errors=%s",
        briefing_data.target_date,
        len(blocks),
        list(briefing_data.source_errors.keys()),
    )

    return fallback, blocks
