"""
External Follow-up Meeting Briefing Formatter (AC 7 Sub-AC 4).

Formats a ``RawBriefingContent`` object into a Slack Block Kit message
tailored for **EXTERNAL_FOLLOWUP** meetings — meetings with companies the
investment team has met before.

The briefing emphasises *relationship context*:
  - Past meeting history with the attendees
  - Notion page / deal records for the company
  - Recent Gmail exchanges with the company
  - Related Slack message history from priority channels

Output format
-------------
``(fallback_text: str, blocks: list[dict])``
  * ``fallback_text`` is shown in mobile push notifications / non-Block-Kit clients.
  * ``blocks`` is a Slack Block Kit payload (always ≤ 50 blocks).

Sections produced
-----------------
1. Header          — "🔄 후속 미팅 브리핑 — {title}"
2. Time / location — countdown, KST start–end, duration, optional location
3. External attendees — enriched profiles with past-meeting history
4. Internal attendees — context element (internal participants)
5. Notion section  — Notion deal / portfolio records for the company
6. Gmail section   — recent email exchanges with the company
7. Slack section   — related Slack message history from priority channels
8. Footer          — accuracy reminder + error summary

Accuracy / data-safety rules
-----------------------------
* Sources that failed are annotated ``⚠️ 확인 불가``.
* ``slack_history_available=False`` → Slack section shows error annotation.
* ``gmail_available=False``         → Gmail section shows error annotation.
* ``notion_available=False``        → Notion section shows error annotation.
* ``calendar_history_available=False`` → attendee shows "이력 확인 불가".
* All times are shown in KST (Asia/Seoul).
* Block count capped at 50 (Slack hard limit).

Usage::

    from src.briefing.external_followup_formatter import format_external_followup_briefing

    text, blocks = format_external_followup_briefing(raw_content)
    ok = bot.send_message(text, blocks=blocks)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.briefing.context_aggregator import RawBriefingContent

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")

# Maximum blocks in a Slack message (hard API limit)
_MAX_BLOCKS: int = 50

# How many Notion records to show before truncating
_MAX_NOTION_SHOWN: int = 5

# How many Gmail threads to show before truncating
_MAX_GMAIL_SHOWN: int = 5

# How many Slack messages to show before truncating
_MAX_SLACK_SHOWN: int = 7

# Max characters for meeting description in the agenda section
_MAX_DESCRIPTION_LEN: int = 400

# Max characters for a Slack message snippet
_MAX_SLACK_SNIPPET_LEN: int = 120

# Max characters for email subject display
_MAX_SUBJECT_LEN: int = 60

# Max characters for attendee domain display
_MAX_DOMAIN_LEN: int = 30


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _fmt_kst(dt: datetime) -> str:
    """Format a tz-aware datetime as KST HH:MM."""
    return dt.astimezone(KST).strftime("%H:%M")


def _fmt_kst_date(dt: datetime) -> str:
    """Format a tz-aware datetime as KST YYYY-MM-DD."""
    return dt.astimezone(KST).strftime("%Y-%m-%d")


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending '…' when shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_add(blocks: list[dict], block: dict) -> bool:
    """
    Append *block* to *blocks* only if under the 50-block Slack limit.

    Reserves 2 slots for the trailing divider + footer context block.
    Returns True if added, False if the limit was hit.
    """
    if len(blocks) >= _MAX_BLOCKS - 2:
        return False
    blocks.append(block)
    return True


# ── Section builders ───────────────────────────────────────────────────────────

def _build_header_block(meeting_title: str) -> dict:
    """Header block: '🔄 후속 미팅 브리핑 — {title}'."""
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"🔄 후속 미팅 브리핑 — {meeting_title}",
            "emoji": True,
        },
    }


def _build_time_block(raw: "RawBriefingContent") -> dict:
    """
    Section block: countdown, KST start–end, duration, optional location/link.

    Example::
        ⏰ *시작까지:* 12.3분  |  🕐 14:00–15:00 (60분)
        📍 강남 회의실 A
        🔗 <url|캘린더에서 보기>
    """
    now_utc = datetime.now(timezone.utc)
    starts_in = round((raw.meeting_start - now_utc).total_seconds() / 60, 1)
    start_kst = _fmt_kst(raw.meeting_start)
    end_kst = _fmt_kst(raw.meeting_end)
    duration = raw.duration_minutes

    lines = [f"⏰ *시작까지:* {starts_in}분  |  🕐 {start_kst}–{end_kst} ({duration}분)"]
    if raw.meeting_location:
        lines.append(f"📍 {raw.meeting_location}")
    if raw.meeting_html_link:
        lines.append(f"🔗 <{raw.meeting_html_link}|캘린더에서 보기>")

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    }


def _build_agenda_blocks(description: str) -> list[dict]:
    """[divider, section] for the agenda / description. Empty when no description."""
    if not description or not description.strip():
        return []
    truncated = description.strip()[:_MAX_DESCRIPTION_LEN]
    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📋 안건/설명*\n{truncated}"},
        },
    ]


def _build_external_attendees_block(raw: "RawBriefingContent") -> dict:
    """
    Section block listing external attendees with full relationship history.

    For EXTERNAL_FOLLOWUP meetings the history counts are the key context,
    so each attendee line shows: past meeting count, last meeting date, and
    a sample of past meeting titles.

    Accuracy: when calendar history is unavailable shows '이력 확인 불가'
    instead of silently presenting zero counts.
    """
    ext = raw.external_attendees

    if not ext:
        return {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*👥 외부 참석자*\n_없음_"},
        }

    lines: list[str] = []
    for profile in ext:
        name = profile.display_name or profile.email
        domain_part = (
            f" _{_truncate(profile.company_domain, _MAX_DOMAIN_LEN)}_"
            if profile.company_domain
            else ""
        )

        if profile.past_meeting_count > 0:
            last_date = (
                f", 최근 {_fmt_kst_date(profile.last_met_date)}"
                if profile.last_met_date
                else ""
            )
            history_part = f" (과거 미팅 *{profile.past_meeting_count}회*{last_date})"

            # Show up to 2 most recent meeting titles for context
            if profile.past_meeting_titles:
                title_samples = [
                    _truncate(t, 40) for t in profile.past_meeting_titles[:2]
                ]
                history_part += f"\n  _최근 미팅: {' / '.join(title_samples)}_"
        elif not raw.calendar_history_available:
            history_part = " _(이력 확인 불가)_"
        else:
            history_part = " _(과거 미팅 기록 없음)_"

        lines.append(f"• *{name}*{domain_part}{history_part}")

    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*👥 외부 참석자 (관계 이력)*\n" + "\n".join(lines),
        },
    }


def _build_internal_attendees_context(raw: "RawBriefingContent") -> list[dict]:
    """Context element listing internal participants. Empty list if none."""
    int_profiles = raw.internal_attendees
    if not int_profiles:
        return []
    names = [p.display_name or p.email for p in int_profiles]
    return [
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"🏢 내부: {', '.join(names)}"}
            ],
        }
    ]


def _build_notion_blocks(raw: "RawBriefingContent") -> list[dict]:
    """
    Notion deal / portfolio section for the follow-up briefing.

    Shows the Notion records associated with the company, emphasising the
    relationship context (status, deal stage) rather than raw deal memo text.
    """
    blocks: list[dict] = [{"type": "divider"}]

    if not raw.notion_available:
        err_msg = next(
            (e.message for e in raw.errors if e.source == "notion"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*📑 Notion 포트폴리오/딜 기록*\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    records = raw.notion_records
    if not records:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📑 Notion 포트폴리오/딜 기록*\n_관련 Notion 항목 없음_",
            },
        })
        return blocks

    record_lines: list[str] = []
    for rec in records[:_MAX_NOTION_SHOWN]:
        title = rec.title or rec.company_name or "(이름 없음)"
        status_part = f"  [{_truncate(rec.status, 20)}]" if rec.status else ""
        title_display = _truncate(title, 40)
        if rec.url:
            title_md = f"<{rec.url}|{title_display}>"
        else:
            title_md = title_display
        record_lines.append(f"• {title_md}{status_part}")

    header_text = (
        f"*📑 Notion 포트폴리오/딜 기록* ({len(records)}개)\n"
        + "\n".join(record_lines)
    )
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": header_text},
    })

    if len(records) > _MAX_NOTION_SHOWN:
        remaining = len(records) - _MAX_NOTION_SHOWN
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 항목_"}
            ],
        })

    return blocks


def _build_gmail_blocks(raw: "RawBriefingContent") -> list[dict]:
    """
    Gmail exchange section: recent email threads with the company.

    Shows each thread with subject, date, and message count so the user
    can quickly assess the state of ongoing email conversations.
    """
    blocks: list[dict] = [{"type": "divider"}]

    if not raw.gmail_available:
        err_msg = next(
            (e.message for e in raw.errors if e.source == "gmail"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*📧 이메일 교신 이력*\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    threads = raw.gmail_threads
    if not threads:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📧 이메일 교신 이력*\n_관련 이메일 없음_",
            },
        })
        return blocks

    thread_lines: list[str] = []
    for thread in threads[:_MAX_GMAIL_SHOWN]:
        subject = thread.subject or "(제목 없음)"
        date_part = ""
        if thread.latest_date:
            date_part = (
                f" `{thread.latest_date.astimezone(KST).strftime('%m/%d')}`"
            )
        count_part = f" ({thread.message_count}개 메시지)"
        thread_lines.append(
            f"• {_truncate(subject, _MAX_SUBJECT_LEN)}{date_part}{count_part}"
        )

    header_text = (
        f"*📧 이메일 교신 이력* ({len(threads)}개)\n" + "\n".join(thread_lines)
    )
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": header_text},
    })

    if len(threads) > _MAX_GMAIL_SHOWN:
        remaining = len(threads) - _MAX_GMAIL_SHOWN
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 스레드_"}
            ],
        })

    return blocks


def _build_slack_history_blocks(raw: "RawBriefingContent") -> list[dict]:
    """
    Slack history section: messages mentioning the company in priority channels.

    This is the key differentiating section for EXTERNAL_FOLLOWUP briefings —
    it surfaces internal team discussions about the company that have happened
    in Slack, providing pre-meeting context the calendar / email data alone
    cannot offer.

    Structure::
        💬 *Slack 내부 논의 이력* (N개 메시지, M개 채널)
        • [#channel]  YYYY-MM-DD  메시지 내용 스니펫…
        …
        (+ N개 더)      ← overflow notice
        검색어: company_name   ← context element

    Accuracy rules
    --------------
    * ``slack_history_available=False`` → ``⚠️ 확인 불가`` annotation.
    * ``slack_history is None`` (not fetched) → section skipped entirely.
    * No messages found → ``_관련 Slack 메시지 없음_``.
    """
    # If Slack history was never fetched (not attempted), skip section entirely
    if raw.slack_history is None and raw.slack_history_available:
        return []

    blocks: list[dict] = [{"type": "divider"}]

    # ── Case 1: fetch failed ────────────────────────────────────────────────────
    if not raw.slack_history_available:
        err_msg = next(
            (e.message for e in raw.errors if e.source == "slack_history"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*💬 Slack 내부 논의 이력*\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    # ── Case 2: fetched but empty ───────────────────────────────────────────────
    result = raw.slack_history
    if not result or not result.messages:
        company_label = f" (`{result.company_name}`)" if result else ""
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*💬 Slack 내부 논의 이력*{company_label}\n_관련 Slack 메시지 없음_",
            },
        })
        return blocks

    # ── Case 3: messages found ─────────────────────────────────────────────────
    messages = result.messages
    channel_count = len({m.channel_name for m in messages})

    summary_header = (
        f"*💬 Slack 내부 논의 이력* "
        f"({len(messages)}개 메시지, {channel_count}개 채널)"
    )

    message_lines: list[str] = []
    for msg in messages[:_MAX_SLACK_SHOWN]:
        channel_label = f"#*{_truncate(msg.channel_name, 25)}*" if msg.channel_name else ""
        date_str = ""
        if msg.message_dt:
            date_str = f"`{msg.message_dt.astimezone(KST).strftime('%m/%d %H:%M')}`"
        # Build clean text snippet (strip mrkdwn formatting)
        text_snippet = _truncate(
            msg.text.replace("\n", " ").replace("*", "").replace("_", ""),
            _MAX_SLACK_SNIPPET_LEN,
        )
        parts = [p for p in [channel_label, date_str] if p]
        prefix = "  ".join(parts)
        if prefix:
            message_lines.append(f"• {prefix}\n  {text_snippet}")
        else:
            message_lines.append(f"• {text_snippet}")

    full_text = summary_header + "\n" + "\n".join(message_lines)
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": full_text},
    })

    # Overflow notice
    if len(messages) > _MAX_SLACK_SHOWN:
        remaining = len(messages) - _MAX_SLACK_SHOWN
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 메시지_"}
            ],
        })

    # Search term footer for transparency
    if result.company_name:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_검색어: `{result.company_name}`_",
                }
            ],
        })

    # Channels-searched context (only when some were skipped)
    if result.channels_skipped:
        skipped_names = ", ".join(
            f"#{_truncate(c.channel_name, 20)}"
            for c in result.channels_skipped[:5]
        )
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_접근 불가 채널: {skipped_names}_",
                }
            ],
        })

    return blocks


def _build_footer_block(raw: "RawBriefingContent") -> dict:
    """Footer context block with accuracy disclaimer and error summary."""
    error_note = ""
    if raw.has_errors:
        sources = sorted({e.source for e in raw.errors})
        error_note = f"  |  ⚠️ 일부 정보 확인 불가: {', '.join(sources)}"

    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    "🤖 Work Assistant (후속 미팅)  |  "
                    "정보 미확인 항목은 *확인 불가* 로 표시됩니다"
                    + error_note
                ),
            }
        ],
    }


def _build_fallback_text(raw: "RawBriefingContent") -> str:
    """
    Plain-text fallback for push notifications / non-Block-Kit clients.

    Captures the key follow-up context in a compact multi-line string:
    meeting time, external attendees with history, data source summaries.
    """
    now_utc = datetime.now(timezone.utc)
    starts_in = round((raw.meeting_start - now_utc).total_seconds() / 60, 1)
    start_kst = _fmt_kst(raw.meeting_start)
    end_kst = _fmt_kst(raw.meeting_end)

    # Attendee summary with history
    att_parts: list[str] = []
    for p in raw.external_attendees:
        name = p.display_name or p.email
        hist = f" ({p.past_meeting_count}회)" if p.past_meeting_count > 0 else ""
        att_parts.append(f"{name}{hist}")
    att_str = ", ".join(att_parts) if att_parts else "없음"

    lines = [
        f"🔄 후속 미팅 브리핑 — {raw.meeting_title}",
        f"⏰ 시작까지: {starts_in}분  ({start_kst}–{end_kst})",
        f"👥 외부 참석자: {att_str}",
    ]

    # Source summaries
    if raw.gmail_available:
        lines.append(f"📧 이메일 교신: {len(raw.gmail_threads)}개 스레드")
    else:
        lines.append("📧 이메일 교신: 확인 불가")

    if raw.notion_available:
        lines.append(f"📑 Notion 기록: {len(raw.notion_records)}개")
    else:
        lines.append("📑 Notion 기록: 확인 불가")

    if raw.slack_history_available and raw.slack_history is not None:
        msg_count = len(raw.slack_history.messages)
        ch_count = len({m.channel_name for m in raw.slack_history.messages})
        lines.append(f"💬 Slack 이력: {msg_count}개 메시지 ({ch_count}개 채널)")
    elif not raw.slack_history_available:
        lines.append("💬 Slack 이력: 확인 불가")

    if raw.has_errors:
        sources = sorted({e.source for e in raw.errors})
        lines.append(f"⚠️ 확인 불가 항목: {', '.join(sources)}")

    return "\n".join(lines)


# ── Public API ──────────────────────────────────────────────────────────────────

def format_external_followup_briefing(
    raw_content: "RawBriefingContent",
) -> tuple[str, list[dict]]:
    """
    Format a ``RawBriefingContent`` into a follow-up meeting briefing DM.

    This is the dedicated formatter for **EXTERNAL_FOLLOWUP** meetings
    (meetings with companies the investment team has previously encountered).
    It emphasises relationship history, ongoing conversations, and internal
    Slack discussion context.

    Parameters
    ----------
    raw_content:
        A fully populated ``RawBriefingContent`` from
        ``MeetingContextAggregator.aggregate()``.  The ``slack_history``
        field (added for AC 7) is optional — when ``None`` the Slack section
        is omitted rather than shown as an error.

    Returns
    -------
    tuple[str, list[dict]]
        ``(fallback_text, blocks)`` ready for
        ``WorkAssistantBot.send_message(fallback_text, blocks=blocks)``.

    Notes
    -----
    * Failed sources are annotated ``⚠️ 확인 불가``; no guessing.
    * Block count is always ≤ 50 (Slack hard limit).
    * The footer block is always preserved (trim back if limit reached).
    * All times are shown in KST (Asia/Seoul).
    """
    blocks: list[dict] = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    blocks.append(_build_header_block(raw_content.meeting_title))

    # ── 2. Time / location ────────────────────────────────────────────────────
    blocks.append(_build_time_block(raw_content))

    # ── 3. Agenda / description (optional) ───────────────────────────────────
    for block in _build_agenda_blocks(raw_content.meeting_description):
        _safe_add(blocks, block)

    # ── 4. External attendees with relationship history ───────────────────────
    _safe_add(blocks, {"type": "divider"})
    _safe_add(blocks, _build_external_attendees_block(raw_content))

    # ── 5. Internal attendees (context element) ───────────────────────────────
    for block in _build_internal_attendees_context(raw_content):
        _safe_add(blocks, block)

    # ── 6. Notion section (deal / portfolio records) ──────────────────────────
    for block in _build_notion_blocks(raw_content):
        _safe_add(blocks, block)

    # ── 7. Gmail exchange section ─────────────────────────────────────────────
    for block in _build_gmail_blocks(raw_content):
        _safe_add(blocks, block)

    # ── 8. Slack history section (new for AC 7 Sub-AC 4) ─────────────────────
    for block in _build_slack_history_blocks(raw_content):
        _safe_add(blocks, block)

    # ── 9. Footer (always present) ────────────────────────────────────────────
    # Trim back excess blocks to guarantee footer fits within 50-block limit
    while len(blocks) > _MAX_BLOCKS - 2:
        blocks.pop()
    blocks.append({"type": "divider"})
    blocks.append(_build_footer_block(raw_content))

    # ── Fallback plain text ────────────────────────────────────────────────────
    fallback = _build_fallback_text(raw_content)

    logger.debug(
        "format_external_followup_briefing: meeting='%s'  blocks=%d  "
        "slack_msgs=%s  errors=%s",
        raw_content.meeting_title,
        len(blocks),
        (
            len(raw_content.slack_history.messages)
            if raw_content.slack_history is not None
            else "n/a"
        ),
        [e.source for e in raw_content.errors],
    )

    return fallback, blocks
