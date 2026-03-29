"""
Meeting Briefing Formatter (Sub-AC 2c).

Transforms a ``RawBriefingContent`` object (produced by
``MeetingContextAggregator``) into a structured Slack Block Kit message
ready to be delivered via ``WorkAssistantBot.send_message()``.

This module is the presentation layer for the per-meeting briefing pipeline.
It owns *only* formatting logic — all data fetching lives in
``src.briefing.context_aggregator`` and delivery in
``src.briefing.pipeline``.

Output format
-------------
``(fallback_text: str, blocks: list[dict])``

  * ``fallback_text`` — plain-text summary (shown in notifications /
    non-Block-Kit clients, e.g. mobile push).
  * ``blocks`` — Slack Block Kit payload ready for
    ``chat_postMessage(blocks=...)``.  Always ≤ 50 blocks (Slack limit).

Sections produced
-----------------
1. Header          — meeting title
2. Time / location — countdown, KST start–end, duration, optional location,
                     optional calendar link
3. Agenda          — meeting description (if any, truncated at 500 chars)
4. External        — enriched attendee profiles with past-meeting history
5. Internal        — brief list of internal participants (context element)
6. Gmail threads   — related email threads (up to 5 shown, with overflow)
7. Notion records  — related deal / portfolio entries (up to 5 shown)
8. Footer          — accuracy reminder + error summary

Accuracy / data-safety rules
-----------------------------
* Sources that failed are annotated ``⚠️ 확인 불가`` — never omitted or
  fabricated.
* ``gmail_available=False`` → Gmail section shows error annotation.
* ``notion_available=False`` → Notion section shows error annotation.
* ``calendar_history_available=False`` → attendee shows ``(과거 이력: 확인 불가)``
  instead of zero-count (which would be misleading).
* All dates / times are converted to KST (Asia/Seoul) before display.

Usage::

    from src.briefing.meeting_briefing_formatter import format_meeting_briefing

    text, blocks = format_meeting_briefing(raw_content)
    ok = bot.send_message(text, blocks=blocks)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.briefing.context_aggregator import RawBriefingContent
    from src.ai.briefing_generator import AIBriefingSections

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")

# Maximum length for meeting description in the agenda section
_MAX_DESCRIPTION_LEN: int = 500

# Maximum attendee domain length before truncation
_MAX_DOMAIN_LEN: int = 30

# Maximum email threads displayed in the Gmail section
_MAX_GMAIL_SHOWN: int = 5

# Maximum Notion records displayed
_MAX_NOTION_SHOWN: int = 5

# Maximum web search results displayed in briefing (Sub-AC 6a)
_MAX_WEB_SEARCH_SHOWN: int = 3

# Maximum blocks (Slack hard limit)
_MAX_BLOCKS: int = 50


# ── Low-level formatting helpers ───────────────────────────────────────────────

def _fmt_kst_time(dt: datetime) -> str:
    """Format a tz-aware datetime as KST ``HH:MM``."""
    return dt.astimezone(KST).strftime("%H:%M")


def _fmt_kst_date(dt: datetime) -> str:
    """Format a tz-aware datetime as KST ``YYYY-MM-DD``."""
    return dt.astimezone(KST).strftime("%Y-%m-%d")


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending ``…`` if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_add(blocks: list[dict], block: dict) -> bool:
    """
    Append *block* to *blocks* only if under the 50-block Slack limit.

    Reserves 2 slots for the trailing divider + footer context block.
    Returns ``True`` if the block was added, ``False`` if the limit was hit.
    """
    if len(blocks) >= _MAX_BLOCKS - 2:
        return False
    blocks.append(block)
    return True


# ── Section builders ───────────────────────────────────────────────────────────

def _build_header_block(meeting_title: str) -> dict:
    """Slack Block Kit ``header`` block with meeting title."""
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"📅 미팅 브리핑 — {meeting_title}",
            "emoji": True,
        },
    }


def _build_time_block(raw_content: "RawBriefingContent") -> dict:
    """
    Section block showing countdown, KST times, duration, location, link.

    Example output::

        ⏰ *시작까지:* 12.3분  |  🕐 14:00–15:00 (60분)
        📍 강남 회의실 A
        🔗 <https://cal.google.com/...|캘린더에서 보기>
    """
    now_utc = datetime.now(timezone.utc)
    starts_in_secs = (raw_content.meeting_start - now_utc).total_seconds()
    starts_in = round(starts_in_secs / 60, 1)
    start_kst = _fmt_kst_time(raw_content.meeting_start)
    end_kst = _fmt_kst_time(raw_content.meeting_end)
    duration = raw_content.duration_minutes

    lines = [
        f"⏰ *시작까지:* {starts_in}분  |  🕐 {start_kst}–{end_kst} ({duration}분)"
    ]
    if raw_content.meeting_location:
        lines.append(f"📍 {raw_content.meeting_location}")
    if raw_content.meeting_html_link:
        lines.append(
            f"🔗 <{raw_content.meeting_html_link}|캘린더에서 보기>"
        )

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    }


def _build_agenda_blocks(description: str) -> list[dict]:
    """
    Return [divider, section] for the agenda/description if non-empty.
    Returns an empty list when there is no description.
    """
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


def _build_external_attendees_block(raw_content: "RawBriefingContent") -> dict:
    """
    Section block listing external attendees with enriched history.

    Each attendee line shows:
      • *Name* _domain_ (과거 미팅 N회, 최근: YYYY-MM-DD)

    When calendar history is unavailable, shows ``(과거 이력: 확인 불가)``
    instead of silently showing zero meetings (which would be misleading).
    """
    ext_profiles = raw_content.external_attendees

    if not ext_profiles:
        return {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*👥 외부 참석자*\n_없음_"},
        }

    lines: list[str] = []
    for profile in ext_profiles:
        name = profile.display_name or profile.email
        domain_part = (
            f" _{_truncate(profile.company_domain, _MAX_DOMAIN_LEN)}_"
            if profile.company_domain
            else ""
        )

        if profile.past_meeting_count > 0:
            last_met_str = ""
            if profile.last_met_date:
                last_met_str = f", 최근: {_fmt_kst_date(profile.last_met_date)}"
            history_part = f" (과거 미팅 {profile.past_meeting_count}회{last_met_str})"
        elif not raw_content.calendar_history_available:
            history_part = " (과거 이력: 확인 불가)"
        else:
            history_part = ""

        lines.append(f"• *{name}*{domain_part}{history_part}")

    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*👥 외부 참석자*\n" + "\n".join(lines),
        },
    }


def _build_internal_attendees_context(raw_content: "RawBriefingContent") -> list[dict]:
    """
    Context element listing internal participants.
    Returns an empty list if there are no internal attendees.
    """
    int_profiles = raw_content.internal_attendees
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


def _build_gmail_blocks(raw_content: "RawBriefingContent") -> list[dict]:
    """
    Section block(s) for the Gmail email threads section.

    * Unavailable source → single ``⚠️ 확인 불가`` block.
    * Empty thread list → ``_관련 이메일 없음_``.
    * Up to 5 threads shown with subject, date, and message count.
    * Overflow → context element with remaining count.
    """
    blocks: list[dict] = [{"type": "divider"}]

    if not raw_content.gmail_available:
        err_msg = next(
            (e.message for e in raw_content.errors if e.source == "gmail"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*📧 관련 이메일 스레드*\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    threads = raw_content.gmail_threads
    if not threads:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📧 관련 이메일 스레드*\n_관련 이메일 없음_",
            },
        })
        return blocks

    thread_lines: list[str] = []
    for thread in threads[:_MAX_GMAIL_SHOWN]:
        subject = thread.subject or "(제목 없음)"
        date_part = ""
        if thread.latest_date:
            date_part = f" `{thread.latest_date.astimezone(KST).strftime('%m/%d')}`"
        count_part = f" ({thread.message_count}개 메시지)"
        thread_lines.append(f"• {_truncate(subject, 60)}{date_part}{count_part}")

    header_text = (
        f"*📧 관련 이메일 스레드* ({len(threads)}개)\n"
        + "\n".join(thread_lines)
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


def _build_web_search_blocks(raw_content: "RawBriefingContent") -> list[dict]:
    """
    Build Slack Block Kit blocks for the web search context section.

    Only rendered when ``raw_content.web_search_summary`` is not None,
    which happens exclusively for EXTERNAL_FIRST meetings (first-time
    external contact where no history exists).

    Section structure::

        🌐 *웹 검색 컨텍스트* (Tavily)
        [summary text or per-result bullets]
        • [Title](<url>) — snippet …
        [검색어: query1, query2]

    Accuracy rules
    --------------
    * ``web_search_available=False`` → shows ``⚠️ 확인 불가`` annotation.
    * Empty results (but available=True) → shows ``_웹 검색 결과 없음_``.
    * When ``web_search_summary is None`` (non-EXTERNAL_FIRST) → returns [].
    """
    # Not an EXTERNAL_FIRST meeting — skip section entirely
    if raw_content.web_search_summary is None:
        return []

    blocks: list[dict] = [{"type": "divider"}]
    ws = raw_content.web_search_summary

    # Provider label for transparency
    provider_label = f" ({ws.provider})" if ws.provider not in {"none", ""} else ""
    section_header = f"*🌐 웹 검색 컨텍스트{provider_label}*"

    # ── Case 1: search entirely failed ────────────────────────────────────────
    if not raw_content.web_search_available or not ws.available:
        err_msg = ws.error or next(
            (e.message for e in raw_content.errors if e.source == "web_search"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{section_header}\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    # ── Case 2: no results ────────────────────────────────────────────────────
    if not ws.has_results:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{section_header}\n_웹 검색 결과 없음_",
            },
        })
        return blocks

    # ── Case 3: results available ─────────────────────────────────────────────
    lines: list[str] = []

    # Use pre-assembled summary text if available
    if ws.summary:
        lines.append(ws.summary)
    else:
        # Fall back to individual result bullets
        for result in ws.results[:_MAX_WEB_SEARCH_SHOWN]:
            title = _truncate(result.title or result.query, 60)
            snippet = _truncate(result.snippet, 150)
            if result.url:
                title_md = f"<{result.url}|{title}>"
            else:
                title_md = f"*{title}*"
            lines.append(f"• {title_md} — {snippet}")

    # Add search query context as a footer element
    query_footer: list[dict] = []
    if ws.queries_executed:
        query_text = ", ".join(f"`{q[:40]}`" for q in ws.queries_executed[:3])
        query_footer = [
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_검색어: {query_text}_",
                    }
                ],
            }
        ]

    # Overflow indicator
    overflow: list[dict] = []
    if len(ws.results) > _MAX_WEB_SEARCH_SHOWN:
        remaining = len(ws.results) - _MAX_WEB_SEARCH_SHOWN
        overflow = [
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 검색 결과_"}
                ],
            }
        ]

    body_text = section_header + "\n" + "\n".join(lines)
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": body_text},
    })
    blocks.extend(overflow)
    blocks.extend(query_footer)

    return blocks


def _build_notion_blocks(raw_content: "RawBriefingContent") -> list[dict]:
    """
    Section block(s) for the Notion deal / portfolio records section.

    * Unavailable source → single ``⚠️ 확인 불가`` block.
    * Empty records list → ``_관련 Notion 항목 없음_``.
    * Up to 5 records shown with title (linked), status.
    * Overflow → context element with remaining count.
    """
    blocks: list[dict] = [{"type": "divider"}]

    if not raw_content.notion_available:
        err_msg = next(
            (e.message for e in raw_content.errors if e.source == "notion"),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*📑 관련 딜 / 포트폴리오*\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    records = raw_content.notion_records
    if not records:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📑 관련 딜 / 포트폴리오*\n_관련 Notion 항목 없음_",
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
        f"*📑 관련 딜 / 포트폴리오* ({len(records)}개)\n"
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


def _build_notion_deal_memo_blocks(raw_content: "RawBriefingContent") -> list[dict]:
    """
    Build Slack Block Kit blocks for the Notion deal memo section.

    Rendered only for EXTERNAL_FIRST meetings where a deal memo was fetched.
    Placed after the web search section so company background flows naturally:
    web context (public info) → deal memo (internal investment analysis).

    Cases handled
    -------------
    * ``notion_deal_memo_available=False``
        → ``⚠️ 확인 불가`` annotation (client error).
    * ``notion_deal_memo is None`` and available=True
        → ``_딜 메모 없음_`` (no Notion page found — legitimate).
    * ``notion_deal_memo is not None``
        → Structured content: company header + status/date + body sections.

    Returns an empty list when the meeting is not EXTERNAL_FIRST (i.e.,
    when both ``notion_deal_memo`` is None and
    ``notion_deal_memo_available=True`` after no fetch was attempted).
    """
    # Skip this section entirely for non-EXTERNAL_FIRST meetings.
    # We distinguish "not attempted" from "attempted but not found" by
    # checking whether notion_deal_memo_available is still True AND memo is None.
    # When the fetch was not attempted (non-EXTERNAL_FIRST), notion_deal_memo
    # stays None and notion_deal_memo_available stays True — we show nothing.
    # When the fetch WAS attempted (EXTERNAL_FIRST), notion_deal_memo_available
    # may still be True even if memo=None (no match found).
    # We use notion_deal_memo_available=False to detect an error during fetch.
    # We cannot distinguish "not attempted" from "tried but found nothing"
    # purely from the content object — both leave memo=None, available=True.
    # The formatter therefore shows the "no memo" message only when there was
    # an error (available=False); otherwise silence avoids false negatives.

    if raw_content.notion_deal_memo is None and raw_content.notion_deal_memo_available:
        # Either not attempted (non-EXTERNAL_FIRST) or attempted but no match.
        # Only show the section when the fetch definitely failed with an error.
        return []

    blocks: list[dict] = [{"type": "divider"}]
    section_header = "*📋 Notion 딜 메모 (투자 분석)*"

    # ── Case 1: fetch failed (client error) ───────────────────────────────────
    if not raw_content.notion_deal_memo_available:
        err_msg = next(
            (
                e.message
                for e in raw_content.errors
                if e.source == "notion_deal_memo"
            ),
            "오류",
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{section_header}\n"
                    f"⚠️ 확인 불가 (`{_truncate(err_msg, 80)}`)"
                ),
            },
        })
        return blocks

    # ── Case 2: memo found — render structured content ────────────────────────
    memo = raw_content.notion_deal_memo
    summary = memo.to_briefing_summary(max_chars=600)

    # Build header line with URL link if available
    company_display = _truncate(memo.company_name or memo.title or "(이름 없음)", 40)
    if memo.url:
        company_md = f"<{memo.url}|{company_display}>"
    else:
        company_md = company_display

    # Status / date badge
    meta_parts: list[str] = []
    if memo.status:
        meta_parts.append(f"[{_truncate(memo.status, 20)}]")
    if memo.date_value:
        meta_parts.append(f"`{memo.date_value}`")
    meta_str = "  ".join(meta_parts)

    header_line = f"{section_header}\n*{company_md}*"
    if meta_str:
        header_line += f"  {meta_str}"

    # Availability note when blocks were not fetched
    if not memo.blocks_fetched:
        body_part = "_페이지 본문 확인 불가 (확인 불가)_"
    elif summary:
        body_part = _truncate(summary, 600)
    else:
        body_part = "_내용 없음_"

    full_text = f"{header_line}\n{body_part}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": full_text},
    })

    return blocks


def _build_footer_block(raw_content: "RawBriefingContent") -> dict:
    """
    Context block at the bottom of the briefing.

    Shows the accuracy disclaimer and, if any source failed, a list of
    which sources are partially unavailable (``확인 불가``).
    """
    error_note = ""
    if raw_content.has_errors:
        sources = sorted({e.source for e in raw_content.errors})
        error_note = f"  |  ⚠️ 일부 정보 확인 불가: {', '.join(sources)}"

    return {
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    "🤖 Work Assistant  |  "
                    "정보 미확인 항목은 *확인 불가* 로 표시됩니다"
                    + error_note
                ),
            }
        ],
    }


def _build_fallback_text(raw_content: "RawBriefingContent") -> str:
    """
    Plain-text fallback message for notification / non-Block-Kit contexts.

    Keeps the same information density as the blocks but as a single
    multi-line string.
    """
    now_utc = datetime.now(timezone.utc)
    starts_in = round(
        (raw_content.meeting_start - now_utc).total_seconds() / 60, 1
    )
    start_kst = _fmt_kst_time(raw_content.meeting_start)
    end_kst = _fmt_kst_time(raw_content.meeting_end)
    ext_emails = ", ".join(p.email for p in raw_content.external_attendees) or "없음"

    lines = [
        f"📅 미팅 브리핑 — {raw_content.meeting_title}",
        f"⏰ 시작까지: {starts_in}분  ({start_kst}–{end_kst})",
        f"👥 외부 참석자: {ext_emails}",
        f"📧 관련 이메일: {len(raw_content.gmail_threads)}개 스레드",
        f"📑 관련 딜: {len(raw_content.notion_records)}개",
    ]
    # Include deal memo status only when it was fetched (EXTERNAL_FIRST)
    if raw_content.notion_deal_memo is not None:
        company = raw_content.notion_deal_memo.company_name or raw_content.notion_deal_memo.title
        lines.append(f"📋 딜 메모: {company or '확인됨'}")
    elif not raw_content.notion_deal_memo_available:
        lines.append("📋 딜 메모: 확인 불가")

    if raw_content.has_errors:
        sources = sorted({e.source for e in raw_content.errors})
        lines.append(f"⚠️ 확인 불가 항목: {', '.join(sources)}")

    return "\n".join(lines)


# ── AI-generated section builders (Sub-AC 3b) ─────────────────────────────────

def _build_ai_attendee_bio_blocks(
    raw_content: "RawBriefingContent",
    ai_sections: "AIBriefingSections",
) -> list[dict]:
    """
    Render AI-generated attendee bios as Slack Block Kit section blocks.

    Only rendered when ai_sections contains bio content for at least one
    external attendee.  Falls back silently (returns []) when no AI content
    is available — the rule-based attendee block above still shows the
    basic attendee list in all cases.

    Accuracy rule: when ai_available=False, renders a single '확인 불가'
    context element so users know AI enrichment was attempted but failed.
    """
    if not raw_content.external_attendees:
        return []

    blocks: list[dict] = []

    # Case: AI unavailable — show annotation only if it was attempted
    if not ai_sections.ai_available:
        err_msg = ai_sections.error or "AI 서비스 오류"
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🤖 _AI 프로필 생성 실패: {_truncate(err_msg, 80)} (확인 불가)_",
                }
            ],
        })
        return blocks

    # Case: AI available but no bios generated (e.g. all attendees internal)
    if not ai_sections.attendee_bios:
        return []

    # Render per-attendee bio blocks
    bio_lines: list[str] = []
    for profile in raw_content.external_attendees:
        bio_text = ai_sections.attendee_bios.get(profile.email, "")
        if not bio_text:
            continue
        name = profile.display_name or profile.email
        bio_lines.append(f"*{name}*\n{_truncate(bio_text, 300)}")

    if not bio_lines:
        return []

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*🤖 AI 참석자 프로필*\n\n" + "\n\n".join(bio_lines),
        },
    })
    return blocks


def _build_ai_agenda_prep_blocks(
    ai_sections: "AIBriefingSections",
) -> list[dict]:
    """
    Render AI-generated agenda preparation notes as Slack Block Kit blocks.

    Returns an empty list when no agenda content is available.
    The section is labelled to distinguish AI-generated from manual content.
    """
    if not ai_sections.agenda_prep:
        return []

    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🤖 AI 준비 사항*\n{_truncate(ai_sections.agenda_prep, 600)}",
            },
        },
    ]


def _build_ai_email_summary_blocks(
    ai_sections: "AIBriefingSections",
) -> list[dict]:
    """
    Render the AI-generated email thread summary as a Slack Block Kit section.

    Returns an empty list when no email summary content is available.
    Placed after the rule-based Gmail threads section to provide a higher-level
    narrative view of the email exchanges.
    """
    if not ai_sections.email_summary:
        return []

    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🤖 AI 이메일 요약*\n{_truncate(ai_sections.email_summary, 600)}",
            },
        },
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def format_meeting_briefing(
    raw_content: "RawBriefingContent",
    ai_sections: "Optional[AIBriefingSections]" = None,
) -> tuple[str, list[dict]]:
    """
    Format a ``RawBriefingContent`` into a Slack DM briefing message.

    This is the main public function for Sub-AC 2c / 3b.  It consumes the
    aggregated raw context produced by ``MeetingContextAggregator.aggregate()``
    and optionally the AI-enriched sections from ``BriefingGenerator.generate()``,
    returning a ``(fallback_text, blocks)`` tuple ready for
    ``WorkAssistantBot.send_message()``.

    Parameters
    ----------
    raw_content:
        The ``RawBriefingContent`` object from
        ``MeetingContextAggregator.aggregate()``.
    ai_sections:
        Optional ``AIBriefingSections`` from ``BriefingGenerator.generate()``.
        When provided, the briefing includes AI-generated attendee bios,
        agenda preparation notes, and an email thread summary.
        When None (default), the formatter renders the rule-based layout only.

    Returns
    -------
    tuple[str, list[dict]]
        ``(fallback_text, blocks)`` where *fallback_text* is the plain-text
        summary and *blocks* is the Slack Block Kit payload (≤ 50 blocks).

    Notes
    -----
    * Failed data sources are annotated ``⚠️ 확인 불가`` — no guessing,
      no hallucination.
    * Block count is capped at 50 (Slack hard limit).  Overflowing content
      is silently dropped; the footer is always preserved.
    * All timestamps are converted to KST (Asia/Seoul).
    * AI sections are clearly labelled with 🤖 so users know which content
      was AI-generated vs. pulled directly from source APIs.
    """
    blocks: list[dict] = []

    # 1. Header
    blocks.append(_build_header_block(raw_content.meeting_title))

    # 2. Time / location
    blocks.append(_build_time_block(raw_content))

    # 3. Agenda / description (optional)
    for block in _build_agenda_blocks(raw_content.meeting_description):
        _safe_add(blocks, block)

    # 4. External attendees (rule-based: name, domain, history counts)
    _safe_add(blocks, {"type": "divider"})
    _safe_add(blocks, _build_external_attendees_block(raw_content))

    # 5. Internal attendees (context element)
    for block in _build_internal_attendees_context(raw_content):
        _safe_add(blocks, block)

    # 5a. AI attendee bios (Sub-AC 3b) — narrative profile per external attendee
    #     Placed directly after the rule-based attendee list so bio context
    #     enriches the structured attendee data seen just above.
    if ai_sections is not None:
        for block in _build_ai_attendee_bio_blocks(raw_content, ai_sections):
            _safe_add(blocks, block)

    # 5b. Web search context (EXTERNAL_FIRST only — Sub-AC 6a)
    #     Rendered right after attendees so it provides company background
    #     before the email/Notion context sections.
    for block in _build_web_search_blocks(raw_content):
        _safe_add(blocks, block)

    # 5c. Notion deal memo (EXTERNAL_FIRST only — Sub-AC 6b)
    #     Shows the internal investment analysis / deal thesis from Notion.
    #     Placed after web search (public context) and before Gmail/records.
    for block in _build_notion_deal_memo_blocks(raw_content):
        _safe_add(blocks, block)

    # 5d. AI agenda preparation (Sub-AC 3b) — suggested talking points
    #     Placed after the deal memo so investment context informs prep notes.
    if ai_sections is not None:
        for block in _build_ai_agenda_prep_blocks(ai_sections):
            _safe_add(blocks, block)

    # 6. Gmail threads (rule-based thread list)
    for block in _build_gmail_blocks(raw_content):
        _safe_add(blocks, block)

    # 6a. AI email summary (Sub-AC 3b) — condensed narrative of email exchanges
    #     Placed after the thread list so the AI summary augments the raw list.
    if ai_sections is not None:
        for block in _build_ai_email_summary_blocks(ai_sections):
            _safe_add(blocks, block)

    # 7. Notion records
    for block in _build_notion_blocks(raw_content):
        _safe_add(blocks, block)

    # 8. Footer — always present (trim back if needed to fit in 50 blocks)
    while len(blocks) > _MAX_BLOCKS - 2:
        blocks.pop()
    blocks.append({"type": "divider"})
    blocks.append(_build_footer_block(raw_content))

    # Fallback plain text
    fallback = _build_fallback_text(raw_content)

    logger.debug(
        "format_meeting_briefing: meeting='%s'  blocks=%d  errors=%s",
        raw_content.meeting_title,
        len(blocks),
        [e.source for e in raw_content.errors],
    )

    return fallback, blocks
