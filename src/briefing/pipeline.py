"""
Meeting Briefing Pipeline.

This module is the integration point for the scheduler.  It provides:

  - trigger_meeting_briefing()       – called by the scheduler when an external
                                       meeting is starting within 15 minutes.
  - run_briefing_for_next_meeting()  – called by the /brief slash command.
  - run_daily_morning_briefing()     – called by the 09:30 KST cron job;
                                       fetches ALL of today's calendar events,
                                       formats a full daily briefing, and sends
                                       it to the user via Slack DM.
  - run_aggregated_brief()          – called by the /brief slash command;
                                       fetches calendar + Gmail + Notion data,
                                       formats a comprehensive briefing, and
                                       delivers it via Slack DM.  This is the
                                       Sub-AC 3.2 implementation.
  - validate_briefing_content()     – AC 8: guard that blocks sending when the
                                       briefing is missing attendees or meeting
                                       purpose; incomplete-but-accurate info is
                                       annotated '확인 불가' but structurally
                                       missing core fields abort the send.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

from src.config import API_RETRY_ATTEMPTS, API_RETRY_DELAY_SECONDS

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting

logger = logging.getLogger(__name__)

# ── AC 8: Completeness validation ──────────────────────────────────────────────

# Title keywords that indicate a generic / unnamed meeting.
# A briefing whose title is *only* one of these tokens and has no description
# is considered to have a missing meeting purpose.
_GENERIC_TITLES: frozenset[str] = frozenset({
    # English
    "meeting", "call", "zoom", "sync", "1:1", "1on1", "chat", "catchup",
    "catch-up", "check-in", "checkin", "standup", "stand-up", "discussion",
    # Korean
    "미팅", "회의", "통화", "화상회의", "화상통화", "싱크", "논의",
    "점검", "업무", "일정", "약속", "대화",
})


def validate_briefing_content(
    raw_content: "RawBriefingContent",
) -> tuple[bool, list[str]]:
    """
    AC 8 guard: ensure a briefing has the minimum required fields before sending.

    A briefing is considered **incomplete** — and must NOT be sent — when:

    1. **Missing attendees** — no external attendees are present in the meeting.
       External meetings always require at least one identified external participant
       for the briefing to be meaningful.

    2. **Missing meeting purpose** — the meeting has neither a non-empty description
       nor a specific (non-generic) title.  A title of just "Meeting" / "미팅" /
       "Zoom" combined with a blank description provides no context for preparing.

    Data-source failures (Gmail, Notion, Calendar history unavailable) do NOT
    trigger this guard; those are annotated '확인 불가' in the message and the
    briefing is still sent.  This function exclusively checks the meeting's own
    structural completeness.

    Parameters
    ----------
    raw_content:
        The ``RawBriefingContent`` from ``MeetingContextAggregator.aggregate()``.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_complete, missing_items)``

        * ``is_complete`` – ``True`` if the briefing may be sent.
        * ``missing_items`` – human-readable Korean descriptions of each missing
          field (empty list when ``is_complete`` is ``True``).

    Examples
    --------
    >>> is_ok, reasons = validate_briefing_content(raw_content)
    >>> if not is_ok:
    ...     logger.warning("Briefing suppressed — %s", reasons)
    ...     return False
    """
    missing: list[str] = []

    # ── Check 1: external attendees ──────────────────────────────────────────
    external = raw_content.external_attendees  # derived property on RawBriefingContent
    if not external:
        missing.append("외부 참석자 정보 없음 (missing external attendees)")

    # ── Check 2: meeting purpose ─────────────────────────────────────────────
    has_description = bool(
        raw_content.meeting_description and raw_content.meeting_description.strip()
    )
    title_normalised = raw_content.meeting_title.strip().lower() if raw_content.meeting_title else ""
    # Strip common Korean/English punctuation for comparison
    title_clean = title_normalised.strip(".,!?~·:：()")

    # A title is "specific" (counts as purpose) when it is non-empty AND not
    # exclusively composed of a generic keyword.
    title_is_specific = bool(title_clean) and (title_clean not in _GENERIC_TITLES)

    if not has_description and not title_is_specific:
        missing.append("미팅 목적/설명 없음 (missing meeting purpose or specific title)")

    return len(missing) == 0, missing


def trigger_meeting_briefing(meeting: "Meeting", bot=None) -> bool:
    """
    Entry point called by the scheduler when an external meeting is detected.

    Aggregates raw briefing context (attendee history, Gmail threads, Notion
    records) via MeetingContextAggregator and sends an enriched notification.

    Args:
        meeting: The external Meeting object about to start (within 15 min).
        bot:     WorkAssistantBot instance for sending DMs.  If None the
                 function logs the trigger and returns True (useful for tests).

    Returns:
        True  – briefing sent successfully (or logged when bot is None).
        False – briefing could not be sent.
    """
    starts_in = round(meeting.starts_in_minutes, 1)
    external_emails = [a.email for a in meeting.external_attendees]

    logger.info(
        "[BRIEFING TRIGGERED] '%s' starts in %.1f min | external: %s",
        meeting.summary,
        starts_in,
        ", ".join(external_emails) or "없음",
    )

    if bot is None:
        # In test / early-dev mode: just log
        return True

    # ── Step 1: Classify meeting (EXTERNAL_FIRST vs EXTERNAL_FOLLOWUP) ────────
    # Web search (Sub-AC 6a) is only performed for EXTERNAL_FIRST meetings to
    # avoid unnecessary API costs on repeat visits.
    is_external_first = _classify_is_external_first(meeting)
    logger.info(
        "[BRIEFING] '%s' classification: %s",
        meeting.summary,
        "EXTERNAL_FIRST" if is_external_first else "EXTERNAL_FOLLOWUP",
    )

    # ── Step 2: Aggregate raw context ─────────────────────────────────────────
    raw_content = _aggregate_meeting_context(meeting, is_external_first=is_external_first)

    # ── Step 2b: AC 8 — completeness guard ────────────────────────────────────
    is_complete, missing_items = validate_briefing_content(raw_content)
    if not is_complete:
        logger.warning(
            "[BRIEFING SUPPRESSED] '%s' — incomplete briefing, not sent. "
            "Missing: %s",
            meeting.summary,
            "; ".join(missing_items),
        )
        # Notify the user so they know a briefing was suppressed
        try:
            bot.send_message(
                f"⚠️ 미팅 브리핑 생략됨: *{meeting.summary}*\n"
                f"필수 정보가 부족하여 브리핑을 발송하지 않았습니다.\n"
                + "\n".join(f"• {item}" for item in missing_items)
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "[BRIEFING SUPPRESSED] Failed to send suppression notice for '%s'",
                meeting.summary,
            )
        return False

    # ── Step 2c: AI-enhanced briefing generation (Sub-AC 3b) ─────────────────
    # Generates attendee bios, agenda prep notes, and email thread summary
    # using Claude.  Failures are non-fatal — ai_sections.ai_available=False
    # causes the formatter to skip AI sections and use rule-based layout.
    ai_sections = _generate_ai_sections(raw_content)

    # ── Step 3: Format structured Slack message from raw content ─────────────
    # EXTERNAL_FOLLOWUP meetings use a dedicated formatter (AC 7 Sub-AC 4)
    # that emphasises relationship history, Gmail exchanges, and Slack context.
    # EXTERNAL_FIRST meetings use the standard formatter with web search /
    # deal memo sections + AI-enriched bios/agenda.
    if is_external_first:
        from src.briefing.meeting_briefing_formatter import format_meeting_briefing
        text, blocks = format_meeting_briefing(raw_content, ai_sections=ai_sections)
    else:
        from src.briefing.external_followup_formatter import (
            format_external_followup_briefing,
        )
        text, blocks = format_external_followup_briefing(raw_content)

    logger.info(
        "[BRIEFING] '%s' formatted as %s — %d blocks",
        meeting.summary,
        "EXTERNAL_FIRST" if is_external_first else "EXTERNAL_FOLLOWUP",
        len(blocks),
    )

    # ── Step 4: Deliver via Slack DM ───────────────────────────────────────────
    success = bot.send_message(text, blocks=blocks)
    if success:
        logger.info("Briefing notification sent for meeting '%s'", meeting.summary)
    else:
        logger.error("Failed to send briefing for meeting '%s'", meeting.summary)

    return success


def _classify_is_external_first(meeting: "Meeting") -> bool:
    """
    Determine whether *meeting* is an EXTERNAL_FIRST event.

    Attempts to load the calendar history cache and run the full event
    classifier.  Falls back to True (conservative: assume EXTERNAL_FIRST)
    if the classifier fails for any reason.

    Returns:
        True  – EXTERNAL_FIRST  (no prior history with attendees/domains)
        False – EXTERNAL_FOLLOWUP (prior meetings found in history cache)
              – when meeting has no external attendees (cannot be EXTERNAL_FIRST)
    """
    # Early exit: meetings without external attendees can never be EXTERNAL_FIRST
    if not meeting.external_attendees:
        return False

    try:
        # Use local imports so tests can patch at the source modules, e.g.:
        # patch("src.calendar.google_calendar.GoogleCalendarClient", ...)
        # patch("src.calendar.history_cache.CalendarHistoryCache", ...)
        # patch("src.calendar.event_classifier.classify_event", ...)
        from src.calendar.event_classifier import classify_event, EventCategory
        from src.calendar.history_cache import CalendarHistoryCache
        from src.calendar.google_calendar import GoogleCalendarClient

        cache = None
        try:
            cal_client = GoogleCalendarClient()
            cal_client.connect()
            cache = CalendarHistoryCache.load_or_build(cal_client)
        except Exception as cache_exc:
            logger.warning(
                "[BRIEFING] History cache unavailable for classification: %s — "
                "defaulting to EXTERNAL_FIRST",
                cache_exc,
            )
            cache = None

        category = classify_event(meeting, history_cache=cache)
        return category == EventCategory.EXTERNAL_FIRST

    except Exception as exc:
        logger.warning(
            "[BRIEFING] Event classification failed: %s — defaulting to EXTERNAL_FIRST",
            exc,
        )
        return True  # Conservative default: assume first-time external


def _aggregate_meeting_context(
    meeting: "Meeting",
    is_external_first: Optional[bool] = None,
):
    """
    Run MeetingContextAggregator for a meeting, returning RawBriefingContent.

    Initialises real API clients (Gmail, Notion, Calendar, WebSearch) and
    aggregates attendee history, Gmail threads, Notion deal records, and
    optionally web search context (only for EXTERNAL_FIRST meetings).

    Any client that fails to initialise is passed as None so aggregation
    continues for the remaining sources.

    Args:
        meeting:          The external Meeting object to brief.
        is_external_first: When True, web search (Sub-AC 6a) is also executed
                           to enrich the briefing with company background.
                           When None (default), automatically determined by
                           calling ``_classify_is_external_first(meeting)``.
    """
    # Determine classification if not pre-computed by caller
    if is_external_first is None:
        is_external_first = _classify_is_external_first(meeting)

    gmail_client = _try_init_gmail()
    notion_client = _try_init_notion()
    calendar_client = _try_init_calendar()
    web_search_client = _try_init_web_search() if is_external_first else None
    # Slack history retriever is used for both EXTERNAL_FIRST and EXTERNAL_FOLLOWUP
    # (AC 7 Sub-AC 3 and AC 6).  A failure to init is non-fatal.
    slack_retriever = _try_init_slack_retriever()

    # Only attempt Slack history search when the retriever was successfully
    # initialised.  Passing fetch_slack_history=False prevents the aggregator
    # from auto-creating a second retriever instance (which would attempt
    # real Slack API calls even when the first init already failed).
    fetch_slack = slack_retriever is not None

    # Use local import so tests can patch at source module:
    # patch("src.briefing.context_aggregator.MeetingContextAggregator", ...)
    from src.briefing.context_aggregator import MeetingContextAggregator

    aggregator = MeetingContextAggregator(
        gmail_client=gmail_client,
        notion_client=notion_client,
        calendar_client=calendar_client,
        web_search_client=web_search_client,
        slack_retriever=slack_retriever,
    )
    return aggregator.aggregate(
        meeting,
        is_external_first=is_external_first,
        fetch_slack_history=fetch_slack,
    )


def _try_init_gmail():
    """Attempt to create a GmailClient; return None on failure."""
    try:
        from src.gmail.gmail_client import GmailClient
        client = GmailClient()
        client.connect()
        return client
    except Exception as exc:
        logger.warning("[BRIEFING] Gmail client init failed: %s", exc)
        return None


def _try_init_notion():
    """Attempt to create a NotionClient; return None on failure."""
    try:
        from src.notion.notion_client import NotionClient
        client = NotionClient()
        client.connect()
        return client
    except Exception as exc:
        logger.warning("[BRIEFING] Notion client init failed: %s", exc)
        return None


def _try_init_calendar():
    """Attempt to create a GoogleCalendarClient; return None on failure."""
    try:
        from src.calendar.google_calendar import GoogleCalendarClient
        client = GoogleCalendarClient()
        client.connect()
        return client
    except Exception as exc:
        logger.warning("[BRIEFING] Calendar client init failed: %s", exc)
        return None


def _try_init_web_search():
    """
    Attempt to create a WebSearchClient for EXTERNAL_FIRST enrichment.

    Returns the client if at least one provider (Tavily or Claude) is
    available; returns None if both are unconfigured or fail to initialise.

    Uses a local import so tests can patch at the source module:
    ``patch("src.ai.web_search.WebSearchClient", ...)``.
    """
    try:
        # Local import: patchable via patch("src.ai.web_search.WebSearchClient", ...)
        from src.ai.web_search import WebSearchClient
        client = WebSearchClient()
        if client.is_available:
            return client
        logger.debug("[BRIEFING] WebSearchClient created but no provider available")
        return client  # Return anyway; aggregator will record the error
    except Exception as exc:
        logger.warning("[BRIEFING] WebSearchClient init failed: %s", exc)
        return None


def _try_init_slack_retriever():
    """
    Attempt to create a SlackHistoryRetriever for Slack history search.

    Used for both EXTERNAL_FIRST and EXTERNAL_FOLLOWUP meetings to surface
    priority-channel discussions about the company (AC 7 Sub-AC 3).

    Returns None on failure so aggregation continues without Slack context.
    """
    try:
        from src.slack.history_retriever import SlackHistoryRetriever
        return SlackHistoryRetriever()
    except Exception as exc:
        logger.warning("[BRIEFING] SlackHistoryRetriever init failed: %s", exc)
        return None


def _generate_ai_sections(raw_content):
    """
    Run the AI briefing generator to produce attendee bios, agenda prep,
    and email thread summary for a meeting (Sub-AC 3b).

    Returns an ``AIBriefingSections`` with ``ai_available=False`` on failure
    so that the formatter degrades gracefully to the rule-based layout.
    Failures do NOT raise — the briefing is always sent, just without the AI
    sections when the generator is unavailable.
    """
    try:
        from src.ai.briefing_generator import BriefingGenerator
        generator = BriefingGenerator()
        return generator.generate(raw_content)
    except Exception as exc:
        logger.warning("[BRIEFING] AI section generation failed: %s", exc)
        # Return a minimal fallback object indicating AI is unavailable
        try:
            from src.ai.briefing_generator import AIBriefingSections
            return AIBriefingSections(
                ai_available=False,
                error=str(exc),
            )
        except Exception:
            return None


def _format_raw_briefing(raw_content) -> tuple[str, list[dict]]:
    """
    Format a RawBriefingContent into a Slack message (fallback text + blocks).

    Renders structured raw data without AI narrative generation.
    Sub-AC 2c will replace this with a Claude-generated briefing.
    """
    from zoneinfo import ZoneInfo
    import datetime as _dt
    KST = ZoneInfo("Asia/Seoul")

    meeting_title = raw_content.meeting_title
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    starts_in_secs = (raw_content.meeting_start - now_utc).total_seconds()
    starts_in = round(starts_in_secs / 60, 1)
    start_kst = raw_content.meeting_start.astimezone(KST).strftime("%H:%M")
    end_kst = raw_content.meeting_end.astimezone(KST).strftime("%H:%M")
    duration = raw_content.duration_minutes

    blocks: list[dict] = []

    # Header
    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"📅 미팅 브리핑 — {meeting_title}",
            "emoji": True,
        },
    })

    # Time / location row
    time_line = f"⏰ *시작까지:* {starts_in}분  |  🕐 {start_kst}–{end_kst} ({duration}분)"
    if raw_content.meeting_location:
        time_line += f"\n📍 {raw_content.meeting_location}"
    if raw_content.meeting_html_link:
        time_line += f"\n🔗 <{raw_content.meeting_html_link}|캘린더에서 보기>"
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": time_line},
    })

    # Agenda / description
    if raw_content.meeting_description:
        desc = raw_content.meeting_description[:500]
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📋 안건/설명*\n{desc}"},
        })

    # ── External attendees ────────────────────────────────────────────────────
    blocks.append({"type": "divider"})
    ext_profiles = raw_content.external_attendees

    if ext_profiles:
        ext_lines: list[str] = []
        for p in ext_profiles:
            name = p.display_name or p.email
            domain = f" _{p.company_domain}_" if p.company_domain else ""
            if p.past_meeting_count > 0:
                last_met = ""
                if p.last_met_date:
                    last_met = (
                        f", 최근: {p.last_met_date.astimezone(KST).strftime('%Y-%m-%d')}"
                    )
                history = f" (과거 미팅 {p.past_meeting_count}회{last_met})"
            elif not raw_content.calendar_history_available:
                history = " (과거 이력: 확인 불가)"
            else:
                history = ""
            ext_lines.append(f"• *{name}*{domain}{history}")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*👥 외부 참석자*\n" + "\n".join(ext_lines),
            },
        })
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*👥 외부 참석자*\n_없음_"},
        })

    # Internal attendees (brief context line)
    int_profiles = raw_content.internal_attendees
    if int_profiles:
        int_names = [p.display_name or p.email for p in int_profiles]
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"🏢 내부: {', '.join(int_names)}"}
            ],
        })

    # ── Gmail threads ─────────────────────────────────────────────────────────
    blocks.append({"type": "divider"})
    if not raw_content.gmail_available:
        err_msg = next(
            (e.message for e in raw_content.errors if e.source == "gmail"), "오류"
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📧 관련 이메일 스레드*\n⚠️ 확인 불가 (`{err_msg[:80]}`)",
            },
        })
    elif raw_content.gmail_threads:
        thread_lines: list[str] = []
        for thread in raw_content.gmail_threads[:5]:
            subject = thread.subject or "(제목 없음)"
            latest = ""
            if thread.latest_date:
                latest = (
                    f" `{thread.latest_date.astimezone(KST).strftime('%m/%d')}`"
                )
            count = f" ({thread.message_count}개 메시지)"
            thread_lines.append(f"• {subject[:60]}{latest}{count}")
        header_text = (
            f"*📧 관련 이메일 스레드* ({len(raw_content.gmail_threads)}개)\n"
            + "\n".join(thread_lines)
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": header_text},
        })
        if len(raw_content.gmail_threads) > 5:
            remaining = len(raw_content.gmail_threads) - 5
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 스레드_"}
                ],
            })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📧 관련 이메일 스레드*\n_관련 이메일 없음_",
            },
        })

    # ── Notion records ────────────────────────────────────────────────────────
    blocks.append({"type": "divider"})
    if not raw_content.notion_available:
        err_msg = next(
            (e.message for e in raw_content.errors if e.source == "notion"), "오류"
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📑 관련 딜 / 포트폴리오*\n⚠️ 확인 불가 (`{err_msg[:80]}`)",
            },
        })
    elif raw_content.notion_records:
        record_lines: list[str] = []
        for rec in raw_content.notion_records[:5]:
            title = rec.title or rec.company_name or "(이름 없음)"
            status = f"  [{rec.status}]" if rec.status else ""
            url_md = f"<{rec.url}|{title[:40]}>" if rec.url else title[:40]
            record_lines.append(f"• {url_md}{status}")
        header_text = (
            f"*📑 관련 딜 / 포트폴리오* ({len(raw_content.notion_records)}개)\n"
            + "\n".join(record_lines)
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": header_text},
        })
        if len(raw_content.notion_records) > 5:
            remaining = len(raw_content.notion_records) - 5
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_… 그 외 {remaining}개 항목_"}
                ],
            })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*📑 관련 딜 / 포트폴리오*\n_관련 Notion 항목 없음_",
            },
        })

    # Footer
    blocks.append({"type": "divider"})
    error_note = ""
    if raw_content.has_errors:
        sources = {e.source for e in raw_content.errors}
        error_note = f"  |  ⚠️ 일부 정보 확인 불가: {', '.join(sorted(sources))}"
    blocks.append({
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
    })

    # Fallback plain text
    ext_email_list = ", ".join(p.email for p in ext_profiles) or "없음"
    fallback = (
        f"📅 미팅 브리핑 — {meeting_title}\n"
        f"⏰ 시작까지: {starts_in}분  ({start_kst}–{end_kst})\n"
        f"👥 외부 참석자: {ext_email_list}\n"
        f"📧 관련 이메일: {len(raw_content.gmail_threads)}개 스레드\n"
        f"📑 관련 딜: {len(raw_content.notion_records)}개"
    )

    return fallback, blocks


def run_briefing_for_next_meeting(lookahead_minutes: int = 15, bot=None) -> bool:
    """
    Called by the /brief slash command.

    Fetches the next external meeting starting within *lookahead_minutes* and
    triggers a briefing.  Returns True if a meeting was found and briefed.
    """
    from src.calendar.google_calendar import GoogleCalendarClient

    client = GoogleCalendarClient()
    meetings = client.get_external_meetings_starting_soon(lookahead_minutes)

    if not meetings:
        logger.info("run_briefing_for_next_meeting: no external meetings in next %d min", lookahead_minutes)
        return False

    # Brief the soonest meeting
    next_meeting = min(meetings, key=lambda m: m.start)
    return trigger_meeting_briefing(next_meeting, bot=bot)


def run_daily_briefing(
    target_date: Optional[date] = None,
    bot=None,
    user_id: Optional[str] = None,
) -> bool:
    """
    Main entry point for the /brief slash command (Sub-AC 3.2).

    Aggregates data from:
      1. Google Calendar — events for target_date
      2. Gmail           — recent inbox emails
      3. Notion          — portfolio company deadline items

    Formats a rich Slack Block Kit briefing and delivers it via DM.

    Args:
        target_date: The date to brief (defaults to today KST).
        bot:         WorkAssistantBot instance.  If None, logs and returns True.
        user_id:     Override DM recipient (defaults to bot's target user).

    Returns:
        True if the briefing was delivered (or bot is None).
        False if delivery failed.
    """
    from zoneinfo import ZoneInfo
    from src.briefing.aggregator import aggregate_briefing_data
    from src.briefing.full_formatter import format_full_briefing

    _KST = ZoneInfo("Asia/Seoul")
    if target_date is None:
        target_date = datetime.now(_KST).date()

    logger.info(
        "run_daily_briefing: target_date=%s  bot=%s",
        target_date,
        "present" if bot else "None",
    )

    # ── Step 1: Aggregate data from all sources ────────────────────────────────
    data = aggregate_briefing_data(target_date=target_date)
    logger.info("Aggregation complete: %s", data.summary())

    if bot is None:
        # Test / dev mode — just log and return
        return True

    # ── Step 2: Format the briefing ────────────────────────────────────────────
    fallback_text, blocks = format_full_briefing(data)

    # ── Step 3: Deliver via Slack DM ───────────────────────────────────────────
    success = bot.send_message(fallback_text, blocks=blocks)
    if success:
        logger.info(
            "run_daily_briefing: full briefing sent for %s", target_date
        )
    else:
        logger.error(
            "run_daily_briefing: failed to send briefing for %s", target_date
        )
        try:
            bot.send_error(
                f"run_daily_briefing ({target_date})",
                RuntimeError("send_message returned False"),
            )
        except Exception:
            logger.exception("run_daily_briefing: error DM also failed")

    return success


def run_daily_morning_briefing(bot=None) -> bool:
    """
    Fetch ALL of today's calendar events and send a full daily schedule briefing.

    This is the primary entry point for the 09:30 KST cron job.  Unlike the
    per-meeting briefings triggered by the 60-second polling job, this sends a
    single comprehensive overview of the full day at the start of each morning.

    Pipeline:
        1. Resolve today's date in KST (Asia/Seoul).
        2. Fetch all calendar events for today (00:00–23:59 KST) via
           ``GoogleCalendarClient.list_upcoming_events()``.
        3. Retry up to ``API_RETRY_ATTEMPTS`` times with
           ``API_RETRY_DELAY_SECONDS`` between attempts.
        4. On persistent failure send an error DM and return False.
        5. Format the event list into Slack Block Kit via
           ``send_daily_briefing_dm()`` (which calls
           ``format_daily_briefing()`` + ``bot.send_message()``).
        6. On DM delivery failure send a best-effort error DM and return False.

    Args:
        bot:  WorkAssistantBot instance used for sending DMs.
              Pass None during testing — the function logs the outcome and
              returns True without making any Slack API calls.

    Returns:
        True  – briefing delivered successfully (or bot is None).
        False – calendar fetch failed or Slack delivery failed after retries.
    """
    from zoneinfo import ZoneInfo
    from src.calendar.google_calendar import GoogleCalendarClient
    from src.slack.dm_sender import send_daily_briefing_dm

    KST = ZoneInfo("Asia/Seoul")
    today: date = datetime.now(KST).date()

    logger.info("[DAILY BRIEFING] Starting morning briefing pipeline for %s", today)

    # ── Early-exit when running without a bot (tests / dry-run) ───────────────
    if bot is None:
        logger.info("[DAILY BRIEFING] bot=None — skipping DM delivery (dry-run mode)")
        return True

    # ── Compute today's window in UTC ─────────────────────────────────────────
    day_start_kst = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=KST)
    day_end_kst = day_start_kst + timedelta(days=1)
    day_start_utc = day_start_kst.astimezone(timezone.utc)
    day_end_utc = day_end_kst.astimezone(timezone.utc)

    # ── Fetch with retry ──────────────────────────────────────────────────────
    meetings = []
    last_exc: Optional[Exception] = None
    calendar_client = GoogleCalendarClient()

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            meetings = calendar_client.list_upcoming_events(
                time_min=day_start_utc,
                time_max=day_end_utc,
                max_results=50,
            )
            last_exc = None
            break
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logger.warning(
                "[DAILY BRIEFING] Calendar fetch failed (attempt %d/%d): %s",
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)

    if last_exc is not None:
        logger.error(
            "[DAILY BRIEFING] Google Calendar API failed after %d attempts: %s",
            API_RETRY_ATTEMPTS,
            last_exc,
        )
        try:
            bot.send_error("Daily morning briefing — Calendar fetch", last_exc)
        except Exception:  # pylint: disable=broad-except
            logger.exception("[DAILY BRIEFING] Failed to send calendar-error DM")
        return False

    logger.info(
        "[DAILY BRIEFING] Fetched %d event(s) for %s; formatting and sending…",
        len(meetings),
        today,
    )

    # ── Format and deliver ────────────────────────────────────────────────────
    ok = send_daily_briefing_dm(bot, meetings, target_date=today)

    if ok:
        logger.info("[DAILY BRIEFING] Morning briefing delivered for %s ✓", today)
    else:
        logger.error(
            "[DAILY BRIEFING] Failed to deliver morning briefing for %s", today
        )
        try:
            bot.send_error(
                "Daily morning briefing — Slack DM delivery",
                RuntimeError("send_message returned False"),
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("[DAILY BRIEFING] Failed to send delivery-error DM")

    return ok


def run_aggregated_brief(
    target_date: Optional[date] = None,
    bot=None,
    user_id: Optional[str] = None,
) -> bool:
    """
    Aggregate briefing data from all sources and deliver it via Slack DM.

    This is the primary entry point for the /brief slash command (Sub-AC 3.2).
    Unlike run_briefing_for_next_meeting() (which only covers calendar), this
    function pulls from all three configured data sources:

      1. Google Calendar  – all events for target_date (KST day boundaries)
      2. Gmail            – recent inbox emails (last 1-2 days)
      3. Notion           – portfolio company deadline items (next 30 days)

    Each source is fetched independently with 3-retry / 10s-delay error
    handling.  If a source fails after all retries, its section in the Slack
    message is annotated '확인 불가' — no incorrect information is sent.

    Parameters
    ----------
    target_date:
        The date to brief.  Defaults to today (KST).
    bot:
        WorkAssistantBot instance used for sending DMs.  Pass None during
        testing — the function logs the outcome and returns True without
        making any Slack API calls.
    user_id:
        Optional Slack user ID to DM.  When provided the DM goes to this
        user; when None it falls back to SLACK_TARGET_USER_ID from config.

    Returns
    -------
    True  – briefing delivered (or bot is None / dry-run mode).
    False – all retries exhausted or DM delivery failed.
    """
    from zoneinfo import ZoneInfo
    from src.briefing.aggregator import aggregate_briefing_data
    from src.briefing.full_formatter import format_full_briefing

    KST = ZoneInfo("Asia/Seoul")
    if target_date is None:
        target_date = datetime.now(KST).date()

    logger.info(
        "[AGGREGATED BRIEF] Starting full briefing pipeline for %s", target_date
    )

    # ── Dry-run mode (no bot) ─────────────────────────────────────────────────
    if bot is None:
        logger.info(
            "[AGGREGATED BRIEF] bot=None — running in dry-run mode (no DM sent)"
        )
        briefing_data = aggregate_briefing_data(target_date=target_date)
        logger.info(
            "[AGGREGATED BRIEF] dry-run result: %s", briefing_data.summary()
        )
        return True

    # ── Aggregate data from all sources ──────────────────────────────────────
    briefing_data = aggregate_briefing_data(target_date=target_date)
    logger.info(
        "[AGGREGATED BRIEF] Aggregation complete: %s", briefing_data.summary()
    )

    # ── Format the full briefing ──────────────────────────────────────────────
    fallback_text, blocks = format_full_briefing(briefing_data)

    # ── Deliver via Slack DM ──────────────────────────────────────────────────
    if user_id:
        # Send directly to the invoking user when user_id is known
        try:
            bot._client.chat_postMessage(
                channel=user_id,
                text=fallback_text,
                blocks=blocks,
            )
            logger.info(
                "[AGGREGATED BRIEF] Briefing delivered to user=%s for %s ✓",
                user_id,
                target_date,
            )
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[AGGREGATED BRIEF] Direct DM to %s failed, falling back to "
                "send_message: %s",
                user_id,
                exc,
            )

    # Fallback: use bot.send_message() which goes to SLACK_TARGET_USER_ID
    ok = bot.send_message(fallback_text, blocks=blocks)
    if ok:
        logger.info(
            "[AGGREGATED BRIEF] Briefing delivered via send_message for %s ✓",
            target_date,
        )
    else:
        logger.error(
            "[AGGREGATED BRIEF] Failed to deliver briefing for %s", target_date
        )
        try:
            bot.send_error(
                f"Aggregated brief for {target_date}",
                RuntimeError("send_message returned False"),
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("[AGGREGATED BRIEF] Failed to send error DM")

    return ok


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_time(dt: datetime) -> str:
    """Format a UTC datetime as KST HH:MM."""
    from zoneinfo import ZoneInfo
    kst = ZoneInfo("Asia/Seoul")
    return dt.astimezone(kst).strftime("%H:%M")
