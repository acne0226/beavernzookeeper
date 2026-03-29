"""
Scheduler  (Sub-AC 2a + Sub-AC 4 + Feature 2 + Feature 3)
===========================================================

Manages recurring jobs via APScheduler:

Job 1 — External-Meeting Checker  (every 60 s)
  Detects external meetings starting within MEETING_LOOKAHEAD_MINUTES and
  triggers the briefing pipeline.

Job 2 — Daily Morning Briefing  (09:30 KST, every day)
  Full-day calendar overview sent as Slack DM.

Job 3 — Calendar History Cache Refresh  (02:00 KST, every day)
  Rebuilds the 1-year calendar event index.

Job 4 — Portfolio Mail Monitor  (every 5 min)
  Scans Gmail for new portfolio company emails; detects deadline approaching
  and overdue states; sends Slack DM alerts.

Job 5 — Missed Reply Check  (09:00 KST, every day)
  Daily scan for portfolio emails without a reply; sends Slack DM alert.

Job 6 — Task/Follow-up Suggestions  (09:00, 12:00, 18:00 KST)
  Natural-language Q&A based suggestions using Calendar+Gmail+Notion+Slack.

Public interface
----------------
  start_scheduler(bot) → BackgroundScheduler  # starts all jobs
  stop_scheduler()                             # graceful shutdown
  get_triggered_ids() → frozenset             # for testing / inspection
  reset_triggered_ids()                        # for testing
  run_check_once(bot)                          # run meeting-check once
  run_mail_check_once(bot)                     # run mail-check once
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import (
    MEETING_LOOKAHEAD_MINUTES,
    SCHEDULER_POLL_INTERVAL_SECONDS,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None
_triggered_event_ids: set[str] = set()
_lock = threading.Lock()  # protects _triggered_event_ids


# ── Core job ───────────────────────────────────────────────────────────────────

def _check_upcoming_external_meetings(bot=None) -> None:
    """
    Scheduler job: detect external meetings starting within the lookahead
    window and trigger the briefing pipeline for any new ones.

    This function is called by APScheduler in the scheduler's thread pool.
    It is safe to call directly for testing.
    """
    from src.calendar.google_calendar import GoogleCalendarClient, Meeting
    from src.briefing.pipeline import trigger_meeting_briefing

    logger.debug("Scheduler tick: checking for external meetings in next %d min", MEETING_LOOKAHEAD_MINUTES)

    # ── Fetch with retry ───────────────────────────────────────────────────────
    meetings: list[Meeting] = []
    last_exc: Optional[Exception] = None

    calendar_client = GoogleCalendarClient()

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            meetings = calendar_client.get_external_meetings_starting_soon(
                lookahead_minutes=MEETING_LOOKAHEAD_MINUTES
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Calendar fetch failed (attempt %d/%d): %s",
                attempt, API_RETRY_ATTEMPTS, exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)

    if last_exc is not None:
        # All retries exhausted
        error_msg = (
            f"Google Calendar API 오류: {API_RETRY_ATTEMPTS}회 재시도 후 실패.\n"
            f"오류: `{last_exc}`"
        )
        logger.error("Calendar API exhausted all retries: %s", last_exc)
        if bot is not None:
            try:
                bot.send_error("Calendar fetch in scheduler", last_exc)
            except Exception:
                logger.exception("Failed to send error DM")
        return

    if not meetings:
        logger.debug("No external meetings found in next %d min.", MEETING_LOOKAHEAD_MINUTES)
        return

    # ── De-duplicate and trigger ───────────────────────────────────────────────
    newly_triggered: list[Meeting] = []

    with _lock:
        for meeting in meetings:
            if meeting.event_id not in _triggered_event_ids:
                _triggered_event_ids.add(meeting.event_id)
                newly_triggered.append(meeting)

    for meeting in newly_triggered:
        logger.info(
            "New external meeting detected: '%s' (id=%s, starts_in=%.1f min)",
            meeting.summary,
            meeting.event_id,
            meeting.starts_in_minutes,
        )
        try:
            trigger_meeting_briefing(meeting, bot=bot)
        except Exception as exc:
            logger.exception(
                "Briefing pipeline error for meeting '%s': %s",
                meeting.summary, exc,
            )
            if bot is not None:
                try:
                    bot.send_error(
                        f"Briefing pipeline for '{meeting.summary}'", exc
                    )
                except Exception:
                    logger.exception("Failed to send briefing-error DM")


# ── History cache refresh job ──────────────────────────────────────────────────

def _run_history_cache_refresh_job(bot=None) -> None:
    """
    APScheduler job: rebuild the 1-year calendar history cache.

    Fires at 02:00 KST every day so the rolling lookback window stays current.
    Uses the module-level ``history_loader.refresh()`` which retries up to
    ``API_RETRY_ATTEMPTS`` times with ``API_RETRY_DELAY_SECONDS`` delay between
    attempts.

    On failure sends an error DM via the bot (if available) but does NOT
    interrupt the running daemon — the previous cache continues to be used.
    """
    from src.calendar.google_calendar import GoogleCalendarClient
    from src.calendar.history_loader import refresh as refresh_history

    logger.info("[HISTORY REFRESH JOB] Cron fired at 02:00 KST — refreshing calendar history cache")
    try:
        cal_client = GoogleCalendarClient()
        cal_client.connect()
        ok = refresh_history(cal_client)
        if ok:
            logger.info("[HISTORY REFRESH JOB] Calendar history cache refreshed successfully.")
        else:
            msg = "Calendar history cache refresh failed after all retries — using stale cache."
            logger.warning("[HISTORY REFRESH JOB] %s", msg)
            if bot is not None:
                try:
                    bot.send_error("History cache refresh job (02:00 cron)", RuntimeError(msg))
                except Exception:
                    logger.exception("[HISTORY REFRESH JOB] Failed to send error DM")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[HISTORY REFRESH JOB] Unexpected error: %s", exc)
        if bot is not None:
            try:
                bot.send_error("History cache refresh job (02:00 cron)", exc)
            except Exception:
                logger.exception("[HISTORY REFRESH JOB] Failed to send top-level error DM")


# ── Daily morning briefing job ─────────────────────────────────────────────────

def _run_daily_morning_briefing_job(bot=None) -> None:
    """
    APScheduler job: send the full daily calendar briefing.

    Fires at 09:30 KST every day via a CronTrigger.
    Delegates all business logic to ``run_daily_morning_briefing()``.
    """
    from src.briefing.pipeline import run_daily_morning_briefing

    logger.info("[DAILY BRIEFING JOB] Cron fired at 09:30 KST — running daily briefing pipeline")
    try:
        run_daily_morning_briefing(bot=bot)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[DAILY BRIEFING JOB] Unexpected error in briefing pipeline: %s", exc)
        if bot is not None:
            try:
                bot.send_error("Daily morning briefing job (09:30 cron)", exc)
            except Exception:  # pylint: disable=broad-except
                logger.exception("[DAILY BRIEFING JOB] Failed to send top-level error DM")


# ── Scheduler lifecycle ────────────────────────────────────────────────────────

def start_scheduler(bot=None) -> BackgroundScheduler:
    """
    Create, configure, and start the APScheduler BackgroundScheduler.

    Registers two jobs:
      * ``meeting_checker``        — interval job every 60 s
      * ``daily_morning_briefing`` — cron job at 09:30 Asia/Seoul every day

    Args:
        bot: WorkAssistantBot instance used for sending DMs.
             Pass None during testing to suppress DM output.

    Returns:
        The running BackgroundScheduler instance.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.warning("start_scheduler called but scheduler is already running.")
        return _scheduler

    _scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,        # Don't pile up missed runs
            "max_instances": 1,      # Only one instance of the job at a time
            "misfire_grace_time": 30,
        },
        timezone="Asia/Seoul",
    )

    # ── Job 1: External-meeting checker (every 60 s) ──────────────────────────
    _scheduler.add_job(
        func=_check_upcoming_external_meetings,
        trigger=IntervalTrigger(seconds=SCHEDULER_POLL_INTERVAL_SECONDS),
        id="meeting_checker",
        name="External Meeting Checker",
        kwargs={"bot": bot},
        replace_existing=True,
    )

    # ── Job 2: Daily morning briefing (09:30 KST) ─────────────────────────────
    _scheduler.add_job(
        func=_run_daily_morning_briefing_job,
        trigger=CronTrigger(hour=9, minute=30, timezone="Asia/Seoul"),
        id="daily_morning_briefing",
        name="Daily Morning Briefing (09:30 KST)",
        kwargs={"bot": bot},
        replace_existing=True,
    )

    # ── Job 3: Daily history cache refresh (02:00 KST) ────────────────────────
    _scheduler.add_job(
        func=_run_history_cache_refresh_job,
        trigger=CronTrigger(hour=2, minute=0, timezone="Asia/Seoul"),
        id="history_cache_refresh",
        name="Calendar History Cache Refresh (02:00 KST)",
        kwargs={"bot": bot},
        replace_existing=True,
    )

    # ── Job 4: Portfolio mail monitor (every 5 minutes) ───────────────────────
    _scheduler.add_job(
        func=_run_portfolio_mail_monitor_job,
        trigger=IntervalTrigger(minutes=5),
        id="portfolio_mail_monitor",
        name="Portfolio Mail Monitor (every 5 min)",
        kwargs={"bot": bot},
        replace_existing=True,
    )

    # ── Job 5: Daily missed-reply check (09:00 KST) ───────────────────────────
    _scheduler.add_job(
        func=_run_missed_reply_check_job,
        trigger=CronTrigger(hour=9, minute=0, timezone="Asia/Seoul"),
        id="missed_reply_check",
        name="Missed Reply Check (09:00 KST)",
        kwargs={"bot": bot},
        replace_existing=True,
    )

    # ── Job 6: Task/follow-up suggestions (09:00, 12:00, 18:00 KST) ──────────
    for suggestion_hour, suggestion_id in [(9, "task_suggest_09"), (12, "task_suggest_12"), (18, "task_suggest_18")]:
        _scheduler.add_job(
            func=_run_task_suggestion_job,
            trigger=CronTrigger(hour=suggestion_hour, minute=0, timezone="Asia/Seoul"),
            id=suggestion_id,
            name=f"Task Suggestions ({suggestion_hour:02d}:00 KST)",
            kwargs={"bot": bot},
            replace_existing=True,
        )

    _scheduler.start()
    logger.info(
        "Scheduler started. Polling every %ds for external meetings in next %d min.",
        SCHEDULER_POLL_INTERVAL_SECONDS,
        MEETING_LOOKAHEAD_MINUTES,
    )
    logger.info(
        "Daily morning briefing scheduled at 09:30 KST (Asia/Seoul) every day."
    )
    logger.info(
        "Calendar history cache refresh scheduled at 02:00 KST (Asia/Seoul) every day."
    )
    logger.info("Portfolio mail monitor scheduled every 5 minutes.")
    logger.info("Missed reply check scheduled at 09:00 KST daily.")
    logger.info("Task suggestions scheduled at 09:00, 12:00, 18:00 KST daily.")

    # Run immediately on startup so we don't miss a meeting in the first minute
    _run_initial_check(bot)

    return _scheduler


def _run_initial_check(bot=None) -> None:
    """Run one synchronous check immediately at startup (non-blocking via thread)."""
    def _initial():
        logger.info("Running initial external-meeting check at startup...")
        _check_upcoming_external_meetings(bot=bot)

    t = threading.Thread(target=_initial, daemon=True, name="initial-meeting-check")
    t.start()


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped.")
    _scheduler = None


def get_triggered_ids() -> frozenset[str]:
    """Return a snapshot of the set of already-triggered event IDs (for testing)."""
    with _lock:
        return frozenset(_triggered_event_ids)


def reset_triggered_ids() -> None:
    """Clear the triggered-event-IDs set (for testing)."""
    with _lock:
        _triggered_event_ids.clear()
    logger.debug("Triggered event ID set cleared.")


# ── Convenience: run the check once synchronously ─────────────────────────────

def run_check_once(bot=None) -> None:
    """
    Run a single synchronous meeting check.

    Useful for:
    - Manual testing from the REPL
    - Verifying connectivity before the scheduler starts
    - Integration tests
    """
    _check_upcoming_external_meetings(bot=bot)


# ── Feature 2: Portfolio Mail Monitor jobs ─────────────────────────────────────

def _send_mail_alerts(bot, alerts) -> None:
    """Send Slack DM alerts for deadline/overdue/missed-reply records."""
    from src.gmail.mail_monitor import (
        format_deadline_approaching_alert,
        format_overdue_alert,
        format_missed_reply_alert,
    )

    for record in alerts.approaching:
        msg = format_deadline_approaching_alert(record)
        logger.info("Sending approaching-deadline alert for %r", record.company_name)
        try:
            bot.send_message(msg)
        except Exception as exc:
            logger.warning("Failed to send approaching alert: %s", exc)

    for record in alerts.overdue:
        msg = format_overdue_alert(record)
        logger.info("Sending overdue alert for %r", record.company_name)
        try:
            bot.send_message(msg)
        except Exception as exc:
            logger.warning("Failed to send overdue alert: %s", exc)

    for record in alerts.missed_reply:
        msg = format_missed_reply_alert(record)
        logger.info("Sending missed-reply alert for %r", record.company_name)
        try:
            bot.send_message(msg)
        except Exception as exc:
            logger.warning("Failed to send missed-reply alert: %s", exc)


def _run_portfolio_mail_monitor_job(bot=None) -> None:
    """
    APScheduler job: scan Gmail for new portfolio emails and send alerts.
    Runs every 5 minutes. Handles deadline approaching (AC 12) and
    overdue (AC 13) alerts.
    """
    from src.gmail.mail_monitor import get_mail_monitor

    logger.debug("[MAIL MONITOR JOB] Running portfolio mail scan…")
    try:
        monitor = get_mail_monitor()
        alerts = monitor.run_scan_and_check()

        if alerts.has_alerts and bot is not None:
            _send_mail_alerts(bot, alerts)
        else:
            logger.debug(
                "[MAIL MONITOR JOB] No new alerts. Approaching=%d, Overdue=%d, MissedReply=%d",
                len(alerts.approaching), len(alerts.overdue), len(alerts.missed_reply),
            )
    except Exception as exc:
        logger.exception("[MAIL MONITOR JOB] Unexpected error: %s", exc)
        if bot is not None:
            try:
                bot.send_error("Portfolio mail monitor job", exc)
            except Exception:
                logger.exception("[MAIL MONITOR JOB] Failed to send error DM")


def _run_missed_reply_check_job(bot=None) -> None:
    """
    APScheduler job: daily missed-reply check at 09:00 KST (AC 14).
    Scans all tracked portfolio emails for unanswered threads.
    """
    from src.gmail.mail_monitor import get_mail_monitor

    logger.info("[MISSED REPLY JOB] Daily missed-reply check at 09:00 KST")
    try:
        monitor = get_mail_monitor()
        # Force a fresh scan first, then check for missed replies
        monitor.scan_emails()
        alerts = monitor.check_alerts()

        if alerts.missed_reply and bot is not None:
            from src.gmail.mail_monitor import format_missed_reply_alert
            for record in alerts.missed_reply:
                msg = format_missed_reply_alert(record)
                logger.info(
                    "Sending missed-reply alert for %r", record.company_name
                )
                try:
                    bot.send_message(msg)
                except Exception as exc:
                    logger.warning("Failed to send missed-reply DM: %s", exc)
        else:
            logger.info("[MISSED REPLY JOB] No missed replies detected.")
    except Exception as exc:
        logger.exception("[MISSED REPLY JOB] Unexpected error: %s", exc)
        if bot is not None:
            try:
                bot.send_error("Missed reply check job (09:00 cron)", exc)
            except Exception:
                logger.exception("[MISSED REPLY JOB] Failed to send error DM")


def run_mail_check_once(bot=None) -> None:
    """Run a single synchronous mail monitor check (for testing/REPL)."""
    _run_portfolio_mail_monitor_job(bot=bot)


# ── Feature 3: Natural language Q&A / Task suggestion jobs ────────────────────

def _run_task_suggestion_job(bot=None) -> None:
    """
    APScheduler job: generate and send automated task/follow-up suggestions.
    Fires at 09:00, 12:00, and 18:00 KST daily (AC 18).
    """
    from src.ai.qa_engine import get_qa_engine

    logger.info("[TASK SUGGESTION JOB] Generating task/follow-up suggestions…")
    try:
        engine = get_qa_engine()
        suggestions = engine.generate_task_suggestions()
        if suggestions and bot is not None:
            bot.send_message(suggestions)
            logger.info("[TASK SUGGESTION JOB] Task suggestions sent.")
        else:
            logger.info("[TASK SUGGESTION JOB] No suggestions generated.")
    except Exception as exc:
        logger.exception("[TASK SUGGESTION JOB] Unexpected error: %s", exc)
        if bot is not None:
            try:
                bot.send_error("Task suggestion job", exc)
            except Exception:
                logger.exception("[TASK SUGGESTION JOB] Failed to send error DM")
