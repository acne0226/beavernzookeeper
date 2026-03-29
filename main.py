"""
Work Assistant Daemon — Entry Point
=====================================

Starts:
  1. Slack Bot (Socket Mode, background thread)
  2. APScheduler (background thread) with two jobs:
       a. External-Meeting Checker — polls Google Calendar every 60 s and
          triggers per-meeting briefings for meetings starting within 15 min.
       b. Daily Morning Briefing   — fires at 09:30 KST every day; fetches
          ALL of today's calendar events, formats a full schedule overview,
          and delivers it as a Slack DM.

Usage:
    python main.py

Stop with Ctrl-C or SIGTERM.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daemon")


def _make_briefing_callback():
    """
    Create a briefing callback for the /brief slash command.

    Uses a 'bot_holder' list to allow the callback to be registered with the
    WorkAssistantBot constructor *before* the bot object is fully initialised,
    avoiding the chicken-and-egg problem.  The holder is filled in immediately
    after the bot is constructed so the callback always has a valid reference
    when it is actually invoked (on the first /brief command).

    The callback aggregates data from all three sources (Google Calendar,
    Gmail, and Notion) and delivers a comprehensive briefing via Slack DM.
    Runs in a background thread to satisfy Slack's 3-second ack requirement.

    Signature expected by the /brief handler:
        fn(target_date: date, user_id: str, channel_id: str) -> None

    Returns:
        (callback, bot_holder)
            callback:   the callable to pass to WorkAssistantBot
            bot_holder: list[bot | None] — set bot_holder[0] after construction
    """
    from datetime import date as _date

    bot_holder: list = [None]

    def _callback(target_date: _date, user_id: str, channel_id: str) -> None:
        """
        Fetch calendar events, pending emails, and Notion deadlines for
        target_date, format a comprehensive briefing, and deliver it via DM.

        The import is deferred to call time (lazy import) so that:
          a) Tests can patch src.briefing.pipeline.run_aggregated_brief and
             have the patch take effect when _callback is invoked.
          b) Module-level import errors in pipeline.py do not prevent the
             bot from starting.
        """
        from src.briefing.pipeline import run_aggregated_brief
        run_aggregated_brief(
            target_date=target_date,
            bot=bot_holder[0],
            user_id=user_id,
        )

    return _callback, bot_holder


def _make_ask_callback():
    """
    Create a callback for the /ask slash command.

    Uses QAEngine to answer natural language questions by combining
    Calendar, Gmail, Notion, and Slack data (no web search).
    Sends the answer as a Slack DM.
    """
    def _callback(question: str, user_id: str, channel_id: str) -> None:
        from src.ai.qa_engine import get_qa_engine
        from slack_sdk import WebClient
        from src.config import SLACK_BOT_TOKEN

        engine = get_qa_engine()
        answer = engine.answer_question(question)

        client = WebClient(token=SLACK_BOT_TOKEN)
        client.chat_postMessage(
            channel=user_id,
            text=f"🤖 *업무 Q&A*\n\n*질문:* {question}\n\n{answer}",
        )
        logger.info("/ask answered for user=%s, question=%r", user_id, question[:60])

    return _callback


def _warm_history_cache() -> None:
    """
    Initialise the 1-year calendar history cache at daemon startup.

    Runs in a background daemon thread so the main startup sequence is not
    blocked by a potentially slow API fetch.  Errors are logged but never
    propagate — the event classifier degrades gracefully when the cache is
    unavailable.
    """
    import threading

    def _load():
        try:
            from src.calendar.google_calendar import GoogleCalendarClient
            from src.calendar.history_loader import initialize as init_history

            cal_client = GoogleCalendarClient()
            cal_client.connect()
            ok = init_history(cal_client)
            if ok:
                logger.info("Calendar history cache warmed successfully at startup.")
            else:
                logger.warning(
                    "Calendar history cache could not be warmed at startup; "
                    "event classifier will run without history context."
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Calendar history warm-up thread failed: %s — daemon continues.", exc
            )

    t = threading.Thread(target=_load, daemon=True, name="history-cache-warmup")
    t.start()
    logger.info("Calendar history cache warm-up started in background thread.")


def main() -> None:
    from src.slack.bot import WorkAssistantBot
    from src.scheduler import start_scheduler, stop_scheduler

    logger.info("=" * 60)
    logger.info("Work Assistant Daemon starting…")
    logger.info("=" * 60)

    # ── 1. Start Slack bot ─────────────────────────────────────────────────────
    # Build the briefing callback before the bot so we can pass it to the
    # constructor (avoiding double-registration).  The bot_holder is filled
    # in immediately after construction so the closure always has a valid ref.
    briefing_callback, bot_holder = _make_briefing_callback()
    ask_callback = _make_ask_callback()
    bot = WorkAssistantBot(
        briefing_callback=briefing_callback,
        ask_callback=ask_callback,
    )
    bot_holder[0] = bot   # callback now has a live bot reference
    bot.start_async()
    logger.info("Slack bot started (Socket Mode) with /brief, /ask callbacks")

    # ── 2. Warm the 1-year calendar history cache (background) ────────────────
    _warm_history_cache()

    # ── 3. Start scheduler ─────────────────────────────────────────────────────
    scheduler = start_scheduler(bot=bot)
    logger.info("APScheduler started:")
    logger.info("  • External-meeting checker — every 60 s")
    logger.info("  • Daily morning briefing   — 09:30 KST every day")
    logger.info("  • History cache refresh    — 02:00 KST every day")

    # ── 3. Graceful shutdown ───────────────────────────────────────────────────
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received (%s). Stopping…", signum)
        stop_scheduler()
        bot.stop()
        logger.info("Daemon stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Daemon running. Press Ctrl-C to stop.")

    # Keep the main thread alive
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
