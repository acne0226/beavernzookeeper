"""
Slack DM Sender Utility.

Provides a standalone, lightweight utility for delivering formatted Slack DM
messages to the authenticated user.  Can be used independently of the full
Bolt App (e.g. in tests or CLI scripts) or alongside WorkAssistantBot.

Architecture
------------
* ``SlackDMSender`` wraps ``slack_sdk.WebClient`` directly — no Bolt overhead.
* DM channel IDs are cached after the first ``conversations_open`` call.
* Every ``send()`` call retries up to ``API_RETRY_ATTEMPTS`` times with
  ``API_RETRY_DELAY_SECONDS`` between attempts, then reports failure.

Main public interface
---------------------
::

    sender = SlackDMSender()                        # reads from config
    ok     = sender.send("Hello!")                  # plain text
    ok     = sender.send(text, blocks=blocks)       # Block Kit message
    sender.send_error("calendar fetch", exc)        # best-effort error DM

Module-level helper
-------------------
::

    from src.slack.dm_sender import send_daily_briefing_dm

    ok = send_daily_briefing_dm(bot, events, target_date=date.today())

    # Equivalent to:
    #   text, blocks = format_daily_briefing(events, target_date)
    #   bot.send_message(text, blocks=blocks)

Design choices
--------------
* The class is intentionally thin: it owns *delivery* logic only; content
  formatting lives in ``src/briefing/formatter.py``.
* ``send_error`` is best-effort (no retry) so it never blocks or raises —
  useful in ``except`` blocks where a secondary failure would be confusing.
* Retry delay is skipped when ``_retry_delay_override`` is set (tests inject
  0 to keep the suite fast).
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config import (
    SLACK_BOT_TOKEN,
    SLACK_TARGET_USER_ID,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


class SlackDMSender:
    """
    Standalone Slack DM sender for the Work Assistant daemon.

    Parameters
    ----------
    token:
        Slack bot token (``xoxb-…``).  Defaults to ``SLACK_BOT_TOKEN`` from
        the environment / ``.env`` file.
    target_user_id:
        Slack user ID of the DM recipient.  Defaults to
        ``SLACK_TARGET_USER_ID`` from config.
    retry_attempts:
        How many times to retry on ``SlackApiError``.  Defaults to
        ``API_RETRY_ATTEMPTS`` (3).
    retry_delay:
        Seconds to wait between retries.  Defaults to
        ``API_RETRY_DELAY_SECONDS`` (10).  Tests may inject 0 for speed.
    """

    def __init__(
        self,
        token: str = SLACK_BOT_TOKEN,
        target_user_id: str = SLACK_TARGET_USER_ID,
        retry_attempts: int = API_RETRY_ATTEMPTS,
        retry_delay: float = API_RETRY_DELAY_SECONDS,
    ) -> None:
        self._client = WebClient(token=token)
        self._target_user_id = target_user_id
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._dm_channel: Optional[str] = None

    # ── DM channel management ──────────────────────────────────────────────────

    def _get_dm_channel(self) -> str:
        """
        Open (or retrieve cached) DM channel with the target user.

        Retries up to ``retry_attempts`` on ``SlackApiError``.

        Raises
        ------
        RuntimeError
            When all retry attempts are exhausted.
        """
        if self._dm_channel:
            return self._dm_channel

        for attempt in range(1, self._retry_attempts + 1):
            try:
                resp = self._client.conversations_open(users=[self._target_user_id])
                self._dm_channel = resp["channel"]["id"]
                logger.info(
                    "SlackDMSender: DM channel with %s → %s",
                    self._target_user_id,
                    self._dm_channel,
                )
                return self._dm_channel
            except SlackApiError as exc:
                error_code = exc.response.get("error", str(exc))
                logger.warning(
                    "conversations_open failed (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    error_code,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay)

        raise RuntimeError(
            f"Cannot open DM channel with {self._target_user_id} "
            f"after {self._retry_attempts} attempts"
        )

    # ── Public send methods ────────────────────────────────────────────────────

    def send(self, text: str, blocks: Optional[list] = None) -> bool:
        """
        Send *text* (and optional Block Kit *blocks*) to the target user's DM.

        Parameters
        ----------
        text:
            Plain-text fallback.  Shown when Block Kit is unavailable (e.g.
            in notifications) and when *blocks* is ``None``.
        blocks:
            Slack Block Kit payload.  When provided the message renders as a
            rich Block Kit message; *text* is used as the notification text.

        Returns
        -------
        bool
            ``True`` on success, ``False`` after all retry attempts failed.
        """
        try:
            channel = self._get_dm_channel()
        except RuntimeError as exc:
            logger.error("SlackDMSender.send: could not open DM channel — %s", exc)
            return False

        kwargs: dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            kwargs["blocks"] = blocks

        for attempt in range(1, self._retry_attempts + 1):
            try:
                self._client.chat_postMessage(**kwargs)
                logger.debug(
                    "SlackDMSender: message sent to %s (blocks=%s)",
                    channel,
                    bool(blocks),
                )
                return True
            except SlackApiError as exc:
                error_code = exc.response.get("error", str(exc))
                logger.warning(
                    "chat_postMessage failed (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    error_code,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay)

        logger.error(
            "SlackDMSender: failed to deliver message after %d attempts",
            self._retry_attempts,
        )
        return False

    def send_error(self, context: str, error: Exception) -> None:
        """
        Send a best-effort error notification DM.

        Never raises — designed to be called safely inside ``except`` blocks.

        Parameters
        ----------
        context:
            Human-readable description of where the error occurred.
        error:
            The exception that triggered the notification.
        """
        text = (
            f":warning: *API 오류 발생*\n"
            f"컨텍스트: {context}\n"
            f"오류: `{error}`"
        )
        try:
            channel = self._get_dm_channel()
            self._client.chat_postMessage(channel=channel, text=text)
            logger.debug("SlackDMSender: error notification sent")
        except Exception:  # pylint: disable=broad-except
            logger.exception("SlackDMSender: could not send error notification")


# ── Module-level convenience helper ───────────────────────────────────────────


def send_daily_briefing_dm(
    bot,
    events: list,
    target_date: Optional[date] = None,
) -> bool:
    """
    Format *events* into a Slack daily briefing and send it via *bot*.

    This is the primary integration point for the ``/brief`` command and the
    scheduler-triggered briefing pipeline.

    Parameters
    ----------
    bot:
        A ``WorkAssistantBot`` instance (or any object that exposes
        ``send_message(text, blocks=...)``).
    events:
        List of calendar events.  Each item may be either a ``Meeting``
        dataclass (``src.calendar.google_calendar``) or a ``dict`` from
        ``src.calendar_fetcher``.  An empty list produces an "no events"
        briefing — never raises.
    target_date:
        The calendar date being briefed.  When ``None`` the date is inferred
        from the first event's start time (KST); falls back to today.

    Returns
    -------
    bool
        ``True`` if the Slack message was delivered successfully.

    Example
    -------
    ::

        from src.slack.dm_sender import send_daily_briefing_dm
        from datetime import date

        ok = send_daily_briefing_dm(bot, meetings, target_date=date.today())
        if not ok:
            logger.error("Failed to deliver daily briefing DM")
    """
    from src.briefing.formatter import format_daily_briefing

    fallback_text, blocks = format_daily_briefing(events, target_date=target_date)
    logger.info(
        "send_daily_briefing_dm: date=%s  events=%d  blocks=%d",
        target_date,
        len(events),
        len(blocks),
    )
    return bot.send_message(fallback_text, blocks=blocks)
