"""
Slack Bot for the Work Assistant Daemon.

Provides:
  WorkAssistantBot  – Slack Bolt App with Socket Mode + slash-command handling
                      and convenience DM helpers.

Architecture
------------
* Uses Slack Bolt (slack_bolt) for slash-command registration and built-in
  request verification.
* Socket Mode (SLACK_APP_TOKEN / xapp-…) means no public HTTP endpoint is
  required; all events arrive over a Bolt-managed WebSocket.
* Request authenticity is guaranteed automatically:
    - Socket Mode: Bolt's SocketModeHandler validates the envelope signature
      using the xapp token before dispatching any event to our code.
    - HTTP mode (future): Bolt checks X-Slack-Signature via SLACK_SIGNING_SECRET.
* Slash commands must call ack() within 3 seconds; heavy work is offloaded to
  daemon threads so the ack is never delayed.

Lifecycle
---------
    bot = WorkAssistantBot()
    bot.start_async()     # non-blocking; useful with APScheduler
    ...
    bot.stop()
    # or
    bot.start()           # blocks – used when the bot is the only component
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Callable, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from src.config import (
    SLACK_APP_TOKEN,
    SLACK_BOT_TOKEN,
    SLACK_TARGET_USER_ID,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)
from src.slack.commands.brief import register_brief_command
from src.slack.commands.mail import register_mail_command

logger = logging.getLogger(__name__)


class WorkAssistantBot:
    """
    Slack Bolt App wrapper with Socket Mode and DM helpers.

    Parameters
    ----------
    briefing_callback:
        Optional callable invoked in a background thread when /brief fires.
        Signature: ``fn(target_date: date, user_id: str, channel_id: str) -> None``
        When omitted, a placeholder DM is sent (useful for sub-AC 3.1 testing).
    """

    def __init__(
        self,
        briefing_callback: Callable[[date, str, str], None] | None = None,
        mail_callback: Callable[[str, str, bool], None] | None = None,
        ask_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._briefing_callback = briefing_callback
        self._mail_callback = mail_callback
        self._ask_callback = ask_callback
        self._handler: Optional[SocketModeHandler] = None
        self._thread: Optional[threading.Thread] = None
        self._dm_channel: Optional[str] = None

        # ── Build the Bolt App ────────────────────────────────────────────────
        # Bolt uses the bot token for all API calls.
        # Signing secret is optional in Socket Mode but validated when present.
        # token_verification_enabled=False: in Socket Mode the xapp token
        # authenticates the WebSocket connection; no separate HTTP auth.test
        # call is needed (and avoids SSL issues in restricted environments).
        self.app = App(
            token=SLACK_BOT_TOKEN,
            token_verification_enabled=False,
            raise_error_for_unhandled_request=False,
        )

        # Expose the underlying WebClient for convenience
        self._client = self.app.client

        # ── Register slash commands ───────────────────────────────────────────
        register_brief_command(
            self.app,
            briefing_callback=self._briefing_callback,
        )
        register_mail_command(
            self.app,
            mail_callback=self._mail_callback,
        )
        # /ask command is registered by Feature 3 module
        if self._ask_callback is not None:
            from src.slack.commands.ask import register_ask_command
            register_ask_command(self.app, ask_callback=self._ask_callback)

        # ── Global error handler ──────────────────────────────────────────────
        @self.app.error
        def _global_error(error, body, logger=logger):  # noqa: ANN001
            logger.exception(
                "Unhandled Bolt error: %s | body: %s", error, body
            )

        logger.info(
            "WorkAssistantBot initialised — /brief, /mail commands registered"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Socket Mode handler and block (e.g. when bot is the main loop)."""
        logger.info("WorkAssistantBot starting (blocking)…")
        handler = SocketModeHandler(self.app, SLACK_APP_TOKEN)
        self._handler = handler
        handler.start()  # blocks until stop() is called

    def start_async(self) -> None:
        """
        Start the Socket Mode handler in a background daemon thread.
        Returns immediately; the bot continues running until stop() is called.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WorkAssistantBot is already running — ignoring start_async()")
            return

        handler = SocketModeHandler(self.app, SLACK_APP_TOKEN)
        self._handler = handler

        def _run() -> None:
            logger.info("WorkAssistantBot Socket-Mode thread started")
            handler.start()

        self._thread = threading.Thread(
            target=_run, daemon=True, name="slack-bot-socket"
        )
        self._thread.start()
        logger.info("WorkAssistantBot started in background thread")

    def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._handler is not None:
            try:
                self._handler.close()
                logger.info("WorkAssistantBot stopped")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error stopping WorkAssistantBot: %s", exc)
        self._handler = None

    # ── DM helpers ────────────────────────────────────────────────────────────

    def _get_dm_channel(self) -> str:
        """Open (or retrieve cached) DM channel with SLACK_TARGET_USER_ID."""
        if self._dm_channel:
            return self._dm_channel

        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                resp = self._client.conversations_open(users=[SLACK_TARGET_USER_ID])
                self._dm_channel = resp["channel"]["id"]
                logger.info(
                    "DM channel with %s: %s",
                    SLACK_TARGET_USER_ID,
                    self._dm_channel,
                )
                return self._dm_channel
            except SlackApiError as exc:
                logger.warning(
                    "conversations_open failed (attempt %d/%d): %s",
                    attempt,
                    API_RETRY_ATTEMPTS,
                    exc.response["error"],
                )
                if attempt < API_RETRY_ATTEMPTS:
                    time.sleep(API_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            "Cannot open DM channel with target user after retries"
        )

    def send_message(self, text: str, blocks: Optional[list] = None) -> bool:
        """
        Send *text* (and optional *blocks*) to the target user's DM.

        Retries up to API_RETRY_ATTEMPTS times with API_RETRY_DELAY_SECONDS
        between attempts.  Returns True on success.
        """
        channel = self._get_dm_channel()

        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                kwargs: dict = {"channel": channel, "text": text}
                if blocks:
                    kwargs["blocks"] = blocks
                self._client.chat_postMessage(**kwargs)
                return True
            except SlackApiError as exc:
                logger.warning(
                    "chat_postMessage failed (attempt %d/%d): %s",
                    attempt,
                    API_RETRY_ATTEMPTS,
                    exc.response["error"],
                )
                if attempt < API_RETRY_ATTEMPTS:
                    time.sleep(API_RETRY_DELAY_SECONDS)

        logger.error(
            "Failed to send Slack DM after %d attempts", API_RETRY_ATTEMPTS
        )
        return False

    def send_dm(self, text: str, user_id: Optional[str] = None) -> None:
        """
        Send a DM to the target user (or an explicit user_id).
        Raises on failure (callers that need best-effort should use send_message).
        """
        target = user_id or SLACK_TARGET_USER_ID
        try:
            self._client.chat_postMessage(channel=target, text=text)
        except SlackApiError as exc:
            logger.error("Failed to send DM to %s: %s", target, exc)
            raise

    def send_error(self, context: str, error: Exception) -> None:
        """Send an API-failure notification DM (best-effort, no retry loop)."""
        text = f":warning: *API 오류 발생*\n컨텍스트: {context}\n오류: `{error}`"
        try:
            channel = self._get_dm_channel()
            self._client.chat_postMessage(channel=channel, text=text)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Could not send error DM")
