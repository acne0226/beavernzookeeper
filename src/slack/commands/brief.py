"""
/brief slash command handler.

Usage:
    /brief          – today's meeting briefing (default)
    /brief tomorrow – tomorrow's meeting briefing
    /brief YYYY-MM-DD – briefing for a specific date

Flow:
    1. Slack sends a POST to the bot (Socket Mode – no public URL required).
    2. Slack Bolt verifies the request automatically (signing secret / app token).
    3. ack() is called immediately to satisfy the 3-second timeout.
    4. A background thread generates the briefing and sends it as a Slack DM.

Request verification in Socket Mode:
    All messages arrive over a WebSocket authenticated with SLACK_APP_TOKEN
    (xapp-…).  Slack Bolt's SocketModeHandler validates the envelope before
    dispatching, so no manual HMAC verification is needed.  When the bot is
    additionally configured with SLACK_SIGNING_SECRET the Bolt App class
    enforces signature checks on any HTTP endpoint as well.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from typing import Callable

from slack_bolt import App

logger = logging.getLogger(__name__)

# ── Human-readable ack messages ───────────────────────────────────────────────
_ACK_MESSAGES = {
    "default": "📋 브리핑을 준비 중입니다… 잠시 후 DM으로 전달해 드릴게요.",
    "tomorrow": "📋 내일 브리핑을 준비 중입니다… 잠시 후 DM으로 전달해 드릴게요.",
    "date": "📋 {date} 브리핑을 준비 중입니다… 잠시 후 DM으로 전달해 드릴게요.",
    "unknown_arg": (
        "⚠️ 인식할 수 없는 날짜 형식입니다. "
        "`/brief`, `/brief tomorrow`, `/brief YYYY-MM-DD` 형식을 사용해 주세요."
    ),
}


def _parse_brief_date(text: str) -> tuple[date | None, str | None]:
    """
    Parse the optional text argument of /brief.

    Returns:
        (target_date, error_message)
        – on success  : (date object, None)
        – on bad input: (None,        error string)
    """
    text = (text or "").strip().lower()

    if text in ("", "today"):
        return date.today(), None

    if text == "tomorrow":
        return date.today() + timedelta(days=1), None

    # Try YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date(), None
        except ValueError:
            continue

    return None, _ACK_MESSAGES["unknown_arg"]


def register_brief_command(
    app: App,
    briefing_callback: Callable[[date, str, str], None] | None = None,
) -> None:
    """
    Register the /brief slash command on the given Slack Bolt App instance.

    Args:
        app:               The Slack Bolt App to attach the handler to.
        briefing_callback: Optional async work function
                           ``fn(target_date, user_id, channel_id)``.
                           When provided it is invoked in a daemon thread after
                           the immediate ack so that Slack's 3-second timeout
                           is never triggered.
                           When omitted a placeholder DM is sent instead (useful
                           during development / testing of this sub-AC alone).
    """

    @app.command("/brief")
    def handle_brief(ack, body, client, logger=logger):  # noqa: ANN001
        """
        Handle /brief slash command.

        Steps
        -----
        1. Parse the optional date argument from `body["text"]`.
        2. Call ack() immediately – this satisfies Slack's 3-second rule and
           displays a transient in-channel acknowledgement to the invoker.
        3. Spin up a background thread to do the heavy lifting and DM the result.
        """
        text: str = body.get("text", "")
        user_id: str = body.get("user_id", "")
        channel_id: str = body.get("channel_id", "")

        logger.info(
            "/brief invoked by user=%s channel=%s text=%r",
            user_id,
            channel_id,
            text,
        )

        # ── Step 1: parse the date argument ──────────────────────────────────
        target_date, parse_error = _parse_brief_date(text)

        if parse_error:
            # Bad argument – acknowledge with an error message immediately
            ack(parse_error)
            logger.warning(
                "/brief rejected for user=%s: bad arg %r", user_id, text
            )
            return

        # ── Step 2: immediate acknowledgement ────────────────────────────────
        if target_date == date.today():
            ack_text = _ACK_MESSAGES["default"]
        elif target_date == date.today() + timedelta(days=1):
            ack_text = _ACK_MESSAGES["tomorrow"]
        else:
            ack_text = _ACK_MESSAGES["date"].format(
                date=target_date.strftime("%Y-%m-%d")
            )

        ack(ack_text)  # Must be called within 3 seconds; nothing blocking above

        # ── Step 3: background processing ────────────────────────────────────
        def _run_briefing() -> None:
            try:
                if briefing_callback is not None:
                    briefing_callback(target_date, user_id, channel_id)
                else:
                    # Placeholder used when no callback has been wired up yet
                    _send_placeholder_dm(client, user_id, target_date)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    "/brief background task failed for user=%s date=%s: %s",
                    user_id,
                    target_date,
                    exc,
                )
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text=(
                            f"⚠️ 브리핑 생성 중 오류가 발생했습니다 ({target_date}). "
                            "잠시 후 다시 시도해 주세요."
                        ),
                    )
                except Exception:  # pylint: disable=broad-except
                    logger.exception(
                        "Failed to send error DM to user=%s", user_id
                    )

        thread = threading.Thread(target=_run_briefing, daemon=True, name="brief-worker")
        thread.start()

        logger.info(
            "/brief background thread started for user=%s date=%s",
            user_id,
            target_date,
        )


def _send_placeholder_dm(client, user_id: str, target_date: date) -> None:
    """
    Send a placeholder DM when no briefing callback has been registered yet.
    Used during development of sub-AC 3.1 before the full briefing pipeline
    (sub-ACs 3.2+) is wired up.
    """
    date_label = (
        "오늘" if target_date == date.today()
        else "내일" if target_date == date.today() + timedelta(days=1)
        else target_date.strftime("%Y-%m-%d")
    )

    client.chat_postMessage(
        channel=user_id,
        text=(
            f"📋 *{date_label} 미팅 브리핑* ({target_date})\n\n"
            "⏳ 브리핑 파이프라인이 아직 연결되지 않았습니다. "
            "이 메시지는 `/brief` 명령어 등록 확인용 placeholder 입니다.\n\n"
            "✅ 명령어 등록 및 인증 검증: 정상\n"
            "✅ 즉시 응답(ack): 정상\n"
            "✅ 백그라운드 처리 스레드: 정상"
        ),
    )
    logger.info(
        "Placeholder DM sent to user=%s for date=%s", user_id, target_date
    )
