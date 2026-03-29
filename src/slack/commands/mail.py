"""
/mail slash command handler (AC 15).

Usage:
    /mail          – show current portfolio email status (all pending, overdue, etc.)
    /mail scan     – force a fresh Gmail scan then show status
    /mail help     – show usage instructions

Flow:
    1. Slack sends a POST to the bot (Socket Mode – no public URL required).
    2. ack() is called immediately to satisfy Slack's 3-second timeout.
    3. A background thread runs the scan (if requested) and sends the status
       report as a Slack DM.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Any

from slack_bolt import App

logger = logging.getLogger(__name__)

_ACK_MESSAGES = {
    "default": "📮 포트폴리오사 메일 현황을 확인 중입니다…",
    "scan": "📮 Gmail을 스캔하고 메일 현황을 확인 중입니다…",
    "help": None,  # handled inline
}

_HELP_TEXT = (
    "📮 *포트폴리오사 메일 모니터 사용법*\n"
    "• `/mail` — 현재 추적 중인 메일 상태 확인\n"
    "• `/mail scan` — Gmail 새로 스캔 후 상태 확인\n"
    "• `/mail help` — 이 도움말 표시\n"
)


def register_mail_command(
    app: App,
    mail_callback: Optional[Callable[[str, str, bool], None]] = None,
) -> None:
    """
    Register the /mail slash command on the given Slack Bolt App instance.

    Parameters
    ----------
    app           : Slack Bolt App instance.
    mail_callback : Optional callback ``fn(user_id, channel_id, force_scan)``.
                    When provided it is invoked in a daemon background thread.
                    When omitted a placeholder DM is sent.
    """

    @app.command("/mail")
    def handle_mail(ack, body, client, logger=logger):  # noqa: ANN001
        text: str = (body.get("text") or "").strip().lower()
        user_id: str = body.get("user_id", "")
        channel_id: str = body.get("channel_id", "")

        logger.info(
            "/mail invoked by user=%s channel=%s text=%r",
            user_id, channel_id, text,
        )

        # Help
        if text in ("help", "도움말", "-h", "--help"):
            ack(_HELP_TEXT)
            return

        force_scan = text in ("scan", "스캔", "refresh")

        # Immediate ack
        if force_scan:
            ack(_ACK_MESSAGES["scan"])
        else:
            ack(_ACK_MESSAGES["default"])

        def _run_mail_check() -> None:
            try:
                if mail_callback is not None:
                    mail_callback(user_id, channel_id, force_scan)
                else:
                    _send_placeholder_dm(client, user_id)
            except Exception as exc:
                logger.exception(
                    "/mail background task failed for user=%s: %s", user_id, exc
                )
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text="⚠️ 메일 상태 확인 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    )
                except Exception:
                    logger.exception("Failed to send error DM to user=%s", user_id)

        thread = threading.Thread(
            target=_run_mail_check, daemon=True, name="mail-worker"
        )
        thread.start()
        logger.info("/mail background thread started for user=%s", user_id)


def _send_placeholder_dm(client: Any, user_id: str) -> None:
    """Send a placeholder DM when no mail callback is wired up."""
    client.chat_postMessage(
        channel=user_id,
        text=(
            "📮 *포트폴리오사 메일 모니터*\n\n"
            "⏳ 메일 모니터링 파이프라인이 아직 연결되지 않았습니다.\n"
            "이 메시지는 `/mail` 명령어 등록 확인용 placeholder 입니다.\n\n"
            "✅ 명령어 등록 및 인증 검증: 정상\n"
            "✅ 즉시 응답(ack): 정상\n"
            "✅ 백그라운드 처리 스레드: 정상"
        ),
    )
    logger.info("Placeholder DM sent to user=%s for /mail", user_id)
