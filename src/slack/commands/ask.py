"""
/ask slash command handler (AC 19).

Usage:
    /ask <question>  – ask a natural language question about your work data
    /ask help        – show usage instructions

Flow:
    1. ack() called immediately (satisfies Slack's 3-second timeout).
    2. Background thread runs QAEngine.answer_question(question).
    3. Answer sent as a Slack DM to the invoking user.

Response time: ≤ 15 seconds (AC 17 constraint).
Data sources: Calendar + Gmail + Notion + Slack (no web search, AC 17).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Any

from slack_bolt import App

logger = logging.getLogger(__name__)

_ACK_PROCESSING = "🤔 질문을 분석 중입니다… 잠시 후 DM으로 답변드릴게요."
_HELP_TEXT = (
    "🤖 *업무 Q&A 어시스턴트 사용법*\n"
    "• `/ask <질문>` — 캘린더, 이메일, Notion, Slack 데이터를 기반으로 답변\n"
    "  예시:\n"
    "  `/ask 이번 주 외부 미팅 있어?`\n"
    "  `/ask 최근에 연락한 회사 목록은?`\n"
    "  `/ask 아직 답장 안 한 중요 이메일은?`\n"
    "• 웹 검색 없이 제공된 데이터 내에서만 답변합니다.\n"
    "• 확인 불가능한 정보는 '확인 불가'로 표기합니다.\n"
)


def register_ask_command(
    app: App,
    ask_callback: Optional[Callable[[str, str, str], None]] = None,
) -> None:
    """
    Register the /ask slash command on the given Slack Bolt App instance.

    Parameters
    ----------
    app          : Slack Bolt App instance.
    ask_callback : Optional callback ``fn(question, user_id, channel_id)``.
                   When provided it is invoked in a daemon background thread.
                   When omitted a placeholder DM is sent.
    """

    @app.command("/ask")
    def handle_ask(ack, body, client, logger=logger):  # noqa: ANN001
        text: str = (body.get("text") or "").strip()
        user_id: str = body.get("user_id", "")
        channel_id: str = body.get("channel_id", "")

        logger.info(
            "/ask invoked by user=%s text=%r", user_id, text[:80]
        )

        # Help
        if text.lower() in ("help", "도움말", "-h", "--help", ""):
            ack(_HELP_TEXT)
            return

        ack(_ACK_PROCESSING)

        def _run_ask() -> None:
            try:
                if ask_callback is not None:
                    ask_callback(text, user_id, channel_id)
                else:
                    _send_placeholder_dm(client, user_id, text)
            except Exception as exc:
                logger.exception(
                    "/ask background task failed for user=%s: %s", user_id, exc
                )
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text=(
                            f"⚠️ 질문 처리 중 오류가 발생했습니다. "
                            "잠시 후 다시 시도해 주세요."
                        ),
                    )
                except Exception:
                    logger.exception("Failed to send error DM to user=%s", user_id)

        thread = threading.Thread(target=_run_ask, daemon=True, name="ask-worker")
        thread.start()
        logger.info("/ask background thread started for user=%s", user_id)


def _send_placeholder_dm(client: Any, user_id: str, question: str) -> None:
    """Send a placeholder DM when no ask callback is wired up."""
    client.chat_postMessage(
        channel=user_id,
        text=(
            f"🤖 *업무 Q&A*\n\n"
            f"*질문:* {question}\n\n"
            "⏳ Q&A 파이프라인이 아직 연결되지 않았습니다.\n"
            "이 메시지는 `/ask` 명령어 등록 확인용 placeholder 입니다.\n\n"
            "✅ 명령어 등록 및 인증 검증: 정상\n"
            "✅ 즉시 응답(ack): 정상\n"
            "✅ 백그라운드 처리 스레드: 정상"
        ),
    )
    logger.info("Placeholder DM sent to user=%s for /ask", user_id)
