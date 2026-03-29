"""
Natural language Q&A engine (Feature 3, ACs 17-19).

Answers questions by combining data from:
- Google Calendar (upcoming/recent meetings)
- Gmail (recent email threads)
- Notion (portfolio company deal records)
- Slack (priority channel history)

Does NOT use web search (AC 17 constraint: "without web search").

Response time target: ≤ 15 seconds (AC 17).

Automated task/follow-up suggestions are sent at 09:00, 12:00, 18:00 KST (AC 18).
The /ask slash command triggers immediate queries (AC 19).

Accuracy guarantee
------------------
- Only reports information found in the combined data context.
- Uses "확인 불가" annotation when data is unavailable (never fabricates).
- If any single source fails, the rest still contribute to the answer.
"""
from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

import anthropic

from src.config import (
    ANTHROPIC_API_KEY,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

# Claude model for Q&A (haiku for speed to meet 15s target)
_QA_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 1024
_CONTEXT_MAX_CHARS = 8000  # Trim total context to avoid token overflow

# Time periods for context gathering
_CALENDAR_LOOKAHEAD_DAYS = 7
_GMAIL_LOOKBACK_DAYS = 14
_NOTION_MAX_RECORDS = 10
_SLACK_LOOKBACK_DAYS = 30


class QAEngine:
    """
    Natural language Q&A over work data sources (Calendar, Gmail, Notion, Slack).

    Thread-safe; a single instance is shared across the daemon threads.
    """

    def __init__(
        self,
        calendar_client: Any = None,
        gmail_client: Any = None,
        notion_client: Any = None,
        slack_retriever: Any = None,
    ) -> None:
        self._calendar_client = calendar_client
        self._gmail_client = gmail_client
        self._notion_client = notion_client
        self._slack_retriever = slack_retriever
        self._anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._lock = threading.Lock()

    # ── Context gathering ──────────────────────────────────────────────────────

    def _gather_calendar_context(self) -> str:
        """Fetch upcoming meetings for the next 7 days."""
        if self._calendar_client is None:
            return "[캘린더: 클라이언트 미설정]"
        try:
            client = self._calendar_client
            now = datetime.now(timezone.utc)
            end = now + timedelta(days=_CALENDAR_LOOKAHEAD_DAYS)
            events = client.list_upcoming_events(time_min=now, time_max=end, max_results=20)
            if not events:
                return "[캘린더: 향후 7일 내 일정 없음]"

            lines = ["[캘린더 - 향후 7일 일정]"]
            for event in events[:10]:
                start = getattr(event, "start", None)
                title = getattr(event, "summary", "(제목 없음)")
                attendees = getattr(event, "attendees", [])
                ext = [a.email for a in attendees if not getattr(a, "is_internal", True)]
                start_str = start.strftime("%Y-%m-%d %H:%M") if start else "미상"
                lines.append(
                    f"- {start_str}: {title}"
                    + (f" (외부참석자: {', '.join(ext[:3])})" if ext else "")
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Calendar context failed: %s", exc)
            return f"[캘린더: 오류 — {exc}]"

    def _gather_gmail_context(self) -> str:
        """Fetch recent email threads from the last 14 days."""
        if self._gmail_client is None:
            return "[Gmail: 클라이언트 미설정]"
        try:
            threads = self._gmail_client.search_threads(
                query=f"newer_than:{_GMAIL_LOOKBACK_DAYS}d",
                max_results=15,
            )
            if not threads:
                return "[Gmail: 최근 14일 이내 이메일 없음]"

            lines = ["[Gmail - 최근 이메일 요약]"]
            for thread in threads[:10]:
                msgs = getattr(thread, "messages", [])
                if not msgs:
                    continue
                first = msgs[0]
                subject = getattr(first, "subject", "(제목 없음)")
                sender = getattr(first, "sender", "")
                snippet = getattr(first, "snippet", "")[:100]
                lines.append(f"- [{sender[:30]}] {subject}: {snippet}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Gmail context failed: %s", exc)
            return f"[Gmail: 오류 — {exc}]"

    def _gather_notion_context(self) -> str:
        """Fetch recent Notion portfolio records."""
        if self._notion_client is None:
            return "[Notion: 클라이언트 미설정]"
        try:
            records = self._notion_client.query_database(max_results=_NOTION_MAX_RECORDS)
            if not records:
                return "[Notion: 포트폴리오 레코드 없음]"

            lines = ["[Notion 딜 DB - 최근 레코드]"]
            for record in records[:_NOTION_MAX_RECORDS]:
                company = getattr(record, "company_name", "") or getattr(record, "title", "")
                status = getattr(record, "status", "")
                lines.append(f"- {company}: {status}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Notion context failed: %s", exc)
            return f"[Notion: 오류 — {exc}]"

    def _gather_slack_context(self, query_hint: str = "") -> str:
        """Search priority Slack channels for recent activity."""
        if self._slack_retriever is None:
            return "[Slack: 클라이언트 미설정]"
        try:
            # Search with the query hint if provided
            search_term = query_hint or "투자"
            result = self._slack_retriever.search_company_history(
                company_name=search_term,
                lookback_days=_SLACK_LOOKBACK_DAYS,
                max_messages_per_channel=10,
            )
            messages = getattr(result, "messages", [])
            if not messages:
                return "[Slack: 최근 30일 내 관련 메시지 없음]"

            lines = ["[Slack - 최근 메시지]"]
            for msg in messages[:10]:
                text = getattr(msg, "text", "")[:120]
                channel = getattr(msg, "channel_name", "")
                lines.append(f"- #{channel}: {text}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Slack context failed: %s", exc)
            return f"[Slack: 오류 — {exc}]"

    def _gather_all_context(self, query: str = "") -> str:
        """
        Gather context from all data sources in parallel.

        Returns a combined string (trimmed to _CONTEXT_MAX_CHARS).
        Thread-safe: uses a lock for shared results dict; timed-out threads
        contribute fallback values rather than partial data.
        """
        results: dict[str, str] = {}
        results_lock = threading.Lock()

        def _fetch(name: str, fn):
            try:
                value = fn()
            except Exception as exc:
                value = f"[{name}: 오류 — {exc}]"
            with results_lock:
                results[name] = value

        # Gather in parallel for speed (target: all complete within 10s)
        threads = [
            threading.Thread(target=_fetch, args=("calendar", self._gather_calendar_context), daemon=True),
            threading.Thread(target=_fetch, args=("gmail", self._gather_gmail_context), daemon=True),
            threading.Thread(target=_fetch, args=("notion", self._gather_notion_context), daemon=True),
            threading.Thread(target=_fetch, args=("slack", lambda: self._gather_slack_context(query)), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        with results_lock:
            combined = "\n\n".join([
                results.get("calendar", "[캘린더: 확인 불가]"),
                results.get("gmail", "[Gmail: 확인 불가]"),
                results.get("notion", "[Notion: 확인 불가]"),
                results.get("slack", "[Slack: 확인 불가]"),
            ])

        # Trim to prevent token overflow
        if len(combined) > _CONTEXT_MAX_CHARS:
            combined = combined[:_CONTEXT_MAX_CHARS] + "\n[... 이하 컨텍스트 생략]"

        return combined

    # ── Claude API call ────────────────────────────────────────────────────────

    def _call_claude(self, system_prompt: str, user_message: str) -> str:
        """Call Claude with retry and return the response text."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                response = self._anthropic.messages.create(
                    model=_QA_MODEL,
                    max_tokens=_MAX_TOKENS,
                    messages=[{"role": "user", "content": user_message}],
                    system=system_prompt,
                )
                text_blocks = [
                    b.text for b in response.content
                    if hasattr(b, "text")
                ]
                return "\n".join(text_blocks).strip()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Claude API call failed (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                if attempt < API_RETRY_ATTEMPTS:
                    time.sleep(API_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"Claude API failed after {API_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    # ── Public API (AC 17, 19) ────────────────────────────────────────────────

    def answer_question(self, question: str) -> str:
        """
        Answer a natural language question using work data (no web search).

        Gathers context from Calendar, Gmail, Notion, and Slack, then asks
        Claude to answer based solely on that context.

        Response time target: ≤ 15 seconds.

        Parameters
        ----------
        question : Natural language question string.

        Returns
        -------
        Answer string (Korean/English) with '확인 불가' for missing info.
        """
        start_time = time.monotonic()

        context = self._gather_all_context(query=question)

        system_prompt = (
            "당신은 개인 업무 어시스턴트입니다. "
            "아래 제공된 데이터(캘린더, 이메일, Notion, Slack)만을 근거로 질문에 답변하세요.\n"
            "- 제공된 데이터에 없는 정보는 '확인 불가'로 표시하세요.\n"
            "- 웹 검색은 하지 마세요.\n"
            "- 답변은 한국어 또는 영어로, 간결하고 명확하게 작성하세요.\n"
            "- 정확한 정보만 전달하고 추측은 명시적으로 표시하세요.\n\n"
            f"[업무 데이터 컨텍스트]\n{context}"
        )

        user_message = f"질문: {question}"

        try:
            answer = self._call_claude(system_prompt, user_message)
        except RuntimeError as exc:
            answer = f"⚠️ AI 응답 오류: {exc}\n확인 불가"

        elapsed = time.monotonic() - start_time
        logger.info(
            "QAEngine.answer_question: elapsed=%.1fs, question=%r",
            elapsed, question[:80],
        )

        return answer

    # ── Task suggestions (AC 18) ──────────────────────────────────────────────

    def generate_task_suggestions(self) -> str:
        """
        Generate automated task and follow-up suggestions.

        Combines upcoming calendar events, recent emails, and pending Notion
        deals to suggest what needs attention.

        Returns a formatted Slack message string.
        Called by the scheduler at 09:00, 12:00, 18:00 KST.
        """
        now = datetime.now(timezone.utc)
        kst_offset = timedelta(hours=9)
        now_kst = now + kst_offset
        time_label = f"{now_kst.strftime('%H:%M')} KST"

        context = self._gather_all_context()

        system_prompt = (
            "당신은 투자팀 업무 어시스턴트입니다. "
            "아래 업무 데이터를 분석하여 지금 당장 해야 할 작업과 팔로업 항목을 제안하세요.\n"
            "- 오늘의 미팅 준비, 미회신 이메일, 주목할 딜 등을 포함하세요.\n"
            "- 제공된 데이터에만 근거하세요 (웹 검색 없음).\n"
            "- 불릿 포인트(•) 형식으로 3-7개 항목을 작성하세요.\n"
            "- 각 항목은 간결하고 실행 가능한 형태로 작성하세요.\n"
            "- 데이터가 없는 항목은 생략하세요 (빈 제안 금지).\n\n"
            f"[업무 데이터 컨텍스트]\n{context}"
        )

        user_message = f"현재 시각 {time_label} 기준으로 지금 처리해야 할 업무와 팔로업을 알려주세요."

        try:
            suggestions = self._call_claude(system_prompt, user_message)
            if not suggestions.strip():
                return ""

            return (
                f"📋 *업무 제안 및 팔로업* ({time_label})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{suggestions}"
            )
        except RuntimeError as exc:
            logger.error("generate_task_suggestions failed: %s", exc)
            return ""


# ── Module-level singleton ────────────────────────────────────────────────────

_qa_singleton: Optional[QAEngine] = None
_qa_lock = threading.Lock()


def get_qa_engine(
    calendar_client: Any = None,
    gmail_client: Any = None,
    notion_client: Any = None,
    slack_retriever: Any = None,
) -> QAEngine:
    """Return the module-level QAEngine singleton."""
    global _qa_singleton
    if _qa_singleton is None:
        with _qa_lock:
            if _qa_singleton is None:
                # Lazy-initialize real clients if not provided
                if calendar_client is None:
                    try:
                        from src.calendar.google_calendar import GoogleCalendarClient
                        calendar_client = GoogleCalendarClient()
                    except Exception as exc:
                        logger.warning("QAEngine: failed to init CalendarClient: %s", exc)

                if gmail_client is None:
                    try:
                        from src.gmail.gmail_client import GmailClient
                        gmail_client = GmailClient()
                    except Exception as exc:
                        logger.warning("QAEngine: failed to init GmailClient: %s", exc)

                if notion_client is None:
                    try:
                        from src.notion.notion_client import NotionClient
                        notion_client = NotionClient()
                    except Exception as exc:
                        logger.warning("QAEngine: failed to init NotionClient: %s", exc)

                if slack_retriever is None:
                    try:
                        from src.slack.history_retriever import SlackHistoryRetriever
                        slack_retriever = SlackHistoryRetriever()
                    except Exception as exc:
                        logger.warning("QAEngine: failed to init SlackHistoryRetriever: %s", exc)

                _qa_singleton = QAEngine(
                    calendar_client=calendar_client,
                    gmail_client=gmail_client,
                    notion_client=notion_client,
                    slack_retriever=slack_retriever,
                )
    return _qa_singleton
