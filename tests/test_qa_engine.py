"""
Tests for src/ai/qa_engine.py (ACs 17-19).

Coverage:
- QAEngine.answer_question(): calls Claude with combined context
- QAEngine.generate_task_suggestions(): returns formatted suggestions
- Context gathering: calendar, gmail, notion, slack (with mocked clients)
- API failure: retry + graceful degradation
- Response time: within 15 seconds (mocked Claude)
- No web search used (verified by absence of web_search calls)
- "확인 불가" annotation when data unavailable
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from src.ai.qa_engine import QAEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_engine(
    calendar_client=None,
    gmail_client=None,
    notion_client=None,
    slack_retriever=None,
) -> QAEngine:
    """Build a QAEngine with mock dependencies."""
    engine = QAEngine(
        calendar_client=calendar_client,
        gmail_client=gmail_client,
        notion_client=notion_client,
        slack_retriever=slack_retriever,
    )
    return engine


def _mock_claude_response(text: str):
    """Return a mock Anthropic message response."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ── Context gathering ─────────────────────────────────────────────────────────

class TestCalendarContext:
    def test_no_client_returns_placeholder(self):
        engine = _make_engine()
        result = engine._gather_calendar_context()
        assert "클라이언트 미설정" in result

    def test_empty_events_returns_empty_message(self):
        cal = MagicMock()
        cal.list_upcoming_events.return_value = []
        engine = _make_engine(calendar_client=cal)
        result = engine._gather_calendar_context()
        assert "일정 없음" in result or "없음" in result

    def test_events_returned_in_context(self):
        cal = MagicMock()
        event = MagicMock()
        event.start = datetime.now(timezone.utc) + timedelta(hours=2)
        event.summary = "Test Meeting"
        event.attendees = []
        cal.list_upcoming_events.return_value = [event]
        engine = _make_engine(calendar_client=cal)
        result = engine._gather_calendar_context()
        assert "Test Meeting" in result

    def test_exception_returns_error_message(self):
        cal = MagicMock()
        cal.list_upcoming_events.side_effect = RuntimeError("Calendar API down")
        engine = _make_engine(calendar_client=cal)
        result = engine._gather_calendar_context()
        assert "오류" in result


class TestGmailContext:
    def test_no_client_returns_placeholder(self):
        engine = _make_engine()
        result = engine._gather_gmail_context()
        assert "클라이언트 미설정" in result

    def test_empty_threads_returns_empty_message(self):
        gmail = MagicMock()
        gmail.search_threads.return_value = []
        engine = _make_engine(gmail_client=gmail)
        result = engine._gather_gmail_context()
        assert "없음" in result or "이메일" in result

    def test_threads_included_in_context(self):
        gmail = MagicMock()
        msg = MagicMock()
        msg.sender = "ceo@acmecorp.com"
        msg.subject = "Investment Update"
        msg.snippet = "Looking forward to discussing..."
        thread = MagicMock()
        thread.messages = [msg]
        gmail.search_threads.return_value = [thread]
        engine = _make_engine(gmail_client=gmail)
        result = engine._gather_gmail_context()
        assert "Investment Update" in result


class TestNotionContext:
    def test_no_client_returns_placeholder(self):
        engine = _make_engine()
        result = engine._gather_notion_context()
        assert "클라이언트 미설정" in result

    def test_records_included_in_context(self):
        notion = MagicMock()
        record = MagicMock()
        record.company_name = "AcmeCorp"
        record.status = "Active"
        notion.query_database.return_value = [record]
        engine = _make_engine(notion_client=notion)
        result = engine._gather_notion_context()
        assert "AcmeCorp" in result


class TestSlackContext:
    def test_no_client_returns_placeholder(self):
        engine = _make_engine()
        result = engine._gather_slack_context()
        assert "클라이언트 미설정" in result

    def test_messages_included_in_context(self):
        retriever = MagicMock()
        msg = MagicMock()
        msg.text = "Investment discussion about AcmeCorp"
        msg.channel_name = "투자팀"
        result_obj = MagicMock()
        result_obj.messages = [msg]
        retriever.search_company_history.return_value = result_obj
        engine = _make_engine(slack_retriever=retriever)
        result = engine._gather_slack_context("AcmeCorp")
        assert "Investment discussion" in result


# ── answer_question (AC 17) ───────────────────────────────────────────────────

class TestAnswerQuestion:
    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_answer_returned_from_claude(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response(
            "이번 주 외부 미팅은 3건입니다."
        )

        engine = _make_engine()
        engine._anthropic = mock_client

        answer = engine.answer_question("이번 주 외부 미팅 있어?")
        assert "이번 주" in answer or "미팅" in answer or "3건" in answer

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_no_web_search_tool_used(self, mock_anthropic_cls):
        """Verify Claude is not called with web_search tool (AC 17)."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response("answer")

        engine = _make_engine()
        engine._anthropic = mock_client

        engine.answer_question("test question")

        # Check that 'tools' parameter (web search) is NOT passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_response_time_target(self, mock_anthropic_cls):
        """Response time should be < 15 seconds with mocked Claude."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response(
            "답변입니다."
        )

        engine = _make_engine()
        engine._anthropic = mock_client

        start = time.monotonic()
        engine.answer_question("간단한 질문")
        elapsed = time.monotonic() - start

        # With mocked API calls, should be well under 15 seconds
        assert elapsed < 15.0

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_api_failure_returns_error_message(self, mock_anthropic_cls):
        """When Claude API fails after all retries, return error string (not raise)."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        engine = _make_engine()
        engine._anthropic = mock_client

        with patch("src.ai.qa_engine.time.sleep"):  # speed up retry
            answer = engine.answer_question("테스트 질문")

        # Should return error message, not raise
        assert answer is not None
        assert "오류" in answer or "확인 불가" in answer

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_context_from_all_sources_in_system_prompt(self, mock_anthropic_cls):
        """Verify that context from all sources appears in the Claude system prompt."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response("ok")

        # Calendar client that returns an event
        cal = MagicMock()
        event = MagicMock()
        event.start = datetime.now(timezone.utc) + timedelta(hours=2)
        event.summary = "IMPORTANT MEETING XYZ"
        event.attendees = []
        cal.list_upcoming_events.return_value = [event]

        engine = _make_engine(calendar_client=cal)
        engine._anthropic = mock_client

        engine.answer_question("미팅 있어?")

        # Check system prompt contains calendar data
        create_call = mock_client.messages.create.call_args
        system_arg = create_call[1].get("system", "")
        assert "IMPORTANT MEETING XYZ" in system_arg


# ── generate_task_suggestions (AC 18) ────────────────────────────────────────

class TestGenerateTaskSuggestions:
    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_returns_formatted_string(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response(
            "• 미팅 준비하세요\n• 이메일 확인하세요"
        )

        engine = _make_engine()
        engine._anthropic = mock_client

        result = engine.generate_task_suggestions()
        assert result is not None
        assert "업무 제안" in result or "미팅" in result

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_returns_empty_on_api_failure(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        engine = _make_engine()
        engine._anthropic = mock_client

        with patch("src.ai.qa_engine.time.sleep"):
            result = engine.generate_task_suggestions()

        assert result == ""

    @patch("src.ai.qa_engine.anthropic.Anthropic")
    def test_no_web_search_in_suggestions(self, mock_anthropic_cls):
        """Suggestions should not use web search tool."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_claude_response(
            "• 팔로업 필요"
        )

        engine = _make_engine()
        engine._anthropic = mock_client

        engine.generate_task_suggestions()

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs


# ── /ask command (AC 19) ──────────────────────────────────────────────────────

class TestAskCommand:
    def test_register_ask_command_no_error(self):
        """register_ask_command should not raise."""
        from src.slack.commands.ask import register_ask_command
        from unittest.mock import MagicMock
        app = MagicMock()
        register_ask_command(app, ask_callback=None)
        app.command.assert_called_once_with("/ask")

    def test_ask_command_ack_empty_text_shows_help(self):
        """Empty /ask shows help text."""
        from src.slack.commands.ask import register_ask_command, _HELP_TEXT

        app = MagicMock()
        ack = MagicMock()
        body = {"text": "", "user_id": "U123", "channel_id": "C123"}
        client = MagicMock()

        # Capture the registered handler
        handler_fn = None

        def _command(path):
            def _decorator(fn):
                nonlocal handler_fn
                handler_fn = fn
                return fn
            return _decorator

        app.command = _command
        register_ask_command(app, ask_callback=None)

        assert handler_fn is not None
        handler_fn(ack=ack, body=body, client=client)
        ack.assert_called_once_with(_HELP_TEXT)


# ── /mail command (AC 15) ─────────────────────────────────────────────────────

class TestMailCommand:
    def test_register_mail_command_no_error(self):
        """register_mail_command should not raise."""
        from src.slack.commands.mail import register_mail_command
        from unittest.mock import MagicMock
        app = MagicMock()
        register_mail_command(app, mail_callback=None)
        app.command.assert_called_once_with("/mail")

    def test_mail_help_returns_help_text(self):
        """'/mail help' should return the help text immediately."""
        from src.slack.commands.mail import register_mail_command, _HELP_TEXT

        app = MagicMock()
        ack = MagicMock()
        body = {"text": "help", "user_id": "U123", "channel_id": "C123"}
        client = MagicMock()

        handler_fn = None

        def _command(path):
            def _decorator(fn):
                nonlocal handler_fn
                handler_fn = fn
                return fn
            return _decorator

        app.command = _command
        register_mail_command(app, mail_callback=None)

        assert handler_fn is not None
        handler_fn(ack=ack, body=body, client=client)
        ack.assert_called_once_with(_HELP_TEXT)
