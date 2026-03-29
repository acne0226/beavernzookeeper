"""
Tests for Sub-AC 3.1: /brief slash command registration, verification, and ack.

Test strategy:
  - Unit-test the command handler logic (date parsing, ack messages, thread dispatch)
    without needing a real Slack connection.
  - Integration-test the WorkAssistantBot initialisation to confirm /brief is
    properly registered in the Bolt App's command dispatch table.
  - Verify that ack() is always called (3-second rule) before any blocking work.

Run with:
    python -m pytest tests/test_brief_command.py -v
"""
from __future__ import annotations

import sys
import os
import threading
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, call
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_body(text: str = "", user_id: str = "U123", channel_id: str = "C456"):
    return {"text": text, "user_id": user_id, "channel_id": channel_id}


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Date-parsing unit tests (no Slack dependency)
# ══════════════════════════════════════════════════════════════════════════════

class TestBriefDateParsing:
    """Tests for _parse_brief_date in commands/brief.py"""

    def setup_method(self):
        from src.slack.commands.brief import _parse_brief_date
        self._parse = _parse_brief_date

    def test_empty_string_returns_today(self):
        d, err = self._parse("")
        assert err is None
        assert d == date.today()

    def test_today_keyword_returns_today(self):
        d, err = self._parse("today")
        assert err is None
        assert d == date.today()

    def test_tomorrow_keyword(self):
        d, err = self._parse("tomorrow")
        assert err is None
        assert d == date.today() + timedelta(days=1)

    def test_iso_date_format(self):
        d, err = self._parse("2026-05-01")
        assert err is None
        assert d.year == 2026
        assert d.month == 5
        assert d.day == 1

    def test_slash_date_format(self):
        d, err = self._parse("2026/05/01")
        assert err is None
        assert d.year == 2026
        assert d.month == 5
        assert d.day == 1

    def test_invalid_string_returns_error(self):
        d, err = self._parse("next week")
        assert d is None
        assert err is not None
        assert "형식" in err or "YYYY" in err

    def test_whitespace_trimmed(self):
        d, err = self._parse("  tomorrow  ")
        assert err is None
        assert d == date.today() + timedelta(days=1)

    def test_case_insensitive(self):
        d, err = self._parse("TOMORROW")
        assert err is None
        assert d == date.today() + timedelta(days=1)


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Handler unit tests (mocked Slack Bolt context)
# ══════════════════════════════════════════════════════════════════════════════

class TestBriefCommandHandler:
    """Tests for the /brief handler function registered via register_brief_command."""

    def _get_handler(self, callback=None):
        """
        Register /brief on a mock Bolt App and return the wrapped handler.
        Returns (app_mock, command_handler).
        """
        from src.slack.commands.brief import register_brief_command

        app_mock = MagicMock()
        registered = {}

        def fake_command(name):
            def decorator(fn):
                registered[name] = fn
                return fn
            return decorator

        app_mock.command.side_effect = fake_command
        register_brief_command(app_mock, briefing_callback=callback)

        assert "/brief" in registered, "/brief was not registered on the app"
        return app_mock, registered["/brief"]

    # ── ack is always called first ─────────────────────────────────────────────

    def test_ack_called_for_empty_text(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body(""), client=client)
        ack.assert_called_once()

    def test_ack_called_for_today(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("today"), client=client)
        ack.assert_called_once()

    def test_ack_called_for_tomorrow(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("tomorrow"), client=client)
        ack.assert_called_once()

    def test_ack_called_for_iso_date(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("2026-05-01"), client=client)
        ack.assert_called_once()

    def test_ack_called_with_error_for_bad_arg(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("not-a-date"), client=client)
        ack.assert_called_once()
        # Error ack text should mention format guidance
        ack_text = ack.call_args[0][0]
        assert "YYYY" in ack_text or "형식" in ack_text

    # ── ack text varies by date ────────────────────────────────────────────────

    def test_ack_text_for_today_contains_briefing_keyword(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body(""), client=client)
        ack_text = ack.call_args[0][0]
        assert "브리핑" in ack_text or "briefing" in ack_text.lower()

    def test_ack_text_for_specific_date_contains_date_string(self):
        _, handler = self._get_handler()
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("2026-05-01"), client=client)
        ack_text = ack.call_args[0][0]
        assert "2026-05-01" in ack_text

    # ── background thread is spawned ──────────────────────────────────────────

    def test_background_thread_spawned_on_valid_input(self):
        """The handler must dispatch work to a thread, not block."""
        event = threading.Event()
        callback_dates = []

        def cb(target_date, user_id, channel_id):
            callback_dates.append(target_date)
            event.set()

        _, handler = self._get_handler(callback=cb)
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body(""), client=client)

        triggered = event.wait(timeout=5)
        assert triggered, "Background callback was not called within 5 seconds"
        assert callback_dates[0] == date.today()

    def test_no_thread_spawned_on_bad_arg(self):
        """Bad arguments must NOT start a background thread."""
        event = threading.Event()

        def cb(target_date, user_id, channel_id):
            event.set()

        _, handler = self._get_handler(callback=cb)
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("invalid-date-xyz"), client=client)

        # The callback should NOT be called
        triggered = event.wait(timeout=1)
        assert not triggered, "Background callback should NOT be called for bad args"

    # ── placeholder DM when no callback ───────────────────────────────────────

    def test_placeholder_dm_sent_when_no_callback(self):
        """When no briefing_callback, a placeholder DM is sent to user."""
        _, handler = self._get_handler(callback=None)
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("", user_id="U999"), client=client)

        # Give the background thread time to run
        import time; time.sleep(1)

        client.chat_postMessage.assert_called_once()
        call_kwargs = client.chat_postMessage.call_args
        assert call_kwargs[1]["channel"] == "U999" or call_kwargs[0][0] == "U999"

    # ── error handling in background thread ───────────────────────────────────

    def test_exception_in_callback_sends_error_dm(self):
        """If the callback raises, the handler must send an error DM (not crash)."""
        def bad_callback(target_date, user_id, channel_id):
            raise RuntimeError("Simulated pipeline failure")

        _, handler = self._get_handler(callback=bad_callback)
        ack = MagicMock()
        client = MagicMock()
        handler(ack=ack, body=_make_mock_body("", user_id="U888"), client=client)

        import time; time.sleep(1)

        # An error DM should be sent
        client.chat_postMessage.assert_called()
        last_call_text = client.chat_postMessage.call_args[1].get("text", "")
        assert "오류" in last_call_text or "error" in last_call_text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: WorkAssistantBot integration test (Bolt App dispatch table)
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkAssistantBotRegistration:
    """
    Verify that WorkAssistantBot registers /brief in the Bolt App's dispatch
    table without starting a real Socket Mode connection or Slack API calls.

    We patch:
      - SocketModeHandler: avoids the real WebSocket connection
      - The WebClient used by Bolt's App: avoids auth.test HTTP calls during init
    """

    # ── Shared patch context for all tests in this class ──────────────────────

    @staticmethod
    def _fake_api_call(method, **kwargs):
        """Stub that returns a minimal successful Slack API response."""
        return {
            "ok": True,
            "url": "https://hooks.slack.com/…",
            "user_id": "UBOT123",
            "bot_id": "BBOT123",
            "team": "TestTeam",
            "user": "bot_user",
        }

    def _make_bot(self, briefing_callback=None):
        """
        Build a WorkAssistantBot with all network I/O patched out.
        Patches the low-level WebClient.api_call so Bolt's auth.test call
        succeeds without real network access.
        Returns (bot, mock_api_call).
        """
        import importlib

        with patch("src.slack.bot.SocketModeHandler"), \
             patch("slack_sdk.WebClient.api_call", side_effect=self._fake_api_call):
            import src.slack.bot as bot_module
            importlib.reload(bot_module)
            bot = bot_module.WorkAssistantBot(briefing_callback=briefing_callback)
            return bot, patch("slack_sdk.WebClient.api_call")

    def test_brief_command_registered_in_bolt_app(self):
        """
        /brief must appear in the Bolt App's command listener map.

        Bolt stores commands as CustomListener objects in app._listeners.
        Each listener has `matchers`, where the first matcher's `func` is a
        closure that returns True for the matching command name.

        We verify by calling matcher.func({'command': '/brief'}) == True.
        """
        bot, _ = self._make_bot()
        app = bot.app

        assert len(app._listeners) > 0, "No listeners registered on the Bolt App"

        # Check if any listener matches '/brief'
        brief_matched = False
        for listener in app._listeners:
            for matcher in listener.matchers:
                try:
                    if matcher.func({"command": "/brief"}):
                        brief_matched = True
                        break
                except Exception:
                    continue
            if brief_matched:
                break

        assert brief_matched, (
            "/brief command not found in Bolt App._listeners matchers. "
            f"Registered {len(app._listeners)} listener(s)."
        )

    def test_brief_command_decorator_called(self):
        """
        register_brief_command must call app.command('/brief') exactly once.
        We verify by mocking a fresh Bolt App and confirming the decorator was used.
        """
        from src.slack.commands.brief import register_brief_command

        app_mock = MagicMock()
        registered_commands = []

        def fake_command(cmd_name):
            def decorator(fn):
                registered_commands.append(cmd_name)
                return fn
            return decorator

        app_mock.command.side_effect = fake_command
        register_brief_command(app_mock)

        assert "/brief" in registered_commands, (
            f"register_brief_command did not register /brief. Got: {registered_commands}"
        )
        assert registered_commands.count("/brief") == 1, (
            f"/brief registered {registered_commands.count('/brief')} times (expected 1)"
        )

    def test_bot_has_send_message_method(self):
        """WorkAssistantBot must expose send_message for DM delivery."""
        bot, _ = self._make_bot()
        assert callable(getattr(bot, "send_message", None))

    def test_bot_has_send_dm_method(self):
        """WorkAssistantBot must expose send_dm for targeted DM delivery."""
        bot, _ = self._make_bot()
        assert callable(getattr(bot, "send_dm", None))

    def test_bot_has_send_error_method(self):
        """WorkAssistantBot must expose send_error for failure notifications."""
        bot, _ = self._make_bot()
        assert callable(getattr(bot, "send_error", None))

    def test_bolt_app_created_with_correct_token(self):
        """
        The Bolt App must be constructed with SLACK_BOT_TOKEN.
        We verify by capturing the App() constructor arguments.
        """
        import importlib
        from src.config import SLACK_BOT_TOKEN

        captured_tokens = []
        original_app_init = __import__("slack_bolt", fromlist=["App"]).App.__init__

        def capturing_init(self_inner, *args, **kwargs):
            captured_tokens.append(kwargs.get("token", args[0] if args else None))
            original_app_init(self_inner, *args, **kwargs)

        with patch("slack_bolt.App.__init__", capturing_init), \
             patch("src.slack.bot.SocketModeHandler"), \
             patch("slack_sdk.WebClient.api_call", side_effect=self._fake_api_call):
            import src.slack.bot as bot_module
            importlib.reload(bot_module)
            bot_module.WorkAssistantBot()

        assert SLACK_BOT_TOKEN in captured_tokens, (
            f"SLACK_BOT_TOKEN not passed to App(). Captured: {captured_tokens}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Request verification documentation test
# ══════════════════════════════════════════════════════════════════════════════

class TestRequestVerification:
    """
    Slack Bolt + Socket Mode provides automatic request verification.
    These tests document and verify the verification guarantees.
    """

    def test_socket_mode_handler_class_imported(self):
        """SocketModeHandler must be importable (verifies dependency installed)."""
        from slack_bolt.adapter.socket_mode import SocketModeHandler
        assert SocketModeHandler is not None

    def test_bolt_app_class_imported(self):
        """Slack Bolt App must be importable."""
        from slack_bolt import App
        assert App is not None

    def test_bolt_middleware_includes_verification(self):
        """
        The Bolt App must include request-verification middleware.

        Bolt's App._middleware_list contains:
          - SslCheck
          - RequestVerification  ← Slack signature verification
          - SingleTeamAuthorization
          - IgnoringSelfEvents
          - UrlVerification
          (among others)

        In Socket Mode, the WebSocket connection is authenticated via the
        xapp token (handled by SocketModeHandler), and RequestVerification
        handles HTTP-level signature checks for any HTTP endpoint.
        We verify both are present in the middleware stack.
        """
        import importlib
        from src.slack.bot import WorkAssistantBot

        bot = WorkAssistantBot()
        middleware_types = [type(m).__name__ for m in bot.app._middleware_list]

        # Middleware list must be non-empty
        assert len(middleware_types) > 0, (
            "Bolt App._middleware_list is empty — middleware not configured"
        )

        # RequestVerification must be present (handles Slack signature checking)
        assert "RequestVerification" in middleware_types, (
            f"RequestVerification not in middleware list. Got: {middleware_types}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
