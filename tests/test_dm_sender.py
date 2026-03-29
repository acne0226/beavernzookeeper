"""
Tests for Sub-AC 3: Slack DM Sender Utility.

Verifies that SlackDMSender correctly:
  - Opens a DM channel with the target user via conversations_open
  - Caches the DM channel ID to avoid redundant API calls
  - Sends plain-text messages via chat_postMessage
  - Sends Block Kit formatted messages (text + blocks)
  - Retries up to API_RETRY_ATTEMPTS (3×) on SlackApiError
  - Waits API_RETRY_DELAY_SECONDS between retries (injected as 0 in tests)
  - Returns True on success, False after exhausting retries
  - Sends error notifications via send_error (best-effort, no raises)
  - Does NOT raise when send_error itself fails

Also tests send_daily_briefing_dm():
  - Calls format_daily_briefing with the supplied events and target_date
  - Passes the formatted (text, blocks) to bot.send_message
  - Returns the bool result of bot.send_message

Run with:
    python -m pytest tests/test_dm_sender.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

# Ensure project root is on sys.path when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared test helpers ────────────────────────────────────────────────────────

KST = ZoneInfo("Asia/Seoul")
_FAKE_DM_CHANNEL = "D_TEST123"
_FAKE_USER_ID = "U_TEST456"
_FAKE_BOT_TOKEN = "xoxb-test-token"

TARGET_DATE = date(2026, 3, 29)


def _make_slack_api_error(error_code: str = "channel_not_found"):
    """Build a SlackApiError with a minimal response payload."""
    from slack_sdk.errors import SlackApiError

    response = MagicMock()
    response.get = lambda key, default=None: error_code if key == "error" else default
    response.__getitem__ = lambda self, key: error_code if key == "error" else None
    return SlackApiError(message=f"Slack API error: {error_code}", response=response)


def _make_sender(dm_channel: str = _FAKE_DM_CHANNEL, retry_delay: float = 0):
    """
    Build a SlackDMSender with a fully mocked WebClient.

    Returns (sender, mock_client).
    """
    from src.slack.dm_sender import SlackDMSender

    with patch("src.slack.dm_sender.WebClient") as MockWebClient:
        mock_client = MagicMock()
        MockWebClient.return_value = mock_client

        # conversations_open returns a DM channel id
        mock_client.conversations_open.return_value = {
            "ok": True,
            "channel": {"id": dm_channel},
        }

        sender = SlackDMSender(
            token=_FAKE_BOT_TOKEN,
            target_user_id=_FAKE_USER_ID,
            retry_delay=retry_delay,
        )
        sender._client = mock_client  # inject mock directly after construction

    return sender, mock_client


def _make_dict_event(
    title: str = "테스트 미팅",
    hour: int = 10,
) -> dict:
    """Minimal calendar_fetcher-style event dict for formatter tests."""
    return {
        "title": title,
        "start": datetime(2026, 3, 29, hour, 0, tzinfo=KST),
        "end": datetime(2026, 3, 29, hour + 1, 0, tzinfo=KST),
        "all_day": False,
        "attendees": [],
        "html_link": "https://calendar.google.com/event/test",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: SlackDMSender — DM channel management
# ══════════════════════════════════════════════════════════════════════════════

class TestGetDmChannel:
    """_get_dm_channel opens and caches the DM channel."""

    def test_opens_dm_channel_on_first_call(self):
        sender, mock_client = _make_sender()
        channel = sender._get_dm_channel()
        assert channel == _FAKE_DM_CHANNEL
        mock_client.conversations_open.assert_called_once_with(users=[_FAKE_USER_ID])

    def test_caches_channel_id_after_first_call(self):
        sender, mock_client = _make_sender()
        _ = sender._get_dm_channel()
        _ = sender._get_dm_channel()  # second call should NOT hit API
        assert mock_client.conversations_open.call_count == 1

    def test_returns_same_channel_on_subsequent_calls(self):
        sender, mock_client = _make_sender()
        ch1 = sender._get_dm_channel()
        ch2 = sender._get_dm_channel()
        assert ch1 == ch2 == _FAKE_DM_CHANNEL

    def test_raises_runtime_error_after_all_retries_fail(self):
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()

        with pytest.raises(RuntimeError, match="after"):
            sender._get_dm_channel()

    def test_retries_correct_number_of_times_on_failure(self):
        from src.config import API_RETRY_ATTEMPTS

        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()

        try:
            sender._get_dm_channel()
        except RuntimeError:
            pass

        assert mock_client.conversations_open.call_count == API_RETRY_ATTEMPTS

    def test_succeeds_on_second_attempt_after_transient_error(self):
        sender, mock_client = _make_sender(retry_delay=0)
        success_response = {"ok": True, "channel": {"id": _FAKE_DM_CHANNEL}}
        mock_client.conversations_open.side_effect = [
            _make_slack_api_error(),  # first attempt fails
            success_response,          # second attempt succeeds
        ]

        channel = sender._get_dm_channel()
        assert channel == _FAKE_DM_CHANNEL
        assert mock_client.conversations_open.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: SlackDMSender.send() — plain text
# ══════════════════════════════════════════════════════════════════════════════

class TestSendPlainText:
    """send(text) with no blocks delivers a plain-text DM."""

    def test_send_returns_true_on_success(self):
        sender, mock_client = _make_sender()
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "12345.678"}
        result = sender.send("안녕하세요")
        assert result is True

    def test_send_calls_chat_post_message(self):
        sender, mock_client = _make_sender()
        sender.send("테스트 메시지")
        mock_client.chat_postMessage.assert_called_once()

    def test_send_posts_to_dm_channel(self):
        sender, mock_client = _make_sender(dm_channel=_FAKE_DM_CHANNEL)
        sender.send("메시지")
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == _FAKE_DM_CHANNEL

    def test_send_passes_text_correctly(self):
        sender, mock_client = _make_sender()
        sender.send("hello world")
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["text"] == "hello world"

    def test_send_without_blocks_does_not_pass_blocks_kwarg(self):
        sender, mock_client = _make_sender()
        sender.send("no blocks")
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" not in call_kwargs

    def test_send_returns_false_after_all_retries_fail(self):
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()
        result = sender.send("실패 메시지")
        assert result is False

    def test_send_does_not_raise_on_api_error(self):
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()
        # Must not raise even after all retries exhausted
        result = sender.send("테스트")
        assert result is False

    def test_send_returns_false_when_dm_channel_unavailable(self):
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()
        result = sender.send("채널 없음")
        assert result is False

    def test_send_retries_on_transient_error(self):
        from src.config import API_RETRY_ATTEMPTS

        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()

        sender.send("재시도 테스트")

        assert mock_client.chat_postMessage.call_count == API_RETRY_ATTEMPTS

    def test_send_succeeds_after_one_transient_failure(self):
        sender, mock_client = _make_sender(retry_delay=0)
        success = {"ok": True, "ts": "111.222"}
        mock_client.chat_postMessage.side_effect = [
            _make_slack_api_error(),
            success,
        ]
        result = sender.send("재시도 성공")
        assert result is True
        assert mock_client.chat_postMessage.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: SlackDMSender.send() — Block Kit formatted messages
# ══════════════════════════════════════════════════════════════════════════════

class TestSendWithBlocks:
    """send(text, blocks=...) delivers a rich Block Kit DM."""

    def _make_blocks(self) -> list[dict]:
        return [
            {"type": "header", "text": {"type": "plain_text", "text": "📅 브리핑"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "미팅 없음"}},
            {"type": "divider"},
        ]

    def test_send_with_blocks_returns_true(self):
        sender, mock_client = _make_sender()
        result = sender.send("fallback", blocks=self._make_blocks())
        assert result is True

    def test_send_passes_blocks_to_chat_post_message(self):
        sender, mock_client = _make_sender()
        blocks = self._make_blocks()
        sender.send("fallback text", blocks=blocks)
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" in call_kwargs
        assert call_kwargs["blocks"] == blocks

    def test_send_passes_fallback_text_alongside_blocks(self):
        sender, mock_client = _make_sender()
        sender.send("fallback text", blocks=self._make_blocks())
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["text"] == "fallback text"

    def test_send_with_blocks_uses_dm_channel(self):
        sender, mock_client = _make_sender(dm_channel="D_RICH_CHANNEL")
        sender.send("rich message", blocks=self._make_blocks())
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "D_RICH_CHANNEL"

    def test_send_with_empty_blocks_list_does_not_pass_blocks(self):
        """An empty blocks list is falsy — should behave like no blocks."""
        sender, mock_client = _make_sender()
        sender.send("text", blocks=[])
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" not in call_kwargs

    def test_send_with_50_blocks_all_passed(self):
        """Block Kit limit is 50 blocks — all should be sent verbatim."""
        sender, mock_client = _make_sender()
        blocks = [{"type": "divider"}] * 50
        sender.send("50 blocks", blocks=blocks)
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert len(call_kwargs["blocks"]) == 50


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: SlackDMSender.send_error() — error notifications
# ══════════════════════════════════════════════════════════════════════════════

class TestSendError:
    """send_error sends an error notification DM (best-effort, no raises)."""

    def test_send_error_calls_chat_post_message(self):
        sender, mock_client = _make_sender()
        sender.send_error("캘린더 API 오류", Exception("timeout"))
        mock_client.chat_postMessage.assert_called_once()

    def test_send_error_includes_context_in_text(self):
        sender, mock_client = _make_sender()
        sender.send_error("Notion DB 조회 실패", Exception("403"))
        text = mock_client.chat_postMessage.call_args[1]["text"]
        assert "Notion DB 조회 실패" in text

    def test_send_error_includes_error_repr_in_text(self):
        sender, mock_client = _make_sender()
        err = Exception("rate_limited")
        sender.send_error("스케줄러", err)
        text = mock_client.chat_postMessage.call_args[1]["text"]
        assert "rate_limited" in text

    def test_send_error_does_not_raise_on_api_failure(self):
        """send_error must be safe to call even when Slack is down."""
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()
        # Must not raise
        sender.send_error("critical section", RuntimeError("boom"))

    def test_send_error_does_not_raise_when_dm_channel_unavailable(self):
        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()
        sender.send_error("test", Exception("x"))  # must not raise

    def test_send_error_posts_to_dm_channel(self):
        sender, mock_client = _make_sender(dm_channel="D_ERROR_CHANNEL")
        sender.send_error("테스트", Exception("e"))
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "D_ERROR_CHANNEL"

    def test_send_error_text_contains_warning_emoji(self):
        sender, mock_client = _make_sender()
        sender.send_error("context", Exception("err"))
        text = mock_client.chat_postMessage.call_args[1]["text"]
        assert ":warning:" in text or "⚠" in text

    def test_send_error_returns_none(self):
        sender, mock_client = _make_sender()
        result = sender.send_error("ctx", Exception("e"))
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: send_daily_briefing_dm() helper
# ══════════════════════════════════════════════════════════════════════════════

class TestSendDailyBriefingDm:
    """
    send_daily_briefing_dm(bot, events, target_date) formats events and
    delivers them via bot.send_message().
    """

    def _make_bot_mock(self, send_message_return: bool = True) -> MagicMock:
        bot = MagicMock()
        bot.send_message.return_value = send_message_return
        return bot

    def test_returns_true_on_success(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock(True)
        result = send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        assert result is True

    def test_returns_false_when_bot_send_fails(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock(False)
        result = send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        assert result is False

    def test_calls_bot_send_message(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        bot.send_message.assert_called_once()

    def test_passes_text_to_send_message(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        call_args = bot.send_message.call_args
        # First positional arg or 'text' kwarg must be a non-empty string
        text_arg = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
        assert isinstance(text_arg, str)
        assert len(text_arg) > 0

    def test_passes_blocks_kwarg_to_send_message(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        call_kwargs = bot.send_message.call_args[1]
        assert "blocks" in call_kwargs
        assert isinstance(call_kwargs["blocks"], list)

    def test_blocks_within_slack_limit(self):
        """Even with many events the formatter caps at 50 blocks."""
        from src.slack.dm_sender import send_daily_briefing_dm

        events = [_make_dict_event(f"미팅 {i}", hour=(i % 8) + 9) for i in range(30)]
        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, events, target_date=TARGET_DATE)
        call_kwargs = bot.send_message.call_args[1]
        assert len(call_kwargs["blocks"]) <= 50

    def test_fallback_text_contains_date(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        text_arg = bot.send_message.call_args[0][0]
        assert "2026" in text_arg

    def test_empty_events_produces_no_meetings_message(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
        # blocks should contain a "no events" section
        blocks = bot.send_message.call_args[1]["blocks"]
        all_block_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "없" in all_block_text  # "없습니다" / "없음"

    def test_with_event_list_title_appears_in_blocks(self):
        from src.slack.dm_sender import send_daily_briefing_dm

        events = [_make_dict_event("킥오프 미팅")]
        bot = self._make_bot_mock()
        send_daily_briefing_dm(bot, events, target_date=TARGET_DATE)
        blocks = bot.send_message.call_args[1]["blocks"]
        all_text = " ".join(
            b.get("text", {}).get("text", "")
            for b in blocks
            if isinstance(b.get("text"), dict)
        )
        assert "킥오프 미팅" in all_text

    def test_target_date_inferred_from_events_when_not_given(self):
        """When target_date is None, format_daily_briefing infers it."""
        from src.slack.dm_sender import send_daily_briefing_dm

        events = [_make_dict_event("자동 날짜 미팅")]
        bot = self._make_bot_mock()
        # No target_date → must not raise; date inferred from event
        result = send_daily_briefing_dm(bot, events)
        assert result is True

    def test_returns_bot_send_message_return_value(self):
        """Return value mirrors bot.send_message's return value."""
        from src.slack.dm_sender import send_daily_briefing_dm

        for expected in (True, False):
            bot = self._make_bot_mock(send_message_return=expected)
            result = send_daily_briefing_dm(bot, [], target_date=TARGET_DATE)
            assert result == expected


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Integration — SlackDMSender + format_daily_briefing
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationWithFormatter:
    """
    Verify that SlackDMSender works end-to-end with the real formatter
    (no mocking of format_daily_briefing).
    """

    def _make_sender_for_integration(self) -> tuple:
        sender, mock_client = _make_sender(retry_delay=0)
        return sender, mock_client

    def test_send_formatted_briefing_no_events(self):
        from src.briefing.formatter import format_daily_briefing

        sender, mock_client = self._make_sender_for_integration()
        text, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        result = sender.send(text, blocks=blocks)
        assert result is True

    def test_send_formatted_briefing_with_events(self):
        from src.briefing.formatter import format_daily_briefing

        events = [_make_dict_event("파트너 미팅", 10), _make_dict_event("내부 스탠드업", 14)]
        sender, mock_client = self._make_sender_for_integration()
        text, blocks = format_daily_briefing(events, target_date=TARGET_DATE)
        result = sender.send(text, blocks=blocks)
        assert result is True

    def test_formatted_blocks_sent_verbatim(self):
        """The blocks produced by the formatter must reach chat_postMessage unchanged."""
        from src.briefing.formatter import format_daily_briefing

        events = [_make_dict_event("테스트 미팅", 11)]
        sender, mock_client = self._make_sender_for_integration()
        text, blocks = format_daily_briefing(events, target_date=TARGET_DATE)
        sender.send(text, blocks=blocks)
        sent_blocks = mock_client.chat_postMessage.call_args[1]["blocks"]
        assert sent_blocks == blocks

    def test_formatted_text_passed_as_fallback(self):
        from src.briefing.formatter import format_daily_briefing

        sender, mock_client = self._make_sender_for_integration()
        text, blocks = format_daily_briefing([], target_date=TARGET_DATE)
        sender.send(text, blocks=blocks)
        sent_text = mock_client.chat_postMessage.call_args[1]["text"]
        assert sent_text == text


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Constructor / configuration
# ══════════════════════════════════════════════════════════════════════════════

class TestSlackDMSenderConfiguration:
    """Verify that constructor parameters are respected."""

    def test_default_construction_uses_config_values(self):
        """Constructor without arguments must not raise."""
        from src.slack.dm_sender import SlackDMSender
        with patch("src.slack.dm_sender.WebClient"):
            sender = SlackDMSender()
            assert sender._retry_attempts > 0

    def test_custom_retry_attempts_respected(self):
        sender, mock_client = _make_sender(retry_delay=0)
        sender._retry_attempts = 2
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()
        sender.send("x")
        assert mock_client.chat_postMessage.call_count == 2

    def test_custom_target_user_id_used_in_conversations_open(self):
        from src.slack.dm_sender import SlackDMSender

        with patch("src.slack.dm_sender.WebClient") as MockWebClient:
            mock_client = MagicMock()
            MockWebClient.return_value = mock_client
            mock_client.conversations_open.return_value = {
                "ok": True,
                "channel": {"id": "D_CUSTOM"},
            }

            sender = SlackDMSender(
                token="xoxb-custom",
                target_user_id="U_CUSTOM_USER",
                retry_delay=0,
            )
            sender._client = mock_client
            sender._get_dm_channel()

        mock_client.conversations_open.assert_called_once_with(users=["U_CUSTOM_USER"])

    def test_dm_channel_initially_none(self):
        from src.slack.dm_sender import SlackDMSender

        with patch("src.slack.dm_sender.WebClient"):
            sender = SlackDMSender(token="xoxb-t", target_user_id="U1")
        assert sender._dm_channel is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: Retry timing (fast path with injected zero delay)
# ══════════════════════════════════════════════════════════════════════════════

class TestRetryTiming:
    """Verify retry count contracts without real sleep."""

    def test_send_retries_exactly_3_times_by_default(self):
        from src.config import API_RETRY_ATTEMPTS

        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.chat_postMessage.side_effect = _make_slack_api_error()
        sender.send("retry test")
        assert mock_client.chat_postMessage.call_count == API_RETRY_ATTEMPTS

    def test_conversations_open_retried_on_failure(self):
        from src.config import API_RETRY_ATTEMPTS

        sender, mock_client = _make_sender(retry_delay=0)
        mock_client.conversations_open.side_effect = _make_slack_api_error()

        try:
            sender._get_dm_channel()
        except RuntimeError:
            pass

        assert mock_client.conversations_open.call_count == API_RETRY_ATTEMPTS

    def test_no_retry_when_first_attempt_succeeds(self):
        sender, mock_client = _make_sender(retry_delay=0)
        sender.send("success first try")
        assert mock_client.chat_postMessage.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
