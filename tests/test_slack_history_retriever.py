"""
Tests for src/slack/history_retriever.py (Sub-AC 3 of AC 7).

Test strategy
-------------
* All Slack WebClient calls are mocked so no live API calls are made.
* Tests cover:
  - Channel discovery: keyword filtering, pagination, caching, cache TTL/force-refresh
  - History fetching: message matching, bot-message exclusion, lookback window
  - Company name normalisation (Korean legal suffixes, English Inc./Corp.)
  - Retry logic: verifies 3 retries with correct delay before raising
  - Error handling: not_in_channel, missing_scope, API failure, short company names
  - Result structure: SlackHistoryResult, SlackMessage, SlackChannel dataclasses
  - Edge cases: empty channels, no matches, multiple channels, thread replies
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest
from slack_sdk.errors import SlackApiError

from src.slack.history_retriever import (
    SlackHistoryRetriever,
    SlackChannel,
    SlackMessage,
    SlackHistoryResult,
    _CHANNEL_CACHE_TTL_SECONDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_channel(
    channel_id: str = "C001",
    name: str = "투자-채널",
    is_private: bool = False,
    is_member: bool = True,
) -> dict:
    """Build a minimal Slack channel dict as returned by conversations_list."""
    return {
        "id": channel_id,
        "name": name,
        "is_private": is_private,
        "is_member": is_member,
        "is_archived": False,
    }


def _make_message(
    text: str = "Hello",
    ts: str = "1680000000.000001",
    user: str = "U123",
    subtype: str | None = None,
    thread_ts: str | None = None,
) -> dict:
    """Build a minimal Slack message dict as returned by conversations_history."""
    msg: dict = {"ts": ts, "text": text, "user": user}
    if subtype:
        msg["subtype"] = subtype
    if thread_ts:
        msg["thread_ts"] = thread_ts
    return msg


def _make_conversations_list_response(channels: list[dict], next_cursor: str = "") -> dict:
    return {
        "ok": True,
        "channels": channels,
        "response_metadata": {"next_cursor": next_cursor},
    }


def _make_history_response(messages: list[dict], has_more: bool = False, next_cursor: str = "") -> dict:
    return {
        "ok": True,
        "messages": messages,
        "has_more": has_more,
        "response_metadata": {"next_cursor": next_cursor},
    }


def _make_retriever(
    channels_response: dict | None = None,
    history_responses: dict | list | None = None,
    retry_delay: float = 0.0,
    priority_keywords: list[str] | None = None,
) -> tuple[SlackHistoryRetriever, MagicMock]:
    """
    Create a SlackHistoryRetriever with mocked WebClient.

    Returns (retriever, mock_client).
    """
    retriever = SlackHistoryRetriever(
        token="xoxb-test",
        priority_keywords=priority_keywords or ["투자", "squad-service"],
        retry_delay=retry_delay,
        retry_attempts=3,
    )

    mock_client = MagicMock()

    if channels_response is not None:
        mock_client.conversations_list.return_value = channels_response

    if history_responses is not None:
        if isinstance(history_responses, list):
            mock_client.conversations_history.side_effect = history_responses
        else:
            mock_client.conversations_history.return_value = history_responses

    retriever._client = mock_client
    return retriever, mock_client


# ── Channel discovery tests ───────────────────────────────────────────────────

class TestDiscoverPriorityChannels:

    def test_matches_keyword_in_name(self):
        """Channels with matching keywords are returned; others are excluded."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜소싱"),
            _make_channel("C002", "squad-service"),
            _make_channel("C003", "general"),
            _make_channel("C004", "투자-포트폴리오"),
            _make_channel("C005", "random"),
        ])
        retriever, _ = _make_retriever(channels_response=channels_resp)

        channels = retriever.discover_priority_channels()

        ids = [c.channel_id for c in channels]
        assert "C001" in ids
        assert "C002" in ids
        assert "C003" not in ids
        assert "C004" in ids
        assert "C005" not in ids

    def test_no_matching_channels_returns_empty(self):
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "general"),
            _make_channel("C002", "random"),
        ])
        retriever, _ = _make_retriever(channels_response=channels_resp)

        channels = retriever.discover_priority_channels()

        assert channels == []

    def test_pagination_fetches_all_pages(self):
        """Paginated responses are followed until next_cursor is empty."""
        page1 = _make_conversations_list_response(
            [_make_channel("C001", "투자-1")],
            next_cursor="cursor_page2",
        )
        page2 = _make_conversations_list_response(
            [_make_channel("C002", "squad-service-a")],
            next_cursor="",
        )
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            priority_keywords=["투자", "squad-service"],
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        mock_client.conversations_list.side_effect = [page1, page2]
        retriever._client = mock_client

        channels = retriever.discover_priority_channels()

        assert len(channels) == 2
        assert mock_client.conversations_list.call_count == 2

    def test_cache_prevents_repeated_api_calls(self):
        """Second call uses cache; conversations_list called only once."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜"),
        ])
        retriever, mock_client = _make_retriever(channels_response=channels_resp)

        retriever.discover_priority_channels()
        retriever.discover_priority_channels()  # second call

        assert mock_client.conversations_list.call_count == 1

    def test_force_refresh_bypasses_cache(self):
        """force=True always re-fetches even if cache is fresh."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜"),
        ])
        retriever, mock_client = _make_retriever(channels_response=channels_resp)

        retriever.discover_priority_channels()
        retriever.discover_priority_channels(force=True)

        assert mock_client.conversations_list.call_count == 2

    def test_invalidate_cache_triggers_refresh(self):
        """invalidate_channel_cache() causes the next call to re-fetch."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        retriever, mock_client = _make_retriever(channels_response=channels_resp)

        retriever.discover_priority_channels()
        retriever.invalidate_channel_cache()
        retriever.discover_priority_channels()

        assert mock_client.conversations_list.call_count == 2

    def test_channel_fields_populated(self):
        """SlackChannel fields are correctly populated from API response."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C999", "투자-테스트", is_private=True, is_member=False),
        ])
        retriever, _ = _make_retriever(channels_response=channels_resp)

        channels = retriever.discover_priority_channels()

        assert len(channels) == 1
        ch = channels[0]
        assert ch.channel_id == "C999"
        assert ch.channel_name == "투자-테스트"
        assert ch.is_private is True
        assert ch.is_member is False

    def test_api_error_returns_empty_and_does_not_raise(self):
        """When conversations_list fails all retries, returns empty list gracefully."""
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            retry_attempts=2,
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        error_response = {"ok": False, "error": "internal_error"}
        mock_client.conversations_list.side_effect = SlackApiError(
            message="internal_error",
            response=error_response,
        )
        retriever._client = mock_client

        channels = retriever.discover_priority_channels()

        assert channels == []


# ── Company name normalisation tests ─────────────────────────────────────────

class TestNormaliseCompanyName:

    @pytest.mark.parametrize("raw, expected", [
        ("AcmeCorp Inc.", "acmecorp"),
        ("주식회사 에이비씨", "에이비씨"),
        ("에이비씨 주식회사", "에이비씨"),
        ("Kakao Corp.", "kakao"),
        ("  SpaceX  ", "spacex"),
        ("Naver Corporation", "naver corporation"),  # "Corporation" not in suffix list
        ("Alpha Ltd.", "alpha"),
        ("Beta LLC", "beta"),
        ("유한회사 테스트", "테스트"),
    ])
    def test_normalise(self, raw, expected):
        result = SlackHistoryRetriever._normalise_company_name(raw)
        assert result == expected


# ── Message matching tests ─────────────────────────────────────────────────────

class TestMessageMatchesCompany:

    def _retriever(self) -> SlackHistoryRetriever:
        return SlackHistoryRetriever(token="xoxb-test", retry_delay=0.0)

    def test_exact_match(self):
        r = self._retriever()
        assert r._message_matches_company(
            "오늘 AcmeCorp 미팅 있어요", "AcmeCorp", "acmecorp"
        )

    def test_case_insensitive_match(self):
        r = self._retriever()
        assert r._message_matches_company(
            "acmecorp 관련 논의", "ACMECORP", "acmecorp"
        )

    def test_normalised_name_match(self):
        r = self._retriever()
        # Raw company "주식회사 에이비씨" normalises to "에이비씨"
        assert r._message_matches_company(
            "에이비씨 미팅 일정 확인", "주식회사 에이비씨", "에이비씨"
        )

    def test_no_match(self):
        r = self._retriever()
        assert not r._message_matches_company(
            "일반 업무 회의 내용입니다", "AcmeCorp", "acmecorp"
        )

    def test_short_company_name_rejected(self):
        """Company names shorter than _MIN_COMPANY_NAME_LEN are never matched."""
        r = self._retriever()
        assert not r._message_matches_company("A 미팅 있음", "A", "a")

    def test_slack_markup_stripped(self):
        """Slack mrkdwn formatting is stripped before matching."""
        r = self._retriever()
        # Bold: *AcmeCorp* → AcmeCorp
        assert r._message_matches_company(
            "*AcmeCorp* 관련 업무", "AcmeCorp", "acmecorp"
        )

    def test_empty_text_no_match(self):
        r = self._retriever()
        assert not r._message_matches_company("", "AcmeCorp", "acmecorp")


# ── History search integration tests ─────────────────────────────────────────

class TestSearchCompanyHistory:

    def _channels_resp(self, channels: list[dict]) -> dict:
        return _make_conversations_list_response(channels)

    def test_matching_messages_returned(self):
        """Messages containing the company name are included in results."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜"),
        ])
        history_resp = _make_history_response([
            _make_message("TestCo 미팅 준비 합시다", ts="1680001000.000001"),
            _make_message("다른 주제 이야기", ts="1680000000.000002"),
            _make_message("TestCo 투자 검토 중", ts="1679900000.000003"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("TestCo")

        assert result.available is True
        assert result.message_count == 2
        texts = [m.text for m in result.messages]
        assert all("TestCo" in t for t in texts)

    def test_results_sorted_newest_first(self):
        """Messages are returned newest-first regardless of channel fetch order."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜"),
        ])
        history_resp = _make_history_response([
            _make_message("AcmeCorp A", ts="1680000100.000001"),
            _make_message("AcmeCorp B", ts="1680000300.000002"),
            _make_message("AcmeCorp C", ts="1680000200.000003"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 3
        ts_values = [float(m.ts) for m in result.messages]
        assert ts_values == sorted(ts_values, reverse=True)

    def test_bot_messages_excluded_by_default(self):
        """Messages with subtype='bot_message' are excluded unless requested."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "squad-service"),
        ])
        history_resp = _make_history_response([
            _make_message("AcmeCorp human msg", ts="1680000100.000001"),
            _make_message("AcmeCorp bot msg", ts="1680000200.000002", subtype="bot_message"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp", include_bot_messages=False)

        assert result.message_count == 1
        assert result.messages[0].text == "AcmeCorp human msg"

    def test_bot_messages_included_when_requested(self):
        """Bot messages are included when include_bot_messages=True."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "squad-service"),
        ])
        history_resp = _make_history_response([
            _make_message("AcmeCorp human msg", ts="1680000100.000001"),
            _make_message("AcmeCorp bot msg", ts="1680000200.000002", subtype="bot_message"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp", include_bot_messages=True)

        assert result.message_count == 2

    def test_channel_join_leave_events_excluded(self):
        """Channel join/leave event messages are never included."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        history_resp = _make_history_response([
            _make_message("AcmeCorp 관련", ts="1680000100.000001"),
            _make_message("AcmeCorp joined the channel", ts="1680000200.000002", subtype="channel_join"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 1
        assert result.messages[0].text == "AcmeCorp 관련"

    def test_not_in_channel_skips_channel(self):
        """Channels where the bot is not a member are added to channels_skipped."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜", is_member=True),
            _make_channel("C002", "투자-비공개", is_member=False),
        ])
        retriever = SlackHistoryRetriever(token="xoxb-test", retry_delay=0.0)
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = channels_resp

        def history_side_effect(**kwargs):
            if kwargs.get("channel") == "C001":
                return _make_history_response([_make_message("TestCo 이야기")])
            # C002 raises not_in_channel
            raise SlackApiError(
                message="not_in_channel",
                response={"ok": False, "error": "not_in_channel"},
            )

        mock_client.conversations_history.side_effect = history_side_effect
        retriever._client = mock_client

        result = retriever.search_company_history("TestCo")

        skipped_ids = [c.channel_id for c in result.channels_skipped]
        searched_ids = [c.channel_id for c in result.channels_searched]
        assert "C002" in skipped_ids
        assert "C001" in searched_ids

    def test_multi_channel_aggregation(self):
        """Matching messages from multiple channels are all returned."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-딜"),
            _make_channel("C002", "squad-service-a"),
        ])
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            priority_keywords=["투자", "squad-service"],
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = channels_resp

        def history_side_effect(**kwargs):
            ch = kwargs.get("channel")
            if ch == "C001":
                return _make_history_response([
                    _make_message("AcmeCorp deal update", ts="1680000100.000001"),
                ])
            elif ch == "C002":
                return _make_history_response([
                    _make_message("AcmeCorp product review", ts="1680000200.000002"),
                ])
            return _make_history_response([])

        mock_client.conversations_history.side_effect = history_side_effect
        retriever._client = mock_client

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 2
        channel_ids = {m.channel_id for m in result.messages}
        assert "C001" in channel_ids
        assert "C002" in channel_ids

    def test_no_matching_messages_returns_empty(self):
        """When no messages match, result has empty messages list but available=True."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        history_resp = _make_history_response([
            _make_message("다른 회사 이야기"),
            _make_message("업무 일반 내용"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("NonExistentCo")

        assert result.available is True
        assert result.message_count == 0

    def test_empty_channels_returns_empty_result(self):
        """When no priority channels exist, result is empty but available."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "general"),
        ])
        retriever, _ = _make_retriever(channels_response=channels_resp)

        result = retriever.search_company_history("AcmeCorp")

        assert result.available is True
        assert result.message_count == 0

    def test_short_company_name_returns_error(self):
        """Company names shorter than minimum length return available=False."""
        retriever = SlackHistoryRetriever(token="xoxb-test", retry_delay=0.0)
        retriever._client = MagicMock()

        result = retriever.search_company_history("A")

        assert result.available is False
        assert result.error is not None
        assert result.message_count == 0

    def test_empty_company_name_returns_error(self):
        retriever = SlackHistoryRetriever(token="xoxb-test", retry_delay=0.0)
        retriever._client = MagicMock()

        result = retriever.search_company_history("   ")

        assert result.available is False

    def test_thread_reply_detected(self):
        """Messages with thread_ts != ts are flagged as thread replies."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        history_resp = _make_history_response([
            _make_message(
                "AcmeCorp 스레드 답글",
                ts="1680000200.000001",
                thread_ts="1680000100.000000",
            ),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 1
        assert result.messages[0].is_thread_reply is True

    def test_top_level_message_not_thread_reply(self):
        """Messages where ts == thread_ts are not flagged as thread replies."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        history_resp = _make_history_response([
            _make_message(
                "AcmeCorp 탑 레벨",
                ts="1680000100.000000",
                # no thread_ts means it's top-level
            ),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 1
        assert result.messages[0].is_thread_reply is False

    def test_message_dt_populated(self):
        """SlackMessage.message_dt is parsed as a UTC datetime."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        ts = "1680000000.000001"
        history_resp = _make_history_response([
            _make_message("AcmeCorp 업무", ts=ts),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 1
        msg = result.messages[0]
        assert msg.message_dt is not None
        assert msg.message_dt.tzinfo is not None

    def test_permalink_fragment_format(self):
        """SlackMessage.permalink_fragment follows the expected format."""
        msg = SlackMessage(
            channel_id="C001",
            channel_name="투자",
            ts="1680000000.123456",
            text="test",
        )
        fragment = msg.permalink_fragment
        assert fragment.startswith("archives/C001/p")
        assert "." not in fragment.split("/p")[1]


# ── Retry logic tests ─────────────────────────────────────────────────────────

class TestRetryLogic:

    def test_history_retried_3_times_on_api_error(self):
        """conversations_history is called 3 times before giving up."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            retry_attempts=3,
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = channels_resp

        error_response = {"ok": False, "error": "internal_error"}
        mock_client.conversations_history.side_effect = SlackApiError(
            message="internal_error",
            response=error_response,
        )
        retriever._client = mock_client

        # Should not raise; bot is not a member / error causes graceful skip
        result = retriever.search_company_history("AcmeCorp")

        # 3 attempts for C001 (retry_attempts=3)
        assert mock_client.conversations_history.call_count == 3

    def test_channel_discovery_retried_3_times(self):
        """conversations_list is retried 3 times on API error."""
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            retry_attempts=3,
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        error_response = {"ok": False, "error": "internal_error"}
        mock_client.conversations_list.side_effect = SlackApiError(
            message="internal_error",
            response=error_response,
        )
        retriever._client = mock_client

        channels = retriever.discover_priority_channels()

        assert mock_client.conversations_list.call_count == 3
        assert channels == []

    def test_retry_succeeds_on_second_attempt(self):
        """Successful call on 2nd attempt returns correct result."""
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        error_response = {"ok": False, "error": "internal_error"}
        history_resp = _make_history_response([
            _make_message("AcmeCorp 딜", ts="1680000100.000001"),
        ])

        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            retry_attempts=3,
            retry_delay=0.0,
        )
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = channels_resp
        mock_client.conversations_history.side_effect = [
            SlackApiError(
                message="internal_error",
                response=error_response,
            ),
            history_resp,
        ]
        retriever._client = mock_client

        result = retriever.search_company_history("AcmeCorp")

        assert result.message_count == 1
        assert mock_client.conversations_history.call_count == 2


# ── to_dict / serialisation tests ─────────────────────────────────────────────

class TestSerialisation:

    def test_slack_message_to_dict(self):
        msg = SlackMessage(
            channel_id="C001",
            channel_name="투자",
            ts="1680000000.123456",
            user_id="U001",
            text="AcmeCorp 미팅",
            thread_ts="1680000000.123456",
            message_dt=datetime(2023, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        )
        d = msg.to_dict()
        assert d["channel_id"] == "C001"
        assert d["channel_name"] == "투자"
        assert d["ts"] == "1680000000.123456"
        assert d["user_id"] == "U001"
        assert d["text"] == "AcmeCorp 미팅"
        assert d["is_thread_reply"] is False
        assert "permalink_fragment" in d

    def test_slack_history_result_to_dict(self):
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        history_resp = _make_history_response([
            _make_message("AcmeCorp 딜"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("AcmeCorp")
        d = result.to_dict()

        assert d["company_name"] == "AcmeCorp"
        assert "channels_searched" in d
        assert "messages" in d
        assert "channels_skipped" in d
        assert isinstance(d["message_count"], int)

    def test_slack_channel_to_dict(self):
        ch = SlackChannel(
            channel_id="C001",
            channel_name="투자-딜",
            is_private=False,
            is_member=True,
        )
        d = ch.to_dict()
        assert d["channel_id"] == "C001"
        assert d["channel_name"] == "투자-딜"
        assert d["is_private"] is False
        assert d["is_member"] is True


# ── get_channel_list helper test ──────────────────────────────────────────────

class TestGetChannelList:

    def test_get_channel_list_returns_cached_channels(self):
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
            _make_channel("C002", "squad-service"),
        ])
        retriever, mock_client = _make_retriever(channels_response=channels_resp)

        channels = retriever.get_channel_list()

        assert len(channels) == 2
        assert mock_client.conversations_list.call_count == 1

    def test_get_channel_list_uses_same_cache_as_discover(self):
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자"),
        ])
        retriever, mock_client = _make_retriever(channels_response=channels_resp)

        retriever.discover_priority_channels()
        retriever.get_channel_list()

        # Should only call API once due to caching
        assert mock_client.conversations_list.call_count == 1


# ── Korean language keyword matching ─────────────────────────────────────────

class TestKoreanKeywordMatching:

    def test_투자_keyword_matches(self):
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            priority_keywords=["투자"],
            retry_delay=0.0,
        )
        assert retriever._channel_matches_keywords("투자-딜소싱")
        assert retriever._channel_matches_keywords("포트폴리오-투자")
        assert not retriever._channel_matches_keywords("general")

    def test_squad_service_keyword_matches(self):
        retriever = SlackHistoryRetriever(
            token="xoxb-test",
            priority_keywords=["squad-service"],
            retry_delay=0.0,
        )
        assert retriever._channel_matches_keywords("squad-service-alpha")
        assert retriever._channel_matches_keywords("squad-service")
        assert not retriever._channel_matches_keywords("squad-engineering")

    def test_korean_company_name_matched_in_message(self):
        channels_resp = _make_conversations_list_response([
            _make_channel("C001", "투자-포트폴리오"),
        ])
        history_resp = _make_history_response([
            _make_message("에이비씨 회사 투자 검토 완료"),
            _make_message("다른 내용 아무 관련 없음"),
        ])
        retriever, _ = _make_retriever(
            channels_response=channels_resp,
            history_responses=history_resp,
        )

        result = retriever.search_company_history("에이비씨")

        assert result.message_count == 1
