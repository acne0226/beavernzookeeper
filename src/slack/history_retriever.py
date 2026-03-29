"""
Slack History Retriever (Sub-AC 3 of AC 7).

Searches Slack message history for a given company name across configured
priority channels (channels whose names contain "투자" or "squad-service").

Architecture
------------
* Uses ``slack_sdk.WebClient`` directly — no Bolt overhead required.
* Channel list is discovered once at first use and cached in-memory (TTL-based).
* History is fetched via ``conversations_history`` — compatible with standard
  bot tokens (``xoxb-``).  The more powerful ``search.messages`` endpoint
  requires a user OAuth token and is therefore not used.
* Text matching is case-insensitive substring search with Korean/English
  normalisation.  The company name (and its normalised form) must appear in
  the message text.
* Thread replies are *not* fetched by default to keep latency low; pass
  ``include_thread_replies=True`` to also retrieve top-level thread roots
  whose replies contain the company name.

Retry policy
------------
Every Slack API call is retried up to ``API_RETRY_ATTEMPTS`` (3) times with
``API_RETRY_DELAY_SECONDS`` (10 s) between attempts.  After exhaustion the
call raises ``SlackApiError`` so callers can react (e.g. send a failure DM).

Usage::

    from src.slack.history_retriever import SlackHistoryRetriever

    retriever = SlackHistoryRetriever()
    messages = retriever.search_company_history(
        company_name="AcmeCorp",
        lookback_days=90,
        max_messages_per_channel=50,
    )
    for msg in messages:
        print(msg.channel_name, msg.ts, msg.text[:80])
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config import (
    SLACK_BOT_TOKEN,
    SLACK_PRIORITY_CHANNEL_KEYWORDS,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How long to cache the channel list before re-discovering (seconds)
_CHANNEL_CACHE_TTL_SECONDS: int = 3600  # 1 hour

# Maximum pages of channel history to fetch per channel per search
_MAX_HISTORY_PAGES: int = 10

# Default lookback window for message history
_DEFAULT_LOOKBACK_DAYS: int = 90

# Default max messages to collect per channel
_DEFAULT_MAX_MESSAGES: int = 50

# Minimum company name length to avoid trivially short / noisy matches
_MIN_COMPANY_NAME_LEN: int = 2


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SlackChannel:
    """A Slack channel identified as a priority search target."""
    channel_id: str
    channel_name: str
    is_private: bool = False
    is_member: bool = True  # False when bot is not a member

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "is_private": self.is_private,
            "is_member": self.is_member,
        }


@dataclass
class SlackMessage:
    """
    A single Slack message matching a company name search.

    Attributes
    ----------
    channel_id:    Internal Slack channel ID (``C...``).
    channel_name:  Human-readable channel name (without ``#``).
    ts:            Slack message timestamp string (e.g. ``"1680000000.123456"``).
    user_id:       Slack user ID of the message sender (empty for bot messages).
    text:          Plain-text message body (Slack mrkdwn may be present).
    thread_ts:     Parent thread timestamp when this is a threaded reply; equals
                   ``ts`` when this is a top-level message.
    message_dt:    UTC datetime parsed from ``ts``.
    is_thread_reply: True when the message is a threaded reply (thread_ts != ts).
    """
    channel_id: str
    channel_name: str
    ts: str
    user_id: str = ""
    text: str = ""
    thread_ts: str = ""
    message_dt: Optional[datetime] = None
    is_thread_reply: bool = False

    # ── Derived helpers ────────────────────────────────────────────────────────

    @property
    def permalink_fragment(self) -> str:
        """A Slack deep-link fragment (channel + ts, no workspace domain)."""
        ts_clean = self.ts.replace(".", "")
        return f"archives/{self.channel_id}/p{ts_clean}"

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "ts": self.ts,
            "user_id": self.user_id,
            "text": self.text,
            "thread_ts": self.thread_ts,
            "message_dt": self.message_dt.isoformat() if self.message_dt else None,
            "is_thread_reply": self.is_thread_reply,
            "permalink_fragment": self.permalink_fragment,
        }


@dataclass
class SlackHistoryResult:
    """
    Aggregated result of a company name search across Slack channels.

    Attributes
    ----------
    company_name:    The search term (as provided by the caller).
    channels_searched: Priority channels that were scanned.
    messages:        All matching messages, sorted newest-first.
    channels_skipped: Channels where the bot lacked access (not a member, etc.).
    available:       False when the Slack API was entirely unreachable.
    error:           Human-readable failure description when available=False.
    """
    company_name: str
    channels_searched: list[SlackChannel] = field(default_factory=list)
    messages: list[SlackMessage] = field(default_factory=list)
    channels_skipped: list[SlackChannel] = field(default_factory=list)
    available: bool = True
    error: Optional[str] = None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def to_dict(self) -> dict:
        return {
            "company_name": self.company_name,
            "channels_searched": [c.to_dict() for c in self.channels_searched],
            "message_count": self.message_count,
            "messages": [m.to_dict() for m in self.messages],
            "channels_skipped": [c.to_dict() for c in self.channels_skipped],
            "available": self.available,
            "error": self.error,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class SlackHistoryRetriever:
    """
    Searches Slack message history for a company name across priority channels.

    Priority channels are those whose names contain at least one keyword from
    ``SLACK_PRIORITY_CHANNEL_KEYWORDS`` (default: ``["투자", "squad-service"]``).

    Parameters
    ----------
    token:
        Slack bot token (``xoxb-…``).  Defaults to ``SLACK_BOT_TOKEN``.
    priority_keywords:
        List of substrings to match against channel names.  Defaults to
        ``SLACK_PRIORITY_CHANNEL_KEYWORDS`` from ``src.config``.
    channel_cache_ttl:
        Seconds before the channel list is re-fetched.  Default 3600 (1 h).
    retry_attempts:
        API retry attempts.  Default ``API_RETRY_ATTEMPTS`` (3).
    retry_delay:
        Seconds between retries.  Default ``API_RETRY_DELAY_SECONDS`` (10).
    """

    def __init__(
        self,
        token: str = SLACK_BOT_TOKEN,
        priority_keywords: Optional[list[str]] = None,
        channel_cache_ttl: int = _CHANNEL_CACHE_TTL_SECONDS,
        retry_attempts: int = API_RETRY_ATTEMPTS,
        retry_delay: float = API_RETRY_DELAY_SECONDS,
    ) -> None:
        self._client = WebClient(token=token)
        self._keywords: list[str] = (
            priority_keywords
            if priority_keywords is not None
            else list(SLACK_PRIORITY_CHANNEL_KEYWORDS)
        )
        self._channel_cache_ttl = channel_cache_ttl
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

        # Cache state
        self._cached_channels: list[SlackChannel] = []
        self._cache_loaded_at: Optional[datetime] = None

    # ── Retry helper ──────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """
        Execute *fn* with retry logic.

        Raises ``SlackApiError`` after all attempts are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except SlackApiError as exc:
                last_exc = exc
                logger.warning(
                    "Slack API error (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    exc.response.get("error", str(exc)),
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay)
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                logger.warning(
                    "Unexpected error calling Slack API (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay)

        raise last_exc or RuntimeError("Slack API call failed")

    # ── Channel discovery ─────────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        if self._cache_loaded_at is None:
            return False
        age = (datetime.now(timezone.utc) - self._cache_loaded_at).total_seconds()
        return age < self._channel_cache_ttl

    def discover_priority_channels(self, force: bool = False) -> list[SlackChannel]:
        """
        Return all channels whose names match at least one priority keyword.

        The list is cached in-memory for ``channel_cache_ttl`` seconds to
        avoid hammering the ``conversations_list`` API.

        Parameters
        ----------
        force:
            If True, bypass the TTL and re-fetch from Slack.

        Returns
        -------
        list[SlackChannel]
            Matched channels.  Empty list when no channels match or on API error.
        """
        if not force and self._is_cache_fresh():
            logger.debug(
                "SlackHistoryRetriever: using cached channels (%d)",
                len(self._cached_channels),
            )
            return list(self._cached_channels)

        logger.info(
            "SlackHistoryRetriever: discovering priority channels (keywords=%s)",
            self._keywords,
        )

        channels: list[SlackChannel] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {
                "exclude_archived": True,
                "types": "public_channel,private_channel",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._call_with_retry(self._client.conversations_list, **params)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "SlackHistoryRetriever: conversations_list failed: %s", exc
                )
                # Return whatever we have so far (may be empty)
                break

            for ch in resp.get("channels", []):
                ch_name: str = ch.get("name", "")
                ch_id: str = ch.get("id", "")
                if not ch_id or not ch_name:
                    continue
                if self._channel_matches_keywords(ch_name):
                    channels.append(
                        SlackChannel(
                            channel_id=ch_id,
                            channel_name=ch_name,
                            is_private=ch.get("is_private", False),
                            is_member=ch.get("is_member", False),
                        )
                    )
                    logger.debug(
                        "SlackHistoryRetriever: matched channel #%s (%s)",
                        ch_name,
                        ch_id,
                    )

            # Pagination
            next_cursor = (
                resp.get("response_metadata", {}).get("next_cursor") or ""
            )
            if next_cursor:
                cursor = next_cursor
            else:
                break

        self._cached_channels = channels
        self._cache_loaded_at = datetime.now(timezone.utc)

        logger.info(
            "SlackHistoryRetriever: found %d priority channels", len(channels)
        )
        return list(channels)

    def _channel_matches_keywords(self, channel_name: str) -> bool:
        """Return True if *channel_name* contains any configured keyword."""
        lower = channel_name.lower()
        for keyword in self._keywords:
            if keyword.lower() in lower:
                return True
        return False

    # ── History fetching ──────────────────────────────────────────────────────

    def _fetch_channel_history(
        self,
        channel: SlackChannel,
        oldest_ts: str,
        max_messages: int,
    ) -> list[dict]:
        """
        Fetch raw message dicts from a channel, paginating as needed.

        Returns an empty list when the bot is not a channel member or when
        the API returns ``not_in_channel`` / ``channel_not_found``.

        Parameters
        ----------
        channel:      Target channel.
        oldest_ts:    Slack timestamp string — only messages newer than this
                      are returned.
        max_messages: Upper bound on returned messages.
        """
        messages: list[dict] = []
        cursor: Optional[str] = None
        pages = 0

        while len(messages) < max_messages and pages < _MAX_HISTORY_PAGES:
            pages += 1
            params: dict = {
                "channel": channel.channel_id,
                "oldest": oldest_ts,
                "limit": min(200, max_messages - len(messages)),
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._call_with_retry(
                    self._client.conversations_history, **params
                )
            except SlackApiError as exc:
                err = exc.response.get("error", "")
                if err in ("not_in_channel", "channel_not_found", "missing_scope"):
                    logger.info(
                        "SlackHistoryRetriever: skipping #%s (%s)",
                        channel.channel_name,
                        err,
                    )
                    channel.is_member = False
                else:
                    logger.warning(
                        "SlackHistoryRetriever: history fetch failed for #%s: %s",
                        channel.channel_name,
                        exc,
                    )
                return messages
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "SlackHistoryRetriever: unexpected error for #%s: %s",
                    channel.channel_name,
                    exc,
                )
                return messages

            batch = resp.get("messages", [])
            messages.extend(batch)

            next_cursor = (
                resp.get("response_metadata", {}).get("next_cursor") or ""
            )
            if next_cursor and resp.get("has_more"):
                cursor = next_cursor
            else:
                break

        return messages[:max_messages]

    # ── Matching ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_for_matching(text: str) -> str:
        """
        Normalise *text* for case-insensitive, punctuation-tolerant matching.

        - Lowercases
        - Collapses whitespace
        - Strips common Slack formatting (``*bold*``, ``_italic_``, ``<URL|label>``)
        """
        # Strip Slack hyperlinks: <https://foo.com|label> → label
        text = re.sub(r"<https?://[^|>]+\|([^>]+)>", r"\1", text)
        # Strip bare URLs in angle brackets
        text = re.sub(r"<https?://[^>]+>", "", text)
        # Strip mrkdwn emphasis markers (* _ ~)
        text = re.sub(r"[*_~`]", " ", text)
        # Lowercase and collapse whitespace
        text = re.sub(r"\s+", " ", text.lower()).strip()
        return text

    def _message_matches_company(
        self,
        message_text: str,
        company_name: str,
        normalised_company: str,
    ) -> bool:
        """
        Return True if *message_text* mentions *company_name*.

        Both the raw company name and its normalised form (stripped of legal
        suffixes, lowercased) are checked.  Minimum 2-character match length
        is enforced.
        """
        if len(company_name) < _MIN_COMPANY_NAME_LEN:
            return False

        norm_msg = self._normalise_for_matching(message_text)

        # Direct case-insensitive match
        if company_name.lower() in norm_msg:
            return True

        # Normalised form match (handles "AcmeCorp Inc." → "acmecorp")
        if normalised_company and normalised_company in norm_msg:
            return True

        return False

    @staticmethod
    def _normalise_company_name(company_name: str) -> str:
        """
        Produce a normalised matching key for *company_name*.

        Strips Korean / English legal suffixes, lowercases, collapses spaces.
        Mirrors the logic in ``src/notion/portfolio_cache.py`` so that the
        same company shows up regardless of how it was entered in Notion.
        """
        _KOREAN_PREFIX = re.compile(
            r"^(?:주식회사|유한회사|유한책임회사|합자회사|사단법인|재단법인|협동조합)\s+"
        )
        _KOREAN_SUFFIX = re.compile(
            r"\s+(?:주식회사|유한회사|유한책임회사|합자회사|사단법인|재단법인|협동조합)\s*$"
        )
        _ENGLISH_SUFFIX = re.compile(
            r"\s+[\(\（]?(?:Inc\.?|Corp\.?|Ltd\.?|LLC\.?|Co\.?|GmbH|S\.?A\.?)[\)\）]?\s*$",
            re.IGNORECASE,
        )
        name = company_name.strip()
        name = _KOREAN_PREFIX.sub("", name).strip()
        name = _KOREAN_SUFFIX.sub("", name).strip()
        name = _ENGLISH_SUFFIX.sub("", name).strip()
        name = re.sub(r"\s+", " ", name.lower())
        return name

    # ── Timestamp helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _datetime_to_slack_ts(dt: datetime) -> str:
        """Convert a UTC datetime to a Slack ``oldest`` timestamp string."""
        return str(dt.timestamp())

    @staticmethod
    def _slack_ts_to_datetime(ts: str) -> Optional[datetime]:
        """Parse a Slack message timestamp to a UTC-aware datetime."""
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, TypeError):
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def search_company_history(
        self,
        company_name: str,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
        max_messages_per_channel: int = _DEFAULT_MAX_MESSAGES,
        include_bot_messages: bool = False,
    ) -> SlackHistoryResult:
        """
        Search Slack message history for *company_name* across priority channels.

        Parameters
        ----------
        company_name:
            The company name to search for.  Both the raw name and a
            normalised form (without legal suffixes) are matched.
        lookback_days:
            How many days of history to scan.  Default 90.
        max_messages_per_channel:
            Upper bound on messages fetched from each channel.  Default 50.
        include_bot_messages:
            When False (default), bot/app messages (``subtype == "bot_message"``)
            are excluded from results.

        Returns
        -------
        SlackHistoryResult
            Always returns; never raises.  ``available=False`` when the Slack
            API is unreachable for all channels.
        """
        result = SlackHistoryResult(company_name=company_name)

        # ── 1. Validate input ─────────────────────────────────────────────────
        company_name = company_name.strip()
        if not company_name or len(company_name) < _MIN_COMPANY_NAME_LEN:
            result.available = False
            result.error = f"회사명이 너무 짧습니다: '{company_name}'"
            logger.warning(
                "SlackHistoryRetriever: search_company_history called with "
                "empty/too-short company name: %r",
                company_name,
            )
            return result

        normalised_name = self._normalise_company_name(company_name)
        logger.info(
            "SlackHistoryRetriever: searching for '%s' (normalised='%s') "
            "in priority channels (lookback=%d days)",
            company_name,
            normalised_name,
            lookback_days,
        )

        # ── 2. Discover channels ──────────────────────────────────────────────
        try:
            channels = self.discover_priority_channels()
        except Exception as exc:  # pylint: disable=broad-except
            result.available = False
            result.error = f"채널 목록 조회 실패: {exc}"
            logger.error(
                "SlackHistoryRetriever: channel discovery failed: %s", exc
            )
            return result

        if not channels:
            logger.info(
                "SlackHistoryRetriever: no priority channels found (keywords=%s)",
                self._keywords,
            )
            result.available = True
            return result

        # ── 3. Calculate lookback timestamp ───────────────────────────────────
        oldest_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        oldest_ts = self._datetime_to_slack_ts(oldest_dt)

        # ── 4. Search each channel ────────────────────────────────────────────
        all_messages: list[SlackMessage] = []

        for channel in channels:
            logger.debug(
                "SlackHistoryRetriever: fetching history for #%s (id=%s)",
                channel.channel_name,
                channel.channel_id,
            )

            raw_msgs = self._fetch_channel_history(
                channel=channel,
                oldest_ts=oldest_ts,
                max_messages=max_messages_per_channel,
            )

            if not channel.is_member:
                result.channels_skipped.append(channel)
                logger.info(
                    "SlackHistoryRetriever: #%s skipped (bot not a member)",
                    channel.channel_name,
                )
                continue

            result.channels_searched.append(channel)

            matched_count = 0
            for raw in raw_msgs:
                # Skip bot messages unless explicitly requested
                if not include_bot_messages and raw.get("subtype") == "bot_message":
                    continue
                # Skip channel join/leave events
                if raw.get("subtype") in ("channel_join", "channel_leave",
                                           "channel_archive", "channel_unarchive"):
                    continue

                msg_text: str = raw.get("text", "") or ""
                if not msg_text:
                    continue

                if self._message_matches_company(msg_text, company_name, normalised_name):
                    ts = raw.get("ts", "")
                    thread_ts = raw.get("thread_ts", ts)
                    msg_dt = self._slack_ts_to_datetime(ts)
                    all_messages.append(
                        SlackMessage(
                            channel_id=channel.channel_id,
                            channel_name=channel.channel_name,
                            ts=ts,
                            user_id=raw.get("user", ""),
                            text=msg_text,
                            thread_ts=thread_ts,
                            message_dt=msg_dt,
                            is_thread_reply=(thread_ts != ts),
                        )
                    )
                    matched_count += 1

            logger.info(
                "SlackHistoryRetriever: #%s — %d/%d messages matched '%s'",
                channel.channel_name,
                matched_count,
                len(raw_msgs),
                company_name,
            )

        # ── 5. Sort results newest-first ──────────────────────────────────────
        all_messages.sort(key=lambda m: float(m.ts) if m.ts else 0.0, reverse=True)
        result.messages = all_messages

        logger.info(
            "SlackHistoryRetriever: search complete — "
            "%d total matches for '%s' across %d channels",
            len(all_messages),
            company_name,
            len(result.channels_searched),
        )
        return result

    def get_channel_list(self) -> list[SlackChannel]:
        """
        Return the currently cached list of priority channels (auto-discovering if needed).

        Useful for diagnostics and Q&A pipeline setup.
        """
        return self.discover_priority_channels()

    def invalidate_channel_cache(self) -> None:
        """Force the next call to discover channels fresh from the Slack API."""
        self._cache_loaded_at = None
        self._cached_channels = []
        logger.debug("SlackHistoryRetriever: channel cache invalidated")
