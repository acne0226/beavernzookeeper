"""
Gmail client for fetching relevant emails.

Uses the Gmail REST API via google-api-python-client with OAuth2 authentication.
Reuses the same Google OAuth2 credentials (google_token.json) which already
includes the gmail.readonly scope.

Provides:
- GmailClient         – authenticated wrapper around the Gmail API
- EmailMessage        – structured email data model
- fetch_recent_emails()  – convenience top-level function
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email import message_from_bytes
from email.header import decode_header as _decode_header
from typing import Any, Optional
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN,
    GOOGLE_TOKEN_FILE,
    INTERNAL_DOMAIN,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

# Gmail requires its own scope; the existing google_token.json already includes it.
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
]

KST = ZoneInfo("Asia/Seoul")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class EmailMessage:
    """Structured email message representation."""

    message_id: str
    thread_id: str
    subject: str
    sender: str          # Display name (may be empty)
    sender_email: str    # Parsed email address
    snippet: str         # Short preview text from Gmail
    received_at: datetime
    is_unread: bool
    labels: list[str] = field(default_factory=list)
    body_preview: str = ""  # First ~300 chars of decoded body (best-effort)

    @property
    def is_external(self) -> bool:
        """True when sender is NOT from the internal domain."""
        return (
            bool(self.sender_email)
            and not self.sender_email.lower().endswith(f"@{INTERNAL_DOMAIN}")
        )

    @property
    def is_important(self) -> bool:
        """True when Gmail has marked this email as important (IMPORTANT label)."""
        return "IMPORTANT" in self.labels

    @property
    def is_urgent(self) -> bool:
        """
        True when the email requires immediate attention.

        Urgency criteria (any of):
        - Is unread AND sent from an external (non-internal) domain
        - Is explicitly marked important by Gmail (IMPORTANT label)

        This intentionally avoids marking internal unread messages as urgent
        so that the briefing focuses the user's attention on actionable external
        communication rather than internal chatter.
        """
        return (self.is_unread and self.is_external) or self.is_important

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "snippet": self.snippet,
            "received_at": self.received_at.isoformat(),
            "is_unread": self.is_unread,
            "labels": self.labels,
            "is_external": self.is_external,
            "is_important": self.is_important,
            "is_urgent": self.is_urgent,
            "body_preview": self.body_preview,
        }


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _build_gmail_credentials() -> Credentials:
    """
    Build Google OAuth2 credentials for Gmail access.

    Reuses the same persisted token file (google_token.json) which already
    contains the gmail.readonly scope from the original OAuth2 grant.
    Falls back to building from env vars and refreshing if the file is absent.
    """
    creds: Optional[Credentials] = None

    # Load existing persisted token (may include gmail.readonly already)
    if GOOGLE_TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(GOOGLE_TOKEN_FILE), GMAIL_SCOPES
            )
        except Exception as exc:
            logger.warning("Gmail: could not load persisted token: %s", exc)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Gmail: refreshing expired Google token…")
            creds.refresh(Request())
        else:
            logger.info("Gmail: building credentials from env vars…")
            creds = Credentials(
                token=None,
                refresh_token=GOOGLE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                scopes=GMAIL_SCOPES,
            )
            creds.refresh(Request())

        # Persist updated token
        try:
            GOOGLE_TOKEN_FILE.write_text(creds.to_json())
            logger.debug("Gmail token persisted to %s", GOOGLE_TOKEN_FILE)
        except Exception as exc:
            logger.warning("Gmail: could not persist token: %s", exc)

    return creds


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_mime_words(s: str) -> str:
    """Decode MIME-encoded header words (e.g. =?UTF-8?B?...?=)."""
    if not s:
        return ""
    parts = _decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _parse_sender(from_header: str) -> tuple[str, str]:
    """
    Parse a From: header into (display_name, email_address).

    Handles formats:
      - "Alice Bob <alice@example.com>"
      - "alice@example.com"
      - "=?UTF-8?B?...?= <alice@example.com>"
    """
    from_header = _decode_mime_words(from_header).strip()
    if "<" in from_header and ">" in from_header:
        name_part, _, addr_part = from_header.partition("<")
        email_addr = addr_part.rstrip(">").strip()
        display_name = name_part.strip().strip('"')
    else:
        email_addr = from_header.strip()
        display_name = ""
    return display_name, email_addr


def _extract_body_preview(payload: dict, max_len: int = 300) -> str:
    """
    Extract a plain-text body preview from a Gmail message payload.

    Walks multipart/text message parts and returns the first *max_len*
    characters of the decoded body. Returns empty string on any failure.
    """
    try:
        mime_type = payload.get("mimeType", "")
        parts = payload.get("parts", [])
        body_data = payload.get("body", {}).get("data", "")

        if mime_type == "text/plain" and body_data:
            raw = base64.urlsafe_b64decode(body_data + "==")
            return raw.decode("utf-8", errors="replace")[:max_len]

        for part in parts:
            if part.get("mimeType") == "text/plain":
                part_data = part.get("body", {}).get("data", "")
                if part_data:
                    raw = base64.urlsafe_b64decode(part_data + "==")
                    return raw.decode("utf-8", errors="replace")[:max_len]

        # Fallback: recurse into nested parts
        for part in parts:
            result = _extract_body_preview(part, max_len)
            if result:
                return result
    except Exception as exc:
        logger.debug("_extract_body_preview error: %s", exc)
    return ""


def _parse_message(msg: dict) -> EmailMessage:
    """Parse a raw Gmail API message dict into an EmailMessage."""
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

    subject = _decode_mime_words(headers.get("Subject", "(제목 없음)"))
    from_raw = headers.get("From", "")
    display_name, email_addr = _parse_sender(from_raw)

    # Gmail internal date is milliseconds since epoch
    internal_date_ms = int(msg.get("internalDate", "0"))
    received_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc)

    label_ids: list[str] = msg.get("labelIds", [])
    is_unread = "UNREAD" in label_ids

    body_preview = _extract_body_preview(msg.get("payload", {}))

    return EmailMessage(
        message_id=msg.get("id", ""),
        thread_id=msg.get("threadId", ""),
        subject=subject,
        sender=display_name,
        sender_email=email_addr,
        snippet=msg.get("snippet", ""),
        received_at=received_at,
        is_unread=is_unread,
        labels=label_ids,
        body_preview=body_preview,
    )


# ── Client ─────────────────────────────────────────────────────────────────────

class GmailClient:
    """Thin wrapper around the Gmail v1 REST API."""

    def __init__(self) -> None:
        self._creds: Optional[Credentials] = None
        self._service: Any = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate and build the Gmail API service object."""
        self._creds = _build_gmail_credentials()
        self._service = build("gmail", "v1", credentials=self._creds, cache_discovery=False)
        logger.info("GmailClient connected.")

    def _ensure_connected(self) -> None:
        if self._service is None:
            self.connect()
        if self._creds and self._creds.expired:
            self._creds.refresh(Request())
            self._service = build("gmail", "v1", credentials=self._creds, cache_discovery=False)

    # ── retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute *fn* with up to API_RETRY_ATTEMPTS retries on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_connected()
                return fn(*args, **kwargs)
            except HttpError as exc:
                logger.warning(
                    "Gmail API HttpError (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "Gmail API error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"Gmail API failed after {API_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_recent_emails(
        self,
        days: int = 1,
        max_results: int = 50,
        query: str = "",
    ) -> list[EmailMessage]:
        """
        Fetch emails received in the last *days* days.

        Parameters
        ----------
        days:
            Lookback window in days (default: 1 = today's emails).
        max_results:
            Maximum number of messages to return (default: 50).
        query:
            Additional Gmail search query string appended to the date filter.
            Example: "is:unread" or "from:startup.com".

        Returns
        -------
        List of EmailMessage objects sorted by received_at descending.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        # Gmail query: after:YYYY/MM/DD format
        date_filter = f"after:{since.strftime('%Y/%m/%d')}"
        full_query = f"{date_filter} {query}".strip()

        logger.info(
            "GmailClient.fetch_recent_emails: query=%r max_results=%d",
            full_query,
            max_results,
        )

        def _list_messages():
            return (
                self._service.users()
                .messages()
                .list(userId="me", q=full_query, maxResults=max_results)
                .execute()
            )

        result = self._call_with_retry(_list_messages)
        message_refs = result.get("messages", [])

        if not message_refs:
            logger.info("GmailClient: no messages found for query %r", full_query)
            return []

        emails: list[EmailMessage] = []
        for ref in message_refs:
            try:
                msg = self._fetch_message(ref["id"])
                if msg:
                    emails.append(msg)
            except Exception as exc:
                logger.warning("Failed to fetch message %s: %s", ref["id"], exc)

        # Sort newest first
        emails.sort(key=lambda e: e.received_at, reverse=True)
        logger.info(
            "GmailClient.fetch_recent_emails: fetched %d messages", len(emails)
        )
        return emails

    def _fetch_message(self, message_id: str) -> Optional[EmailMessage]:
        """Fetch a single message by ID with full payload."""
        def _get():
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

        msg = self._call_with_retry(_get)
        return _parse_message(msg)

    def fetch_unread_emails(
        self,
        days: int = 1,
        max_results: int = 50,
    ) -> list[EmailMessage]:
        """
        Convenience wrapper: fetch only unread emails from the last *days* days.
        """
        return self.fetch_recent_emails(
            days=days, max_results=max_results, query="is:unread"
        )

    def fetch_inbox_emails(
        self,
        days: int = 1,
        max_results: int = 50,
    ) -> list[EmailMessage]:
        """
        Fetch emails in the INBOX (any read status) from the last *days* days.
        This is the primary method for daily briefing email collection.
        """
        return self.fetch_recent_emails(
            days=days, max_results=max_results, query="in:inbox"
        )


# ── Module-level convenience function ─────────────────────────────────────────

def fetch_recent_emails(days: int = 1, max_results: int = 50) -> list[EmailMessage]:
    """
    Top-level convenience function used by the briefing aggregator.

    Creates a GmailClient, fetches inbox emails for the last *days* days,
    and returns them sorted newest-first.

    Returns an empty list (not an exception) on API failure.
    """
    try:
        client = GmailClient()
        return client.fetch_inbox_emails(days=days, max_results=max_results)
    except Exception as exc:
        logger.error("fetch_recent_emails failed: %s", exc)
        return []
