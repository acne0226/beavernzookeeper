"""
Gmail client for fetching email threads relevant to upcoming meetings.

Provides:
- GmailClient: OAuth2-authenticated wrapper around the Gmail v1 API
- EmailMessage / EmailThread: structured data models
- get_threads_for_meeting(): high-level helper used by the briefing aggregator
"""
from __future__ import annotations

import base64
import email.utils
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
    GMAIL_TOKEN_FILE,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN,
)

logger = logging.getLogger(__name__)

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ── Korean / English stop-words to ignore when building keyword queries ─────
_STOP_WORDS = {
    # Korean
    "미팅", "회의", "미팅", "논의", "검토", "협의", "관련", "건", "관해", "대해",
    "및", "와", "과", "이", "가", "을", "를", "의", "에", "는", "은",
    # English
    "meeting", "call", "sync", "discussion", "regarding", "re", "about",
    "with", "the", "and", "for", "of", "in", "on", "at", "to", "a", "an",
}


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class EmailMessage:
    """A single Gmail message with key headers and body text."""

    message_id: str
    thread_id: str
    subject: str
    sender: str
    recipients: list[str] = field(default_factory=list)
    date: Optional[datetime] = None
    snippet: str = ""
    body_text: str = ""
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "recipients": self.recipients,
            "date": self.date.isoformat() if self.date else None,
            "snippet": self.snippet,
            # Truncate body to 2 000 chars for briefing context
            "body_text": self.body_text[:2000] if self.body_text else "",
        }


@dataclass
class EmailThread:
    """A Gmail thread (conversation) with one or more messages."""

    thread_id: str
    subject: str
    messages: list[EmailMessage] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def latest_date(self) -> Optional[datetime]:
        dates = [m.date for m in self.messages if m.date]
        return max(dates) if dates else None

    @property
    def participants(self) -> list[str]:
        """Unique sender/recipient email addresses across the thread."""
        seen: set[str] = set()
        result: list[str] = []
        for msg in self.messages:
            for raw in [msg.sender] + msg.recipients:
                addr = _extract_address(raw)
                if addr and addr not in seen:
                    seen.add(addr)
                    result.append(addr)
        return result

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "message_count": self.message_count,
            "latest_date": self.latest_date.isoformat() if self.latest_date else None,
            "participants": self.participants,
            "messages": [m.to_dict() for m in self.messages],
        }


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _build_gmail_credentials() -> Credentials:
    """
    Build Gmail OAuth2 credentials from env vars, refreshing as needed.
    Persists the token to GMAIL_TOKEN_FILE to avoid repeated logins.
    """
    creds: Optional[Credentials] = None

    if GMAIL_TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN_FILE), GMAIL_SCOPES)
        except Exception as exc:
            logger.warning("Could not load Gmail token: %s", exc)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Gmail token...")
            creds.refresh(Request())
        else:
            logger.info("Building Gmail credentials from env vars...")
            creds = Credentials(
                token=None,
                refresh_token=GOOGLE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                scopes=GMAIL_SCOPES,
            )
            creds.refresh(Request())

        try:
            GMAIL_TOKEN_FILE.write_text(creds.to_json())
            logger.debug("Gmail token persisted to %s", GMAIL_TOKEN_FILE)
        except Exception as exc:
            logger.warning("Could not persist Gmail token: %s", exc)

    return creds


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_email_date(date_str: str) -> Optional[datetime]:
    """Parse an RFC 2822 date string from an email header into a UTC datetime."""
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_address(raw: str) -> str:
    """Extract bare email address from a 'Display Name <addr>' string."""
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip().lower()
    # Bare address
    return raw.strip().lower()


def _decode_part(part: dict) -> str:
    """Base64url-decode a single MIME part body."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_text(payload: dict, depth: int = 0) -> str:
    """
    Recursively walk MIME parts and extract plain-text content.
    Prefers text/plain; falls back to stripping HTML tags from text/html.
    """
    if depth > 10:
        return ""

    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        return _decode_part(payload)

    if mime_type == "text/html" and not parts:
        html = _decode_part(payload)
        # Strip tags crudely
        return re.sub(r"<[^>]+>", " ", html).strip()

    # multipart/* – recurse
    texts: list[str] = []
    for p in parts:
        text = _extract_body_text(p, depth + 1)
        if text.strip():
            texts.append(text.strip())
            # Stop after first useful plain-text block for brevity
            if "text/plain" in p.get("mimeType", ""):
                break

    return "\n".join(texts)


def _extract_keywords(title: str, max_keywords: int = 5) -> list[str]:
    """
    Extract meaningful search keywords from a meeting title.
    Strips stop-words and short tokens; returns up to *max_keywords*.
    """
    # Split on whitespace, brackets, slashes, dashes, underscores
    tokens = re.split(r"[\s/\-_\[\]()]+", title)
    keywords: list[str] = []
    for tok in tokens:
        tok_lower = tok.lower().strip(".,!?;:")
        if len(tok_lower) >= 2 and tok_lower not in _STOP_WORDS:
            keywords.append(tok_lower)
        if len(keywords) >= max_keywords:
            break
    return keywords


# ── Client ───────────────────────────────────────────────────────────────────

class GmailClient:
    """Thin wrapper around the Gmail v1 REST API focused on thread search."""

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

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse_message(self, msg_data: dict) -> Optional[EmailMessage]:
        """Parse a Gmail API message resource into an EmailMessage."""
        payload = msg_data.get("payload", {})
        headers = {
            h["name"].lower(): h["value"]
            for h in payload.get("headers", [])
        }

        subject = headers.get("subject", "")
        sender = headers.get("from", "")
        to_raw = headers.get("to", "")
        cc_raw = headers.get("cc", "")
        date_str = headers.get("date", "")

        recipients: list[str] = []
        for raw in [to_raw, cc_raw]:
            for part in raw.split(","):
                part = part.strip()
                if part:
                    recipients.append(part)

        date = _parse_email_date(date_str)
        snippet = msg_data.get("snippet", "")
        body_text = _extract_body_text(payload)
        labels = msg_data.get("labelIds", [])

        return EmailMessage(
            message_id=msg_data.get("id", ""),
            thread_id=msg_data.get("threadId", ""),
            subject=subject,
            sender=sender,
            recipients=recipients,
            date=date,
            snippet=snippet,
            body_text=body_text,
            labels=labels,
        )

    def _fetch_thread(self, thread_id: str) -> Optional[EmailThread]:
        """Fetch and parse a complete Gmail thread."""
        def _get():
            return (
                self._service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )

        try:
            data = self._call_with_retry(_get)
        except RuntimeError:
            logger.error("Failed to fetch thread %s", thread_id)
            return None

        messages_data = data.get("messages", [])
        messages = [self._parse_message(m) for m in messages_data]
        messages = [m for m in messages if m is not None]

        subject = messages[0].subject if messages else ""
        return EmailThread(thread_id=thread_id, subject=subject, messages=messages)

    # ── public API ────────────────────────────────────────────────────────────

    def search_threads(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[EmailThread]:
        """
        Search Gmail for threads matching *query* (Gmail search syntax).
        Returns parsed EmailThread objects sorted newest-first.
        """
        def _list():
            return (
                self._service.users()
                .threads()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

        try:
            result = self._call_with_retry(_list)
        except RuntimeError:
            logger.error("Thread list search failed for query: %s", query)
            return []

        thread_stubs = result.get("threads", [])
        threads: list[EmailThread] = []
        for stub in thread_stubs:
            thread = self._fetch_thread(stub["id"])
            if thread:
                threads.append(thread)

        return sorted(
            threads,
            key=lambda t: t.latest_date or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    def get_threads_for_meeting(
        self,
        external_emails: list[str],
        meeting_title: str,
        lookback_days: int = 30,
        max_threads: int = 10,
    ) -> list[EmailThread]:
        """
        Fetch Gmail threads relevant to an upcoming meeting.

        Strategy:
        1. For each external attendee email (up to 3), search threads from/to
           that address within the past *lookback_days* days.
        2. Search threads matching keywords extracted from *meeting_title*.
        3. Deduplicate by thread_id, sort newest-first, cap at *max_threads*.

        Returns an empty list (with a warning log) on any API failure rather
        than raising, so the briefing pipeline can still run.
        """
        seen: dict[str, EmailThread] = {}
        lookback_str = f"newer_than:{lookback_days}d"

        # 1. Per-attendee email search (cap at 3 to limit API calls)
        for email_addr in external_emails[:3]:
            query = f"(from:{email_addr} OR to:{email_addr}) {lookback_str}"
            logger.debug("Gmail search: %s", query)
            threads = self.search_threads(query, max_results=5)
            for t in threads:
                seen[t.thread_id] = t

        # 2. Keyword search from meeting title
        keywords = _extract_keywords(meeting_title)
        if keywords:
            kw_query = " OR ".join(f'"{kw}"' for kw in keywords)
            query = f"({kw_query}) {lookback_str}"
            logger.debug("Gmail keyword search: %s", query)
            threads = self.search_threads(query, max_results=5)
            for t in threads:
                seen[t.thread_id] = t

        sorted_threads = sorted(
            seen.values(),
            key=lambda t: t.latest_date or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        logger.info(
            "Gmail context: found %d unique threads for meeting '%s'",
            len(sorted_threads),
            meeting_title,
        )
        return sorted_threads[:max_threads]
