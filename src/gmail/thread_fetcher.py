"""
Gmail Thread Fetcher — Sub-AC 7.2

Retrieves the most recent email exchange for a given company domain or
specific contact, and returns a structured summary of that exchange.

Provides:
  - ThreadSummary             – structured summary data model
  - get_threads_for_company() – fetch EmailThread objects by domain/contact
  - build_thread_summary()    – convert an EmailThread to a ThreadSummary
  - get_latest_thread_summary() – top-level: most recent thread summary
  - get_all_thread_summaries()  – top-level: all recent thread summaries

Design notes
------------
* Domain-only search uses Gmail's @domain syntax so a query like
  ``from:@startup.com OR to:@startup.com`` matches all addresses at that domain.
* When a specific contact_email is given it takes precedence (narrower query).
* Direction is determined from the latest message's sender:
    inbound  = external company sent last
    outbound = internal domain sent last (we're waiting for their reply)
* Retries follow the project-wide 3-attempts / 10s-interval policy.
* On total failure an empty list (or None) is returned rather than raising,
  so callers can annotate with '확인 불가' without crashing.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
    INTERNAL_DOMAIN,
)
from src.gmail.gmail_client import GmailClient, EmailThread, EmailMessage

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_MAX_THREADS: int = 5
_DEFAULT_LOOKBACK_DAYS: int = 90
# Thread is "active" if its latest message is within this many days
_ACTIVE_DAYS_THRESHOLD: int = 7
# "waiting_reply" if we sent last AND it was within this many days
_WAITING_REPLY_DAYS: int = 3


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ThreadSummary:
    """
    Structured summary of an email thread with a company or contact.

    Fields
    ------
    thread_id        : Gmail thread identifier.
    subject          : Thread subject line.
    company_domain   : Domain that was searched (e.g. "startup.com").
    contact_email    : Specific contact email that was searched, or None when
                       only the domain was used.
    latest_date      : UTC datetime of the most recent message in the thread.
    message_count    : Total number of messages in the thread.
    participants     : Unique email addresses across all messages.
    last_sender      : Display name of the latest message's sender.
    last_sender_email: Bare email address of the latest message's sender.
    last_snippet     : Gmail-provided snippet (short preview) of the latest msg.
    last_body_preview: First 500 chars of the latest message body (plain text).
    direction        : "inbound"  – external contact sent last;
                       "outbound" – we (internal domain) sent last;
                       "unknown"  – cannot determine.
    status           : "active"        – latest msg ≤ _ACTIVE_DAYS_THRESHOLD days ago;
                       "waiting_reply" – latest msg was outbound AND ≤ _WAITING_REPLY_DAYS;
                       "stale"         – latest msg older than _ACTIVE_DAYS_THRESHOLD;
                       "unknown"       – no date available.
    messages         : Serialised list of all messages (to_dict output).
    """

    thread_id: str
    subject: str

    # Search context
    company_domain: str
    contact_email: Optional[str]  # None when only domain was specified

    # Thread metadata
    latest_date: Optional[datetime]
    message_count: int
    participants: list[str]

    # Latest message details
    last_sender: str       # Display name (may be empty)
    last_sender_email: str # Bare email address
    last_snippet: str
    last_body_preview: str  # First ≤500 chars of body

    # Thread characterisation
    direction: str  # "inbound" | "outbound" | "unknown"
    status: str     # "active" | "waiting_reply" | "stale" | "unknown"

    # All messages in serialised form (optional deeper analysis)
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "company_domain": self.company_domain,
            "contact_email": self.contact_email,
            "latest_date": self.latest_date.isoformat() if self.latest_date else None,
            "message_count": self.message_count,
            "participants": self.participants,
            "last_sender": self.last_sender,
            "last_sender_email": self.last_sender_email,
            "last_snippet": self.last_snippet,
            "last_body_preview": self.last_body_preview,
            "direction": self.direction,
            "status": self.status,
            "messages": self.messages,
        }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_domain(email_addr: str) -> str:
    """Return the domain portion of an email address (lower-cased)."""
    try:
        return email_addr.strip().lower().split("@")[1]
    except (IndexError, AttributeError):
        return ""


def _extract_address(raw: str) -> str:
    """Extract the bare email address from a 'Display Name <addr>' string."""
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def _is_internal_email(email_addr: str) -> bool:
    """Return True when *email_addr* belongs to the configured internal domain."""
    domain = _extract_domain(email_addr)
    return bool(domain) and domain == INTERNAL_DOMAIN.lower()


def _latest_message(thread: EmailThread) -> Optional[EmailMessage]:
    """
    Return the message with the most recent date from *thread*.
    Falls back to the last element of the list if no dates are present.
    """
    dated = [(m, m.date) for m in thread.messages if m.date]
    if dated:
        return max(dated, key=lambda x: x[1])[0]
    return thread.messages[-1] if thread.messages else None


def _determine_direction(thread: EmailThread) -> str:
    """
    Classify the thread's last communication direction.

    Returns
    -------
    "inbound"  – latest message came from an external (non-internal) sender.
    "outbound" – latest message came from an internal-domain sender.
    "unknown"  – sender address could not be parsed.
    """
    last_msg = _latest_message(thread)
    if last_msg is None:
        return "unknown"

    raw_sender = last_msg.sender or ""
    sender_addr = _extract_address(raw_sender) if raw_sender else ""
    if not sender_addr:
        return "unknown"

    return "outbound" if _is_internal_email(sender_addr) else "inbound"


def _determine_status(thread: EmailThread, direction: str) -> str:
    """
    Classify the thread's current status.

    Returns
    -------
    "active"        – latest message arrived within _ACTIVE_DAYS_THRESHOLD days.
    "waiting_reply" – we sent last (outbound) AND it was within _WAITING_REPLY_DAYS.
    "stale"         – latest message older than _ACTIVE_DAYS_THRESHOLD days.
    "unknown"       – no date information available.
    """
    latest = thread.latest_date
    if latest is None:
        return "unknown"

    now = datetime.now(timezone.utc)
    # Ensure latest is tz-aware for subtraction
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    days_ago = (now - latest).days

    if days_ago > _ACTIVE_DAYS_THRESHOLD:
        return "stale"
    if direction == "outbound" and days_ago <= _WAITING_REPLY_DAYS:
        return "waiting_reply"
    return "active"


def _parse_display_name(raw_sender: str) -> str:
    """
    Extract just the display name from a 'Display Name <email>' header.
    Returns the bare address when no display name is present.
    """
    if "<" in raw_sender:
        name_part = raw_sender.split("<")[0].strip().strip('"').strip("'")
        return name_part if name_part else _extract_address(raw_sender)
    return raw_sender.strip()


# ── Core public functions ──────────────────────────────────────────────────────

def get_threads_for_company(
    client: GmailClient,
    domain: str,
    contact_email: Optional[str] = None,
    max_threads: int = _DEFAULT_MAX_THREADS,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> list[EmailThread]:
    """
    Fetch recent Gmail threads involving a company domain or specific contact.

    Strategy
    --------
    1. If *contact_email* is given:  ``from:<email> OR to:<email>``
    2. Otherwise (domain only):      ``from:@<domain> OR to:@<domain>``
    Both queries are restricted to the last *lookback_days* days.

    The returned list is sorted newest-first (by latest message date).
    On any API failure the client's own retry logic runs first; if the error
    propagates here the exception bubbles up so callers can handle it.

    Parameters
    ----------
    client        : Authenticated and connected GmailClient.
    domain        : Company domain to search (e.g. "startup.com").
    contact_email : Specific contact address.  When provided, domain is still
                    recorded in summaries but the query is narrowed to this address.
    max_threads   : Maximum threads to return.
    lookback_days : Lookback window in days.

    Returns
    -------
    List of EmailThread objects sorted newest-first; empty list when no matches.
    """
    lookback_str = f"newer_than:{lookback_days}d"

    if contact_email:
        query = f"(from:{contact_email} OR to:{contact_email}) {lookback_str}"
    else:
        # @domain matches all addresses at that domain in Gmail search syntax
        query = f"(from:@{domain} OR to:@{domain}) {lookback_str}"

    logger.info(
        "get_threads_for_company: domain=%r contact=%r query=%r max=%d",
        domain, contact_email, query, max_threads,
    )

    threads = client.search_threads(query, max_results=max_threads)
    logger.info(
        "get_threads_for_company: found %d thread(s) for domain=%r contact=%r",
        len(threads), domain, contact_email,
    )
    return threads


def build_thread_summary(
    thread: EmailThread,
    domain: str,
    contact_email: Optional[str] = None,
) -> ThreadSummary:
    """
    Convert an EmailThread into a structured ThreadSummary.

    All fields are populated from the thread's messages.  The function never
    raises; if a field cannot be determined it defaults to "" / None / "unknown".

    Parameters
    ----------
    thread        : The EmailThread to summarise.
    domain        : The company domain this thread was found for.
    contact_email : The specific contact that was queried (or None).

    Returns
    -------
    Fully populated ThreadSummary.
    """
    last_msg = _latest_message(thread)

    if last_msg is not None:
        raw_sender = last_msg.sender or ""
        last_sender_email = _extract_address(raw_sender) if raw_sender else ""
        last_sender = _parse_display_name(raw_sender) if raw_sender else last_sender_email
        last_snippet = last_msg.snippet or ""
        # body_text is already truncated to 2000 chars in EmailMessage.to_dict();
        # we take at most 500 chars here for the summary preview.
        last_body_preview = (last_msg.body_text or "")[:500]
    else:
        last_sender = ""
        last_sender_email = ""
        last_snippet = ""
        last_body_preview = ""

    direction = _determine_direction(thread)
    status = _determine_status(thread, direction)

    return ThreadSummary(
        thread_id=thread.thread_id,
        subject=thread.subject,
        company_domain=domain,
        contact_email=contact_email,
        latest_date=thread.latest_date,
        message_count=thread.message_count,
        participants=thread.participants,
        last_sender=last_sender,
        last_sender_email=last_sender_email,
        last_snippet=last_snippet,
        last_body_preview=last_body_preview,
        direction=direction,
        status=status,
        messages=[m.to_dict() for m in thread.messages],
    )


def get_latest_thread_summary(
    domain: str,
    contact_email: Optional[str] = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    client: Optional[GmailClient] = None,
) -> Optional[ThreadSummary]:
    """
    High-level function: return the most recent email exchange summary for a
    company domain or contact.

    Implements 3-retry / 10s-delay error handling per project requirements.
    Returns None (never raises) when no threads are found or all retries fail.

    Parameters
    ----------
    domain        : Company email domain (e.g. "startup.com").
    contact_email : Specific contact address (optional).
    lookback_days : How far back to search (default 90 days).
    client        : Pre-created GmailClient to reuse.  Created internally when None.

    Returns
    -------
    ThreadSummary for the most recent thread, or None.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            _client: GmailClient
            if client is None:
                _client = GmailClient()
                _client.connect()
            else:
                _client = client

            threads = get_threads_for_company(
                client=_client,
                domain=domain,
                contact_email=contact_email,
                max_threads=1,
                lookback_days=lookback_days,
            )

            if not threads:
                logger.info(
                    "get_latest_thread_summary: no threads for domain=%r contact=%r",
                    domain, contact_email,
                )
                return None

            summary = build_thread_summary(
                thread=threads[0],
                domain=domain,
                contact_email=contact_email,
            )
            logger.info(
                "get_latest_thread_summary: thread=%r subject=%r direction=%r status=%r",
                summary.thread_id, summary.subject,
                summary.direction, summary.status,
            )
            return summary

        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logger.warning(
                "get_latest_thread_summary: attempt %d/%d failed for domain=%r: %s",
                attempt, API_RETRY_ATTEMPTS, domain, exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)

    logger.error(
        "get_latest_thread_summary: all %d attempts failed for domain=%r. "
        "Last error: %s",
        API_RETRY_ATTEMPTS, domain, last_exc,
    )
    return None


def get_all_thread_summaries(
    domain: str,
    contact_email: Optional[str] = None,
    max_threads: int = _DEFAULT_MAX_THREADS,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    client: Optional[GmailClient] = None,
) -> list[ThreadSummary]:
    """
    Return summaries for all recent email threads with a company domain or contact.

    Implements 3-retry / 10s-delay error handling.  Returns an empty list
    (never raises) on total failure so callers can annotate with '확인 불가'.

    Parameters
    ----------
    domain        : Company email domain (e.g. "startup.com").
    contact_email : Specific contact address (optional).
    max_threads   : Maximum number of thread summaries to return (default 5).
    lookback_days : How far back to search (default 90 days).
    client        : Pre-created GmailClient to reuse.  Created internally when None.

    Returns
    -------
    List of ThreadSummary objects sorted newest-first; empty list on failure.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            _client: GmailClient
            if client is None:
                _client = GmailClient()
                _client.connect()
            else:
                _client = client

            threads = get_threads_for_company(
                client=_client,
                domain=domain,
                contact_email=contact_email,
                max_threads=max_threads,
                lookback_days=lookback_days,
            )

            summaries = [
                build_thread_summary(thread=t, domain=domain, contact_email=contact_email)
                for t in threads
            ]

            logger.info(
                "get_all_thread_summaries: %d summary(ies) for domain=%r contact=%r",
                len(summaries), domain, contact_email,
            )
            return summaries

        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logger.warning(
                "get_all_thread_summaries: attempt %d/%d failed for domain=%r: %s",
                attempt, API_RETRY_ATTEMPTS, domain, exc,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)

    logger.error(
        "get_all_thread_summaries: all %d attempts failed for domain=%r. "
        "Last error: %s",
        API_RETRY_ATTEMPTS, domain, last_exc,
    )
    return []
