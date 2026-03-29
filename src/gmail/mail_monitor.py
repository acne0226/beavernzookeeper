"""
Portfolio company email monitor (Feature 2, ACs 9–16).

Responsibilities
----------------
* Scan Gmail inbox for emails from portfolio companies (AC 9).
* Identify portfolio companies by matching sender names/domains against the
  Notion startup DB via SenderMatcher; fall back to web search for brand-vs-legal
  name mismatches (AC 9).
* Cache all company-name mappings persistently so no sender is resolved twice
  (AC 10).
* Extract deadlines from email bodies using regex + keyword heuristics; default
  to ``received_date + 3 days`` when no explicit deadline is found (AC 11).
* Detect *deadline approaching* (1 day before) and *overdue* (past deadline)
  states and report them so the scheduler can alert via Slack DM (ACs 12, 13).
* Detect *missed replies*: portfolio emails that have not been answered within
  the reply window (AC 14).
* Expose ``run_mail_status_check()`` for the /mail slash command (AC 15).
* Persist mail status to JSON so state survives daemon restarts (AC 16).

Threading / concurrency
-----------------------
``PortfolioMailMonitor`` is safe to call from multiple threads; an internal
threading.Lock protects all writes to the ``_mail_records`` dictionary.

Retry policy
------------
All external API calls (Gmail, Notion, web-search) retry up to
``API_RETRY_ATTEMPTS`` times with ``API_RETRY_DELAY_SECONDS`` delay.
"""
from __future__ import annotations

import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Any

from src.config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
    ROOT_DIR,
    INTERNAL_DOMAIN,
    INVESTMENT_TEAM_EMAILS,
)
from src.gmail.gmail_client import GmailClient
from src.gmail.company_name_cache import get_company_name_cache, CompanyNameCache
from src.gmail.sender_matcher import SenderMatcher
from src.notion.portfolio_cache import get_portfolio_cache, PortfolioCache

logger = logging.getLogger(__name__)

# ── Configuration constants ───────────────────────────────────────────────────

# Persist mail status records here
MAIL_STATUS_FILE: Path = ROOT_DIR / "portfolio_mail_status.json"

# How many days back to scan Gmail for portfolio emails
MAIL_SCAN_LOOKBACK_DAYS: int = 30

# Default deadline = received_date + this many days (when not extractable)
DEFAULT_DEADLINE_DAYS: int = 3

# Days before deadline to send "approaching" alert
DEADLINE_ALERT_DAYS: int = 1

# Days without reply before flagging as "missed reply"
MISSED_REPLY_DAYS: int = 3

# Max emails to scan per run (avoids very long scans on large inboxes)
MAX_EMAILS_PER_SCAN: int = 100

# ── Korean + English deadline patterns ───────────────────────────────────────

_DEADLINE_PATTERNS: list[re.Pattern] = [
    # Explicit date formats
    re.compile(
        r"(?:deadline|due|by|마감|기한|제출|회신|답변)\s*[:\-\s]*"
        r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})\s*"
        r"(?:까지|기한|마감|deadline|due|by)",
        re.IGNORECASE,
    ),
    # "MM/DD까지" or "M월 D일까지"
    re.compile(
        r"(\d{1,2})[./월]\s*(\d{1,2})[일]?\s*(?:까지|기한|마감)",
        re.IGNORECASE,
    ),
    # "by March 15" or "by 3/15"
    re.compile(
        r"by\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)",
        re.IGNORECASE,
    ),
    # N days from now patterns like "within 3 days", "3일 이내"
    re.compile(
        r"(?:within|이내|내로)\s*(\d+)\s*(?:days?|일)",
        re.IGNORECASE,
    ),
]

_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PortfolioMailRecord:
    """Tracks a single portfolio-company email through its lifecycle."""

    message_id: str
    thread_id: str
    subject: str
    sender: str                  # raw From: header
    company_name: str            # matched portfolio company name
    received_date: str           # ISO-8601 UTC
    deadline: str                # ISO-8601 UTC
    deadline_source: str         # "extracted" | "default"
    is_replied: bool = False
    alerted_approaching: bool = False   # True after 1-day alert sent
    alerted_overdue: bool = False       # True after overdue alert sent
    missed_reply_alerted: bool = False  # True after missed-reply alert sent
    last_updated: str = ""       # ISO-8601 UTC

    @property
    def received_dt(self) -> datetime:
        return datetime.fromisoformat(self.received_date)

    @property
    def deadline_dt(self) -> datetime:
        return datetime.fromisoformat(self.deadline)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioMailRecord":
        # Only pass fields the dataclass knows about
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MailAlertResult:
    """Return value from alert-checking methods."""

    approaching: list[PortfolioMailRecord] = field(default_factory=list)
    overdue: list[PortfolioMailRecord] = field(default_factory=list)
    missed_reply: list[PortfolioMailRecord] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return bool(self.approaching or self.overdue or self.missed_reply)


# ── Deadline extraction ───────────────────────────────────────────────────────

def _extract_deadline(body_text: str, received_dt: datetime) -> tuple[datetime, str]:
    """
    Try to extract an explicit deadline from *body_text*.

    Returns (deadline_datetime, source) where source is one of:
    - "extracted": a specific date was found in the email body
    - "default": no date found; default is received_date + DEFAULT_DEADLINE_DAYS

    This function never raises; all parse errors fall through to the default.
    """
    now = datetime.now(timezone.utc)
    current_year = now.year

    # ── Pattern 1 & 2: full YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD ─────────────
    for pat in _DEADLINE_PATTERNS[:2]:
        m = pat.search(body_text)
        if m:
            date_str = m.group(1).replace(".", "-").replace("/", "-")
            try:
                dt = datetime.fromisoformat(date_str).replace(
                    tzinfo=timezone.utc
                )
                if dt >= received_dt - timedelta(days=1):  # sanity check
                    return dt, "extracted"
            except ValueError:
                pass

    # ── Pattern 3: MM/DD까지 ─────────────────────────────────────────────────
    m = _DEADLINE_PATTERNS[2].search(body_text)
    if m:
        try:
            month = int(m.group(1))
            day = int(m.group(2))
            year = current_year
            dt = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
            if dt < received_dt:
                dt = dt.replace(year=year + 1)
            return dt, "extracted"
        except (ValueError, AttributeError):
            pass

    # ── Pattern 4: "by March 15" ─────────────────────────────────────────────
    m = _DEADLINE_PATTERNS[3].search(body_text)
    if m:
        date_text = m.group(1).strip().rstrip("stndrh").strip()
        parts = date_text.split()
        if len(parts) == 2:
            month_word = parts[0].lower()
            try:
                month = _MONTH_NAMES.get(month_word)
                day = int(parts[1])
                if month:
                    dt = datetime(current_year, month, day, 23, 59, 59, tzinfo=timezone.utc)
                    if dt < received_dt:
                        dt = dt.replace(year=current_year + 1)
                    return dt, "extracted"
            except (ValueError, AttributeError):
                pass

    # ── Pattern 5: "within N days" ───────────────────────────────────────────
    m = _DEADLINE_PATTERNS[4].search(body_text)
    if m:
        try:
            days = int(m.group(1))
            if 1 <= days <= 90:  # sanity range
                return received_dt + timedelta(days=days), "extracted"
        except ValueError:
            pass

    # ── Default: received_date + DEFAULT_DEADLINE_DAYS ───────────────────────
    return received_dt + timedelta(days=DEFAULT_DEADLINE_DAYS), "default"


# ── Reply detection ───────────────────────────────────────────────────────────

def _is_thread_replied(thread_messages: list[Any]) -> bool:
    """
    Return True if the investment team has replied to any message in the thread.

    A reply is counted when a message sender is in INVESTMENT_TEAM_EMAILS or
    has the INTERNAL_DOMAIN.
    """
    internal_emails_lower = {e.lower() for e in INVESTMENT_TEAM_EMAILS}
    for msg in thread_messages:
        sender = getattr(msg, "sender", "").lower()
        if INTERNAL_DOMAIN in sender:
            return True
        if sender in internal_emails_lower:
            return True
    return False


# ── PortfolioMailMonitor ──────────────────────────────────────────────────────

class PortfolioMailMonitor:
    """
    Core class for portfolio-company email monitoring.

    Parameters
    ----------
    gmail_client      : GmailClient instance (or None to create on-demand)
    portfolio_cache   : PortfolioCache instance (or None to use singleton)
    company_name_cache: CompanyNameCache instance (or None to use singleton)
    status_file       : Path to the JSON file for persisting mail status
    web_search_client : Optional web-search client for brand-name fallback (AC 9)
    """

    def __init__(
        self,
        gmail_client: Optional[GmailClient] = None,
        portfolio_cache: Optional[PortfolioCache] = None,
        company_name_cache: Optional[CompanyNameCache] = None,
        status_file: Path = MAIL_STATUS_FILE,
        web_search_client: Any = None,
    ) -> None:
        self._gmail = gmail_client
        self._portfolio_cache = portfolio_cache or get_portfolio_cache()
        self._name_cache = company_name_cache or get_company_name_cache()
        self._status_file = status_file
        self._web_search = web_search_client
        self._records: dict[str, PortfolioMailRecord] = {}  # message_id → record
        self._lock = threading.Lock()
        self._status_loaded = False
        self._sender_matcher: Optional[SenderMatcher] = None

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _get_gmail(self) -> GmailClient:
        if self._gmail is None:
            self._gmail = GmailClient()
        return self._gmail

    def _get_sender_matcher(self) -> SenderMatcher:
        if self._sender_matcher is None:
            self._sender_matcher = SenderMatcher(cache=self._portfolio_cache)
        return self._sender_matcher

    # ── Status persistence ────────────────────────────────────────────────────

    def _load_status(self) -> None:
        """Load persisted mail records from JSON file."""
        if self._status_file.exists():
            try:
                raw = json.loads(self._status_file.read_text(encoding="utf-8"))
                records = raw.get("records", {})
                with self._lock:
                    self._records = {
                        mid: PortfolioMailRecord.from_dict(r)
                        for mid, r in records.items()
                    }
                logger.info(
                    "PortfolioMailMonitor: loaded %d mail records", len(self._records)
                )
            except Exception as exc:
                logger.warning("PortfolioMailMonitor: could not load status file: %s", exc)
        self._status_loaded = True

    def _save_status(self) -> None:
        """Persist current mail records to JSON file."""
        try:
            with self._lock:
                records_dict = {mid: r.to_dict() for mid, r in self._records.items()}
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(records_dict),
                "records": records_dict,
            }
            self._status_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("PortfolioMailMonitor: could not save status file: %s", exc)

    def _ensure_status_loaded(self) -> None:
        if not self._status_loaded:
            self._load_status()

    # ── Portfolio company matching (AC 9) ─────────────────────────────────────

    def _identify_portfolio_company(self, sender: str, subject: str, body: str) -> Optional[str]:
        """
        Try to match an email sender to a portfolio company.

        Strategy:
        1. Check the persistent name cache (AC 10) — instant return if known.
        2. Use SenderMatcher (domain + display name channels).
        3. If no match found, optionally use web search to resolve brand vs legal
           name differences (AC 9).

        Returns the portfolio company name, or None if no match found.
        """
        # 1. Check persistent cache first (AC 10)
        # Try domain root as cache key
        domain_root = ""
        if "@" in sender:
            domain_part = sender.split("@")[-1].split(">")[0].strip().lower()
            # Extract root (first part of domain, before the first dot in TLD)
            domain_parts = domain_part.split(".")
            domain_root = domain_parts[0] if domain_parts else ""

        if domain_root:
            cached = self._name_cache.get(domain_root)
            if cached is not None:
                logger.debug(
                    "CompanyNameCache HIT: domain %r → %r", domain_root, cached
                )
                return cached

        # 2. SenderMatcher
        matcher = self._get_sender_matcher()
        summary = matcher.match(sender)
        if summary.matched and summary.top_match:
            company_name = summary.top_match.company.name
            # Persist to cache (AC 10)
            if domain_root:
                self._name_cache.set(domain_root, company_name)
            logger.debug(
                "SenderMatcher: %r → %r (conf=%.2f)",
                sender, company_name,
                summary.top_match.confidence,
            )
            return company_name

        # 3. Web search fallback for brand-name vs legal-name mismatches (AC 9)
        if self._web_search is not None and domain_root:
            try:
                company_name = self._web_search_company_name(domain_root, subject)
                if company_name:
                    # Verify against portfolio cache
                    verified = self._verify_against_portfolio(company_name)
                    if verified:
                        self._name_cache.set(domain_root, verified)
                        logger.info(
                            "Web search resolved %r → %r (verified: %r)",
                            domain_root, company_name, verified,
                        )
                        return verified
            except Exception as exc:
                logger.warning("Web search fallback failed: %s", exc)

        return None

    def _web_search_company_name(self, domain_root: str, subject: str) -> Optional[str]:
        """Use web search to find the official company name for a domain."""
        if self._web_search is None:
            return None
        query = f"{domain_root} company name official"
        try:
            results = self._web_search.search(query, max_results=3)
            if results:
                # Use the first result's title/snippet to extract company name
                return results[0].get("title", "").split("|")[0].strip()
        except Exception as exc:
            logger.warning("Web search for company name failed: %s", exc)
        return None

    def _verify_against_portfolio(self, company_name: str) -> Optional[str]:
        """Check if a company name matches any portfolio company; return canonical name."""
        try:
            companies = self._portfolio_cache.get_all_companies()
            name_lower = company_name.lower()
            for company in companies:
                if (company.normalised and
                        (company.normalised in name_lower or name_lower in company.normalised)):
                    return company.name
        except Exception as exc:
            logger.warning("Portfolio verification failed: %s", exc)
        return None

    # ── Email scanning (ACs 9-11) ─────────────────────────────────────────────

    def _fetch_recent_emails_with_retry(self, lookback_days: int) -> list[Any]:
        """Fetch emails from the last *lookback_days* days with retry logic."""
        gmail = self._get_gmail()
        last_exc: Optional[Exception] = None

        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                threads = gmail.search_threads(
                    query=f"newer_than:{lookback_days}d",
                    max_results=MAX_EMAILS_PER_SCAN,
                )
                return threads
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Gmail fetch failed (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                if attempt < API_RETRY_ATTEMPTS:
                    time.sleep(API_RETRY_DELAY_SECONDS)

        logger.error("Gmail API exhausted all retries: %s", last_exc)
        raise RuntimeError(f"Gmail API failed after {API_RETRY_ATTEMPTS} attempts") from last_exc

    def scan_emails(self, lookback_days: int = MAIL_SCAN_LOOKBACK_DAYS) -> int:
        """
        Scan recent emails and identify portfolio company mails.

        For each newly identified email:
        - Extract deadline (AC 11)
        - Cache company name mapping (AC 10)
        - Add to tracking records

        Returns the number of newly identified portfolio emails.
        """
        self._ensure_status_loaded()

        try:
            threads = self._fetch_recent_emails_with_retry(lookback_days)
        except RuntimeError as exc:
            logger.error("scan_emails: Gmail fetch failed: %s", exc)
            return 0

        new_count = 0

        for thread in threads:
            # Get messages in thread
            messages = getattr(thread, "messages", [])
            if not messages:
                continue

            # Use the first (oldest) message as the "primary" mail
            primary = messages[0]
            message_id = getattr(primary, "message_id", "")

            if not message_id:
                continue

            # Skip if already tracked
            with self._lock:
                if message_id in self._records:
                    # Update reply status
                    if not self._records[message_id].is_replied:
                        replied = _is_thread_replied(messages)
                        if replied:
                            self._records[message_id].is_replied = True
                            self._records[message_id].last_updated = (
                                datetime.now(timezone.utc).isoformat()
                            )
                    continue

            sender = getattr(primary, "sender", "")
            subject = getattr(primary, "subject", "")
            body = getattr(primary, "body_text", "") or getattr(primary, "snippet", "")

            # Skip internal emails (from our own domain)
            if INTERNAL_DOMAIN in sender.lower():
                continue

            # Try to match to portfolio company (AC 9)
            company_name = self._identify_portfolio_company(sender, subject, body)
            if not company_name:
                continue  # not a portfolio company email

            # Parse received date
            received_dt = getattr(primary, "date", None)
            if received_dt is None:
                received_dt = datetime.now(timezone.utc)
            if received_dt.tzinfo is None:
                received_dt = received_dt.replace(tzinfo=timezone.utc)

            # Extract deadline (AC 11)
            deadline_dt, deadline_source = _extract_deadline(body, received_dt)

            # Check if thread already has a reply
            is_replied = _is_thread_replied(messages)

            record = PortfolioMailRecord(
                message_id=message_id,
                thread_id=getattr(thread, "thread_id", ""),
                subject=subject,
                sender=sender,
                company_name=company_name,
                received_date=received_dt.isoformat(),
                deadline=deadline_dt.isoformat(),
                deadline_source=deadline_source,
                is_replied=is_replied,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

            with self._lock:
                self._records[message_id] = record

            logger.info(
                "PortfolioMailMonitor: NEW email tracked — company=%r, "
                "subject=%r, deadline=%s (%s), replied=%s",
                company_name, subject,
                deadline_dt.strftime("%Y-%m-%d"),
                deadline_source,
                is_replied,
            )
            new_count += 1

        if new_count > 0:
            self._save_status()

        return new_count

    # ── Alert detection (ACs 12-14) ──────────────────────────────────────────

    def check_alerts(self) -> MailAlertResult:
        """
        Inspect all tracked records and classify into three alert categories:

        - **approaching**: deadline within DEADLINE_ALERT_DAYS (1 day) and not
          yet alerted (AC 12).
        - **overdue**: deadline has passed (AC 13).
        - **missed_reply**: no reply and email is older than MISSED_REPLY_DAYS
          days (AC 14).

        Marks ``alerted_approaching`` / ``alerted_overdue`` flags to prevent
        duplicate alerts across calls.
        """
        self._ensure_status_loaded()

        now = datetime.now(timezone.utc)
        result = MailAlertResult()
        changed = False

        with self._lock:
            for record in self._records.values():
                deadline_dt = record.deadline_dt
                received_dt = record.received_dt

                # ── AC 12: Approaching (1 day before) ────────────────────────
                time_to_deadline = (deadline_dt - now).total_seconds() / 3600  # hours
                if (
                    0 < time_to_deadline <= DEADLINE_ALERT_DAYS * 24
                    and not record.alerted_approaching
                    and not record.is_replied
                ):
                    result.approaching.append(record)
                    record.alerted_approaching = True
                    record.last_updated = now.isoformat()
                    changed = True

                # ── AC 13: Overdue ────────────────────────────────────────────
                if (
                    deadline_dt <= now
                    and not record.alerted_overdue
                    and not record.is_replied
                ):
                    result.overdue.append(record)
                    record.alerted_overdue = True
                    record.last_updated = now.isoformat()
                    changed = True

                # ── AC 14: Missed reply ───────────────────────────────────────
                age_days = (now - received_dt).days
                if (
                    age_days >= MISSED_REPLY_DAYS
                    and not record.is_replied
                    and not record.missed_reply_alerted
                ):
                    result.missed_reply.append(record)
                    record.missed_reply_alerted = True
                    record.last_updated = now.isoformat()
                    changed = True

        if changed:
            self._save_status()

        return result

    # ── On-demand status report (AC 15 — /mail command) ──────────────────────

    def get_status_report(self) -> dict:
        """
        Generate a summary status report for the /mail slash command.

        Returns a dict with:
        - total_tracked: int
        - pending_reply: list[PortfolioMailRecord]
        - approaching_deadline: list[PortfolioMailRecord]
        - overdue: list[PortfolioMailRecord]
        - replied: list[PortfolioMailRecord]
        """
        self._ensure_status_loaded()

        now = datetime.now(timezone.utc)
        pending_reply = []
        approaching = []
        overdue = []
        replied = []

        with self._lock:
            for record in self._records.values():
                if record.is_replied:
                    replied.append(record)
                    continue

                deadline_dt = record.deadline_dt
                time_to_deadline = (deadline_dt - now).total_seconds() / 3600

                if deadline_dt <= now:
                    overdue.append(record)
                elif time_to_deadline <= DEADLINE_ALERT_DAYS * 24:
                    approaching.append(record)
                else:
                    pending_reply.append(record)

        return {
            "total_tracked": len(self._records),
            "pending_reply": sorted(pending_reply, key=lambda r: r.deadline_dt),
            "approaching_deadline": sorted(approaching, key=lambda r: r.deadline_dt),
            "overdue": sorted(overdue, key=lambda r: r.deadline_dt),
            "replied": sorted(replied, key=lambda r: r.received_dt, reverse=True),
            "generated_at": now.isoformat(),
        }

    # ── High-level entry points ───────────────────────────────────────────────

    def run_scan_and_check(self) -> MailAlertResult:
        """
        Full cycle: scan new emails, then check all alerts.

        Called by the scheduler job and /mail command.
        """
        self.scan_emails()
        return self.check_alerts()

    def get_all_records(self) -> list[PortfolioMailRecord]:
        """Return a snapshot of all tracked portfolio mail records."""
        self._ensure_status_loaded()
        with self._lock:
            return list(self._records.values())

    def get_pending_records(self) -> list[PortfolioMailRecord]:
        """Return records that haven't been replied to yet."""
        return [r for r in self.get_all_records() if not r.is_replied]


# ── Slack message formatting ─────────────────────────────────────────────────

def format_deadline_approaching_alert(record: PortfolioMailRecord) -> str:
    """Format a human-readable Slack DM for a deadline-approaching alert (AC 12)."""
    deadline_dt = record.deadline_dt
    kst_offset = timedelta(hours=9)
    deadline_kst = deadline_dt + kst_offset

    return (
        f"⏰ *[데드라인 임박 알림]* — {record.company_name}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*제목:* {record.subject}\n"
        f"*발신:* {record.sender}\n"
        f"*마감일:* {deadline_kst.strftime('%Y-%m-%d %H:%M')} KST "
        f"({'추정' if record.deadline_source == 'default' else '본문 기재'})\n"
        f"*상태:* 미회신\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ 내일까지 회신이 필요합니다."
    )


def format_overdue_alert(record: PortfolioMailRecord) -> str:
    """Format a human-readable Slack DM for an overdue deadline alert (AC 13)."""
    deadline_dt = record.deadline_dt
    kst_offset = timedelta(hours=9)
    deadline_kst = deadline_dt + kst_offset
    now = datetime.now(timezone.utc)
    overdue_hours = int((now - deadline_dt).total_seconds() / 3600)

    return (
        f"🚨 *[마감 초과 알림]* — {record.company_name}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*제목:* {record.subject}\n"
        f"*발신:* {record.sender}\n"
        f"*마감일:* {deadline_kst.strftime('%Y-%m-%d %H:%M')} KST "
        f"({'추정' if record.deadline_source == 'default' else '본문 기재'})\n"
        f"*초과 시간:* {overdue_hours}시간\n"
        f"*상태:* 미회신\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔴 즉시 회신이 필요합니다!"
    )


def format_missed_reply_alert(record: PortfolioMailRecord) -> str:
    """Format a Slack DM for a missed-reply alert (AC 14)."""
    received_dt = record.received_dt
    kst_offset = timedelta(hours=9)
    received_kst = received_dt + kst_offset
    now = datetime.now(timezone.utc)
    age_days = (now - received_dt).days

    return (
        f"📬 *[미회신 알림]* — {record.company_name}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*제목:* {record.subject}\n"
        f"*발신:* {record.sender}\n"
        f"*수신일:* {received_kst.strftime('%Y-%m-%d %H:%M')} KST\n"
        f"*경과일:* {age_days}일\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 포트폴리오사 메일 {age_days}일 경과 미회신 — 확인 필요"
    )


def format_mail_status_report(status: dict) -> str:
    """
    Format the full mail status report for the /mail slash command (AC 15).

    Returns a text message suitable for Slack DM.
    """
    now_str = datetime.fromisoformat(status["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📮 *포트폴리오사 메일 현황* ({now_str})",
        f"총 추적 메일: {status['total_tracked']}건",
        "",
    ]

    overdue = status.get("overdue", [])
    approaching = status.get("approaching_deadline", [])
    pending = status.get("pending_reply", [])
    replied = status.get("replied", [])

    if overdue:
        lines.append(f"🔴 *마감 초과* ({len(overdue)}건)")
        for r in overdue[:5]:
            dl = r.deadline_dt.strftime("%m/%d")
            lines.append(f"  • {r.company_name} — {r.subject[:40]} (마감: {dl})")
        lines.append("")

    if approaching:
        lines.append(f"⏰ *마감 임박* ({len(approaching)}건)")
        for r in approaching[:5]:
            dl = r.deadline_dt.strftime("%m/%d %H:%M")
            lines.append(f"  • {r.company_name} — {r.subject[:40]} (마감: {dl} KST)")
        lines.append("")

    if pending:
        lines.append(f"📨 *회신 대기중* ({len(pending)}건)")
        for r in pending[:5]:
            dl = r.deadline_dt.strftime("%m/%d")
            lines.append(f"  • {r.company_name} — {r.subject[:40]} (기한: {dl})")
        lines.append("")

    if replied:
        lines.append(f"✅ *회신 완료* ({len(replied)}건)")

    if not overdue and not approaching and not pending and not replied:
        lines.append("추적 중인 포트폴리오사 메일이 없습니다.")

    return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_monitor_singleton: Optional[PortfolioMailMonitor] = None
_monitor_lock = threading.Lock()


def get_mail_monitor(
    gmail_client: Optional[GmailClient] = None,
    web_search_client: Any = None,
) -> PortfolioMailMonitor:
    """Return the module-level PortfolioMailMonitor singleton."""
    global _monitor_singleton
    if _monitor_singleton is None:
        with _monitor_lock:
            if _monitor_singleton is None:
                _monitor_singleton = PortfolioMailMonitor(
                    gmail_client=gmail_client,
                    web_search_client=web_search_client,
                )
    return _monitor_singleton
