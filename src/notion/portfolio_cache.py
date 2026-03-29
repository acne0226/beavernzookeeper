"""
Portfolio company name cache for email matching (Sub-AC 9a of AC 9).

Fetches ALL portfolio company names from the Notion startup database at startup,
caches them in-memory with TTL-based refresh, and exposes a queryable list
for identifying portfolio-related emails.

Key design decisions:
- Paginates through all Notion DB records (not just the first page).
- Normalises company names for robust matching (case-insensitive, strips legal
  suffixes like 주식회사/Inc/Corp, etc.).
- match_email() returns matches based on subject/sender domain/body text, so
  callers do not need to make Notion API calls per email.
- Thread-safe: uses a lock around cache refresh so concurrent callers never
  see a half-populated cache.
- Respects retry constraints: 3 attempts, 10 s apart, then raises / returns
  empty list (caller's choice via strict= flag).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from notion_client import Client
from notion_client.errors import APIResponseError

from src.config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
    NOTION_DB_ID,
    NOTION_TOKEN,
)

logger = logging.getLogger(__name__)

# ── Legal-suffix patterns to strip when normalising company names ─────────────
# English legal suffixes at the END: must be preceded by whitespace to avoid
# stripping "Corp" from inside compound names like "AcmeCorp".
_LEGAL_SUFFIX_END = re.compile(
    r"\s+[\(\（]?(?:Inc\.?|Corp\.?|Ltd\.?|LLC\.?|Co\.?|GmbH|S\.?A\.?|B\.?V\.?)[\)\）]?\s*$",
    re.IGNORECASE,
)
# Korean legal designators at the START of a name (e.g. "주식회사 에이비씨")
_KOREAN_PREFIX = re.compile(
    r"^(?:주식회사|유한회사|유한책임회사|합자회사|사단법인|재단법인|협동조합)\s+",
)
# Korean legal designators at the END of a name (e.g. "에이비씨 주식회사")
_KOREAN_SUFFIX_END = re.compile(
    r"\s+(?:주식회사|유한회사|유한책임회사|합자회사|사단법인|재단법인|협동조합)\s*$",
)

# Minimum token length to consider for matching (avoids single-char false positives)
_MIN_TOKEN_LEN = 2

# Default TTL for the in-memory cache (seconds)
_DEFAULT_CACHE_TTL_SECONDS: int = 3600  # 1 hour


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PortfolioCompany:
    """A portfolio company entry fetched from the Notion database."""

    page_id: str
    name: str                # Raw name from Notion
    normalised: str          # Lower-cased, suffix-stripped name for matching
    status: str = ""         # Deal stage / status (if available)
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "name": self.name,
            "normalised": self.normalised,
            "status": self.status,
            "url": self.url,
        }


@dataclass
class EmailMatchResult:
    """Result of matching an email against the portfolio company list."""

    matched: bool
    companies: list[PortfolioCompany] = field(default_factory=list)
    # Which part of the email triggered the match
    match_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "companies": [c.to_dict() for c in self.companies],
            "match_sources": self.match_sources,
        }


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise_name(raw: str) -> str:
    """
    Normalise a company name for robust matching.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove Korean legal designators from start (주식회사 ...) or end.
    3. Remove English legal suffixes at the end when preceded by a space
       (so "AcmeCorp" stays "AcmeCorp" but "Acme Corp" → "Acme").
    4. Lowercase.
    5. Collapse multiple spaces.
    """
    name = raw.strip()
    # Korean prefix: "주식회사 에이비씨" → "에이비씨"
    name = _KOREAN_PREFIX.sub("", name).strip()
    # Korean suffix: "에이비씨 주식회사" → "에이비씨"
    name = _KOREAN_SUFFIX_END.sub("", name).strip()
    # English suffix (space-separated): "Acme Corp" → "Acme", "AcmeCorp" unchanged
    name = _LEGAL_SUFFIX_END.sub("", name).strip()
    name = name.lower()
    name = re.sub(r"\s+", " ", name)
    return name


def _tokenise(text: str) -> list[str]:
    """
    Split text into tokens for matching.
    Splits on whitespace and common delimiters; keeps tokens >= _MIN_TOKEN_LEN.
    """
    raw_tokens = re.split(r"[\s/\-_\[\]()「」『』【】<>《》,.:;!?@#~]+", text)
    return [
        t.lower()
        for t in raw_tokens
        if len(t) >= _MIN_TOKEN_LEN
    ]


def _extract_sender_domain_root(sender: str) -> str:
    """
    Extract the main domain label from a sender address.
    'CEO <ceo@acmecorp.com>'  → 'acmecorp'
    'user@sub.example.co.kr' → 'example'
    """
    # Extract bare email address
    match = re.search(r"<([^>]+)>", sender)
    addr = match.group(1).strip() if match else sender.strip()

    try:
        domain = addr.split("@")[1].lower()
        parts = domain.split(".")
        _TLDS = {"com", "co", "kr", "net", "org", "io", "ai", "vc", "biz", "app"}
        meaningful = [p for p in parts if p not in _TLDS and len(p) > 2]
        return meaningful[-1] if meaningful else ""
    except (IndexError, AttributeError):
        return ""


# ── Cache ─────────────────────────────────────────────────────────────────────

class PortfolioCache:
    """
    In-memory cache of portfolio company names fetched from Notion.

    Usage:
    ------
        cache = PortfolioCache()
        cache.load()                        # blocks until loaded
        matches = cache.match_email(
            subject="AcmeCorp 투자 검토 요청",
            sender="ceo@acmecorp.com",
        )

    Thread safety:
    --------------
    load() acquires _lock before modifying state. get_all_companies() and
    match_email() acquire _lock for reads, ensuring callers always see a
    consistent snapshot.
    """

    def __init__(
        self,
        db_id: str = NOTION_DB_ID,
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._db_id = db_id
        self._cache_ttl = cache_ttl_seconds
        self._client: Optional[Client] = None

        # Protected by _lock
        self._lock = threading.Lock()
        self._companies: list[PortfolioCompany] = []
        self._loaded_at: Optional[datetime] = None
        self._schema_title_field: Optional[str] = None
        self._schema_status_field: Optional[str] = None

    # ── Connectivity ──────────────────────────────────────────────────────────

    def _ensure_client(self) -> Client:
        if self._client is None:
            self._client = Client(auth=NOTION_TOKEN)
        return self._client

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute *fn* with up to API_RETRY_ATTEMPTS retries at 10 s intervals."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_client()
                return fn(*args, **kwargs)
            except APIResponseError as exc:
                logger.warning(
                    "PortfolioCache Notion API error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "PortfolioCache unexpected error (attempt %d/%d): %s",
                    attempt, API_RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            f"PortfolioCache: Notion API failed after {API_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    # ── Schema discovery ──────────────────────────────────────────────────────

    def _discover_schema(self) -> None:
        """
        Retrieve DB metadata and identify the title and status field names.
        Results are stored on self so _fetch_all_companies() can use them.
        """
        def _retrieve():
            return self._ensure_client().databases.retrieve(
                database_id=self._db_id
            )

        db_meta = self._call_with_retry(_retrieve)
        raw_props = db_meta.get("properties", {})

        title_field: Optional[str] = None
        status_field: Optional[str] = None

        _STATUS_PATTERNS = re.compile(
            r"status|stage|단계|상태|진행|phase", re.IGNORECASE
        )

        for prop_name, prop_meta in raw_props.items():
            ptype = prop_meta.get("type", "")
            if ptype == "title" and title_field is None:
                title_field = prop_name
            if ptype in ("select", "status") and status_field is None:
                if _STATUS_PATTERNS.search(prop_name):
                    status_field = prop_name

        # Fallback: first select/status field if none matched the keyword pattern
        if status_field is None:
            for prop_name, prop_meta in raw_props.items():
                if prop_meta.get("type") in ("select", "status"):
                    status_field = prop_name
                    break

        self._schema_title_field = title_field
        self._schema_status_field = status_field

        logger.info(
            "PortfolioCache schema: title_field=%r status_field=%r",
            title_field, status_field,
        )

    # ── Fetching ──────────────────────────────────────────────────────────────

    def _extract_title(self, prop_data: dict) -> str:
        parts = prop_data.get("title", [])
        return "".join(p.get("plain_text", "") for p in parts).strip()

    def _extract_status(self, prop_data: dict) -> str:
        ptype = prop_data.get("type", "")
        if ptype in ("select", "status"):
            sel = prop_data.get(ptype) or {}
            return sel.get("name", "")
        return ""

    def _parse_page(self, page: dict) -> Optional[PortfolioCompany]:
        """Convert a raw Notion page dict into a PortfolioCompany."""
        props = page.get("properties", {})

        # Extract company name
        name = ""
        if self._schema_title_field:
            title_prop = props.get(self._schema_title_field, {})
            name = self._extract_title(title_prop)

        if not name:
            return None  # Skip unnamed records

        # Extract status
        status = ""
        if self._schema_status_field:
            status_prop = props.get(self._schema_status_field, {})
            status = self._extract_status(status_prop)

        return PortfolioCompany(
            page_id=page.get("id", ""),
            name=name,
            normalised=_normalise_name(name),
            status=status,
            url=page.get("url", ""),
        )

    _MAX_PAGES = 200  # Safety limit: 200 pages × 100 records = 20 000 companies max

    def _fetch_all_companies(self) -> list[PortfolioCompany]:
        """
        Paginate through ALL records in the Notion database and return
        a list of PortfolioCompany objects.

        Handles Notion's pagination (has_more / next_cursor) automatically.
        A hard limit of _MAX_PAGES prevents infinite loops on unexpected responses.
        """
        companies: list[PortfolioCompany] = []
        start_cursor: Optional[str] = None
        page_num = 0

        while page_num < self._MAX_PAGES:
            page_num += 1

            def _query(cursor=start_cursor):
                params: dict = {
                    "database_id": self._db_id,
                    "page_size": 100,
                }
                if cursor:
                    params["start_cursor"] = cursor
                return self._ensure_client().databases.query(**params)

            result = self._call_with_retry(_query)
            pages = result.get("results", [])

            for page in pages:
                company = self._parse_page(page)
                if company is not None:
                    companies.append(company)

            logger.debug(
                "PortfolioCache: fetched page %d (%d records so far)",
                page_num, len(companies),
            )

            # Explicitly check for boolean True to avoid truthy non-bool values
            # (e.g. MagicMock objects) causing an infinite pagination loop.
            if result.get("has_more") is True:
                start_cursor = result.get("next_cursor")
            else:
                break

        return companies

    # ── Public load / refresh ──────────────────────────────────────────────────

    def load(self, force: bool = False) -> None:
        """
        Load (or refresh) the portfolio company cache from Notion.

        Parameters
        ----------
        force:
            If True, bypass the TTL check and always reload from Notion.
            Default is False (respects TTL).

        Raises
        ------
        RuntimeError
            If the Notion API is unreachable after all retries.
        """
        with self._lock:
            if not force and self._is_cache_fresh():
                logger.debug("PortfolioCache: cache is fresh, skipping reload")
                return

            logger.info(
                "PortfolioCache: loading portfolio companies from Notion DB %s …",
                self._db_id,
            )
            self._discover_schema()
            companies = self._fetch_all_companies()
            self._companies = companies
            self._loaded_at = datetime.now(timezone.utc)

            logger.info(
                "PortfolioCache: loaded %d portfolio companies (TTL=%ds)",
                len(companies),
                self._cache_ttl,
            )

    def _is_cache_fresh(self) -> bool:
        """Return True if the cache was loaded within the TTL window."""
        if self._loaded_at is None or not self._companies:
            return False
        age = (datetime.now(timezone.utc) - self._loaded_at).total_seconds()
        return age < self._cache_ttl

    def ensure_loaded(self) -> None:
        """
        Auto-refresh the cache if it has expired or was never loaded.
        Safe to call on every email check — no-op if cache is still fresh.
        """
        if not self._is_cache_fresh():
            self.load()

    # ── Public query API ──────────────────────────────────────────────────────

    def get_all_companies(self) -> list[PortfolioCompany]:
        """
        Return a snapshot of all cached portfolio companies.

        Automatically refreshes the cache if it has expired.
        """
        self.ensure_loaded()
        with self._lock:
            return list(self._companies)

    def get_company_names(self) -> list[str]:
        """
        Return a list of raw company name strings.
        Convenience method for callers that just need the names.
        """
        return [c.name for c in self.get_all_companies()]

    def find_matches(self, text: str) -> list[PortfolioCompany]:
        """
        Find portfolio companies whose normalised name appears as a
        substring in *text* (case-insensitive).

        Returns a deduplicated list of matching PortfolioCompany objects.
        """
        self.ensure_loaded()
        norm_text = text.lower()
        seen: set[str] = set()
        results: list[PortfolioCompany] = []

        with self._lock:
            for company in self._companies:
                if not company.normalised:
                    continue
                if company.page_id in seen:
                    continue
                if company.normalised in norm_text:
                    seen.add(company.page_id)
                    results.append(company)

        return results

    def match_email(
        self,
        subject: str = "",
        sender: str = "",
        body: str = "",
        recipients: Optional[list[str]] = "",
    ) -> EmailMatchResult:
        """
        Determine whether an email relates to one or more portfolio companies.

        Matching strategy (in order, results de-duplicated):
        1. Subject substring match (highest signal — checked first).
        2. Sender domain root match against normalised company names.
        3. Body text substring match (lowest signal — avoids false positives
           by requiring a token of >= 3 chars that is also in the company name).

        Parameters
        ----------
        subject:
            Email subject line.
        sender:
            From: header value (e.g. "CEO <ceo@acmecorp.com>").
        body:
            Plaintext email body (optional; checked last to limit noise).
        recipients:
            List of recipient address strings (not currently used for
            matching, reserved for future use).

        Returns
        -------
        EmailMatchResult with matched=True and companies filled in when
        at least one portfolio company is identified.
        """
        self.ensure_loaded()

        seen: set[str] = set()
        matched_companies: list[PortfolioCompany] = []
        match_sources: list[str] = []

        def _add(company: PortfolioCompany, source: str) -> None:
            if company.page_id not in seen:
                seen.add(company.page_id)
                matched_companies.append(company)
                if source not in match_sources:
                    match_sources.append(source)

        # 1. Subject match
        if subject:
            for c in self.find_matches(subject):
                _add(c, "subject")

        # 2. Sender domain match
        if sender:
            domain_root = _extract_sender_domain_root(sender)
            if domain_root and len(domain_root) >= 3:
                with self._lock:
                    for company in self._companies:
                        if domain_root in company.normalised or \
                                company.normalised in domain_root:
                            _add(company, "sender_domain")

        # 3. Body text match (only check if body is provided and not too short)
        if body and len(body.strip()) >= 10:
            for c in self.find_matches(body):
                _add(c, "body")

        return EmailMatchResult(
            matched=bool(matched_companies),
            companies=matched_companies,
            match_sources=match_sources,
        )

    def is_portfolio_email(
        self,
        subject: str = "",
        sender: str = "",
        body: str = "",
    ) -> bool:
        """
        Convenience method: returns True if the email relates to any
        portfolio company, False otherwise.
        """
        return self.match_email(subject=subject, sender=sender, body=body).matched

    # ── Cache introspection ───────────────────────────────────────────────────

    def cache_info(self) -> dict:
        """Return metadata about the current cache state."""
        with self._lock:
            age_seconds: Optional[float] = None
            if self._loaded_at is not None:
                age_seconds = (
                    datetime.now(timezone.utc) - self._loaded_at
                ).total_seconds()
            return {
                "count": len(self._companies),
                "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
                "age_seconds": age_seconds,
                "ttl_seconds": self._cache_ttl,
                "is_fresh": self._is_cache_fresh(),
                "title_field": self._schema_title_field,
                "status_field": self._schema_status_field,
            }


# ── Module-level singleton ────────────────────────────────────────────────────

_default_cache: Optional[PortfolioCache] = None
_cache_lock = threading.Lock()


def get_portfolio_cache(
    cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
) -> PortfolioCache:
    """
    Return (creating if necessary) the module-level singleton PortfolioCache.

    The cache is not loaded until the first call to load() or ensure_loaded().
    Callers that need the cache pre-warmed should call:

        get_portfolio_cache().load()

    at daemon startup.
    """
    global _default_cache
    with _cache_lock:
        if _default_cache is None:
            _default_cache = PortfolioCache(
                cache_ttl_seconds=cache_ttl_seconds,
            )
    return _default_cache
