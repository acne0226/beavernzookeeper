"""
Email sender matching logic (Sub-AC 9b of AC 9).

Compares email sender display name and domain against cached Notion portfolio
company names using both exact substring matching and fuzzy similarity scoring
via Python's built-in ``difflib.SequenceMatcher``.

Matching strategy (two independent channels):

A) Sender **NAME** channel:
   1. Parse the display name from the From: header
      e.g. ``"김민준 (AcmeCorp)" → display_name = "김민준 (AcmeCorp)"``
   2. Normalise the display name (same rules as PortfolioCache normalisation):
      strip legal suffixes, lowercase, collapse spaces.
   3. For every portfolio company:

      a. **Exact** – normalised company name is a substring of the normalised
         display name (or vice-versa when the company name is multi-token).
         Confidence = 1.0
      b. **Fuzzy** – SequenceMatcher ratio ≥ FUZZY_NAME_THRESHOLD.
         Confidence = ratio value.

B) Sender **DOMAIN** channel:
   1. Extract the "root" label from the sender email domain:
      ``"ceo@acmecorp.co.kr" → "acmecorp"``
      (reuses ``_extract_sender_domain_root`` from portfolio_cache).
   2. For every portfolio company:

      a. **Exact** – domain root is a substring of normalised company name
         OR the normalised company name is a substring of the domain root.
         Confidence = 1.0
      b. **Fuzzy** – SequenceMatcher ratio ≥ FUZZY_DOMAIN_THRESHOLD.
         Confidence = ratio value.

Results from both channels are merged and deduplicated: when the same company
is matched via both channels the **higher**-confidence entry is kept.

Only companies whose normalised name has at least ``MIN_COMPANY_NAME_LEN``
characters are considered, preventing false positives from very short names
like "AI" or "IO".

Public API
----------
* ``SenderMatchResult``   – structured result for a single company match.
* ``SenderMatchSummary``  – aggregate result for one email sender.
* ``SenderMatcher``       – stateful matcher that wraps a ``PortfolioCache``.
* ``match_sender()``      – module-level convenience function.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.notion.portfolio_cache import (
    PortfolioCache,
    PortfolioCompany,
    _normalise_name,
    _extract_sender_domain_root,
    get_portfolio_cache,
)

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Minimum fuzzy ratio (0–1) to accept a name-channel match.
# 0.82 is intentionally conservative to reduce false positives from personal
# names that share tokens with company names ("Kim" matching "KimPay").
FUZZY_NAME_THRESHOLD: float = 0.82

# Minimum fuzzy ratio (0–1) to accept a domain-channel match.
# Domain roots are shorter and more distinctive, so a slightly lower bar
# (0.80) is appropriate.
FUZZY_DOMAIN_THRESHOLD: float = 0.80

# Minimum character length of a normalised company name before it is
# considered for matching; skips very short names ("AI", "IO", etc.).
MIN_COMPANY_NAME_LEN: int = 3

# Minimum character length of the sender name token before fuzzy-matching;
# avoids comparing short tokens like "of" or "at" against company names.
MIN_TOKEN_LEN_FOR_FUZZY: int = 3


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SenderMatchResult:
    """
    A single portfolio company matched to an email sender.

    Attributes
    ----------
    company      : The matched ``PortfolioCompany`` from the Notion cache.
    match_type   : One of:
                   ``"exact_name"``   – display name contains the company name.
                   ``"fuzzy_name"``   – fuzzy similarity on display name.
                   ``"exact_domain"`` – domain root exactly matches company name.
                   ``"fuzzy_domain"`` – fuzzy similarity on domain root.
    confidence   : Float in [0, 1]; 1.0 for exact matches, ratio for fuzzy.
    matched_text : The sender-side token that triggered the match (for logging).
    """

    company: PortfolioCompany
    match_type: str   # "exact_name" | "fuzzy_name" | "exact_domain" | "fuzzy_domain"
    confidence: float  # 0.0–1.0
    matched_text: str  # e.g. "acmecorp" or "alpha ventures"

    def to_dict(self) -> dict:
        return {
            "company": self.company.to_dict(),
            "match_type": self.match_type,
            "confidence": round(self.confidence, 4),
            "matched_text": self.matched_text,
        }


@dataclass
class SenderMatchSummary:
    """
    Aggregate matching result for one email From: header.

    Attributes
    ----------
    sender_raw      : The original From: header value.
    display_name    : Parsed and normalised display name (may be empty).
    domain_root     : Extracted domain root label (may be empty).
    matches         : Deduplicated list of ``SenderMatchResult`` objects,
                      sorted by confidence descending.
    """

    sender_raw: str
    display_name: str       # normalised
    domain_root: str
    matches: list[SenderMatchResult] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        """True when at least one portfolio company was identified."""
        return bool(self.matches)

    @property
    def top_match(self) -> Optional[SenderMatchResult]:
        """Highest-confidence match, or None when no matches."""
        return self.matches[0] if self.matches else None

    def to_dict(self) -> dict:
        return {
            "sender_raw": self.sender_raw,
            "display_name": self.display_name,
            "domain_root": self.domain_root,
            "matched": self.matched,
            "matches": [m.to_dict() for m in self.matches],
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_display_name(sender_raw: str) -> str:
    """
    Extract the display name portion from a From: header string.

    Handles:
    * ``"Alice Kim <alice@startup.com>"`` → ``"Alice Kim"``
    * ``"alice@startup.com"``             → ``""``  (bare address has no name)
    * ``'"AcmeCorp CEO" <ceo@acmecorp.com>'`` → ``"AcmeCorp CEO"``
    * MIME-encoded words are *not* decoded here (plain ASCII expected at this
      stage; MIME decoding is the caller's responsibility if needed).
    """
    if "<" not in sender_raw:
        return ""  # bare address — no display name

    name_part = sender_raw.split("<")[0].strip()
    # Strip surrounding quotes
    name_part = name_part.strip('"').strip("'").strip()
    return name_part


def _normalise_display_name(display_name: str) -> str:
    """
    Apply the same normalisation used for company names to a display name.

    This strips legal suffixes, lowercases, and collapses spaces so that
    ``"AcmeCorp Inc."`` in a sender display name normalises to ``"acmecorp"``,
    enabling direct comparison with normalised company names from Notion.
    """
    return _normalise_name(display_name)


def _tokenise_name(normalised_text: str) -> list[str]:
    """
    Split a normalised text into meaningful tokens for fuzzy matching.

    Tokens shorter than ``MIN_TOKEN_LEN_FOR_FUZZY`` characters are discarded.
    Splitting is done on whitespace, brackets, slashes, dashes, underscores,
    dots, and Korean corner-brackets.
    """
    raw_tokens = re.split(r"[\s/\-_\[\]()\u300c\u300d\u300e\u300f.,<>:;!?@#~]+",
                          normalised_text)
    return [t for t in raw_tokens if len(t) >= MIN_TOKEN_LEN_FOR_FUZZY]


def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Return the SequenceMatcher similarity ratio for strings *a* and *b*.

    Uses ``difflib.SequenceMatcher`` with ``autojunk=False`` so that all
    characters are considered (important for short company names).
    """
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# ── Core matching logic ───────────────────────────────────────────────────────

def _match_name_channel(
    display_name_normalised: str,
    companies: list[PortfolioCompany],
) -> list[SenderMatchResult]:
    """
    Match normalised sender display name against portfolio company names.

    Three-pass approach:
    1. **Exact substring**: company's normalised name ⊂ display_name (or v-v).
    2. **Full-text fuzzy**: SequenceMatcher ratio of the entire display name
       against the company name; handles cases like ``"acme corp"`` vs
       ``"acmecorp"`` where token splitting would lose the similarity signal.
    3. **Token-level fuzzy**: for each *token* in the display name compute
       SequenceMatcher ratio against each company's normalised name;
       accept when ratio ≥ FUZZY_NAME_THRESHOLD.

    Pass 2 and 3 are combined: the highest ratio from either approach is used.

    Applies defensive lowercasing so the function is safe even when called with
    a non-normalised string (the caller should normalise, but we guard here).

    Returns a list of ``SenderMatchResult`` (may contain duplicates — caller
    deduplicates to keep the highest confidence entry per company).
    """
    if not display_name_normalised:
        return []

    # Defensive lowercase — normalisation should already have lowercased but
    # guards against callers that skip normalisation.
    display_name_normalised = display_name_normalised.lower()

    results: list[SenderMatchResult] = []

    for company in companies:
        cn = company.normalised
        if not cn or len(cn) < MIN_COMPANY_NAME_LEN:
            continue

        # 1. Exact substring check (bidirectional)
        if cn in display_name_normalised or display_name_normalised in cn:
            results.append(SenderMatchResult(
                company=company,
                match_type="exact_name",
                confidence=1.0,
                matched_text=display_name_normalised,
            ))
            continue  # No need to check fuzzy if exact already found

        # 2. Full-text fuzzy — compare the whole display name against the
        #    company name.  This catches cases like "acme corp" ↔ "acmecorp"
        #    where token splitting would yield low individual-token ratios.
        full_ratio = _fuzzy_ratio(display_name_normalised, cn)

        # 3. Token-level fuzzy — useful when the display name contains the
        #    company name as one distinct token among many others (e.g.
        #    "김민준 (BetaStart) 대표" where "betastart" is one token).
        tokens = _tokenise_name(display_name_normalised)
        best_token_ratio: float = 0.0
        best_token: str = ""
        for token in tokens:
            ratio = _fuzzy_ratio(token, cn)
            if ratio > best_token_ratio:
                best_token_ratio = ratio
                best_token = token

        # Pick the higher signal between full-text and token-level
        if full_ratio >= best_token_ratio:
            best_ratio = full_ratio
            best_matched_text = display_name_normalised
        else:
            best_ratio = best_token_ratio
            best_matched_text = best_token

        if best_ratio >= FUZZY_NAME_THRESHOLD:
            results.append(SenderMatchResult(
                company=company,
                match_type="fuzzy_name",
                confidence=best_ratio,
                matched_text=best_matched_text,
            ))

    return results


def _match_domain_channel(
    domain_root: str,
    companies: list[PortfolioCompany],
) -> list[SenderMatchResult]:
    """
    Match sender domain root against portfolio company names.

    Two-pass approach:
    1. **Exact substring**: domain_root ⊂ company.normalised OR vice-versa.
    2. **Fuzzy**: SequenceMatcher ratio ≥ FUZZY_DOMAIN_THRESHOLD.

    The domain root is already a compact token (e.g. "acmecorp"), so token
    splitting is not needed here — the full root is compared directly.

    Returns a list of ``SenderMatchResult``.
    """
    if not domain_root or len(domain_root) < MIN_TOKEN_LEN_FOR_FUZZY:
        return []

    results: list[SenderMatchResult] = []

    for company in companies:
        cn = company.normalised
        if not cn or len(cn) < MIN_COMPANY_NAME_LEN:
            continue

        # 1. Exact substring (bidirectional)
        if domain_root in cn or cn in domain_root:
            results.append(SenderMatchResult(
                company=company,
                match_type="exact_domain",
                confidence=1.0,
                matched_text=domain_root,
            ))
            continue

        # 2. Fuzzy ratio on full domain root vs full company name
        ratio = _fuzzy_ratio(domain_root, cn)
        if ratio >= FUZZY_DOMAIN_THRESHOLD:
            results.append(SenderMatchResult(
                company=company,
                match_type="fuzzy_domain",
                confidence=ratio,
                matched_text=domain_root,
            ))

    return results


def _merge_results(
    name_results: list[SenderMatchResult],
    domain_results: list[SenderMatchResult],
) -> list[SenderMatchResult]:
    """
    Merge and deduplicate results from both channels.

    When the same company appears in both channels, keep the entry with the
    higher confidence.  Sort final list by confidence descending.
    """
    # page_id → SenderMatchResult
    best: dict[str, SenderMatchResult] = {}

    for result in name_results + domain_results:
        pid = result.company.page_id
        if pid not in best or result.confidence > best[pid].confidence:
            best[pid] = result

    return sorted(best.values(), key=lambda r: r.confidence, reverse=True)


# ── SenderMatcher class ───────────────────────────────────────────────────────

class SenderMatcher:
    """
    Matches email sender information against the portfolio company cache.

    Parameters
    ----------
    cache : ``PortfolioCache`` instance to use.  When *None*, the module-level
            singleton returned by ``get_portfolio_cache()`` is used.

    Usage
    -----
    ::

        matcher = SenderMatcher()
        summary = matcher.match("CEO <ceo@acmecorp.co.kr>")
        if summary.matched:
            print(summary.top_match.company.name)

    Thread safety
    -------------
    ``SenderMatcher`` has no mutable state beyond the ``PortfolioCache``
    reference (which is itself thread-safe).  It is safe to share a single
    ``SenderMatcher`` instance across threads.
    """

    def __init__(self, cache: Optional[PortfolioCache] = None) -> None:
        self._cache: PortfolioCache = cache or get_portfolio_cache()

    # ── Public API ────────────────────────────────────────────────────────────

    def match(self, sender_raw: str) -> SenderMatchSummary:
        """
        Match a raw From: header value against the portfolio company cache.

        Parameters
        ----------
        sender_raw : The full From: header string, e.g.
                     ``"Alice Kim <alice@acmecorp.co.kr>"``
                     or ``"noreply@newsletter.example.com"``.

        Returns
        -------
        ``SenderMatchSummary`` with all matched companies sorted by confidence.
        No exception is raised; on cache load failure an empty summary is
        returned and the error is logged.
        """
        try:
            self._cache.ensure_loaded()
        except Exception as exc:
            logger.error("SenderMatcher: cache load failed: %s", exc)
            return SenderMatchSummary(
                sender_raw=sender_raw,
                display_name="",
                domain_root="",
                matches=[],
            )

        # Extract and normalise display name
        raw_display = _parse_display_name(sender_raw)
        norm_display = _normalise_display_name(raw_display) if raw_display else ""

        # Extract domain root
        domain_root = _extract_sender_domain_root(sender_raw)

        # Fetch company snapshot (thread-safe copy)
        companies = self._cache.get_all_companies()

        # Run both channels
        name_results = _match_name_channel(norm_display, companies)
        domain_results = _match_domain_channel(domain_root, companies)

        merged = _merge_results(name_results, domain_results)

        summary = SenderMatchSummary(
            sender_raw=sender_raw,
            display_name=norm_display,
            domain_root=domain_root,
            matches=merged,
        )

        if merged:
            logger.debug(
                "SenderMatcher: sender=%r matched %d company(ies): %s",
                sender_raw,
                len(merged),
                [r.company.name for r in merged],
            )
        else:
            logger.debug(
                "SenderMatcher: sender=%r → no portfolio match", sender_raw
            )

        return summary

    def is_portfolio_sender(self, sender_raw: str) -> bool:
        """
        Convenience method: True when the sender matches any portfolio company.
        """
        return self.match(sender_raw).matched

    def get_matched_companies(
        self,
        sender_raw: str,
        min_confidence: float = 0.0,
    ) -> list[PortfolioCompany]:
        """
        Return a list of matched ``PortfolioCompany`` objects for *sender_raw*.

        Parameters
        ----------
        sender_raw      : The full From: header string.
        min_confidence  : Only return matches with confidence ≥ this value.
                          Default 0.0 returns all matches.

        Returns
        -------
        List of ``PortfolioCompany`` sorted by confidence descending.
        """
        summary = self.match(sender_raw)
        return [
            r.company
            for r in summary.matches
            if r.confidence >= min_confidence
        ]


# ── Module-level convenience functions ───────────────────────────────────────

def match_sender(
    sender_raw: str,
    cache: Optional[PortfolioCache] = None,
) -> SenderMatchSummary:
    """
    Module-level convenience function: match *sender_raw* against the
    portfolio company cache.

    Creates a fresh ``SenderMatcher`` on every call.  For repeated matching
    in a loop, prefer creating one ``SenderMatcher`` and reusing it.

    Parameters
    ----------
    sender_raw : Full From: header value.
    cache      : Optional ``PortfolioCache`` to use.  Defaults to the
                 module-level singleton.

    Returns
    -------
    ``SenderMatchSummary``.
    """
    return SenderMatcher(cache=cache).match(sender_raw)


def is_portfolio_sender(
    sender_raw: str,
    cache: Optional[PortfolioCache] = None,
) -> bool:
    """
    Module-level convenience: True when *sender_raw* matches any portfolio
    company name or domain in the Notion cache.
    """
    return match_sender(sender_raw, cache=cache).matched
