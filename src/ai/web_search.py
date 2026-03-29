"""
Web Search Integration for External-First Meeting Briefings (Sub-AC 6a).

Provides web search capabilities to enrich meeting briefings for first-time
external meetings (EXTERNAL_FIRST category). Queries company and meeting
context using external search APIs and returns structured results.

Search providers (in priority order)
--------------------------------------
1. Tavily API   — structured search designed for AI/RAG use cases.
                  Requires TAVILY_API_KEY in .env
2. Claude API   — fallback using Claude's built-in web_search tool.
                  Uses ANTHROPIC_API_KEY (already configured in the project).

When both providers are unavailable, the module returns a WebSearchSummary
with ``available=False`` and records an error — the briefing pipeline
annotates the web-search section as '확인 불가' instead of silently omitting
it or fabricating information.

Usage::

    from src.ai.web_search import WebSearchClient

    client = WebSearchClient()
    summary = client.search_company_context(
        company_domains=["acme.com"],
        meeting_title="Product Demo - ACME Corp",
        attendee_names=["John Smith"],
    )
    # summary.results  → list[WebSearchResult]
    # summary.summary  → plain-text overview for the Slack briefing block
    # summary.available → False if all providers failed

Design constraints
------------------
* Retry logic: 3 attempts × 10-second delay on provider failure.
* Data safety: never fabricate information; annotate failures as '확인 불가'.
* Scope: only called for EXTERNAL_FIRST meetings to avoid unnecessary API costs.
* Query cap: ≤ 3 queries per meeting; ≤ 3 results per query.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum search results requested from the provider per query
_MAX_RESULTS_PER_QUERY: int = 3

# Maximum distinct queries executed per meeting
_MAX_QUERIES: int = 3

# Maximum characters for an individual result snippet
_MAX_SNIPPET_LEN: int = 300

# Maximum characters for the assembled briefing summary
_MAX_SUMMARY_LEN: int = 800

# Retry logic (mirrors global API_RETRY_ATTEMPTS / API_RETRY_DELAY_SECONDS)
_SEARCH_RETRY_ATTEMPTS: int = 3
_SEARCH_RETRY_DELAY: float = 10.0


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class WebSearchResult:
    """
    A single web search result from a provider.

    Attributes
    ----------
    query:    The query string that produced this result.
    title:    Page / article title from the search result.
    url:      Source URL (empty string when provider omits it).
    snippet:  Short text excerpt (≤ _MAX_SNIPPET_LEN characters).
    provider: Which provider produced this result: "tavily" | "claude" | "none".
    score:    Provider-supplied relevance score (0.0–1.0); 0.0 if not available.
    """

    query: str
    title: str
    url: str
    snippet: str
    provider: str = "unknown"
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
            "score": self.score,
        }


@dataclass
class WebSearchSummary:
    """
    Aggregated web search output for a single meeting briefing.

    Produced by ``WebSearchClient.search_company_context()`` and consumed by
    ``MeetingContextAggregator`` which attaches it to ``RawBriefingContent``.
    The briefing formatter renders it as a distinct Slack Block Kit section.

    Attributes
    ----------
    company_names:    Human-readable company names derived from the searched
                      domains (e.g. ["Acme Corp"] from ["acme.com"]).
    queries_executed: Actual query strings that were submitted to the provider.
    results:          All collected WebSearchResult objects.
    summary:          Pre-assembled plain-text overview for the Slack block
                      (pulled from top-scored snippets; ≤ _MAX_SUMMARY_LEN).
    provider:         Provider that was used ("tavily" | "claude" | "none").
    available:        False when all retries were exhausted or no provider is
                      configured.
    error:            Human-readable error message when available=False.
    """

    company_names: list[str] = field(default_factory=list)
    queries_executed: list[str] = field(default_factory=list)
    results: list[WebSearchResult] = field(default_factory=list)
    summary: str = ""
    provider: str = "none"
    available: bool = True
    error: Optional[str] = None

    @property
    def has_results(self) -> bool:
        """True if at least one search result was returned."""
        return bool(self.results)

    def to_dict(self) -> dict:
        return {
            "company_names": self.company_names,
            "queries_executed": self.queries_executed,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "provider": self.provider,
            "available": self.available,
            "error": self.error,
        }


# ── Query builder ─────────────────────────────────────────────────────────────

def build_search_queries(
    company_domains: list[str],
    meeting_title: str,
    attendee_names: Optional[list[str]] = None,
    max_queries: int = _MAX_QUERIES,
) -> list[str]:
    """
    Build targeted search queries from meeting context.

    Query generation priority:
      1. ``"{CompanyName}" company overview`` for each unique domain.
      2. Meeting title as-is (when informative, i.e. not generic).
      3. ``"{AttendaneName} {CompanyName}"`` for the top external attendee.

    Duplicate queries are suppressed.  Returns at most *max_queries* entries.

    Args:
        company_domains: External attendee email domains (e.g. ["acme.com"]).
        meeting_title:   Calendar event summary / title string.
        attendee_names:  Display names of external attendees (optional).
        max_queries:     Upper bound on returned queries.

    Returns:
        List of deduplicated query strings.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> bool:
        """Add query if not duplicate and below limit.  Returns True if added."""
        normalised = q.strip()
        if not normalised or normalised in seen:
            return False
        seen.add(normalised)
        queries.append(normalised)
        return True

    # 1. Company-name queries from domains
    for domain in company_domains:
        if len(queries) >= max_queries:
            break
        company_name = _domain_to_company_name(domain)
        if company_name:
            _add(f'"{company_name}" company overview')

    # 2. Meeting title (only if non-generic and long enough)
    if len(queries) < max_queries:
        clean_title = meeting_title.strip()
        if len(clean_title) > 10 and not _is_generic_title(clean_title):
            _add(clean_title)

    # 3. Attendee name + company for top external participant
    if len(queries) < max_queries and attendee_names and company_domains:
        top_company = _domain_to_company_name(company_domains[0])
        top_attendee = (attendee_names[0] or "").strip()
        if top_attendee and top_company and top_company.lower() not in top_attendee.lower():
            _add(f"{top_attendee} {top_company}")

    return queries[:max_queries]


def _domain_to_company_name(domain: str) -> str:
    """
    Extract a human-readable company name from an email domain.

    Examples::

        "acme-corp.com"      → "Acme Corp"
        "openai.com"         → "Openai"
        "mail.samsung.co.kr" → "Samsung"
        ""                   → ""
    """
    if not domain:
        return ""

    # Normalise
    parts = domain.lower().split(".")

    # Strip common subdomains (mail.*, www.*, app.*, api.*)
    if parts and parts[0] in {"mail", "www", "app", "api", "corp", "m"}:
        parts = parts[1:]

    if not parts:
        return ""

    # For "something.co.kr" or "something.com" take the last meaningful segment
    # i.e. the part just before the TLD(s)
    # Remove trailing known TLDs: com, co, kr, io, net, org, ai, jp, cn, de ...
    tlds = {"com", "co", "kr", "io", "net", "org", "ai", "jp", "cn", "de",
            "uk", "au", "fr", "in", "sg", "tech", "app"}
    while len(parts) > 1 and parts[-1] in tlds:
        parts = parts[:-1]

    name_part = parts[-1] if parts else ""
    # Convert hyphens / underscores to spaces, title-case
    return name_part.replace("-", " ").replace("_", " ").title()


def _is_generic_title(title: str) -> bool:
    """
    Return True if the meeting title is too generic to produce a useful query.

    Treats titles as generic if they consist only of common scheduling words
    (e.g. "meeting", "1:1", "call") or are very short.
    """
    generic_keywords = {
        "meeting", "미팅", "call", "콜", "sync", "싱크", "chat",
        "discussion", "intro", "kickoff", "catch-up", "catchup", "회의",
        "1:1", "coffee", "lunch", "zoom", "teams", "meet",
    }
    lower = title.lower().strip()

    # Pure generic word
    if lower in generic_keywords:
        return True

    # Very short after stripping punctuation and spaces
    stripped = lower.replace(" ", "").replace("-", "").replace(":", "")
    if len(stripped) < 4:
        return True

    return False


# ── Provider: Tavily ──────────────────────────────────────────────────────────

class _TavilyProvider:
    """
    Tavily search provider (primary).

    Uses the ``tavily-python`` SDK.  Requires TAVILY_API_KEY.
    Install with: ``pip install tavily-python``
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from tavily import TavilyClient  # type: ignore[import]
            self._client = TavilyClient(api_key=self._api_key)
        except ImportError as exc:
            raise RuntimeError(
                "tavily-python is not installed. "
                "Run: pip install 'tavily-python>=0.3.0'"
            ) from exc

    def search(
        self,
        query: str,
        max_results: int = _MAX_RESULTS_PER_QUERY,
    ) -> list[WebSearchResult]:
        """Execute *query* and return up to *max_results* results."""
        self._ensure_client()
        assert self._client is not None  # noqa: S101

        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        results: list[WebSearchResult] = []
        for item in (response.get("results") or [])[:max_results]:
            snippet = (item.get("content") or "")[:_MAX_SNIPPET_LEN]
            results.append(
                WebSearchResult(
                    query=query,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    provider="tavily",
                    score=float(item.get("score") or 0.0),
                )
            )
        return results


# ── Provider: Claude web_search ───────────────────────────────────────────────

class _ClaudeWebSearchProvider:
    """
    Fallback search provider using Anthropic Claude with the web_search tool.

    Sends a single-turn message to Claude asking for a factual summary of the
    query, with the ``web_search_20250305`` tool enabled.  Claude's response
    (and any inline citations it produces) are converted to WebSearchResult
    objects.

    Notes
    -----
    * Model used: ``claude-opus-4-5`` (supports web_search tool).
    * One API call per query; each call counts toward Anthropic rate limits.
    * This provider is only activated when TAVILY_API_KEY is absent or broken.
    """

    # Model that supports the built-in web_search tool
    _MODEL = "claude-opus-4-5"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # type: ignore[import]
            self._client = anthropic.Anthropic(api_key=self._api_key)
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. "
                "Run: pip install 'anthropic>=0.23.0'"
            ) from exc

    def search(
        self,
        query: str,
        max_results: int = _MAX_RESULTS_PER_QUERY,  # noqa: ARG002 (unused here)
    ) -> list[WebSearchResult]:
        """
        Query Claude with the web_search tool and return a single result.

        Returns a list with one WebSearchResult whose snippet is Claude's
        response text (sourced from live web content via the tool).
        """
        self._ensure_client()
        assert self._client is not None  # noqa: S101

        import anthropic  # type: ignore[import]

        prompt = (
            f"Search the web for: {query}\n\n"
            "Provide a concise factual summary (3–5 sentences) with key "
            "details about the company or topic.  Do not speculate — only "
            "report information you actually find.  Include source URLs when "
            "available."
        )

        try:
            response = self._client.messages.create(
                model=self._MODEL,
                max_tokens=512,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.BadRequestError as exc:
            # web_search tool may not be available in all regions/tiers
            logger.warning(
                "[WebSearch/Claude] web_search tool unavailable: %s", exc
            )
            return []

        # Collect text blocks from the response
        full_text = ""
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                full_text += block.text

        if not full_text.strip():
            return []

        snippet = full_text.strip()[:_MAX_SNIPPET_LEN]

        return [
            WebSearchResult(
                query=query,
                title=f"Web overview: {query[:60]}",
                url="",   # Claude does not always emit citable URLs
                snippet=snippet,
                provider="claude",
                score=0.7,
            )
        ]


# ── Main client ───────────────────────────────────────────────────────────────

class WebSearchClient:
    """
    Web search client for enriching EXTERNAL_FIRST meeting briefings.

    Provider selection (auto-detected at first use):
      - Tavily     — if TAVILY_API_KEY is set and ``tavily-python`` is installed.
      - Claude     — if ANTHROPIC_API_KEY is set (fallback).
      - None       — both unavailable; returns a summary with available=False.

    Retry policy
    ------------
    Every query is retried up to ``_SEARCH_RETRY_ATTEMPTS`` (3) times with
    ``_SEARCH_RETRY_DELAY`` (10 s) between attempts, mirroring the global
    API retry policy in src/config.py.

    Usage::

        client = WebSearchClient()
        summary = client.search_company_context(
            company_domains=["acme.com"],
            meeting_title="ACME Product Demo",
            attendee_names=["Jane Doe"],
        )
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        # Auto-load keys from src/config if not explicitly provided
        if tavily_api_key is None:
            tavily_api_key = _load_config_key("TAVILY_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = _load_config_key("ANTHROPIC_API_KEY")

        self._tavily_key: Optional[str] = tavily_api_key or None
        self._anthropic_key: Optional[str] = anthropic_api_key or None

        self._provider: Optional[_TavilyProvider | _ClaudeWebSearchProvider] = None
        self._provider_name: str = "none"
        self._initialized: bool = False

    # ── Provider initialisation ───────────────────────────────────────────────

    def _init_provider(self) -> None:
        """Select and lazy-initialise the best available search provider."""
        if self._initialized:
            return
        self._initialized = True

        # 1. Try Tavily (preferred)
        if self._tavily_key:
            try:
                candidate = _TavilyProvider(self._tavily_key)
                candidate._ensure_client()
                self._provider = candidate
                self._provider_name = "tavily"
                logger.info("[WebSearch] Provider: Tavily")
                return
            except Exception as exc:
                logger.warning(
                    "[WebSearch] Tavily init failed (%s); trying Claude fallback", exc
                )

        # 2. Fallback to Claude web_search
        if self._anthropic_key:
            try:
                candidate = _ClaudeWebSearchProvider(self._anthropic_key)
                candidate._ensure_client()
                self._provider = candidate
                self._provider_name = "claude"
                logger.info("[WebSearch] Provider: Claude web_search (fallback)")
                return
            except Exception as exc:
                logger.warning(
                    "[WebSearch] Claude web_search init failed (%s)", exc
                )

        # 3. No provider available
        logger.warning(
            "[WebSearch] No search provider available. "
            "Set TAVILY_API_KEY for Tavily or ANTHROPIC_API_KEY for Claude fallback."
        )

    @property
    def is_available(self) -> bool:
        """True if at least one search provider was successfully initialised."""
        self._init_provider()
        return self._provider is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def search_company_context(
        self,
        company_domains: list[str],
        meeting_title: str,
        attendee_names: Optional[list[str]] = None,
    ) -> WebSearchSummary:
        """
        Search for company/meeting context for an EXTERNAL_FIRST briefing.

        Builds targeted queries from external attendee domains, meeting title,
        and optional attendee display names.  Executes each query with retry
        logic and assembles the results into a ``WebSearchSummary``.

        Args:
            company_domains: Email domains of external attendees.
            meeting_title:   Calendar event title / summary.
            attendee_names:  Display names of external attendees (optional).

        Returns:
            ``WebSearchSummary`` — always returns, never raises.
        """
        self._init_provider()

        company_names = [
            _domain_to_company_name(d)
            for d in company_domains
            if d
        ]
        company_names = [n for n in company_names if n]

        summary = WebSearchSummary(
            company_names=company_names,
            provider=self._provider_name,
        )

        if not self.is_available:
            summary.available = False
            summary.error = (
                "검색 제공자 미설정 (TAVILY_API_KEY 또는 ANTHROPIC_API_KEY 필요)"
            )
            logger.debug("[WebSearch] No provider — skipping search")
            return summary

        queries = build_search_queries(
            company_domains=company_domains,
            meeting_title=meeting_title,
            attendee_names=attendee_names,
        )

        if not queries:
            summary.available = True
            summary.summary = ""
            logger.debug("[WebSearch] No queries generated for this meeting context")
            return summary

        summary.queries_executed = queries

        # Execute each query with retry
        all_results: list[WebSearchResult] = []
        all_failed = True  # becomes False as soon as one query succeeds

        for query in queries:
            results, succeeded = self._search_with_retry(query)
            if succeeded:
                all_failed = False
            all_results.extend(results)

        if all_failed and not all_results:
            summary.available = False
            summary.error = f"모든 검색 쿼리 실패 (쿼리: {queries})"
            logger.error("[WebSearch] All queries failed for meeting '%s'", meeting_title)
            return summary

        summary.results = all_results
        if all_results:
            summary.summary = _build_summary_text(
                company_names=company_names,
                results=all_results,
            )

        logger.info(
            "[WebSearch] Done: %d queries, %d results, provider=%s, meeting='%s'",
            len(queries),
            len(all_results),
            self._provider_name,
            meeting_title[:60],
        )
        return summary

    def _search_with_retry(
        self, query: str
    ) -> tuple[list[WebSearchResult], bool]:
        """
        Execute *query* with up to ``_SEARCH_RETRY_ATTEMPTS`` retries.

        Returns ``(results, succeeded)`` where *succeeded* is False only when
        all attempts raised exceptions (i.e. an empty result list from the
        provider still counts as succeeded=True).
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, _SEARCH_RETRY_ATTEMPTS + 1):
            try:
                assert self._provider is not None  # noqa: S101
                results = self._provider.search(query)
                return results, True
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                logger.warning(
                    "[WebSearch] Query '%s' failed (attempt %d/%d): %s",
                    query[:60],
                    attempt,
                    _SEARCH_RETRY_ATTEMPTS,
                    exc,
                )
                if attempt < _SEARCH_RETRY_ATTEMPTS:
                    time.sleep(_SEARCH_RETRY_DELAY)

        logger.error(
            "[WebSearch] Query '%s' exhausted all %d retries: %s",
            query[:60],
            _SEARCH_RETRY_ATTEMPTS,
            last_exc,
        )
        return [], False


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary_text(
    company_names: list[str],
    results: list[WebSearchResult],
) -> str:
    """
    Assemble a concise briefing summary from search results.

    Combines the highest-scored snippets into a short paragraph suitable for
    rendering in a Slack Block Kit ``section`` block.

    Returns a string of at most ``_MAX_SUMMARY_LEN`` characters.
    """
    if not results:
        return ""

    # Sort by score descending; take top 3 non-empty snippets
    top_results = sorted(results, key=lambda r: r.score, reverse=True)[:3]
    snippets = [r.snippet.strip() for r in top_results if r.snippet.strip()]

    if not snippets:
        return ""

    header = ""
    if company_names:
        company_label = ", ".join(company_names[:2])
        header = f"*{company_label}* 웹 검색 결과:\n"

    body = "\n\n".join(snippets)
    full = header + body

    if len(full) > _MAX_SUMMARY_LEN:
        full = full[: _MAX_SUMMARY_LEN - 1] + "…"

    return full


# ── Config helper ─────────────────────────────────────────────────────────────

def _load_config_key(attr: str) -> Optional[str]:
    """Safely load a string config attribute; return None on any failure."""
    try:
        import importlib
        config = importlib.import_module("src.config")
        value = getattr(config, attr, None)
        return value if value else None
    except Exception:  # pylint: disable=broad-except
        return None
