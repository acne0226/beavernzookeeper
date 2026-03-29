"""
Tests for src/notion/portfolio_cache.py (Sub-AC 9a of AC 9).

Coverage:
- _normalise_name(): legal suffix stripping, lowercasing, whitespace collapsing
- _tokenise(): splitting on delimiters
- _extract_sender_domain_root(): various email formats
- PortfolioCompany dataclass (to_dict)
- EmailMatchResult dataclass (to_dict)
- PortfolioCache._parse_page(): happy path, missing title returns None
- PortfolioCache._fetch_all_companies(): pagination (has_more / next_cursor)
- PortfolioCache.load(): calls schema discovery + fetch, sets _loaded_at
- PortfolioCache._is_cache_fresh(): TTL logic
- PortfolioCache.ensure_loaded(): skips reload when fresh; reloads when stale
- PortfolioCache.get_all_companies(): returns snapshot, triggers ensure_loaded
- PortfolioCache.get_company_names(): returns plain strings
- PortfolioCache.find_matches(): substring search, case-insensitive, dedup
- PortfolioCache.match_email(): subject / sender_domain / body matching
- PortfolioCache.is_portfolio_email(): boolean convenience wrapper
- PortfolioCache.cache_info(): introspection dict
- get_portfolio_cache(): singleton semantics
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest

from src.notion.portfolio_cache import (
    PortfolioCache,
    PortfolioCompany,
    EmailMatchResult,
    _normalise_name,
    _tokenise,
    _extract_sender_domain_root,
    get_portfolio_cache,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_notion_page(
    page_id: str,
    title: str,
    status: str = "",
    url: str = "",
) -> dict:
    """Build a minimal Notion page dict as returned by the API."""
    props: dict = {
        "Name": {
            "type": "title",
            "title": [{"plain_text": title}],
        }
    }
    if status:
        props["Stage"] = {
            "type": "select",
            "select": {"name": status},
        }
    return {
        "id": page_id,
        "url": url or f"https://notion.so/{page_id}",
        "properties": props,
    }


def _make_cache() -> PortfolioCache:
    """Create a PortfolioCache with a mocked Notion client."""
    cache = PortfolioCache.__new__(PortfolioCache)
    cache._db_id = "test-db-id"
    cache._cache_ttl = 3600
    cache._client = MagicMock()
    cache._lock = threading.Lock()
    cache._companies = []
    cache._loaded_at = None
    cache._schema_title_field = "Name"
    cache._schema_status_field = "Stage"
    return cache


# ── _normalise_name ──────────────────────────────────────────────────────────

class TestNormaliseName:
    def test_lowercase(self):
        assert _normalise_name("AcmeCorp") == "acmecorp"

    def test_strips_inc_suffix_space_separated(self):
        # "AcmeCorp Inc." has a space before Inc → Inc is stripped
        assert _normalise_name("AcmeCorp Inc.") == "acmecorp"

    def test_compound_name_corp_not_stripped(self):
        # "AcmeCorp" has no space before "Corp" → should not strip
        assert _normalise_name("AcmeCorp") == "acmecorp"

    def test_strips_corp_suffix_space_separated(self):
        # "BetaTech Corp" has a space before Corp → Corp is stripped
        assert _normalise_name("BetaTech Corp") == "betatech"

    def test_strips_ltd_suffix_space_separated(self):
        # "Gamma Ltd." has a space before Ltd → stripped
        assert _normalise_name("Gamma Ltd.") == "gamma"

    def test_strips_korean_jusikhweasa_prefix(self):
        # "주식회사 에이비씨" → 주식회사 is a leading prefix
        result = _normalise_name("주식회사 에이비씨")
        assert "주식회사" not in result
        assert "에이비씨" in result

    def test_strips_korean_jusikhweasa_suffix(self):
        # "에이비씨 주식회사" → trailing 주식회사
        result = _normalise_name("에이비씨 주식회사")
        assert "주식회사" not in result
        assert "에이비씨" in result

    def test_strips_trailing_whitespace(self):
        assert _normalise_name("  SpacePad  ") == "spacepad"

    def test_collapses_multiple_spaces(self):
        result = _normalise_name("Foo  Bar   Baz")
        assert "  " not in result

    def test_empty_string(self):
        assert _normalise_name("") == ""

    def test_already_normalised_no_spaces(self):
        # Compound name without spaces: no suffix stripping
        assert _normalise_name("acmecorp") == "acmecorp"


# ── _tokenise ────────────────────────────────────────────────────────────────

class TestTokenise:
    def test_splits_on_whitespace(self):
        assert "hello" in _tokenise("hello world")
        assert "world" in _tokenise("hello world")

    def test_filters_short_tokens(self):
        tokens = _tokenise("a bb ccc")
        assert "a" not in tokens
        assert "bb" in tokens
        assert "ccc" in tokens

    def test_splits_on_at_sign(self):
        tokens = _tokenise("user@example.com")
        assert "user" in tokens or "example" in tokens

    def test_lowercases(self):
        tokens = _tokenise("FooBar Baz")
        assert "foobar" in tokens
        assert "baz" in tokens


# ── _extract_sender_domain_root ──────────────────────────────────────────────

class TestExtractSenderDomainRoot:
    def test_bare_email(self):
        assert _extract_sender_domain_root("user@acmecorp.com") == "acmecorp"

    def test_display_name_format(self):
        assert _extract_sender_domain_root("CEO <ceo@acmecorp.com>") == "acmecorp"

    def test_subdomain(self):
        result = _extract_sender_domain_root("user@mail.betastart.io")
        assert result == "betastart"

    def test_co_kr_domain(self):
        result = _extract_sender_domain_root("ceo@gammatech.co.kr")
        assert result == "gammatech"

    def test_empty_string(self):
        assert _extract_sender_domain_root("") == ""

    def test_no_at_sign(self):
        assert _extract_sender_domain_root("not-an-email") == ""


# ── PortfolioCompany ─────────────────────────────────────────────────────────

class TestPortfolioCompany:
    def test_to_dict(self):
        company = PortfolioCompany(
            page_id="p1",
            name="AcmeCorp",
            normalised="acmecorp",
            status="심사 중",
            url="https://notion.so/p1",
        )
        d = company.to_dict()
        assert d["page_id"] == "p1"
        assert d["name"] == "AcmeCorp"
        assert d["normalised"] == "acmecorp"
        assert d["status"] == "심사 중"
        assert d["url"] == "https://notion.so/p1"


# ── EmailMatchResult ─────────────────────────────────────────────────────────

class TestEmailMatchResult:
    def test_matched_false_default(self):
        result = EmailMatchResult(matched=False)
        assert result.matched is False
        assert result.companies == []
        assert result.match_sources == []

    def test_to_dict(self):
        company = PortfolioCompany(
            page_id="p1", name="AcmeCorp", normalised="acmecorp"
        )
        result = EmailMatchResult(
            matched=True,
            companies=[company],
            match_sources=["subject"],
        )
        d = result.to_dict()
        assert d["matched"] is True
        assert len(d["companies"]) == 1
        assert d["match_sources"] == ["subject"]


# ── PortfolioCache._parse_page ────────────────────────────────────────────────

class TestParsePage:
    def test_happy_path(self):
        cache = _make_cache()
        page = _make_notion_page("p1", "AcmeCorp", status="심사 중")
        result = cache._parse_page(page)
        assert result is not None
        assert result.page_id == "p1"
        assert result.name == "AcmeCorp"
        # "AcmeCorp" has no space before "Corp" → normalised is "acmecorp"
        assert result.normalised == "acmecorp"
        assert result.status == "심사 중"

    def test_missing_title_returns_none(self):
        cache = _make_cache()
        page = {
            "id": "p2",
            "url": "https://notion.so/p2",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [],  # empty title
                }
            },
        }
        result = cache._parse_page(page)
        assert result is None

    def test_no_title_field_configured_returns_none(self):
        cache = _make_cache()
        cache._schema_title_field = None
        page = _make_notion_page("p3", "SomeCompany")
        result = cache._parse_page(page)
        assert result is None

    def test_missing_status_field_still_parses(self):
        cache = _make_cache()
        cache._schema_status_field = None
        page = _make_notion_page("p4", "BetaStart")
        result = cache._parse_page(page)
        assert result is not None
        assert result.status == ""

    def test_url_set_from_page(self):
        cache = _make_cache()
        page = _make_notion_page("p5", "GammaTech", url="https://notion.so/custom")
        result = cache._parse_page(page)
        assert result is not None
        assert result.url == "https://notion.so/custom"

    def test_normalised_strips_suffix(self):
        cache = _make_cache()
        # "DeltaCorp Inc." has a space before "Inc." → suffix is stripped
        page = _make_notion_page("p6", "DeltaCorp Inc.")
        result = cache._parse_page(page)
        assert result is not None
        assert "inc" not in result.normalised
        assert "deltacorp" in result.normalised


# ── PortfolioCache._fetch_all_companies ───────────────────────────────────────

class TestFetchAllCompanies:
    def test_single_page_no_pagination(self):
        cache = _make_cache()
        pages = [
            _make_notion_page("p1", "Alpha"),
            _make_notion_page("p2", "Beta"),
        ]
        cache._client.databases.query.return_value = {
            "results": pages,
            "has_more": False,
        }

        companies = cache._fetch_all_companies()
        assert len(companies) == 2
        assert companies[0].name == "Alpha"
        assert companies[1].name == "Beta"
        cache._client.databases.query.assert_called_once()

    def test_pagination_follows_next_cursor(self):
        cache = _make_cache()
        page1 = [_make_notion_page(f"p{i}", f"Company {i}") for i in range(3)]
        page2 = [_make_notion_page(f"p{i}", f"Company {i}") for i in range(3, 5)]

        cache._client.databases.query.side_effect = [
            {"results": page1, "has_more": True, "next_cursor": "cursor-2"},
            {"results": page2, "has_more": False},
        ]

        companies = cache._fetch_all_companies()
        assert len(companies) == 5
        assert cache._client.databases.query.call_count == 2

    def test_skips_unnamed_records(self):
        cache = _make_cache()
        pages = [
            _make_notion_page("p1", "Named Co"),
            {  # unnamed
                "id": "p2",
                "url": "https://notion.so/p2",
                "properties": {
                    "Name": {"type": "title", "title": []},
                },
            },
        ]
        cache._client.databases.query.return_value = {
            "results": pages,
            "has_more": False,
        }

        companies = cache._fetch_all_companies()
        assert len(companies) == 1
        assert companies[0].name == "Named Co"

    def test_api_failure_raises_after_retries(self):
        cache = _make_cache()
        cache._client.databases.query.side_effect = Exception("Notion down")

        with pytest.raises(RuntimeError, match="Notion API failed"):
            cache._fetch_all_companies()


# ── PortfolioCache.load ───────────────────────────────────────────────────────

class TestLoad:
    def test_load_populates_cache(self):
        cache = _make_cache()
        pages = [_make_notion_page("p1", "StartupA")]
        cache._client.databases.query.return_value = {
            "results": pages, "has_more": False
        }
        cache._client.databases.retrieve.return_value = {
            "title": [{"plain_text": "Deal DB"}],
            "properties": {
                "Name": {"type": "title"},
                "Stage": {"type": "select"},
            },
        }

        cache.load()
        assert len(cache._companies) == 1
        assert cache._loaded_at is not None

    def test_load_skips_when_cache_fresh(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="Cached", normalised="cached")
        ]
        cache._loaded_at = datetime.now(timezone.utc)

        # Should not call Notion API
        cache._client.databases.retrieve = MagicMock()
        cache._client.databases.query = MagicMock()

        cache.load()  # Should be a no-op

        cache._client.databases.retrieve.assert_not_called()
        cache._client.databases.query.assert_not_called()

    def test_load_force_bypasses_ttl(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="Old", normalised="old")
        ]
        cache._loaded_at = datetime.now(timezone.utc)

        new_pages = [_make_notion_page("p2", "NewCompany")]
        cache._client.databases.query.return_value = {
            "results": new_pages, "has_more": False
        }
        cache._client.databases.retrieve.return_value = {
            "title": [{"plain_text": "Deal DB"}],
            "properties": {"Name": {"type": "title"}, "Stage": {"type": "select"}},
        }

        cache.load(force=True)
        names = [c.name for c in cache._companies]
        assert "NewCompany" in names


# ── PortfolioCache._is_cache_fresh ────────────────────────────────────────────

class TestIsCacheFresh:
    def test_fresh_when_loaded_recently(self):
        cache = _make_cache()
        cache._companies = [PortfolioCompany(page_id="p1", name="Co", normalised="co")]
        cache._loaded_at = datetime.now(timezone.utc)
        cache._cache_ttl = 3600
        assert cache._is_cache_fresh() is True

    def test_stale_when_loaded_long_ago(self):
        cache = _make_cache()
        cache._companies = [PortfolioCompany(page_id="p1", name="Co", normalised="co")]
        cache._loaded_at = datetime.now(timezone.utc) - timedelta(hours=2)
        cache._cache_ttl = 3600
        assert cache._is_cache_fresh() is False

    def test_not_fresh_when_never_loaded(self):
        cache = _make_cache()
        cache._loaded_at = None
        assert cache._is_cache_fresh() is False

    def test_not_fresh_when_empty(self):
        cache = _make_cache()
        cache._loaded_at = datetime.now(timezone.utc)
        cache._companies = []
        assert cache._is_cache_fresh() is False


# ── PortfolioCache.ensure_loaded ──────────────────────────────────────────────

class TestEnsureLoaded:
    def test_triggers_load_when_not_loaded(self):
        cache = _make_cache()
        with patch.object(cache, "load") as mock_load:
            cache.ensure_loaded()
            mock_load.assert_called_once()

    def test_no_op_when_fresh(self):
        cache = _make_cache()
        cache._companies = [PortfolioCompany(page_id="p1", name="Co", normalised="co")]
        cache._loaded_at = datetime.now(timezone.utc)
        with patch.object(cache, "load") as mock_load:
            cache.ensure_loaded()
            mock_load.assert_not_called()


# ── PortfolioCache.get_all_companies ─────────────────────────────────────────

class TestGetAllCompanies:
    def test_returns_snapshot(self):
        cache = _make_cache()
        companies = [
            PortfolioCompany(page_id="p1", name="Alpha", normalised="alpha"),
            PortfolioCompany(page_id="p2", name="Beta", normalised="beta"),
        ]
        cache._companies = companies
        cache._loaded_at = datetime.now(timezone.utc)

        result = cache.get_all_companies()
        assert len(result) == 2
        assert result[0].name == "Alpha"
        assert result[1].name == "Beta"

    def test_returns_copy_not_reference(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="Alpha", normalised="alpha")
        ]
        cache._loaded_at = datetime.now(timezone.utc)

        result = cache.get_all_companies()
        result.append(PortfolioCompany(page_id="p99", name="Rogue", normalised="rogue"))
        # Internal list should be unchanged
        assert len(cache._companies) == 1

    def test_triggers_ensure_loaded(self):
        cache = _make_cache()
        with patch.object(cache, "ensure_loaded") as mock_el:
            cache.get_all_companies()
            mock_el.assert_called_once()


# ── PortfolioCache.get_company_names ─────────────────────────────────────────

class TestGetCompanyNames:
    def test_returns_list_of_strings(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="Alpha Inc.", normalised="alpha"),
            PortfolioCompany(page_id="p2", name="베타 주식회사", normalised="베타"),
        ]
        cache._loaded_at = datetime.now(timezone.utc)

        names = cache.get_company_names()
        assert names == ["Alpha Inc.", "베타 주식회사"]


# ── PortfolioCache.find_matches ───────────────────────────────────────────────

class TestFindMatches:
    def _cache_with_companies(self) -> PortfolioCache:
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="AcmeCorp", normalised="acmecorp"),
            PortfolioCompany(page_id="p2", name="Beta Start", normalised="beta start"),
            PortfolioCompany(page_id="p3", name="Gamma AI", normalised="gamma ai"),
        ]
        cache._loaded_at = datetime.now(timezone.utc)
        return cache

    def test_exact_normalised_match(self):
        cache = self._cache_with_companies()
        results = cache.find_matches("AcmeCorp 투자 검토")
        assert any(c.page_id == "p1" for c in results)

    def test_case_insensitive_match(self):
        cache = self._cache_with_companies()
        results = cache.find_matches("ACMECORP meeting")
        assert any(c.page_id == "p1" for c in results)

    def test_no_match_returns_empty(self):
        cache = self._cache_with_companies()
        results = cache.find_matches("completely unrelated text")
        assert results == []

    def test_multiple_companies_matched(self):
        cache = self._cache_with_companies()
        results = cache.find_matches("meeting with acmecorp and gamma ai team")
        page_ids = {c.page_id for c in results}
        assert "p1" in page_ids
        assert "p3" in page_ids

    def test_deduplicates_results(self):
        cache = self._cache_with_companies()
        # 'acmecorp' appears twice in text — should only return 1 match
        results = cache.find_matches("acmecorp acmecorp acmecorp")
        acme_matches = [c for c in results if c.page_id == "p1"]
        assert len(acme_matches) == 1

    def test_full_name_matches_in_text(self):
        cache = self._cache_with_companies()
        # "gamma ai" (full normalised name) should match when it appears in text
        results = cache.find_matches("gamma ai quarterly report")
        assert any(c.page_id == "p3" for c in results)

    def test_partial_token_not_matched(self):
        cache = self._cache_with_companies()
        # "gamma" alone is not enough to match "gamma ai" (substring match
        # looks for the full normalised name)
        results = cache.find_matches("gamma report")
        # "gamma" is NOT a substring of "gamma ai" reversed; we check if
        # "gamma ai" (normalised) is a substring of the text → it is NOT
        # So no match expected
        assert not any(c.page_id == "p3" for c in results)


# ── PortfolioCache.match_email ────────────────────────────────────────────────

class TestMatchEmail:
    def _cache_with_companies(self) -> PortfolioCache:
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(
                page_id="p1", name="AcmeCorp", normalised="acmecorp",
                url="https://notion.so/p1",
            ),
            PortfolioCompany(
                page_id="p2", name="Beta Start", normalised="beta start",
                url="https://notion.so/p2",
            ),
        ]
        cache._loaded_at = datetime.now(timezone.utc)
        return cache

    def test_subject_match(self):
        cache = self._cache_with_companies()
        result = cache.match_email(subject="AcmeCorp 투자 검토 요청")
        assert result.matched is True
        assert any(c.page_id == "p1" for c in result.companies)
        assert "subject" in result.match_sources

    def test_sender_domain_match(self):
        cache = self._cache_with_companies()
        result = cache.match_email(sender="CEO <ceo@acmecorp.com>")
        assert result.matched is True
        assert "sender_domain" in result.match_sources

    def test_body_match(self):
        cache = self._cache_with_companies()
        result = cache.match_email(
            body="첨부된 서류는 acmecorp 관련 자료입니다."
        )
        assert result.matched is True
        assert "body" in result.match_sources

    def test_no_match_returns_false(self):
        cache = self._cache_with_companies()
        result = cache.match_email(
            subject="General newsletter",
            sender="noreply@unrelated.com",
            body="No portfolio company mentioned here.",
        )
        assert result.matched is False
        assert result.companies == []

    def test_deduplicates_across_sources(self):
        """If subject AND body both match AcmeCorp, only one entry returned."""
        cache = self._cache_with_companies()
        result = cache.match_email(
            subject="AcmeCorp deal",
            body="Follow up on acmecorp investment",
        )
        acme_matches = [c for c in result.companies if c.page_id == "p1"]
        assert len(acme_matches) == 1

    def test_multiple_companies_in_one_email(self):
        cache = self._cache_with_companies()
        result = cache.match_email(
            subject="AcmeCorp and Beta Start joint meeting"
        )
        assert result.matched is True
        page_ids = {c.page_id for c in result.companies}
        assert "p1" in page_ids
        assert "p2" in page_ids

    def test_empty_inputs_no_match(self):
        cache = self._cache_with_companies()
        result = cache.match_email()
        assert result.matched is False

    def test_short_body_not_matched(self):
        """Body shorter than 10 chars should be skipped."""
        cache = self._cache_with_companies()
        result = cache.match_email(body="hi")
        assert result.matched is False

    def test_result_to_dict(self):
        cache = self._cache_with_companies()
        result = cache.match_email(subject="AcmeCorp 투자")
        d = result.to_dict()
        assert d["matched"] is True
        assert isinstance(d["companies"], list)
        assert "subject" in d["match_sources"]


# ── PortfolioCache.is_portfolio_email ────────────────────────────────────────

class TestIsPortfolioEmail:
    def test_returns_true_on_match(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="AcmeCorp", normalised="acmecorp"),
        ]
        cache._loaded_at = datetime.now(timezone.utc)
        assert cache.is_portfolio_email(subject="AcmeCorp 계약서") is True

    def test_returns_false_on_no_match(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="AcmeCorp", normalised="acmecorp"),
        ]
        cache._loaded_at = datetime.now(timezone.utc)
        assert cache.is_portfolio_email(subject="Weekly newsletter") is False


# ── PortfolioCache.cache_info ────────────────────────────────────────────────

class TestCacheInfo:
    def test_empty_cache(self):
        cache = _make_cache()
        info = cache.cache_info()
        assert info["count"] == 0
        assert info["loaded_at"] is None
        assert info["is_fresh"] is False

    def test_loaded_cache(self):
        cache = _make_cache()
        cache._companies = [
            PortfolioCompany(page_id="p1", name="Co", normalised="co")
        ]
        cache._loaded_at = datetime.now(timezone.utc)
        cache._schema_title_field = "Name"
        cache._schema_status_field = "Stage"

        info = cache.cache_info()
        assert info["count"] == 1
        assert info["is_fresh"] is True
        assert info["ttl_seconds"] == 3600
        assert info["title_field"] == "Name"
        assert info["status_field"] == "Stage"
        assert info["age_seconds"] is not None
        assert info["age_seconds"] < 5  # Should be very fresh


# ── get_portfolio_cache (singleton) ──────────────────────────────────────────

class TestGetPortfolioCache:
    def test_returns_portfolio_cache_instance(self):
        # Reset the global singleton for test isolation
        import src.notion.portfolio_cache as pc_module
        original = pc_module._default_cache
        pc_module._default_cache = None
        try:
            cache = get_portfolio_cache()
            assert isinstance(cache, PortfolioCache)
        finally:
            pc_module._default_cache = original

    def test_singleton_returns_same_instance(self):
        import src.notion.portfolio_cache as pc_module
        original = pc_module._default_cache
        pc_module._default_cache = None
        try:
            c1 = get_portfolio_cache()
            c2 = get_portfolio_cache()
            assert c1 is c2
        finally:
            pc_module._default_cache = original
