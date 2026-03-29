"""
Tests for src/gmail/company_name_cache.py (AC 10).

Coverage:
- CompanyNameCache.get() / set() / contains() / all_mappings() / size()
- Persistent round-trip: save then reload
- Thread-safety: concurrent sets don't corrupt state
- get_or_set() helper
- Cache singleton get_company_name_cache()
- Tolerance for missing/corrupt cache file
"""
from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.gmail.company_name_cache import CompanyNameCache, get_company_name_cache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path):
    """Return a fresh CompanyNameCache backed by a temp directory."""
    return CompanyNameCache(cache_file=tmp_path / "test_company_name_cache.json")


# ── get / set / contains ──────────────────────────────────────────────────────

class TestBasicOperations:
    def test_get_missing_key_returns_none(self, tmp_cache):
        assert tmp_cache.get("unknowncompany") is None

    def test_set_then_get_returns_value(self, tmp_cache):
        tmp_cache.set("acmecorp", "AcmeCorp")
        assert tmp_cache.get("acmecorp") == "AcmeCorp"

    def test_get_is_case_insensitive(self, tmp_cache):
        tmp_cache.set("AcmeCorp", "AcmeCorp Inc.")
        assert tmp_cache.get("acmecorp") == "AcmeCorp Inc."
        assert tmp_cache.get("ACMECORP") == "AcmeCorp Inc."

    def test_contains_true_after_set(self, tmp_cache):
        tmp_cache.set("betastart", "Beta Start")
        assert tmp_cache.contains("betastart") is True

    def test_contains_false_before_set(self, tmp_cache):
        assert tmp_cache.contains("gamma") is False

    def test_size_increases_with_new_entries(self, tmp_cache):
        assert tmp_cache.size() == 0
        tmp_cache.set("a", "A")
        assert tmp_cache.size() == 1
        tmp_cache.set("b", "B")
        assert tmp_cache.size() == 2

    def test_set_same_key_does_not_increase_size(self, tmp_cache):
        tmp_cache.set("acmecorp", "AcmeCorp")
        tmp_cache.set("acmecorp", "AcmeCorp")
        assert tmp_cache.size() == 1

    def test_all_mappings_returns_snapshot(self, tmp_cache):
        tmp_cache.set("a", "A")
        tmp_cache.set("b", "B")
        mappings = tmp_cache.all_mappings()
        assert "a" in mappings
        assert "b" in mappings
        assert len(mappings) == 2

    def test_clear_empties_cache(self, tmp_cache):
        tmp_cache.set("a", "A")
        tmp_cache.clear()
        assert tmp_cache.size() == 0


class TestGetOrSet:
    def test_get_or_set_returns_cached_when_exists(self, tmp_cache):
        tmp_cache.set("acmecorp", "Original Name")
        result = tmp_cache.get_or_set("acmecorp", "New Name")
        assert result == "Original Name"

    def test_get_or_set_stores_and_returns_new_name(self, tmp_cache):
        result = tmp_cache.get_or_set("newcompany", "New Company Inc.")
        assert result == "New Company Inc."
        assert tmp_cache.get("newcompany") == "New Company Inc."


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_data_survives_reload(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        c1 = CompanyNameCache(cache_file=cache_file)
        c1.set("acmecorp", "AcmeCorp")
        c1.set("betastart", "Beta Start")

        # Create a fresh instance pointing to the same file
        c2 = CompanyNameCache(cache_file=cache_file)
        assert c2.get("acmecorp") == "AcmeCorp"
        assert c2.get("betastart") == "Beta Start"

    def test_reload_method_refreshes_data(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        c1 = CompanyNameCache(cache_file=cache_file)
        c1.set("acmecorp", "AcmeCorp")

        # Manually write new data to file
        payload = {
            "mappings": {"acmecorp": "AcmeCorp Updated"},
            "timestamps": {},
        }
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        # Reload and check
        c1.reload()
        assert c1.get("acmecorp") == "AcmeCorp Updated"

    def test_missing_file_starts_fresh(self, tmp_path):
        cache_file = tmp_path / "nonexistent.json"
        cache = CompanyNameCache(cache_file=cache_file)
        assert cache.size() == 0

    def test_corrupt_file_starts_fresh(self, tmp_path):
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("not valid json {{", encoding="utf-8")
        cache = CompanyNameCache(cache_file=cache_file)
        assert cache.size() == 0

    def test_file_created_on_first_set(self, tmp_path):
        cache_file = tmp_path / "newcache.json"
        assert not cache_file.exists()
        cache = CompanyNameCache(cache_file=cache_file)
        cache.set("acmecorp", "AcmeCorp")
        assert cache_file.exists()


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_sets_do_not_corrupt(self, tmp_path):
        cache_file = tmp_path / "concurrent.json"
        cache = CompanyNameCache(cache_file=cache_file)
        errors = []

        def _worker(key: str, value: str):
            try:
                cache.set(key, value)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(f"company{i}", f"Company {i}"))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert cache.size() == 20


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_singleton_returns_same_instance(self, tmp_path):
        # Reset the singleton for this test
        import src.gmail.company_name_cache as mod
        original = mod._singleton
        mod._singleton = None
        try:
            c1 = get_company_name_cache(cache_file=tmp_path / "s.json")
            c2 = get_company_name_cache(cache_file=tmp_path / "s.json")
            assert c1 is c2
        finally:
            mod._singleton = original
