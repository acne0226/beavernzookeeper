"""
Persistent company name mapping cache (AC 10).

Maps email sender addresses (or domains) to portfolio company names so that
the same sender is never looked up more than once via web search.

Storage: JSON file at COMPANY_NAME_CACHE_FILE (project root).
Thread-safe: uses a threading.Lock for all reads and writes.

Public API
----------
* ``CompanyNameCache``   – class with load/save/get/set operations
* ``get_company_name_cache()`` – module-level singleton
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

# Path to the persistent JSON cache file
COMPANY_NAME_CACHE_FILE: Path = ROOT_DIR / "company_name_cache.json"


class CompanyNameCache:
    """
    Persistent mapping: sender_key → company_name.

    The ``sender_key`` is a normalised string uniquely identifying a sender:
    - For domain-based matches: the email domain root (e.g. ``"acmecorp"``)
    - For name-based matches: the normalised display name

    The cache is loaded from disk at first access and persisted after every
    new entry is added, ensuring zero re-lookups across daemon restarts.
    """

    def __init__(self, cache_file: Path = COMPANY_NAME_CACHE_FILE) -> None:
        self._cache_file = cache_file
        self._data: dict[str, str] = {}  # sender_key → company_name
        self._metadata: dict[str, str] = {}  # sender_key → ISO timestamp
        self._lock = threading.Lock()
        self._loaded = False

    # ── Load / Save ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load cache from JSON file. No-op if file does not exist."""
        if not self._cache_file.exists():
            logger.debug("CompanyNameCache: no existing cache file at %s", self._cache_file)
            self._loaded = True
            return

        try:
            raw = json.loads(self._cache_file.read_text(encoding="utf-8"))
            self._data = raw.get("mappings", {})
            self._metadata = raw.get("timestamps", {})
            logger.info(
                "CompanyNameCache: loaded %d mappings from %s",
                len(self._data),
                self._cache_file,
            )
        except Exception as exc:
            logger.warning(
                "CompanyNameCache: failed to load %s — starting fresh: %s",
                self._cache_file, exc,
            )
            self._data = {}
            self._metadata = {}
        self._loaded = True

    def _save(self) -> None:
        """Persist current cache to JSON file (called under self._lock)."""
        try:
            payload = {
                "mappings": self._data,
                "timestamps": self._metadata,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "total_entries": len(self._data),
            }
            self._cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("CompanyNameCache: failed to save cache: %s", exc)

    def _ensure_loaded(self) -> None:
        """Load from disk if not yet done (lazy init, thread-safe)."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, sender_key: str) -> Optional[str]:
        """
        Look up a sender key and return the cached company name, or None.

        Parameters
        ----------
        sender_key : Normalised sender identifier (domain root or display name).
        """
        self._ensure_loaded()
        with self._lock:
            return self._data.get(sender_key.lower())

    def set(self, sender_key: str, company_name: str) -> None:
        """
        Store a new mapping and immediately persist to disk.

        Parameters
        ----------
        sender_key   : Normalised sender identifier.
        company_name : Human-readable company name (brand/canonical name).
        """
        key = sender_key.lower()
        self._ensure_loaded()
        with self._lock:
            if self._data.get(key) == company_name:
                return  # already stored — skip write
            self._data[key] = company_name
            self._metadata[key] = datetime.now(timezone.utc).isoformat()
            self._save()
            logger.debug(
                "CompanyNameCache: cached %r → %r", sender_key, company_name
            )

    def get_or_set(self, sender_key: str, company_name: str) -> str:
        """
        Return cached name for *sender_key*, or store *company_name* and
        return it.  Convenient one-call helper for the matching pipeline.
        """
        cached = self.get(sender_key)
        if cached is not None:
            return cached
        self.set(sender_key, company_name)
        return company_name

    def contains(self, sender_key: str) -> bool:
        """Return True if *sender_key* has a cached mapping."""
        return self.get(sender_key) is not None

    def all_mappings(self) -> dict[str, str]:
        """Return a snapshot copy of all mappings."""
        self._ensure_loaded()
        with self._lock:
            return dict(self._data)

    def size(self) -> int:
        """Return the number of cached mappings."""
        self._ensure_loaded()
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        """Clear all cached mappings (used in tests)."""
        with self._lock:
            self._data = {}
            self._metadata = {}
            self._loaded = True

    def reload(self) -> None:
        """Force reload from disk (used in tests)."""
        with self._lock:
            self._loaded = False
        self._ensure_loaded()


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: Optional[CompanyNameCache] = None
_singleton_lock = threading.Lock()


def get_company_name_cache(
    cache_file: Path = COMPANY_NAME_CACHE_FILE,
) -> CompanyNameCache:
    """
    Return the module-level CompanyNameCache singleton.

    The singleton is created on first call and reused for the lifetime of
    the process.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CompanyNameCache(cache_file=cache_file)
    return _singleton
