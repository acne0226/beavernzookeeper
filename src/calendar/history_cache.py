"""
CalendarHistoryCache — local in-memory (+ optional JSON) cache of historical
Google Calendar events.

Purpose
-------
The constraints require a 1-year calendar history lookback to classify
meetings as "external" and to build attendee relationship context.  Fetching
365 days of raw events on every briefing run would be slow and wasteful.
This module:

1.  Provides ``CalendarHistoryCache``, which stores every event from the past
    year indexed by **attendee email** and **attendee domain**.
2.  Persists the index to ``CALENDAR_CACHE_FILE`` (JSON) so the daemon can
    warm-start quickly on restart.
3.  Exposes query helpers used by the briefing pipeline:
    - ``get_meetings_by_email(email)``  → meetings that included this person
    - ``get_meetings_by_domain(domain)`` → meetings from a company domain
    - ``is_known_external_domain(domain)`` → True if we've ever met this domain

Typical usage::

    from src.calendar.history_cache import CalendarHistoryCache
    from src.calendar.google_calendar import GoogleCalendarClient

    client = GoogleCalendarClient()
    client.connect()

    cache = CalendarHistoryCache.build(client)          # warm from API
    # or fast-start:
    cache = CalendarHistoryCache.load_or_build(client)  # load JSON first

    meetings = cache.get_meetings_by_domain("acme.com")
    is_new   = not cache.is_known_external_domain("newco.io")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.config import (
    CALENDAR_CACHE_FILE,
    CALENDAR_HISTORY_LOOKBACK_DAYS,
    INTERNAL_DOMAIN,
)

if TYPE_CHECKING:
    from src.calendar.google_calendar import Meeting, GoogleCalendarClient

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CachedEvent:
    """
    Minimal representation of a calendar event stored in the history cache.

    Only the fields needed for external-meeting classification and attendee
    relationship context are kept — raw description / HTML links are omitted
    to keep the cache compact.
    """
    event_id: str
    title: str
    start_iso: str            # ISO-8601 UTC
    end_iso: str              # ISO-8601 UTC
    all_day: bool
    organizer_email: str
    is_external: bool         # True if ≥1 non-internal attendee
    attendee_emails: list[str] = field(default_factory=list)
    attendee_domains: list[str] = field(default_factory=list)

    # ── derived (not persisted — recomputed on load) ────────────────────────

    @property
    def start(self) -> datetime:
        return datetime.fromisoformat(self.start_iso)

    @property
    def external_domains(self) -> list[str]:
        """Unique non-internal domains present in this event."""
        return [d for d in dict.fromkeys(self.attendee_domains) if d != INTERNAL_DOMAIN]

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CachedEvent":
        return cls(
            event_id=data["event_id"],
            title=data["title"],
            start_iso=data["start_iso"],
            end_iso=data["end_iso"],
            all_day=data["all_day"],
            organizer_email=data["organizer_email"],
            is_external=data["is_external"],
            attendee_emails=data.get("attendee_emails", []),
            attendee_domains=data.get("attendee_domains", []),
        )

    @classmethod
    def from_meeting(cls, meeting: "Meeting") -> "CachedEvent":
        """Convert a live ``Meeting`` object to a ``CachedEvent``."""
        emails = [a.email.lower() for a in meeting.attendees if a.email]
        domains = [_email_domain(e) for e in emails]
        return cls(
            event_id=meeting.event_id,
            title=meeting.summary,
            start_iso=meeting.start.isoformat(),
            end_iso=meeting.end.isoformat(),
            all_day=meeting.all_day,
            organizer_email=meeting.organizer_email or "",
            is_external=meeting.is_external,
            attendee_emails=emails,
            attendee_domains=domains,
        )


# ── Cache ─────────────────────────────────────────────────────────────────────

class CalendarHistoryCache:
    """
    In-memory index of historical calendar events (up to 1 year back).

    Attributes
    ----------
    events : list[CachedEvent]
        All cached events, sorted ascending by start time.
    built_at : datetime
        UTC timestamp when this cache instance was built / loaded.
    lookback_days : int
        Number of days of history represented in the cache.

    Indexes (built lazily or on construction):
    - ``_by_email``  : dict[email_lower, list[CachedEvent]]
    - ``_by_domain`` : dict[domain_lower, list[CachedEvent]]
    """

    def __init__(
        self,
        events: list[CachedEvent],
        built_at: Optional[datetime] = None,
        lookback_days: int = CALENDAR_HISTORY_LOOKBACK_DAYS,
    ) -> None:
        self.events: list[CachedEvent] = events
        self.built_at: datetime = built_at or datetime.now(timezone.utc)
        self.lookback_days: int = lookback_days

        # Lazy indexes
        self._by_email: dict[str, list[CachedEvent]] = {}
        self._by_domain: dict[str, list[CachedEvent]] = {}
        self._indexes_built: bool = False

    # ── Index construction ────────────────────────────────────────────────────

    def _build_indexes(self) -> None:
        """Build email and domain lookup indexes from ``self.events``."""
        by_email: dict[str, list[CachedEvent]] = {}
        by_domain: dict[str, list[CachedEvent]] = {}

        for ev in self.events:
            for email in ev.attendee_emails:
                by_email.setdefault(email, []).append(ev)
            for domain in ev.attendee_domains:
                by_domain.setdefault(domain, []).append(ev)

        self._by_email = by_email
        self._by_domain = by_domain
        self._indexes_built = True
        logger.debug(
            "CalendarHistoryCache: indexed %d events, %d unique emails, %d unique domains",
            len(self.events),
            len(self._by_email),
            len(self._by_domain),
        )

    def _ensure_indexes(self) -> None:
        if not self._indexes_built:
            self._build_indexes()

    # ── Public query API ──────────────────────────────────────────────────────

    def get_meetings_by_email(self, email: str) -> list[CachedEvent]:
        """
        Return all cached events that include *email* as an attendee.

        Results are sorted descending (most recent first).
        """
        self._ensure_indexes()
        events = self._by_email.get(email.lower(), [])
        return sorted(events, key=lambda e: e.start_iso, reverse=True)

    def get_meetings_by_domain(self, domain: str) -> list[CachedEvent]:
        """
        Return all cached events that include at least one attendee from
        *domain* (e.g. ``"acme.com"``).

        Results are sorted descending (most recent first).
        """
        self._ensure_indexes()
        events = self._by_domain.get(domain.lower(), [])
        return sorted(events, key=lambda e: e.start_iso, reverse=True)

    def is_known_external_domain(self, domain: str) -> bool:
        """
        Return True if *domain* has appeared as an external attendee domain
        in any meeting during the lookback period.
        """
        d = domain.lower()
        if d == INTERNAL_DOMAIN.lower():
            return False
        self._ensure_indexes()
        return d in self._by_domain and len(self._by_domain[d]) > 0

    def past_meeting_count_for_email(self, email: str) -> int:
        """Number of historical meetings with this email address."""
        return len(self.get_meetings_by_email(email))

    def past_meeting_count_for_domain(self, domain: str) -> int:
        """Number of historical meetings with any attendee from this domain."""
        return len(self.get_meetings_by_domain(domain))

    def last_meeting_with_email(self, email: str) -> Optional[CachedEvent]:
        """Most recent cached event involving *email*, or None."""
        meetings = self.get_meetings_by_email(email)
        return meetings[0] if meetings else None

    def last_meeting_with_domain(self, domain: str) -> Optional[CachedEvent]:
        """Most recent cached event involving *domain*, or None."""
        meetings = self.get_meetings_by_domain(domain)
        return meetings[0] if meetings else None

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def external_events(self) -> list[CachedEvent]:
        return [e for e in self.events if e.is_external]

    @property
    def known_external_domains(self) -> set[str]:
        """Set of all non-internal attendee domains seen in history."""
        self._ensure_indexes()
        return {d for d in self._by_domain if d != INTERNAL_DOMAIN.lower()}

    def summary(self) -> dict:
        """Return a human-readable summary dict (useful for logging)."""
        self._ensure_indexes()
        return {
            "total_events": self.total_events,
            "external_events": len(self.external_events),
            "unique_attendee_emails": len(self._by_email),
            "unique_attendee_domains": len(self._by_domain),
            "known_external_domains": len(self.known_external_domains),
            "lookback_days": self.lookback_days,
            "built_at": self.built_at.isoformat(),
        }

    # ── Build from live API ───────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        calendar_client: "GoogleCalendarClient",
        lookback_days: int = CALENDAR_HISTORY_LOOKBACK_DAYS,
    ) -> "CalendarHistoryCache":
        """
        Populate the cache by fetching *all* calendar events from the past
        *lookback_days* days via the Google Calendar API.

        Raises ``RuntimeError`` (propagated from the client) if the API call
        fails after all retries.
        """
        logger.info(
            "CalendarHistoryCache.build: fetching %d days of history …",
            lookback_days,
        )
        meetings = calendar_client.list_all_historical_events(
            lookback_days=lookback_days,
        )
        cached = [CachedEvent.from_meeting(m) for m in meetings]
        instance = cls(events=cached, lookback_days=lookback_days)
        logger.info(
            "CalendarHistoryCache.build: cached %d events", len(cached)
        )
        return instance

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """
        Persist the cache to a JSON file.

        The file is written atomically (write to ``<path>.tmp``, then rename)
        to avoid corruption on crash.
        """
        target = path or CALENDAR_CACHE_FILE
        payload = {
            "version": 1,
            "built_at": self.built_at.isoformat(),
            "lookback_days": self.lookback_days,
            "events": [e.to_dict() for e in self.events],
        }
        tmp = Path(str(target) + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=None))
            tmp.replace(target)
            logger.info("CalendarHistoryCache saved: %d events → %s", len(self.events), target)
        except Exception as exc:
            logger.warning("CalendarHistoryCache.save failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CalendarHistoryCache":
        """
        Load a previously persisted cache from JSON.

        Raises ``FileNotFoundError`` if the file does not exist.
        Raises ``ValueError`` if the file is malformed.
        """
        source = path or CALENDAR_CACHE_FILE
        raw = source.read_text(encoding="utf-8")
        data = json.loads(raw)

        if data.get("version") != 1:
            raise ValueError(f"Unsupported cache version: {data.get('version')}")

        built_at = datetime.fromisoformat(data["built_at"])
        lookback_days = int(data.get("lookback_days", CALENDAR_HISTORY_LOOKBACK_DAYS))
        events = [CachedEvent.from_dict(e) for e in data.get("events", [])]

        instance = cls(events=events, built_at=built_at, lookback_days=lookback_days)
        logger.info(
            "CalendarHistoryCache loaded: %d events from %s (built %s)",
            len(events),
            source,
            built_at.date(),
        )
        return instance

    @classmethod
    def load_or_build(
        cls,
        calendar_client: "GoogleCalendarClient",
        lookback_days: int = CALENDAR_HISTORY_LOOKBACK_DAYS,
        max_cache_age_hours: float = 12.0,
        path: Optional[Path] = None,
    ) -> "CalendarHistoryCache":
        """
        Return a ``CalendarHistoryCache`` — loading from disk when a fresh
        enough cache exists, otherwise building from the live API and saving.

        Args:
            calendar_client:     Authenticated ``GoogleCalendarClient``.
            lookback_days:       History window in days (default 365).
            max_cache_age_hours: How old (in hours) a cached file is still
                                 considered fresh. Default 12 hours.
            path:                Override the default ``CALENDAR_CACHE_FILE``.

        Returns:
            A populated ``CalendarHistoryCache`` instance.
        """
        cache_path = path or CALENDAR_CACHE_FILE

        # Try loading an existing, fresh-enough cache
        if cache_path.exists():
            try:
                cache = cls.load(cache_path)
                age = datetime.now(timezone.utc) - cache.built_at.replace(
                    tzinfo=timezone.utc if cache.built_at.tzinfo is None else cache.built_at.tzinfo
                )
                if age <= timedelta(hours=max_cache_age_hours):
                    logger.info(
                        "Using cached calendar history (age %.1f h, max %.1f h)",
                        age.total_seconds() / 3600,
                        max_cache_age_hours,
                    )
                    return cache
                else:
                    logger.info(
                        "Cache is stale (age %.1f h > %.1f h); rebuilding …",
                        age.total_seconds() / 3600,
                        max_cache_age_hours,
                    )
            except Exception as exc:
                logger.warning("Could not load cache file (%s); rebuilding …", exc)

        # Build from API and persist
        cache = cls.build(calendar_client, lookback_days=lookback_days)
        cache.save(cache_path)
        return cache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _email_domain(email: str) -> str:
    """Return the lower-cased domain part of an email address."""
    try:
        return email.split("@")[1].lower()
    except (IndexError, AttributeError):
        return ""
