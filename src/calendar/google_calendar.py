"""
Google Calendar client with OAuth2 authentication.

Provides:
- GoogleCalendarClient: authenticated wrapper around the Google Calendar API
- Meeting dataclass with parsed attendee info
- External meeting detection logic based on attendee domains
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN,
    GOOGLE_TOKEN_FILE,
    INVESTMENT_TEAM_EMAILS,
    INTERNAL_DOMAIN,
    PRIMARY_CALENDAR_ID,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Attendee:
    email: str
    display_name: str = ""
    response_status: str = "needsAction"

    @property
    def is_internal(self) -> bool:
        return self.email.lower().endswith(f"@{INTERNAL_DOMAIN}")

    @property
    def is_investment_team(self) -> bool:
        return self.email.lower() in [e.lower() for e in INVESTMENT_TEAM_EMAILS]


@dataclass
class Meeting:
    event_id: str
    summary: str
    start: datetime
    end: datetime
    attendees: list[Attendee] = field(default_factory=list)
    description: str = ""
    location: str = ""
    html_link: str = ""
    organizer_email: str = ""
    calendar_id: str = PRIMARY_CALENDAR_ID
    all_day: bool = False

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def external_attendees(self) -> list[Attendee]:
        return [a for a in self.attendees if not a.is_internal]

    @property
    def is_external(self) -> bool:
        """True when at least one non-internal attendee is present."""
        return len(self.external_attendees) > 0

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    @property
    def starts_in_minutes(self) -> float:
        now = datetime.now(timezone.utc)
        return (self.start - now).total_seconds() / 60

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "summary": self.summary,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "all_day": self.all_day,
            "attendees": [
                {
                    "email": a.email,
                    "display_name": a.display_name,
                    "is_internal": a.is_internal,
                    "response_status": a.response_status,
                }
                for a in self.attendees
            ],
            "external_attendees": [a.email for a in self.external_attendees],
            "is_external": self.is_external,
            "description": self.description,
            "location": self.location,
            "html_link": self.html_link,
            "organizer_email": self.organizer_email,
            "duration_minutes": self.duration_minutes,
        }


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _refresh_token_manually(refresh_token: str, client_id: str, client_secret: str) -> dict:
    """
    Directly call Google's token endpoint to exchange a refresh token for a new
    access token WITHOUT sending any ``scope`` parameter.

    Sending scopes during a refresh request is optional (RFC 6749 §6) and can
    cause ``invalid_scope`` errors if the stored scope list doesn't exactly
    match what the original OAuth consent granted.  Omitting the scope parameter
    tells the server to use the full set of scopes from the original grant.
    """
    import urllib.request
    import urllib.parse

    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json as _json
        return _json.loads(resp.read().decode())


def _build_credentials() -> Credentials:
    """
    Build Google OAuth2 credentials from env vars, refreshing the access token
    as needed. Persists the refreshed token to GOOGLE_TOKEN_FILE so subsequent
    runs start without an extra round-trip.

    NOTE: We do NOT pass scopes in the token refresh request because the scopes
    are already embedded in the original OAuth grant.  Sending a stale or mismatched
    scope list causes ``invalid_scope: Bad Request`` errors from Google.  We use a
    direct HTTP call instead of the google-auth library's refresh() method so we can
    omit the scope parameter entirely.
    """
    from datetime import timedelta
    import json as _json

    # ── Helper: build a valid Credentials object from raw token fields ────────
    def _make_creds(token: str, expires_in: int, scopes_list: list) -> Credentials:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=max(0, expires_in - 30))
        creds = Credentials(
            token=token,
            refresh_token=GOOGLE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
        )
        creds.expiry = expiry
        return creds

    # ── Try loading a still-valid persisted token ─────────────────────────────
    if GOOGLE_TOKEN_FILE.exists():
        try:
            raw = _json.loads(GOOGLE_TOKEN_FILE.read_text())
            creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_FILE))
            if creds.valid:
                logger.debug("Loaded valid credentials from %s", GOOGLE_TOKEN_FILE)
                return creds
            logger.info("Persisted token is expired — refreshing without scopes …")
        except Exception as exc:
            logger.warning("Could not load persisted token: %s", exc)

    # ── Refresh without sending scopes ───────────────────────────────────────
    logger.info("Refreshing Google OAuth2 token via direct HTTP (no scope param) …")
    try:
        data = _refresh_token_manually(
            GOOGLE_REFRESH_TOKEN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
        )
        creds = _make_creds(
            token=data["access_token"],
            expires_in=data.get("expires_in", 3600),
            scopes_list=data.get("scope", "").split(),
        )
        # Persist updated token (with correct scopes from the response)
        token_payload = {
            "token": data["access_token"],
            "refresh_token": GOOGLE_REFRESH_TOKEN,
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "scopes": data.get("scope", "").split(),
            "universe_domain": "googleapis.com",
            "account": "",
            "expiry": creds.expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        GOOGLE_TOKEN_FILE.write_text(_json.dumps(token_payload))
        logger.info("Google OAuth2 token refreshed and persisted to %s", GOOGLE_TOKEN_FILE)
        return creds
    except Exception as exc:
        raise RuntimeError(f"Failed to refresh Google OAuth2 token: {exc}") from exc


# ── Client ─────────────────────────────────────────────────────────────────────

class GoogleCalendarClient:
    """Thin wrapper around the Google Calendar v3 REST API."""

    def __init__(self) -> None:
        self._creds: Optional[Credentials] = None
        self._service: Any = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate and build the API service object."""
        self._creds = _build_credentials()
        self._service = build("calendar", "v3", credentials=self._creds, cache_discovery=False)
        logger.info("GoogleCalendarClient connected.")

    def _ensure_connected(self) -> None:
        if self._service is None:
            self.connect()
        # Refresh token if expired
        if self._creds and self._creds.expired:
            self._creds.refresh(Request())
            self._service = build("calendar", "v3", credentials=self._creds, cache_discovery=False)

    # ── retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute *fn* with up to API_RETRY_ATTEMPTS retries on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_connected()
                return fn(*args, **kwargs)
            except HttpError as exc:
                logger.warning("Google API HttpError (attempt %d/%d): %s", attempt, API_RETRY_ATTEMPTS, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("Google API error (attempt %d/%d): %s", attempt, API_RETRY_ATTEMPTS, exc)
                last_exc = exc
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_DELAY_SECONDS)
        raise RuntimeError(f"Google Calendar API failed after {API_RETRY_ATTEMPTS} attempts") from last_exc

    # ── parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_dt(dt_obj: dict) -> datetime:
        """Parse a Google Calendar dateTime or date field into a UTC datetime."""
        if "dateTime" in dt_obj:
            dt = datetime.fromisoformat(dt_obj["dateTime"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        # All-day event: treat as midnight UTC
        date_str = dt_obj["date"]
        return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)

    @staticmethod
    def _parse_event(event: dict, calendar_id: str = PRIMARY_CALENDAR_ID) -> Meeting:
        start_raw = event.get("start", {})
        end_raw = event.get("end", {})
        # All-day events use "date" key; timed events use "dateTime"
        all_day = "date" in start_raw and "dateTime" not in start_raw
        start = GoogleCalendarClient._parse_dt(start_raw)
        end = GoogleCalendarClient._parse_dt(end_raw)
        attendees = [
            Attendee(
                email=a.get("email", ""),
                display_name=a.get("displayName", ""),
                response_status=a.get("responseStatus", "needsAction"),
            )
            for a in event.get("attendees", [])
            if a.get("email")
        ]
        organizer = event.get("organizer", {})
        return Meeting(
            event_id=event.get("id", ""),
            summary=event.get("summary", "(제목 없음)"),
            start=start,
            end=end,
            attendees=attendees,
            description=event.get("description", ""),
            location=event.get("location", ""),
            html_link=event.get("htmlLink", ""),
            organizer_email=organizer.get("email", ""),
            calendar_id=calendar_id,
            all_day=all_day,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def list_upcoming_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        calendar_id: str = PRIMARY_CALENDAR_ID,
        max_results: int = 50,
    ) -> list[Meeting]:
        """Return events in [time_min, time_max) sorted by start time."""
        if time_min is None:
            time_min = datetime.now(timezone.utc)
        if time_max is None:
            time_max = time_min + timedelta(hours=24)

        def _fetch():
            return (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=max_results,
                )
                .execute()
            )

        result = self._call_with_retry(_fetch)
        events = result.get("items", [])
        return [self._parse_event(e, calendar_id) for e in events]

    def list_todays_events(
        self,
        tz_name: str = "Asia/Seoul",
        calendar_id: str = PRIMARY_CALENDAR_ID,
        include_cancelled: bool = False,
    ) -> list[Meeting]:
        """
        Fetch all calendar events for today (in the given timezone), sorted
        chronologically by start time.

        This is the canonical Sub-AC 1.1 method: it returns structured Meeting
        objects with title, start/end time, attendees, location, and any
        conference links, covering the full 24-hour window from midnight to
        midnight in *tz_name*.

        Cancelled events are excluded by default (set include_cancelled=True
        to keep them).  All-day events appear first (sorted to midnight).

        Args:
            tz_name:           IANA timezone for determining "today"
                               (default: "Asia/Seoul").
            calendar_id:       Google Calendar to query (default: "primary").
            include_cancelled: When True, cancelled events are also returned.

        Returns:
            List of ``Meeting`` objects sorted ascending by start time.
        """
        from zoneinfo import ZoneInfo
        from datetime import date as _date

        tz = ZoneInfo(tz_name)
        today: _date = datetime.now(tz).date()

        # Build the full-day time window in the given timezone
        day_start = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=tz)
        day_end = day_start + timedelta(days=1)

        # Convert to UTC for the API call
        time_min = day_start.astimezone(timezone.utc)
        time_max = day_end.astimezone(timezone.utc)

        def _fetch():
            return (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=250,
                )
                .execute()
            )

        result = self._call_with_retry(_fetch)
        raw_events = result.get("items", [])

        meetings: list[Meeting] = []
        for event in raw_events:
            if not include_cancelled and event.get("status") == "cancelled":
                continue
            try:
                meetings.append(self._parse_event(event, calendar_id))
            except Exception as exc:
                logger.warning(
                    "Skipping unparseable event '%s': %s",
                    event.get("summary", "?"),
                    exc,
                )

        # Sort: all-day events (stored as midnight UTC) come first; then by time
        meetings.sort(key=lambda m: m.start)
        logger.info(
            "list_todays_events: %d events for %s (%s)", len(meetings), today, timezone
        )
        return meetings

    def get_external_meetings_starting_soon(
        self,
        lookahead_minutes: int = 15,
    ) -> list[Meeting]:
        """
        Return external meetings (at least one non-kakaoventures attendee)
        whose start time is within the next *lookahead_minutes* minutes.

        A meeting is considered 'starting soon' if:
            0 < starts_in_minutes <= lookahead_minutes
        (already-started meetings are excluded)
        """
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(minutes=lookahead_minutes)

        meetings = self.list_upcoming_events(time_min=now, time_max=window_end)
        external = [m for m in meetings if m.is_external]

        logger.info(
            "Found %d meetings in next %d min, %d are external.",
            len(meetings),
            lookahead_minutes,
            len(external),
        )
        return external

    def list_historical_external_meetings(
        self,
        lookback_days: int = 365,
        calendar_id: str = PRIMARY_CALENDAR_ID,
        max_results: int = 2500,
    ) -> list[Meeting]:
        """
        Fetch past external meetings for context (used by the briefing pipeline
        to enrich attendee relationship history).
        """
        now = datetime.now(timezone.utc)
        time_min = now - timedelta(days=lookback_days)

        def _fetch():
            return (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=now.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=max_results,
                )
                .execute()
            )

        result = self._call_with_retry(_fetch)
        events = result.get("items", [])
        all_meetings = [self._parse_event(e, calendar_id) for e in events]
        return [m for m in all_meetings if m.is_external]

    def list_all_historical_events(
        self,
        lookback_days: int = 365,
        calendar_id: str = PRIMARY_CALENDAR_ID,
        page_size: int = 500,
    ) -> list[Meeting]:
        """
        Fetch ALL calendar events (internal + external) from the past
        *lookback_days* days, paging through the API until exhausted.

        Unlike ``list_historical_external_meetings``, this method returns
        every event so the history cache can index all attendee domains
        regardless of whether they are external.

        Args:
            lookback_days: How far back to look (default 365 = 1 year).
            calendar_id:   Calendar to query (default "primary").
            page_size:     Number of events per API page (max 2500).

        Returns:
            Flat list of Meeting objects sorted ascending by start time.
        """
        now = datetime.now(timezone.utc)
        time_min = now - timedelta(days=lookback_days)

        meetings: list[Meeting] = []
        page_token: Optional[str] = None

        while True:
            token = page_token  # capture for closure

            def _fetch(pt=token):
                params = dict(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat(),
                    timeMax=now.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=page_size,
                )
                if pt:
                    params["pageToken"] = pt
                return self._service.events().list(**params).execute()

            result = self._call_with_retry(_fetch)
            items = result.get("items", [])
            for event in items:
                try:
                    meetings.append(self._parse_event(event, calendar_id))
                except Exception as exc:
                    logger.warning(
                        "Skipping unparseable historical event '%s': %s",
                        event.get("summary", "?"),
                        exc,
                    )

            page_token = result.get("nextPageToken")
            if not page_token:
                break

        logger.info(
            "list_all_historical_events: fetched %d events over last %d days",
            len(meetings),
            lookback_days,
        )
        return meetings

    def get_event_by_id(
        self,
        event_id: str,
        calendar_id: str = PRIMARY_CALENDAR_ID,
    ) -> Optional[Meeting]:
        """Fetch a single event by ID."""
        def _fetch():
            return self._service.events().get(calendarId=calendar_id, eventId=event_id).execute()

        try:
            event = self._call_with_retry(_fetch)
            return self._parse_event(event, calendar_id)
        except Exception:
            return None


# ── Module-level filter utilities ──────────────────────────────────────────────

def filter_external_meetings(
    meetings: list[Meeting],
    internal_domain: str = INTERNAL_DOMAIN,
) -> list[Meeting]:
    """
    Filter a list of ``Meeting`` objects to those with at least one external
    (non-company) attendee.

    A meeting is considered *external* when at least one attendee's email
    does **not** end with ``@<internal_domain>``.  The ``internal_domain``
    parameter defaults to ``INTERNAL_DOMAIN`` (``"kakaoventures.co.kr"``) but
    can be overridden for testing or multi-tenant scenarios.

    Args:
        meetings:        Iterable of ``Meeting`` objects to filter.
        internal_domain: The company's internal email domain (without the
                         leading ``@``).  Case-insensitive comparison is used.

    Returns:
        A new list containing only ``Meeting`` objects where
        ``Meeting.is_external`` is ``True`` (or, when *internal_domain* differs
        from the compiled-in constant, where at least one attendee email does
        not end with ``@<internal_domain>``).

    Notes:
        - Meetings with **no attendees** are treated as internal (not external).
        - Attendees with empty or malformed email addresses are skipped.
        - The original list is **not** mutated; a new list is returned.

    Examples::

        from src.calendar.google_calendar import filter_external_meetings

        today_meetings = calendar_client.list_upcoming_events()
        external = filter_external_meetings(today_meetings)
    """
    domain_lower = internal_domain.lower()

    def _is_external_for_domain(meeting: Meeting) -> bool:
        """Return True if *meeting* has at least one non-internal attendee."""
        for attendee in meeting.attendees:
            email = (attendee.email or "").strip().lower()
            if not email or "@" not in email:
                continue  # skip malformed / empty emails
            if not email.endswith(f"@{domain_lower}"):
                return True
        return False

    # When using the default domain, delegate to the pre-computed Meeting.is_external
    # property to avoid duplicating domain-check logic.
    if domain_lower == INTERNAL_DOMAIN.lower():
        return [m for m in meetings if m.is_external]

    # Custom domain override path (used in tests or multi-tenant deployments)
    return [m for m in meetings if _is_external_for_domain(m)]


def get_external_attendee_domains(meeting: Meeting) -> set[str]:
    """
    Return the set of unique external (non-internal) attendee domains for a
    ``Meeting``.

    Only email addresses containing ``@`` are considered.  The internal domain
    (``INTERNAL_DOMAIN``) is excluded from the result.

    Args:
        meeting: A ``Meeting`` object.

    Returns:
        A ``set[str]`` of lower-cased domain strings, e.g. ``{"acme.com"}``.
        Returns an empty set if all attendees are internal or if no valid
        external email addresses are present.

    Examples::

        domains = get_external_attendee_domains(meeting)
        # → {"startup.io", "partner.com"}
    """
    domains: set[str] = set()
    for attendee in meeting.external_attendees:
        email = (attendee.email or "").strip().lower()
        if "@" not in email:
            continue
        domain = email.split("@")[1]
        if domain:
            domains.add(domain)
    return domains
