"""
Google Calendar Fetching Module

Retrieves all calendar events for the current day and returns structured event data
including title, time, attendees, location/link.

Google OAuth2 credentials are loaded from .env (CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
and persisted to google_token.json after first successful use.
"""

import os
import json
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# NOTE: Scopes are embedded in the refresh token from the original OAuth consent.
# We do NOT pass a scope list when constructing Credentials from a refresh_token,
# as that causes "invalid_scope" errors if the lists don't match exactly.
# This constant is kept for documentation purposes only.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
]
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", "google_token.json")
RETRY_COUNT = 3
RETRY_DELAY = 10  # seconds
DEFAULT_TIMEZONE = "Asia/Seoul"


# ── Credential helpers ───────────────────────────────────────────────────────

def _load_credentials() -> Credentials:
    """
    Load Google OAuth2 credentials from token file or .env refresh token.

    Priority:
    1. Existing google_token.json (persisted OAuth2 token)
    2. GOOGLE_REFRESH_TOKEN from .env  (first-time bootstrap)

    Returns refreshed Credentials ready for API calls.
    Raises RuntimeError if credentials cannot be obtained.
    """
    creds: Optional[Credentials] = None

    # 1. Try loading from persisted token file
    token_path = os.path.abspath(TOKEN_FILE)
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            logger.debug("Loaded credentials from %s", token_path)
        except Exception as exc:
            logger.warning("Failed to load token file: %s", exc)
            creds = None

    # 2. Bootstrap from .env refresh token
    if creds is None:
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN", "").strip()

        if not all([client_id, client_secret, refresh_token]):
            raise RuntimeError(
                "Google credentials missing. "
                "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN in .env"
            )

        # Do NOT pass scopes when bootstrapping from a refresh token —
        # scopes are already embedded in the token from the original OAuth consent.
        # Passing a different scope list causes "invalid_scope" on refresh.
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
        )
        logger.info("Bootstrapped credentials from .env refresh token")

    # 3. Refresh the access token if expired or missing
    if not creds.valid:
        if creds.refresh_token:
            logger.info("Refreshing Google OAuth2 access token …")
            creds.refresh(Request())
        else:
            raise RuntimeError("Google credentials are invalid and cannot be refreshed.")

    # 4. Persist to token file for next run
    try:
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())
        logger.debug("Saved refreshed credentials to %s", token_path)
    except Exception as exc:
        logger.warning("Could not persist token file: %s", exc)

    return creds


def _build_service():
    """Return an authenticated Google Calendar service object."""
    creds = _load_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_event_time(event: dict, timezone: str) -> dict:
    """
    Extract start/end times from an event dict.

    Returns:
        {
            "start": datetime | date,
            "end":   datetime | date,
            "all_day": bool,
            "start_iso": str,
            "end_iso":   str,
        }
    """
    tz = ZoneInfo(timezone)
    start_raw = event.get("start", {})
    end_raw = event.get("end", {})

    if "dateTime" in start_raw:
        start_dt = datetime.fromisoformat(start_raw["dateTime"]).astimezone(tz)
        end_dt = datetime.fromisoformat(end_raw["dateTime"]).astimezone(tz)
        all_day = False
    else:
        # All-day event
        start_dt = date.fromisoformat(start_raw["date"])
        end_dt = date.fromisoformat(end_raw["date"])
        all_day = True

    return {
        "start": start_dt,
        "end": end_dt,
        "all_day": all_day,
        "start_iso": start_raw.get("dateTime") or start_raw.get("date", ""),
        "end_iso": end_raw.get("dateTime") or end_raw.get("date", ""),
    }


def _extract_attendees(event: dict) -> list[dict]:
    """
    Extract attendee list from event.

    Returns list of:
        {"email": str, "name": str, "response_status": str, "is_organizer": bool}
    """
    raw_attendees = event.get("attendees", [])
    organizer_email = event.get("organizer", {}).get("email", "")

    attendees = []
    for att in raw_attendees:
        attendees.append(
            {
                "email": att.get("email", ""),
                "name": att.get("displayName") or att.get("email", ""),
                "response_status": att.get("responseStatus", "needsAction"),
                "is_organizer": att.get("email", "") == organizer_email or att.get("organizer", False),
            }
        )
    return attendees


def _extract_location_or_link(event: dict) -> dict:
    """
    Extract location and/or video conference link from event.

    Returns:
        {
            "location": str | None,
            "video_link": str | None,
            "conference_type": str | None,
        }
    """
    location = event.get("location") or None

    video_link = None
    conference_type = None

    # Google Meet / Hangouts link embedded in conferenceData
    conf_data = event.get("conferenceData")
    if conf_data:
        for ep in conf_data.get("entryPoints", []):
            if ep.get("entryPointType") == "video":
                video_link = ep.get("uri")
                break
        solution = conf_data.get("conferenceSolution", {})
        conference_type = solution.get("name")  # e.g. "Google Meet"

    # Fallback: look for Zoom / Teams links in description or location
    if video_link is None:
        for field in [event.get("description", ""), location or ""]:
            if field and ("zoom.us/j" in field or "zoom.us/w" in field):
                # Extract the first zoom URL
                import re
                match = re.search(r'https?://[^\s<>"]+zoom\.us/[^\s<>"]+', field)
                if match:
                    video_link = match.group(0).rstrip(".,;)")
                    conference_type = "Zoom"
                    break
            if field and "teams.microsoft.com" in field:
                import re
                # Use * (zero or more) before teams.microsoft.com to match
                # both subdomain URLs (xxx.teams.microsoft.com) and the
                # direct root URL (teams.microsoft.com/...)
                match = re.search(r'https?://[^\s<>"]*teams\.microsoft\.com[^\s<>"]+', field)
                if match:
                    video_link = match.group(0).rstrip(".,;)")
                    conference_type = "Microsoft Teams"
                    break

    return {
        "location": location,
        "video_link": video_link,
        "conference_type": conference_type,
    }


def _parse_event(event: dict, timezone: str) -> dict:
    """
    Parse a raw Google Calendar event dict into a clean structured dict.

    Returns:
        {
            "id":               str,
            "title":            str,
            "status":           str,          # confirmed | tentative | cancelled
            "organizer_email":  str,
            "organizer_name":   str,
            "start":            datetime | date,
            "end":              datetime | date,
            "all_day":          bool,
            "start_iso":        str,
            "end_iso":          str,
            "attendees":        list[dict],
            "location":         str | None,
            "video_link":       str | None,
            "conference_type":  str | None,
            "description":      str | None,
            "html_link":        str,
            "recurring_event_id": str | None,
        }
    """
    time_info = _parse_event_time(event, timezone)
    attendees = _extract_attendees(event)
    location_info = _extract_location_or_link(event)

    organizer = event.get("organizer", {})

    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(No Title)"),
        "status": event.get("status", "confirmed"),
        "organizer_email": organizer.get("email", ""),
        "organizer_name": organizer.get("displayName") or organizer.get("email", ""),
        "start": time_info["start"],
        "end": time_info["end"],
        "all_day": time_info["all_day"],
        "start_iso": time_info["start_iso"],
        "end_iso": time_info["end_iso"],
        "attendees": attendees,
        "location": location_info["location"],
        "video_link": location_info["video_link"],
        "conference_type": location_info["conference_type"],
        "description": event.get("description") or None,
        "html_link": event.get("htmlLink", ""),
        "recurring_event_id": event.get("recurringEventId") or None,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_todays_events(
    target_date: Optional[date] = None,
    timezone: str = DEFAULT_TIMEZONE,
    calendar_id: str = "primary",
) -> list[dict]:
    """
    Fetch all calendar events for ``target_date`` (defaults to today).

    Retries up to RETRY_COUNT times on API errors, with RETRY_DELAY seconds
    between attempts. On persistent failure raises the last exception.

    Args:
        target_date: Date to fetch events for. Defaults to today in ``timezone``.
        timezone:    IANA timezone string (default: "Asia/Seoul").
        calendar_id: Google Calendar ID (default: "primary").

    Returns:
        Sorted list of parsed event dicts (see _parse_event for schema).
        Cancelled events are excluded.
    """
    tz = ZoneInfo(timezone)

    if target_date is None:
        target_date = datetime.now(tz).date()

    # Build RFC3339 time window for the full day
    day_start = datetime(
        target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz
    )
    day_end = day_start + timedelta(days=1)

    time_min = day_start.isoformat()
    time_max = day_end.isoformat()

    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            service = _build_service()
            events_result = (
                service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,       # Expand recurring events
                    orderBy="startTime",
                    maxResults=250,
                )
                .execute()
            )
            raw_events = events_result.get("items", [])
            logger.info(
                "Fetched %d raw events for %s (attempt %d)",
                len(raw_events), target_date, attempt,
            )

            parsed = []
            for ev in raw_events:
                if ev.get("status") == "cancelled":
                    continue
                try:
                    parsed.append(_parse_event(ev, timezone))
                except Exception as parse_err:
                    logger.warning(
                        "Failed to parse event '%s': %s",
                        ev.get("summary", "?"), parse_err,
                    )

            # Sort by start time (all-day events come first)
            def sort_key(e):
                s = e["start"]
                if isinstance(s, datetime):
                    return s
                # all-day: treat as midnight
                return datetime(s.year, s.month, s.day, tzinfo=tz)

            parsed.sort(key=sort_key)
            logger.info("Returning %d parsed events for %s", len(parsed), target_date)
            return parsed

        except HttpError as exc:
            last_error = exc
            logger.error(
                "Google Calendar API error on attempt %d/%d: %s",
                attempt, RETRY_COUNT, exc,
            )
        except Exception as exc:
            last_error = exc
            logger.error(
                "Unexpected error on attempt %d/%d: %s",
                attempt, RETRY_COUNT, exc,
            )

        if attempt < RETRY_COUNT:
            logger.info("Retrying in %d seconds …", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    # All retries exhausted
    raise RuntimeError(
        f"Failed to fetch Google Calendar events after {RETRY_COUNT} attempts. "
        f"Last error: {last_error}"
    ) from last_error


def fetch_events_range(
    start_date: date,
    end_date: date,
    timezone: str = DEFAULT_TIMEZONE,
    calendar_id: str = "primary",
) -> list[dict]:
    """
    Fetch calendar events for a date range [start_date, end_date) (exclusive end).

    Useful for weekly briefings or historical lookback.
    Same retry/error-handling behaviour as fetch_todays_events.
    """
    tz = ZoneInfo(timezone)

    day_start = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=tz)
    day_end = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=tz)

    time_min = day_start.isoformat()
    time_max = day_end.isoformat()

    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            service = _build_service()
            events_result = (
                service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                )
                .execute()
            )
            raw_events = events_result.get("items", [])
            logger.info(
                "Fetched %d raw events for %s→%s (attempt %d)",
                len(raw_events), start_date, end_date, attempt,
            )

            parsed = []
            for ev in raw_events:
                if ev.get("status") == "cancelled":
                    continue
                try:
                    parsed.append(_parse_event(ev, timezone))
                except Exception as parse_err:
                    logger.warning("Failed to parse event '%s': %s", ev.get("summary", "?"), parse_err)

            def sort_key(e):
                s = e["start"]
                if isinstance(s, datetime):
                    return s
                return datetime(s.year, s.month, s.day, tzinfo=tz)

            parsed.sort(key=sort_key)
            return parsed

        except HttpError as exc:
            last_error = exc
            logger.error("Google Calendar API error on attempt %d/%d: %s", attempt, RETRY_COUNT, exc)
        except Exception as exc:
            last_error = exc
            logger.error("Unexpected error on attempt %d/%d: %s", attempt, RETRY_COUNT, exc)

        if attempt < RETRY_COUNT:
            logger.info("Retrying in %d seconds …", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Failed to fetch Google Calendar events after {RETRY_COUNT} attempts. "
        f"Last error: {last_error}"
    ) from last_error
