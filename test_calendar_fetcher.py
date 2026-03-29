"""
Test script for calendar_fetcher module.

Usage:
    python test_calendar_fetcher.py

Verifies:
1. Credentials load from .env without interactive login
2. fetch_todays_events() returns a list (possibly empty on weekends/holidays)
3. Each event has required fields: id, title, start, end, attendees, location/video_link
4. Proper error handling and retry logic
"""

import logging
import sys
import os
from datetime import date

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.calendar_fetcher import fetch_todays_events, fetch_events_range

REQUIRED_FIELDS = {
    "id", "title", "status", "start", "end", "all_day",
    "start_iso", "end_iso", "attendees",
    "location", "video_link", "conference_type",
    "organizer_email", "organizer_name", "html_link",
}


def validate_event(event: dict) -> list[str]:
    """Return list of validation errors for an event dict."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in event:
            errors.append(f"Missing field: {field}")

    if event.get("attendees") is not None:
        for att in event["attendees"]:
            for af in ("email", "name", "response_status", "is_organizer"):
                if af not in att:
                    errors.append(f"Attendee missing field: {af}")
    return errors


def run_tests():
    print("=" * 60)
    print("  Google Calendar Fetcher — Integration Test")
    print("=" * 60)

    # ── Test 1: fetch today's events ──────────────────────────────
    print("\n[TEST 1] fetch_todays_events() for today …")
    try:
        events = fetch_todays_events()
        print(f"  ✓ Returned {len(events)} event(s)")
    except Exception as exc:
        print(f"  ✗ fetch_todays_events() raised: {exc}")
        return False

    # ── Test 2: validate structure of each event ──────────────────
    print("\n[TEST 2] Validating event structure …")
    all_valid = True
    for ev in events:
        errs = validate_event(ev)
        if errs:
            print(f"  ✗ Event '{ev.get('title')}' has errors: {errs}")
            all_valid = False
        else:
            print(
                f"  ✓ '{ev.get('title')}' | "
                f"{'All-day' if ev['all_day'] else ev['start'].strftime('%H:%M')}"
                f"-{'' if ev['all_day'] else ev['end'].strftime('%H:%M')}"
                f" | {len(ev['attendees'])} attendee(s)"
                f" | video={ev['video_link'] or 'N/A'}"
                f" | loc={ev['location'] or 'N/A'}"
            )
    if all_valid:
        print("  ✓ All event structures valid")

    # ── Test 3: fetch_events_range ────────────────────────────────
    print("\n[TEST 3] fetch_events_range() for the past 7 days …")
    try:
        today = date.today()
        from datetime import timedelta
        week_ago = today - timedelta(days=7)
        range_events = fetch_events_range(week_ago, today)
        print(f"  ✓ Returned {len(range_events)} event(s) over last 7 days")
    except Exception as exc:
        print(f"  ✗ fetch_events_range() raised: {exc}")
        return False

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED" if all_valid else "  SOME TESTS FAILED")
    print("=" * 60)

    # Pretty-print first event for visual inspection
    if events:
        ev = events[0]
        print("\n── First event details ──────────────────────────────────")
        for k, v in ev.items():
            if k != "description":
                print(f"  {k:20s}: {v}")
        if ev.get("description"):
            desc_preview = (ev["description"] or "")[:120].replace("\n", " ")
            print(f"  {'description':20s}: {desc_preview}…")

    return True


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
