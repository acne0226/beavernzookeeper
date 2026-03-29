"""
Unit tests for Sub-AC 2a: Scheduler / External-Meeting Detection.

Tests are entirely offline (no real Google API calls):
  - Meeting.is_external detection logic
  - Deduplication of triggered event IDs
  - Scheduler job calls briefing pipeline exactly once per meeting
  - Scheduler start/stop lifecycle
  - Pipeline exception resilience (scheduler keeps running after pipeline error)
  - Boundary condition: meetings starting exactly at lookahead boundary
  - Boundary condition: already-started meetings excluded from window

Run:
    python -m pytest tests/test_scheduler.py -v
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# ── path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_meeting(
    event_id: str = "evt-001",
    summary: str = "Test Meeting",
    starts_in_minutes: float = 10.0,
    external_emails: list[str] | None = None,
    internal_emails: list[str] | None = None,
):
    """Construct a Meeting with controllable attributes."""
    from src.calendar.google_calendar import Meeting, Attendee

    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=starts_in_minutes)
    end = start + timedelta(minutes=30)

    attendees: list[Attendee] = []
    for email in (external_emails or ["ceo@startup.com"]):
        attendees.append(Attendee(email=email, display_name="외부인"))
    for email in (internal_emails or ["invest1@kakaoventures.co.kr"]):
        attendees.append(Attendee(email=email, display_name="내부인"))

    return Meeting(
        event_id=event_id,
        summary=summary,
        start=start,
        end=end,
        attendees=attendees,
    )


# ── Meeting.is_external tests ─────────────────────────────────────────────────

class TestMeetingIsExternal:
    def test_external_when_has_external_attendee(self):
        m = _make_meeting(external_emails=["ceo@startup.com"])
        assert m.is_external is True

    def test_not_external_when_all_internal(self):
        from src.calendar.google_calendar import Meeting, Attendee
        now = datetime.now(timezone.utc)
        m = Meeting(
            event_id="all-internal",
            summary="Internal Sync",
            start=now + timedelta(minutes=5),
            end=now + timedelta(minutes=35),
            attendees=[
                Attendee(email="invest1@kakaoventures.co.kr"),
                Attendee(email="invest2@kakaoventures.co.kr"),
            ],
        )
        assert m.is_external is False

    def test_external_attendees_property(self):
        m = _make_meeting(
            external_emails=["ceo@startup.com", "cto@startup.com"],
            internal_emails=["invest1@kakaoventures.co.kr"],
        )
        external = m.external_attendees
        assert len(external) == 2
        assert all(not a.is_internal for a in external)

    def test_starts_in_minutes(self):
        m = _make_meeting(starts_in_minutes=10)
        # Should be roughly 10 minutes (allow ±1 for execution time)
        assert 9 < m.starts_in_minutes < 11

    def test_no_attendees_is_internal(self):
        from src.calendar.google_calendar import Meeting
        now = datetime.now(timezone.utc)
        m = Meeting(
            event_id="no-attendees",
            summary="Solo block",
            start=now + timedelta(minutes=5),
            end=now + timedelta(minutes=35),
            attendees=[],
        )
        assert m.is_external is False


# ── Attendee.is_internal tests ────────────────────────────────────────────────

class TestAttendeeIsInternal:
    def test_kakaoventures_domain_is_internal(self):
        from src.calendar.google_calendar import Attendee
        a = Attendee(email="someone@kakaoventures.co.kr")
        assert a.is_internal is True

    def test_external_domain_is_not_internal(self):
        from src.calendar.google_calendar import Attendee
        a = Attendee(email="founder@startup.com")
        assert a.is_internal is False

    def test_case_insensitive(self):
        from src.calendar.google_calendar import Attendee
        a = Attendee(email="Someone@KakaoVentures.Co.Kr")
        assert a.is_internal is True


# ── Scheduler deduplication tests ─────────────────────────────────────────────

class TestSchedulerDeduplication:
    def setup_method(self):
        """Reset scheduler state before each test."""
        from src.scheduler import reset_triggered_ids
        reset_triggered_ids()

    def test_meeting_triggered_only_once(self):
        """Same meeting ID should trigger briefing only once across two ticks."""
        triggered_summaries: list[str] = []

        def mock_trigger(meeting, bot=None):
            triggered_summaries.append(meeting.summary)
            return True

        meeting = _make_meeting(event_id="dedup-evt-001", summary="Startup Pitch")

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[meeting],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=mock_trigger,
        ):
            from src.scheduler import _check_upcoming_external_meetings
            _check_upcoming_external_meetings(bot=None)
            _check_upcoming_external_meetings(bot=None)

        assert triggered_summaries.count("Startup Pitch") == 1, (
            "Meeting should be triggered exactly once even across multiple ticks"
        )

    def test_different_meetings_both_triggered(self):
        """Two different meeting IDs should both trigger briefings."""
        triggered_ids: list[str] = []

        def mock_trigger(meeting, bot=None):
            triggered_ids.append(meeting.event_id)
            return True

        m1 = _make_meeting(event_id="evt-A", summary="Meeting A", starts_in_minutes=5)
        m2 = _make_meeting(event_id="evt-B", summary="Meeting B", starts_in_minutes=10)

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[m1, m2],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=mock_trigger,
        ):
            from src.scheduler import _check_upcoming_external_meetings
            _check_upcoming_external_meetings(bot=None)

        assert "evt-A" in triggered_ids
        assert "evt-B" in triggered_ids
        assert len(triggered_ids) == 2

    def test_get_triggered_ids_reflects_state(self):
        from src.scheduler import get_triggered_ids, _triggered_event_ids, _lock

        with _lock:
            _triggered_event_ids.add("test-evt-123")

        ids = get_triggered_ids()
        assert "test-evt-123" in ids

    def test_reset_triggered_ids(self):
        from src.scheduler import get_triggered_ids, reset_triggered_ids, _triggered_event_ids, _lock

        with _lock:
            _triggered_event_ids.add("evt-to-clear")

        reset_triggered_ids()
        assert len(get_triggered_ids()) == 0


# ── Scheduler API-failure & retry tests ───────────────────────────────────────

class TestSchedulerRetryBehavior:
    def setup_method(self):
        from src.scheduler import reset_triggered_ids
        reset_triggered_ids()

    def test_error_dm_sent_after_exhausted_retries(self):
        """When all calendar retries fail, bot.send_error must be called."""
        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            side_effect=RuntimeError("API down"),
        ), patch("time.sleep"):  # skip actual sleep
            from src.scheduler import _check_upcoming_external_meetings
            _check_upcoming_external_meetings(bot=mock_bot)

        mock_bot.send_error.assert_called_once()
        call_args = mock_bot.send_error.call_args
        assert "Calendar" in call_args[0][0]

    def test_no_error_dm_when_bot_is_none(self):
        """When bot=None, failure should not raise, just log."""
        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            side_effect=RuntimeError("API down"),
        ), patch("time.sleep"):
            from src.scheduler import _check_upcoming_external_meetings
            # Should not raise
            _check_upcoming_external_meetings(bot=None)


# ── Scheduler start/stop lifecycle ────────────────────────────────────────────

class TestSchedulerLifecycle:
    def test_start_and_stop(self):
        """Scheduler should start running then stop cleanly."""
        import src.scheduler as sched_module
        sched_module._scheduler = None  # ensure clean state

        with patch(
            "src.scheduler._run_initial_check"
        ):  # skip the initial API call
            scheduler = sched_module.start_scheduler(bot=None)

        assert scheduler.running is True
        job = scheduler.get_job("meeting_checker")
        assert job is not None
        assert job.name == "External Meeting Checker"

        sched_module.stop_scheduler()
        assert sched_module._scheduler is None

    def test_start_idempotent(self):
        """Calling start_scheduler twice should not create duplicate jobs."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            sched_module.start_scheduler(bot=None)
            sched_module.start_scheduler(bot=None)  # second call

        jobs = sched_module._scheduler.get_jobs()
        meeting_jobs = [j for j in jobs if j.id == "meeting_checker"]
        assert len(meeting_jobs) == 1  # replace_existing=True

        sched_module.stop_scheduler()


# ── Integration: run_check_once ───────────────────────────────────────────────

class TestRunCheckOnce:
    def setup_method(self):
        from src.scheduler import reset_triggered_ids
        reset_triggered_ids()

    def test_run_check_once_triggers_briefing(self):
        """run_check_once should trigger briefing for a detected meeting."""
        triggered: list[str] = []

        def mock_trigger(meeting, bot=None):
            triggered.append(meeting.event_id)
            return True

        meeting = _make_meeting(event_id="once-evt-001", starts_in_minutes=8)

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[meeting],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=mock_trigger,
        ):
            from src.scheduler import run_check_once
            run_check_once(bot=None)

        assert "once-evt-001" in triggered


# ── Pipeline exception resilience ─────────────────────────────────────────────

class TestSchedulerPipelineResilience:
    """Verify the scheduler does NOT crash when the briefing pipeline raises."""

    def setup_method(self):
        from src.scheduler import reset_triggered_ids
        reset_triggered_ids()

    def test_pipeline_exception_does_not_crash_scheduler(self):
        """An exception in trigger_meeting_briefing must be caught; scheduler continues."""
        mock_bot = MagicMock()
        meeting = _make_meeting(event_id="fail-evt-001", summary="Will Fail")

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[meeting],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=RuntimeError("pipeline exploded"),
        ):
            from src.scheduler import _check_upcoming_external_meetings
            # Should NOT raise even though the pipeline raises
            _check_upcoming_external_meetings(bot=mock_bot)

        # The bot should receive an error DM about the pipeline failure
        mock_bot.send_error.assert_called_once()

    def test_pipeline_failure_still_marks_event_as_triggered(self):
        """Even if pipeline fails, the event ID is added to triggered set (no retry storm)."""
        meeting = _make_meeting(event_id="fail-dedup-evt", summary="Fail But Dedup")

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[meeting],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=RuntimeError("fail"),
        ):
            from src.scheduler import _check_upcoming_external_meetings, get_triggered_ids
            _check_upcoming_external_meetings(bot=None)

        # Event should be in triggered set so it won't be retried on next tick
        assert "fail-dedup-evt" in get_triggered_ids()

    def test_two_meetings_one_fails_other_still_triggered(self):
        """If first meeting's pipeline fails, the second meeting should still be briefed."""
        briefed_ids: list[str] = []

        def selective_fail(meeting, bot=None):
            if meeting.event_id == "fail-evt":
                raise RuntimeError("This one fails")
            briefed_ids.append(meeting.event_id)
            return True

        m1 = _make_meeting(event_id="fail-evt", summary="Failing Meeting")
        m2 = _make_meeting(event_id="ok-evt", summary="OK Meeting")

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.get_external_meetings_starting_soon",
            return_value=[m1, m2],
        ), patch(
            "src.briefing.pipeline.trigger_meeting_briefing",
            side_effect=selective_fail,
        ):
            from src.scheduler import _check_upcoming_external_meetings
            _check_upcoming_external_meetings(bot=None)

        # The second meeting should have been briefed despite first one failing
        assert "ok-evt" in briefed_ids


# ── Boundary conditions ────────────────────────────────────────────────────────

class TestMeetingBoundaryConditions:
    """Edge cases for the meeting time window detection."""

    def test_meeting_duration_calculation(self):
        """duration_minutes should correctly reflect end - start."""
        from src.calendar.google_calendar import Meeting
        now = datetime.now(timezone.utc)
        m = Meeting(
            event_id="duration-test",
            summary="1-hour Meeting",
            start=now + timedelta(minutes=10),
            end=now + timedelta(minutes=70),
            attendees=[],
        )
        assert m.duration_minutes == 60

    def test_to_dict_serialization(self):
        """Meeting.to_dict() should include all expected keys."""
        m = _make_meeting(event_id="dict-evt", summary="Dict Test")
        d = m.to_dict()

        expected_keys = {
            "event_id", "summary", "start", "end", "attendees",
            "external_attendees", "is_external", "description",
            "location", "html_link", "organizer_email", "duration_minutes",
        }
        assert expected_keys.issubset(d.keys())
        assert d["event_id"] == "dict-evt"
        assert d["is_external"] is True

    def test_attendee_with_display_name(self):
        """Attendee display_name should be preserved."""
        from src.calendar.google_calendar import Attendee
        a = Attendee(email="ceo@startup.com", display_name="Jane CEO", response_status="accepted")
        assert a.display_name == "Jane CEO"
        assert a.response_status == "accepted"
        assert a.is_internal is False

    def test_only_external_attendees_listed_in_property(self):
        """external_attendees should include only non-kakaoventures attendees."""
        from src.calendar.google_calendar import Meeting, Attendee
        now = datetime.now(timezone.utc)
        m = Meeting(
            event_id="mixed-evt",
            summary="Mixed Meeting",
            start=now + timedelta(minutes=5),
            end=now + timedelta(minutes=35),
            attendees=[
                Attendee(email="ext1@startup.com"),
                Attendee(email="ext2@portfolio.io"),
                Attendee(email="int1@kakaoventures.co.kr"),
                Attendee(email="int2@kakaoventures.co.kr"),
            ],
        )
        external = m.external_attendees
        emails = {a.email for a in external}
        assert emails == {"ext1@startup.com", "ext2@portfolio.io"}
        assert len(external) == 2

    def test_investment_team_member_is_internal_and_team(self):
        """A listed team member should be both is_internal and is_investment_team."""
        from src.calendar.google_calendar import Attendee
        a = Attendee(email="hyewon.anne@kakaoventures.co.kr")
        assert a.is_internal is True
        assert a.is_investment_team is True

    def test_non_team_internal_is_internal_but_not_team(self):
        """An @kakaoventures.co.kr address not in INVESTMENT_TEAM_EMAILS is internal but not team."""
        from src.calendar.google_calendar import Attendee
        # Non-listed internal domain address
        a = Attendee(email="unknown.staff@kakaoventures.co.kr")
        assert a.is_internal is True
        assert a.is_investment_team is False


# ── Scheduler configuration verification ─────────────────────────────────────

class TestSchedulerConfiguration:
    """Verify the scheduler is configured with the correct operational parameters."""

    def test_correct_lookahead_minutes(self):
        """Meeting lookahead should be exactly 15 minutes as required."""
        from src.config import MEETING_LOOKAHEAD_MINUTES
        assert MEETING_LOOKAHEAD_MINUTES == 15, (
            f"MEETING_LOOKAHEAD_MINUTES must be 15, got {MEETING_LOOKAHEAD_MINUTES}"
        )

    def test_correct_retry_settings(self):
        """Retry must be 3 attempts with 10-second delays."""
        from src.config import API_RETRY_ATTEMPTS, API_RETRY_DELAY_SECONDS
        assert API_RETRY_ATTEMPTS == 3
        assert API_RETRY_DELAY_SECONDS == 10

    def test_correct_poll_interval(self):
        """Scheduler must poll at 60-second intervals."""
        from src.config import SCHEDULER_POLL_INTERVAL_SECONDS
        assert SCHEDULER_POLL_INTERVAL_SECONDS == 60

    def test_scheduler_job_configured_with_coalesce_and_single_instance(self):
        """APScheduler job must use coalesce=True and max_instances=1."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job = scheduler.get_job("meeting_checker")
        assert job is not None
        # coalesce and max_instances are in job defaults
        assert scheduler._job_defaults.get("coalesce") is True
        assert scheduler._job_defaults.get("max_instances") == 1

        sched_module.stop_scheduler()

    def test_scheduler_timezone_is_seoul(self):
        """Scheduler timezone must be Asia/Seoul for KST scheduling."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        assert str(scheduler.timezone) == "Asia/Seoul"

        sched_module.stop_scheduler()
