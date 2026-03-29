"""
Unit tests for Sub-AC 4: Daily Morning Briefing Scheduler.

Tests are entirely offline (no real Google / Slack API calls):

  - run_daily_morning_briefing() fetch → format → send pipeline
  - Calendar fetch failure triggers error DM and returns False
  - Slack delivery failure triggers error DM and returns False
  - Bot=None mode (dry-run): returns True without any Slack calls
  - Empty event list still produces and sends a briefing (no-meetings day)
  - Daily briefing job registered in scheduler at correct cron time
  - Daily briefing cron uses Asia/Seoul timezone
  - Scheduler has both the meeting_checker and daily_morning_briefing jobs
  - _run_daily_morning_briefing_job calls pipeline and handles exceptions
  - CronTrigger fires at 09:30 KST (validated via next_fire_time)

Run:
    python -m pytest tests/test_daily_briefing.py -v
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
    starts_in_minutes: float = 60.0,
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


# ── run_daily_morning_briefing() — core pipeline tests ────────────────────────

class TestRunDailyMorningBriefing:
    """Tests for src.briefing.pipeline.run_daily_morning_briefing."""

    def test_bot_none_returns_true_without_api_calls(self):
        """When bot=None (dry-run), function should return True and skip all API calls."""
        from src.briefing.pipeline import run_daily_morning_briefing

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events"
        ) as mock_fetch:
            result = run_daily_morning_briefing(bot=None)

        assert result is True
        mock_fetch.assert_not_called()

    def test_successful_pipeline_returns_true(self):
        """Full pipeline: fetch → format → send should return True."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()
        meetings = [
            _make_meeting("evt-A", "Meeting A", 120),
            _make_meeting("evt-B", "Meeting B", 240),
        ]

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            return_value=meetings,
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ) as mock_send:
            result = run_daily_morning_briefing(bot=mock_bot)

        assert result is True
        mock_send.assert_called_once()
        # Verify the correct date and events were passed to send_daily_briefing_dm
        call_args = mock_send.call_args
        assert call_args[0][0] is mock_bot    # first positional arg = bot
        assert call_args[0][1] == meetings    # second positional arg = events list

    def test_empty_events_still_sends_briefing(self):
        """An empty calendar day should still produce and deliver a briefing."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            return_value=[],
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ) as mock_send:
            result = run_daily_morning_briefing(bot=mock_bot)

        assert result is True
        # send_daily_briefing_dm should be called with an empty list (not skipped)
        mock_send.assert_called_once()
        events_arg = mock_send.call_args[0][1]
        assert events_arg == []

    def test_target_date_passed_as_today_kst(self):
        """The target_date passed to send_daily_briefing_dm must be today in KST."""
        from src.briefing.pipeline import run_daily_morning_briefing
        from datetime import date
        from zoneinfo import ZoneInfo

        mock_bot = MagicMock()
        KST = ZoneInfo("Asia/Seoul")
        expected_date = datetime.now(KST).date()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            return_value=[],
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ) as mock_send:
            run_daily_morning_briefing(bot=mock_bot)

        call_kwargs = mock_send.call_args[1]
        assert call_kwargs.get("target_date") == expected_date

    def test_calendar_fetch_failure_sends_error_dm_and_returns_false(self):
        """When calendar API fails after all retries, error DM is sent and False is returned."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=RuntimeError("Calendar API down"),
        ), patch("time.sleep"):
            result = run_daily_morning_briefing(bot=mock_bot)

        assert result is False
        mock_bot.send_error.assert_called_once()
        error_context = mock_bot.send_error.call_args[0][0]
        assert "briefing" in error_context.lower() or "calendar" in error_context.lower()

    def test_calendar_failure_does_not_call_send_dm(self):
        """If calendar fetch fails, no briefing DM should be sent."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=RuntimeError("down"),
        ), patch("time.sleep"), patch(
            "src.slack.dm_sender.send_daily_briefing_dm"
        ) as mock_send:
            run_daily_morning_briefing(bot=mock_bot)

        mock_send.assert_not_called()

    def test_retry_count_matches_config(self):
        """Calendar fetch should be retried exactly API_RETRY_ATTEMPTS times."""
        from src.briefing.pipeline import run_daily_morning_briefing
        from src.config import API_RETRY_ATTEMPTS

        mock_bot = MagicMock()
        attempt_count = 0

        def counting_fail(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError(f"Attempt {attempt_count} failed")

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=counting_fail,
        ), patch("time.sleep"):
            run_daily_morning_briefing(bot=mock_bot)

        assert attempt_count == API_RETRY_ATTEMPTS

    def test_retry_sleeps_between_attempts(self):
        """time.sleep must be called between retry attempts (not after the last one)."""
        from src.briefing.pipeline import run_daily_morning_briefing
        from src.config import API_RETRY_ATTEMPTS, API_RETRY_DELAY_SECONDS

        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=RuntimeError("down"),
        ), patch("time.sleep") as mock_sleep:
            run_daily_morning_briefing(bot=mock_bot)

        # sleep is called between attempts: (API_RETRY_ATTEMPTS - 1) times
        assert mock_sleep.call_count == API_RETRY_ATTEMPTS - 1
        for c in mock_sleep.call_args_list:
            assert c[0][0] == API_RETRY_DELAY_SECONDS

    def test_success_on_second_attempt(self):
        """Briefing should succeed if the calendar API recovers on the second attempt."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()
        meetings = [_make_meeting("retry-evt", "Recovery Meeting", 120)]
        call_count = 0

        def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient failure")
            return meetings

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=fail_then_succeed,
        ), patch("time.sleep"), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ) as mock_send:
            result = run_daily_morning_briefing(bot=mock_bot)

        assert result is True
        mock_send.assert_called_once()
        mock_bot.send_error.assert_not_called()

    def test_delivery_failure_sends_error_dm_and_returns_false(self):
        """When send_daily_briefing_dm returns False, an error DM is sent."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            return_value=[],
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=False,
        ):
            result = run_daily_morning_briefing(bot=mock_bot)

        assert result is False
        mock_bot.send_error.assert_called_once()


# ── Scheduler integration tests ───────────────────────────────────────────────

class TestDailyBriefingSchedulerRegistration:
    """Verify the daily_morning_briefing job is correctly registered in the scheduler."""

    def test_daily_briefing_job_registered(self):
        """After start_scheduler(), a 'daily_morning_briefing' job must exist."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job = scheduler.get_job("daily_morning_briefing")
        assert job is not None, "daily_morning_briefing job must be registered"
        assert "09:30" in job.name or "Morning" in job.name or "Briefing" in job.name

        sched_module.stop_scheduler()

    def test_both_jobs_registered(self):
        """Scheduler must register both meeting_checker and daily_morning_briefing."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job_ids = {j.id for j in scheduler.get_jobs()}
        assert "meeting_checker" in job_ids, "meeting_checker job missing"
        assert "daily_morning_briefing" in job_ids, "daily_morning_briefing job missing"

        sched_module.stop_scheduler()

    def test_daily_briefing_job_uses_cron_trigger(self):
        """The daily morning briefing job must use a CronTrigger (not IntervalTrigger)."""
        import src.scheduler as sched_module
        from apscheduler.triggers.cron import CronTrigger

        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job = scheduler.get_job("daily_morning_briefing")
        assert isinstance(job.trigger, CronTrigger), (
            f"daily_morning_briefing must use CronTrigger, got {type(job.trigger).__name__}"
        )

        sched_module.stop_scheduler()

    def test_daily_briefing_cron_fires_at_09_30(self):
        """The CronTrigger must fire at 09:30 (h=9, m=30)."""
        import src.scheduler as sched_module
        from apscheduler.triggers.cron import CronTrigger
        from zoneinfo import ZoneInfo

        sched_module._scheduler = None
        KST = ZoneInfo("Asia/Seoul")

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job = scheduler.get_job("daily_morning_briefing")
        trigger: CronTrigger = job.trigger

        # Calculate the next fire time from a known reference point
        # Reference: any time before 09:30 today KST
        today_start_kst = datetime.now(KST).replace(hour=8, minute=0, second=0, microsecond=0)
        next_fire = trigger.get_next_fire_time(None, today_start_kst.astimezone(timezone.utc))

        assert next_fire is not None
        next_fire_kst = next_fire.astimezone(KST)
        assert next_fire_kst.hour == 9, (
            f"Expected fire hour 9, got {next_fire_kst.hour}"
        )
        assert next_fire_kst.minute == 30, (
            f"Expected fire minute 30, got {next_fire_kst.minute}"
        )

        sched_module.stop_scheduler()

    def test_daily_briefing_cron_timezone_is_seoul(self):
        """The CronTrigger timezone must be Asia/Seoul."""
        import src.scheduler as sched_module
        from apscheduler.triggers.cron import CronTrigger

        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            scheduler = sched_module.start_scheduler(bot=None)

        job = scheduler.get_job("daily_morning_briefing")
        trigger: CronTrigger = job.trigger
        trigger_tz = str(trigger.timezone)
        assert "Seoul" in trigger_tz or "Asia/Seoul" in trigger_tz, (
            f"CronTrigger timezone must be Asia/Seoul, got {trigger_tz}"
        )

        sched_module.stop_scheduler()

    def test_daily_briefing_job_uses_replace_existing(self):
        """Calling start_scheduler twice must not create duplicate daily_morning_briefing jobs."""
        import src.scheduler as sched_module
        sched_module._scheduler = None

        with patch("src.scheduler._run_initial_check"):
            sched_module.start_scheduler(bot=None)
            sched_module.start_scheduler(bot=None)  # idempotent second call

        jobs = sched_module._scheduler.get_jobs()
        daily_jobs = [j for j in jobs if j.id == "daily_morning_briefing"]
        assert len(daily_jobs) == 1, (
            f"Expected exactly 1 daily_morning_briefing job, found {len(daily_jobs)}"
        )

        sched_module.stop_scheduler()


# ── _run_daily_morning_briefing_job() tests ───────────────────────────────────

class TestRunDailyMorningBriefingJob:
    """Tests for the scheduler job wrapper function."""

    def test_job_calls_pipeline(self):
        """_run_daily_morning_briefing_job must delegate to run_daily_morning_briefing."""
        from src.scheduler import _run_daily_morning_briefing_job

        mock_bot = MagicMock()

        with patch(
            "src.briefing.pipeline.run_daily_morning_briefing",
            return_value=True,
        ) as mock_pipeline:
            _run_daily_morning_briefing_job(bot=mock_bot)

        mock_pipeline.assert_called_once_with(bot=mock_bot)

    def test_job_bot_none_does_not_raise(self):
        """Calling the job with bot=None must not raise."""
        from src.scheduler import _run_daily_morning_briefing_job

        with patch("src.briefing.pipeline.run_daily_morning_briefing", return_value=True):
            _run_daily_morning_briefing_job(bot=None)  # must not raise

    def test_job_pipeline_exception_caught_and_error_dm_sent(self):
        """If run_daily_morning_briefing raises, the job must catch it and send error DM."""
        from src.scheduler import _run_daily_morning_briefing_job

        mock_bot = MagicMock()

        with patch(
            "src.briefing.pipeline.run_daily_morning_briefing",
            side_effect=RuntimeError("unexpected pipeline error"),
        ):
            # Must not raise
            _run_daily_morning_briefing_job(bot=mock_bot)

        mock_bot.send_error.assert_called_once()

    def test_job_pipeline_exception_with_no_bot_does_not_raise(self):
        """Exception in pipeline with bot=None must be caught silently."""
        from src.scheduler import _run_daily_morning_briefing_job

        with patch(
            "src.briefing.pipeline.run_daily_morning_briefing",
            side_effect=RuntimeError("pipeline exploded"),
        ):
            # Must not raise even without a bot to send DM to
            _run_daily_morning_briefing_job(bot=None)


# ── Integration: pipeline uses correct time window ────────────────────────────

class TestDailyBriefingTimeWindow:
    """Verify the time window passed to list_upcoming_events covers the full day."""

    def test_fetch_covers_full_day_kst(self):
        """list_upcoming_events must be called with midnight-to-midnight KST bounds."""
        from src.briefing.pipeline import run_daily_morning_briefing
        from zoneinfo import ZoneInfo

        KST = ZoneInfo("Asia/Seoul")
        mock_bot = MagicMock()
        captured_args: list[dict] = []

        def capture_call(**kwargs):
            captured_args.append(kwargs)
            return []

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=capture_call,
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ):
            run_daily_morning_briefing(bot=mock_bot)

        assert len(captured_args) == 1
        call_kwargs = captured_args[0]

        time_min: datetime = call_kwargs["time_min"]
        time_max: datetime = call_kwargs["time_max"]

        # Convert to KST for assertions
        time_min_kst = time_min.astimezone(KST)
        time_max_kst = time_max.astimezone(KST)

        # Window must span exactly 24 hours
        delta = time_max - time_min
        assert delta.total_seconds() == 86400, (
            f"Expected 24-hour window, got {delta}"
        )

        # time_min must be 00:00 KST
        assert time_min_kst.hour == 0
        assert time_min_kst.minute == 0
        assert time_min_kst.second == 0

        # time_max must be 00:00 KST next day
        assert time_max_kst.hour == 0
        assert time_max_kst.minute == 0

    def test_max_results_is_50(self):
        """list_upcoming_events should be called with max_results=50."""
        from src.briefing.pipeline import run_daily_morning_briefing

        mock_bot = MagicMock()
        captured_args: list[dict] = []

        def capture_call(**kwargs):
            captured_args.append(kwargs)
            return []

        with patch(
            "src.calendar.google_calendar.GoogleCalendarClient.list_upcoming_events",
            side_effect=capture_call,
        ), patch(
            "src.slack.dm_sender.send_daily_briefing_dm",
            return_value=True,
        ):
            run_daily_morning_briefing(bot=mock_bot)

        assert captured_args[0]["max_results"] == 50
