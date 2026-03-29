"""
Tests for src/calendar/title_classifier.py (Sub-AC 4b).

Covers:
- MeetingLabel enum values
- classify_by_title: English internal patterns (1:1, standup, all-hands, retro,
  sprint, OKR, team/squad sync, weekly, internal, HR, lunch & learn)
- classify_by_title: Korean internal patterns (스탠드업, 올핸즈, 전사, 팀 회의,
  주간 싱크, 회고, 내부 미팅, 면접, OKR 리뷰 …)
- classify_by_title: External patterns (IR, pitch, 투자 미팅, partner, client,
  고객, 외부 미팅, conference, MOU, advisory board …)
- classify_by_title: UNKNOWN for unrecognised titles
- classify_by_title: empty / whitespace titles → UNKNOWN
- classify_by_title: case-insensitivity
- classify_by_title: internal patterns take priority over external patterns
- is_title_internal / is_title_external convenience wrappers
- matched_internal_pattern / matched_external_pattern debug helpers
- Custom pattern injection via keyword arguments
"""
from __future__ import annotations

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar.title_classifier import (
    MeetingLabel,
    classify_by_title,
    is_title_internal,
    is_title_external,
    matched_internal_pattern,
    matched_external_pattern,
)


# ── MeetingLabel ──────────────────────────────────────────────────────────────

class TestMeetingLabel:

    def test_enum_values(self):
        assert MeetingLabel.INTERNAL.value == "internal"
        assert MeetingLabel.EXTERNAL.value == "external"
        assert MeetingLabel.UNKNOWN.value  == "unknown"

    def test_enum_is_str_subclass(self):
        assert isinstance(MeetingLabel.INTERNAL, str)

    def test_string_equality(self):
        assert MeetingLabel.INTERNAL == "internal"
        assert MeetingLabel.EXTERNAL == "external"
        assert MeetingLabel.UNKNOWN  == "unknown"


# ── Internal patterns — English ───────────────────────────────────────────────

class TestInternalEnglish:

    @pytest.mark.parametrize("title", [
        "1:1 with Alice",
        "1-on-1 Jane",
        "1on1 check-in",
        "one-on-one with manager",
        "Weekly 1:1",
    ])
    def test_one_on_one_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "Daily Standup",
        "morning standup",
        "Stand-up Meeting",
        "stand up",
        "daily sync",
        "daily scrum",
        "morning sync",
    ])
    def test_standup_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "All Hands Meeting",
        "All-Hands Q1",
        "allhands",
        "Town Hall",
        "townhall 2026",
    ])
    def test_all_hands_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "Retrospective",
        "Sprint Retro",
        "Q2 Retro",
        "team retrospective",
    ])
    def test_retrospective_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "Sprint Planning",
        "Sprint Review",
        "Sprint Demo",
        "sprint grooming",
        "Sprint",
    ])
    def test_sprint_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "OKR Review",
        "OKR check-in",
        "Q1 OKR",
        "OKR Planning",
        "OKR Kickoff",
    ])
    def test_okr_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "Team Sync",
        "Team Meeting",
        "Squad Sync",
        "squad standup",
        "Weekly Sync",
        "weekly meeting",
        "bi-weekly check-in",
        "biweekly team",
        "Internal Meeting",
        "internal call",
        "company-internal sync",
    ])
    def test_team_sync_variants(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "Hiring Interview",
        "Onboarding session",
        "Performance Review",
        "perf check-in",
        "Lunch & Learn",
        "lunch and learn",
        "Brown Bag session",
    ])
    def test_hr_and_knowledge_sharing(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL


# ── Internal patterns — Korean ────────────────────────────────────────────────

class TestInternalKorean:

    @pytest.mark.parametrize("title", [
        "데일리 스탠드업",
        "스탠드-업 미팅",
        "스탠드업",
    ])
    def test_korean_standup(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "올핸즈",
        "전사 회의",
        "전사 미팅",
        "전사 행사",
        "전사",
    ])
    def test_korean_all_hands(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "팀 회의",
        "팀 싱크",
        "팀 미팅",
        "팀 스탠드업",
    ])
    def test_korean_team_meeting(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "주간 회의",
        "주간 싱크",
        "주간 미팅",
        "격주 싱크",
        "격주 회의",
        "월간 회의",
    ])
    def test_korean_periodic_sync(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "회고",
        "스프린트 회고",
        "스프린트 플래닝",
        "스프린트 리뷰",
    ])
    def test_korean_sprint_ceremonies(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "내부 회의",
        "내부 미팅",
        "내부 싱크",
    ])
    def test_korean_internal_meeting(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "OKR 리뷰",
        "OKR 체크인",
        "OKR 세션",
    ])
    def test_korean_okr(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    @pytest.mark.parametrize("title", [
        "면접",
        "채용 면접",
        "온보딩",
        "인사 평가",
        "인사 면담",
    ])
    def test_korean_hr(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL


# ── External patterns ─────────────────────────────────────────────────────────

class TestExternalPatterns:

    @pytest.mark.parametrize("title", [
        "ABC Corp IR Pitch",
        "IR Meeting",
        "Pitch Deck Review",
        "Series A Pitch",
    ])
    def test_ir_pitch(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "투자 미팅",
        "투자 심사",
        "투자 검토",
        "투자 상담",
        "투자심사",
    ])
    def test_korean_investment_meeting(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "Deal Review",
        "Deal Discussion",
        "Deal Call",
        "딜 리뷰",
        "딜 미팅",
    ])
    def test_deal_review(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "Partner Meeting",
        "Partnership Call",
        "파트너 미팅",
        "파트너 협의",
        "파트너 방문",
    ])
    def test_partner_meeting(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "Client Call",
        "Client Demo",
        "Customer Meeting",
        "customer success sync",
        "고객 미팅",
        "고객 상담",
        "고객 방문",
    ])
    def test_client_customer(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "외부 미팅",
        "외부 방문",
        "외부 회의",
        "External Meeting",
        "external call",
    ])
    def test_external_explicit(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "Product Demo",
        "Sales Meeting",
        "Sales Call",
        "세일즈 미팅",
        "세일즈 콜",
    ])
    def test_sales_demo(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "TechConference 2026",
        "Industry Summit",
        "Startup Conference",
        "컨퍼런스",
        "Webinar",
        "웨비나",
        "Networking Event",
        "네트워킹",
    ])
    def test_conference_networking(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "MOU signing",
        "NDA review",
        "LOI discussion",
    ])
    def test_legal_milestone(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "Advisory Board Meeting",
        "Advisory Council",
        "자문 위원회",
        "자문 회의",
    ])
    def test_advisory_board(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "포트폴리오 미팅",
        "포트폴리오 방문",
        "portfolio company meeting",
        "portfolio check-in",
    ])
    def test_portfolio_meeting(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL

    @pytest.mark.parametrize("title", [
        "심사",
        "투자 논의",
    ])
    def test_investment_screening(self, title: str):
        assert classify_by_title(title) == MeetingLabel.EXTERNAL


# ── Unknown titles ────────────────────────────────────────────────────────────

class TestUnknownTitles:

    @pytest.mark.parametrize("title", [
        "Birthday Party",
        "Doctor Appointment",
        "Lunch",
        "Travel",
        "Focus Time",
        "Deep Work",
        "🎂 Happy Birthday",
        "random event",
        "예약됨",                           # "Reserved" in Korean — no keyword match
    ])
    def test_unknown_titles(self, title: str):
        assert classify_by_title(title) == MeetingLabel.UNKNOWN


# ── Block / time-block events (AC 5) ──────────────────────────────────────────

class TestBlockEvents:
    """
    Calendar 'block' events are personal/team time reservations and must be
    classified as INTERNAL so they are skipped from all briefings (AC 5).
    """

    @pytest.mark.parametrize("title", [
        "block",
        "Block",
        "BLOCK",
        "blocked",
        "Blocked",
        "BLOCKED",
        "blocking",
        "block time",
        "Block Time",
        "focus block",
        "Focus Block",
        "deep work block",
        "block: project review",
        "[block]",
        "블록",
    ])
    def test_block_titles_are_internal(self, title: str):
        """Events whose titles contain 'block'/'blocked' must be INTERNAL."""
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    def test_block_standalone_word(self):
        """Bare 'block' on its own must be INTERNAL."""
        assert classify_by_title("block") == MeetingLabel.INTERNAL

    def test_block_case_insensitive(self):
        for variant in ("BLOCK", "Block", "bLoCk"):
            assert classify_by_title(variant) == MeetingLabel.INTERNAL

    def test_block_does_not_match_partial_word(self):
        """
        'blockchain' should NOT match the block heuristic because there is a
        word boundary assertion.  'blockchain' is not a calendar time-block.
        """
        # 'blockchain' contains no other internal/external keyword → UNKNOWN
        result = classify_by_title("blockchain workshop")
        # blockchain → the \bblock(ed|ing)?\b pattern won't match because
        # 'block' in 'blockchain' is not followed by a word boundary right after.
        # Actually \bblock\b WOULD match the 'block' in 'blockchain' if 'chain' starts
        # a new token. Let's check: \bblock(ed|ing)?\b — 'blockchain' has block+chain,
        # so \bblock\b would NOT match because after 'block' comes 'c' (not a
        # word-boundary).  So this should remain UNKNOWN or EXTERNAL if another
        # pattern fires.  Since "workshop" has no keyword → UNKNOWN.
        assert result != MeetingLabel.INTERNAL  # blockchain ≠ block


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    @pytest.mark.parametrize("title", [
        "",
        "   ",
        "\t\n",
    ])
    def test_empty_or_whitespace(self, title: str):
        assert classify_by_title(title) == MeetingLabel.UNKNOWN

    def test_none_like_empty_string(self):
        # Empty string must not raise
        result = classify_by_title("")
        assert result == MeetingLabel.UNKNOWN

    @pytest.mark.parametrize("title", [
        "WEEKLY SYNC",
        "weekly sync",
        "Weekly Sync",
        "wEeKlY sYnC",
    ])
    def test_case_insensitive(self, title: str):
        assert classify_by_title(title) == MeetingLabel.INTERNAL

    def test_internal_wins_over_external(self):
        """
        A title containing both an internal keyword and an external keyword
        should be classified as INTERNAL (internal patterns checked first).
        """
        # "internal" keyword wins over "partner" keyword
        result = classify_by_title("Internal Partner Review")
        assert result == MeetingLabel.INTERNAL

    def test_sprint_demo_classified_internal_not_external(self):
        """
        'Sprint Demo' should be INTERNAL (sprint ceremony), not EXTERNAL
        (demo to a client).  Internal patterns take priority.
        """
        result = classify_by_title("Sprint Demo")
        assert result == MeetingLabel.INTERNAL

    def test_title_with_extra_whitespace(self):
        """Leading/trailing whitespace should not prevent a match."""
        assert classify_by_title("   Weekly Sync   ") == MeetingLabel.INTERNAL

    def test_mixed_korean_english(self):
        """Titles mixing Korean and English should still match."""
        assert classify_by_title("Squad 주간 싱크") == MeetingLabel.INTERNAL
        assert classify_by_title("외부 Partner 미팅") == MeetingLabel.EXTERNAL

    def test_partial_word_does_not_spuriously_match(self):
        """
        Words like 'sprint' embedded in longer words should not match.
        e.g., 'constraintPlanning' — not a sprint ceremony.
        The patterns use word-boundary assertions (\b) where appropriate.
        """
        # 'sprinter' should not match the bare `\bsprint\b` pattern
        # Note: this title contains no other keyword so UNKNOWN expected
        result = classify_by_title("sprinter convention")
        # 'sprint' pattern uses \bsprint\b so "sprinter" won't match
        assert result == MeetingLabel.UNKNOWN


# ── Convenience wrappers ──────────────────────────────────────────────────────

class TestConvenienceWrappers:

    def test_is_title_internal_true(self):
        assert is_title_internal("Weekly Team Sync") is True

    def test_is_title_internal_false_for_external(self):
        assert is_title_internal("IR Pitch") is False

    def test_is_title_internal_false_for_unknown(self):
        assert is_title_internal("Doctor Appointment") is False

    def test_is_title_external_true(self):
        assert is_title_external("Client Demo") is True

    def test_is_title_external_false_for_internal(self):
        assert is_title_external("Daily Standup") is False

    def test_is_title_external_false_for_unknown(self):
        assert is_title_external("Lunch") is False


# ── Debug helpers ─────────────────────────────────────────────────────────────

class TestDebugHelpers:

    def test_matched_internal_pattern_returns_pattern(self):
        pat = matched_internal_pattern("Weekly Team Sync")
        assert pat is not None
        # Should be a regex pattern string
        assert isinstance(pat, str)

    def test_matched_internal_pattern_returns_none_for_unknown(self):
        assert matched_internal_pattern("Birthday Party") is None

    def test_matched_internal_pattern_returns_none_for_external(self):
        # IR Pitch matches external, not internal
        assert matched_internal_pattern("IR Pitch") is None

    def test_matched_external_pattern_returns_pattern(self):
        pat = matched_external_pattern("IR Pitch")
        assert pat is not None
        assert isinstance(pat, str)

    def test_matched_external_pattern_returns_none_for_unknown(self):
        assert matched_external_pattern("Birthday Party") is None

    def test_matched_external_pattern_returns_none_for_internal(self):
        # Internal meetings don't match external patterns
        assert matched_external_pattern("Daily Standup") is None


# ── Custom pattern injection ──────────────────────────────────────────────────

class TestCustomPatternInjection:

    def test_custom_internal_patterns(self):
        """Callers can inject custom compiled patterns."""
        custom_internal = [re.compile(r"\bcake\b", re.IGNORECASE)]
        custom_external = [re.compile(r"\bparty\b", re.IGNORECASE)]

        assert classify_by_title(
            "Birthday cake",
            internal_patterns=custom_internal,
            external_patterns=custom_external,
        ) == MeetingLabel.INTERNAL

    def test_custom_external_patterns(self):
        custom_internal = [re.compile(r"\bcake\b", re.IGNORECASE)]
        custom_external = [re.compile(r"\bparty\b", re.IGNORECASE)]

        assert classify_by_title(
            "Birthday party",
            internal_patterns=custom_internal,
            external_patterns=custom_external,
        ) == MeetingLabel.EXTERNAL

    def test_empty_custom_patterns_all_unknown(self):
        """If both pattern lists are empty, everything is UNKNOWN."""
        assert classify_by_title(
            "Daily Standup",
            internal_patterns=[],
            external_patterns=[],
        ) == MeetingLabel.UNKNOWN

    def test_internal_takes_priority_with_custom_patterns(self):
        both_pattern = [re.compile(r"\bfoo\b", re.IGNORECASE)]
        result = classify_by_title(
            "foo bar",
            internal_patterns=both_pattern,
            external_patterns=both_pattern,
        )
        assert result == MeetingLabel.INTERNAL


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
