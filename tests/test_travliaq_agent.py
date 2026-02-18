"""Tests for PhaseTracker, _update_phase_tracker, and _build_on_step_end."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.travliaq_agent import PhaseTracker
from src.orchestrator import _update_phase_tracker, _build_on_step_end


def _make_persona(phases=None):
    """Create a minimal persona mock with conversation_goals."""
    if phases is None:
        phases = [
            "greeting", "preferences", "destination", "dates",
            "travelers", "logistics", "deep_conversation",
            "completion", "send_logs",
        ]
    persona = MagicMock()
    goals = []
    for phase in phases:
        goal = MagicMock()
        goal.phase = phase
        goals.append(goal)
    persona.conversation_goals = goals
    return persona


class TestPhaseTracker:
    def test_initial_state(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        assert pt.current_phase_index == 0
        assert pt.total_phases == 9
        assert pt.feedback_submitted is False
        assert pt.feedback_phase_index == 8
        assert pt.phases_remaining == 9
        assert pt.is_feedback_reached is False

    def test_feedback_reached(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        pt.current_phase_index = 8
        assert pt.is_feedback_reached is True
        assert pt.phases_remaining == 1

    def test_phases_remaining_at_end(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        pt.current_phase_index = 9
        assert pt.phases_remaining == 0

    def test_feedback_phase_index_always_last(self):
        pt = PhaseTracker(total_phases=5)
        assert pt.feedback_phase_index == 4


class TestUpdatePhaseTracker:
    def test_advances_phase_on_keyword(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "Looking at destination suggestions", ["click"])
        assert pt.current_phase_index >= 2  # destination is index 2

    def test_does_not_go_backwards(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        pt.current_phase_index = 5
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "greeting bonjour", ["type"])
        assert pt.current_phase_index == 5  # stays at 5, doesn't go to 0

    def test_detects_feedback_submission(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "clicking on nous aider link", ["click"])
        assert pt.feedback_submitted is True

    def test_no_feedback_without_click(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "I see the feedback link", ["scroll"])
        assert pt.feedback_submitted is False

    def test_handles_none_tracker(self):
        persona = _make_persona()
        # Should not raise
        _update_phase_tracker(None, persona, "hello", ["click"])

    def test_handles_none_thinking(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, None, ["datepicker"])
        assert pt.current_phase_index >= 3  # dates is index 3

    def test_advances_through_multiple_phases(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        # Step 1: greeting
        _update_phase_tracker(pt, persona, "bonjour je cherche un voyage", ["type"])
        idx_after_greeting = pt.current_phase_index
        # Step 2: destination
        _update_phase_tracker(pt, persona, "Looking at destination cards", ["click"])
        assert pt.current_phase_index > idx_after_greeting

    def test_airport_advances_to_logistics(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "selecting aéroport CDG", ["click"])
        assert pt.current_phase_index >= 5  # logistics is index 5

    def test_recap_advances_to_completion(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        pt.current_phase_index = 6  # at deep_conversation
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "je vois le récapitulatif du voyage", ["scroll"])
        assert pt.current_phase_index >= 7  # completion is index 7

    def test_popup_submit_marks_feedback(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        persona = _make_persona()
        _update_phase_tracker(pt, persona, "submitted the popup feedback form", ["click"])
        assert pt.feedback_submitted is True


class TestBuildOnStepEnd:
    def test_no_sleep_when_zero_failures(self):
        callback = _build_on_step_end("test-persona")
        agent = MagicMock()
        agent.state.consecutive_failures = 0
        with patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_not_called()

    def test_sleeps_10s_on_first_failure(self):
        callback = _build_on_step_end("test-persona")
        agent = MagicMock()
        agent.state.consecutive_failures = 1
        with patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(10)

    def test_sleeps_20s_on_second_failure(self):
        callback = _build_on_step_end("test-persona")
        agent = MagicMock()
        agent.state.consecutive_failures = 2
        with patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(20)

    def test_caps_at_60s(self):
        callback = _build_on_step_end("test-persona")
        agent = MagicMock()
        agent.state.consecutive_failures = 5  # 10 * 2^4 = 160 → capped at 60
        with patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(60)
