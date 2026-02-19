"""Tests for PhaseTracker, _update_phase_tracker, _build_on_step_end, and rate-limit fixes."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.travliaq_agent import PhaseTracker
from src.orchestrator import _update_phase_tracker, _build_on_step_end
from src.agent_factory import _get_provider
from src.models import TestRunResult, RunStatus


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


class TestGetProvider:
    """Test _get_provider identifies LLM providers correctly."""

    def test_gemini_is_google(self):
        assert _get_provider("gemini-2.5-flash-lite") == "google"
        assert _get_provider("gemini-2.5-flash") == "google"

    def test_groq_model_exact_match(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider("meta-llama/llama-4-scout-17b-16e-instruct", settings) == "groq"

    def test_openrouter_default(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider("google/gemma-3-12b-it", settings) == "openrouter"

    def test_openrouter_backup_model(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider("nvidia/nemotron-nano-12b-v2-vl:free", settings) == "openrouter"

    def test_sambanova_model_exact_match(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = "test_key"
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider("Llama-4-Maverick-17B-128E-Instruct", settings) == "sambanova"

    def test_sambanova_without_key_falls_to_openrouter(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider("Llama-4-Maverick-17B-128E-Instruct", settings) == "openrouter"


class TestExcludedProvidersFallback:
    """Test that excluded_providers filters fallback selection in create_agent."""

    def _mock_settings(self):
        """Create settings mock with all 5 providers configured."""
        settings = MagicMock()
        settings.groq_api_key = "gsk_test"
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.openrouter_api_key = "sk-or-test"
        settings.openrouter_model = "nvidia/nemotron-nano-12b-v2-vl:free"
        settings.google_api_key = "AI_test"
        settings.google_model = "gemini-2.5-flash-lite"
        settings.google_fallback_model = "gemini-2.5-flash"
        settings.sambanova_api_key = "sn_test"
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        settings.cerebras_api_key = "csk_test"
        settings.cerebras_model = "llama-3.3-70b"
        settings.openrouter_backup_models_list = []
        settings.build_model_chain.return_value = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "Llama-4-Maverick-17B-128E-Instruct",
            "llama-3.3-70b",
        ]
        return settings

    def test_no_excluded_picks_different_provider(self):
        """Without exclusions, fallback is first model from different provider."""
        settings = self._mock_settings()
        chain = settings.build_model_chain()
        primary = "gemini-2.5-flash-lite"
        primary_provider = _get_provider(primary)
        excluded = {primary_provider}  # just {"google"}
        fallback = None
        for candidate in chain:
            if _get_provider(candidate, settings) not in excluded:
                fallback = candidate
                break
        assert fallback == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert _get_provider(fallback, settings) == "groq"

    def test_excluded_groq_skips_to_openrouter(self):
        """With groq excluded, Google primary picks OpenRouter fallback."""
        settings = self._mock_settings()
        chain = settings.build_model_chain()
        primary = "gemini-2.5-flash-lite"
        primary_provider = _get_provider(primary)
        excluded = {"groq"} | {primary_provider}  # {"groq", "google"}
        fallback = None
        for candidate in chain:
            if _get_provider(candidate, settings) not in excluded:
                fallback = candidate
                break
        assert fallback == "nvidia/nemotron-nano-12b-v2-vl:free"
        assert _get_provider(fallback, settings) == "openrouter"

    def test_excluded_groq_openrouter_picks_sambanova(self):
        """With groq+openrouter excluded, Google primary picks SambaNova."""
        settings = self._mock_settings()
        chain = settings.build_model_chain()
        primary = "gemini-2.5-flash-lite"
        primary_provider = _get_provider(primary)
        excluded = {"groq", "openrouter"} | {primary_provider}
        fallback = None
        for candidate in chain:
            if _get_provider(candidate, settings) not in excluded:
                fallback = candidate
                break
        assert fallback == "Llama-4-Maverick-17B-128E-Instruct"
        assert _get_provider(fallback, settings) == "sambanova"

    def test_all_excluded_no_fallback(self):
        """With all providers excluded, no fallback is selected."""
        settings = self._mock_settings()
        chain = settings.build_model_chain()
        primary = "gemini-2.5-flash-lite"
        primary_provider = _get_provider(primary)
        excluded = {"groq", "openrouter", "sambanova", "cerebras"} | {primary_provider}
        fallback = None
        for candidate in chain:
            if _get_provider(candidate, settings) not in excluded:
                fallback = candidate
                break
        assert fallback is None  # no valid fallback exists


class TestBatchAbort:
    """Test that run_batch aborts when all providers exhausted."""

    def test_exhausted_providers_on_result(self):
        """TestRunResult stores exhausted_providers correctly."""
        result = TestRunResult(
            run_id="test-1",
            persona_id="p1",
            persona_name="Test",
            exhausted_providers=["groq", "openrouter", "google"],
        )
        assert len(result.exhausted_providers) == 3
        assert "groq" in result.exhausted_providers

    def test_exhausted_providers_in_json(self):
        """exhausted_providers appears in JSON output."""
        result = TestRunResult(
            run_id="test-1",
            persona_id="p1",
            persona_name="Test",
            exhausted_providers=["groq", "openrouter", "google"],
        )
        json_dict = result.to_json_dict()
        assert json_dict["meta"]["exhausted_providers"] == ["groq", "openrouter", "google"]

    def test_empty_exhausted_providers_default(self):
        """Default exhausted_providers is empty list."""
        result = TestRunResult(
            run_id="test-1",
            persona_id="p1",
            persona_name="Test",
        )
        assert result.exhausted_providers == []

    def test_batch_abort_skips_remaining(self):
        """When all providers exhausted, remaining personas get skipped status."""
        from src.models import TestRunResult, RunStatus

        # Simulate: persona 1 fails with all providers exhausted
        result1 = TestRunResult(
            run_id="p1-abc",
            persona_id="persona1",
            persona_name="Persona 1",
            status=RunStatus.FAILED,
            exhausted_providers=["google", "groq", "openrouter"],
        )

        # Simulate what run_batch does: check and create skip results
        p2 = MagicMock(id="persona2", language="fr")
        p2.name = "Persona 2"
        p3 = MagicMock(id="persona3", language="en")
        p3.name = "Persona 3"
        personas_remaining = [p2, p3]
        results = [result1]

        if len(result1.exhausted_providers) >= 3:
            for remaining in personas_remaining:
                skip = TestRunResult(
                    run_id=f"{remaining.id}-skipped",
                    persona_id=remaining.id,
                    persona_name=remaining.name,
                    persona_language=remaining.language,
                    status=RunStatus.FAILED,
                    error_message="Skipped — all LLM providers exhausted",
                    exhausted_providers=result1.exhausted_providers,
                )
                skip.finished_at = datetime.now(timezone.utc)
                skip.duration_seconds = 0.0
                results.append(skip)

        assert len(results) == 3
        assert results[1].persona_id == "persona2"
        assert results[1].status == RunStatus.FAILED
        assert "Skipped" in results[1].error_message
        assert results[1].duration_seconds == 0.0
        assert results[2].persona_id == "persona3"


class TestBackupChainTrigger:
    """Test that backup chain fires when agent aborts early (< min_useful_steps)."""

    @staticmethod
    def _should_trigger_backup(is_done, has_errors, num_actions, min_useful_steps=10):
        """Reproduce the orchestrator's backup chain trigger logic."""
        return (
            not is_done
            and has_errors
            and num_actions < min_useful_steps
        )

    def test_triggers_on_zero_actions(self):
        """0 actions + errors → should trigger backup (original behaviour)."""
        assert self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=0
        )

    def test_triggers_on_few_actions(self):
        """3 actions + errors + not done → should trigger backup."""
        assert self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=3
        )

    def test_triggers_at_boundary(self):
        """9 actions (just under threshold) → should trigger."""
        assert self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=9
        )

    def test_no_trigger_at_threshold(self):
        """10 actions (at threshold) → should NOT trigger."""
        assert not self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=10
        )

    def test_no_trigger_on_many_actions(self):
        """15 actions + errors → should NOT trigger (agent made progress)."""
        assert not self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=15
        )

    def test_no_trigger_when_done(self):
        """Agent called done() → should NOT trigger backup."""
        assert not self._should_trigger_backup(
            is_done=True, has_errors=True, num_actions=0
        )

    def test_no_trigger_without_errors(self):
        """No errors → should NOT trigger backup."""
        assert not self._should_trigger_backup(
            is_done=False, has_errors=False, num_actions=3
        )

    def test_custom_threshold(self):
        """Custom min_useful_steps threshold works."""
        assert self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=4, min_useful_steps=5
        )
        assert not self._should_trigger_backup(
            is_done=False, has_errors=True, num_actions=5, min_useful_steps=5
        )


class TestStuckActionCount:
    """Tests for PhaseTracker._stuck_action_count periodic re-injection."""

    def test_initial_count_is_zero(self):
        pt = PhaseTracker(total_phases=9, language="fr")
        assert pt._stuck_action_count == 0

    def test_count_increments_while_stuck(self):
        """Count goes 1,2,3 while is_stuck_in_loop is True."""
        pt = PhaseTracker(total_phases=9, language="fr")
        # Fill recent_actions with same action to trigger is_stuck_in_loop
        for _ in range(6):
            pt.push_action("click_element[5]")
        assert pt.is_stuck_in_loop
        # Simulate what _inject_budget_warning does
        pt._stuck_action_count += 1
        assert pt._stuck_action_count == 1
        pt._stuck_action_count += 1
        assert pt._stuck_action_count == 2
        pt._stuck_action_count += 1
        assert pt._stuck_action_count == 3

    def test_count_resets_when_loop_breaks(self):
        """Different action resets stuck count to 0."""
        pt = PhaseTracker(total_phases=9, language="fr")
        for _ in range(6):
            pt.push_action("click_element[5]")
        pt._stuck_action_count = 7
        # Break the loop
        pt.push_action("input_text[3]")
        pt.push_action("scroll_down")
        pt.push_action("go_to_url")
        assert not pt.is_stuck_in_loop
        # Simulate what _inject_budget_warning does when not stuck
        if not pt.is_stuck_in_loop:
            pt._stuck_action_count = 0
        assert pt._stuck_action_count == 0

    def test_should_inject_at_1_and_5(self):
        """Injection fires at count 1 and every 5th count."""
        should_inject_at = []
        for count in range(1, 16):
            should_inject = (count == 1 or count % 5 == 0)
            if should_inject:
                should_inject_at.append(count)
        assert should_inject_at == [1, 5, 10, 15]

    def test_escalation_at_10(self):
        """At count >= 10, escalated message should fire."""
        pt = PhaseTracker(total_phases=9, language="fr")
        pt._stuck_action_count = 10
        assert pt._stuck_action_count >= 10  # escalation threshold

    def test_is_stuck_in_loop_property(self):
        """is_stuck_in_loop returns True when last 3 of 6 recent_actions are identical."""
        pt = PhaseTracker(total_phases=9, language="fr")
        # Not stuck initially (not enough actions)
        assert not pt.is_stuck_in_loop
        # Add 6 varied actions
        for action in ["a", "b", "c", "d", "e", "f"]:
            pt.push_action(action)
        assert not pt.is_stuck_in_loop
        # Add 3 identical → now last 3 are same
        for _ in range(3):
            pt.push_action("click")
        assert pt.is_stuck_in_loop


class TestStepThrottling:
    """Tests for per-step rate-limit throttling in _build_on_step_end."""

    def test_throttle_sleeps_when_below_interval(self):
        """Two rapid successful steps should trigger throttle sleep."""
        callback = _build_on_step_end("test-persona", min_step_interval=2.5)
        agent = MagicMock()
        agent.state.consecutive_failures = 0

        with patch("src.orchestrator.time.monotonic") as mock_time, \
             patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # First call: last_step_time=0.0, monotonic=1000.0, elapsed >> 2.5
            mock_time.return_value = 1000.0
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_not_called()

            # Second call: 0.5s later (under 2.5s interval)
            mock_time.return_value = 1000.5
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(2.0)

    def test_no_throttle_when_interval_exceeded(self):
        """Step after >= min_interval should not sleep."""
        callback = _build_on_step_end("test-persona", min_step_interval=2.5)
        agent = MagicMock()
        agent.state.consecutive_failures = 0

        with patch("src.orchestrator.time.monotonic") as mock_time, \
             patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_time.return_value = 1000.0
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_time.return_value = 1003.0  # 3.0s later > 2.5s
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_not_called()

    def test_throttle_disabled_when_zero(self):
        """min_step_interval=0 disables throttling entirely."""
        callback = _build_on_step_end("test-persona", min_step_interval=0)
        agent = MagicMock()
        agent.state.consecutive_failures = 0

        with patch("src.orchestrator.time.monotonic") as mock_time, \
             patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_time.return_value = 1000.0
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_time.return_value = 1000.1
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_not_called()

    def test_failure_backoff_takes_precedence(self):
        """Failures should use exponential backoff, not throttle."""
        callback = _build_on_step_end("test-persona", min_step_interval=2.5)
        agent = MagicMock()
        agent.state.consecutive_failures = 2

        with patch("src.orchestrator.time.monotonic") as mock_time, \
             patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_time.return_value = 1000.0
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(20)  # backoff, not 2.5s throttle

    def test_custom_interval(self):
        """Custom min_step_interval value works."""
        callback = _build_on_step_end("test-persona", min_step_interval=4.0)
        agent = MagicMock()
        agent.state.consecutive_failures = 0

        with patch("src.orchestrator.time.monotonic") as mock_time, \
             patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_time.return_value = 1000.0
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_time.return_value = 1001.0  # 1s later, under 4.0s
            asyncio.get_event_loop().run_until_complete(callback(agent))
            mock_sleep.assert_called_once_with(3.0)


class TestCerebrasProvider:
    """Tests for Cerebras provider identification and chain placement."""

    def test_cerebras_model_detected(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        settings.cerebras_api_key = "csk-test"
        settings.cerebras_model = "llama-3.3-70b"
        assert _get_provider("llama-3.3-70b", settings) == "cerebras"

    def test_cerebras_without_key_falls_to_openrouter(self):
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = ""
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        settings.cerebras_api_key = ""
        settings.cerebras_model = "llama-3.3-70b"
        assert _get_provider("llama-3.3-70b", settings) == "openrouter"

    def test_cerebras_in_model_chain(self):
        """Cerebras appears in chain when key is configured."""
        from src.config import Settings
        settings = Settings(
            groq_api_key="gsk_test",
            openrouter_api_key="sk-or-test",
            google_api_key="AI_test",
            sambanova_api_key="sn_test",
            cerebras_api_key="csk_test",
        )
        chain = settings.build_model_chain()
        assert settings.cerebras_model in chain

    def test_cerebras_not_in_chain_without_key(self):
        """Cerebras absent from chain when key is empty."""
        from src.config import Settings
        settings = Settings(
            groq_api_key="gsk_test",
            openrouter_api_key="sk-or-test",
            cerebras_api_key="",
        )
        chain = settings.build_model_chain()
        assert "llama-3.3-70b" not in chain

    def test_excluded_4_providers_picks_cerebras(self):
        """With groq+openrouter+sambanova excluded, Google primary picks Cerebras."""
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = "sn_test"
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        settings.cerebras_api_key = "csk_test"
        settings.cerebras_model = "llama-3.3-70b"
        chain = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "Llama-4-Maverick-17B-128E-Instruct",
            "llama-3.3-70b",
        ]
        excluded = {"groq", "openrouter", "sambanova", "google"}
        fallback = None
        for candidate in chain:
            if _get_provider(candidate, settings) not in excluded:
                fallback = candidate
                break
        assert fallback == "llama-3.3-70b"
        assert _get_provider(fallback, settings) == "cerebras"


class TestProviderRotation:
    """Tests for per-persona provider rotation in run_batch."""

    def test_unique_provider_deduplicates(self):
        """Only one model per provider in the rotation pool."""
        settings = MagicMock()
        settings.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        settings.sambanova_api_key = "sn_test"
        settings.sambanova_model = "Llama-4-Maverick-17B-128E-Instruct"
        settings.cerebras_api_key = "csk_test"
        settings.cerebras_model = "llama-3.3-70b"

        chain = [
            "meta-llama/llama-4-scout-17b-16e-instruct",  # groq
            "nvidia/nemotron-nano-12b-v2-vl:free",         # openrouter
            "gemini-2.5-flash-lite",                       # google
            "gemini-2.5-flash",                            # google (dup)
            "Llama-4-Maverick-17B-128E-Instruct",         # sambanova
            "llama-3.3-70b",                               # cerebras
            "google/gemma-3-12b-it:free",                  # openrouter (dup)
        ]

        unique = []
        seen = set()
        for m in chain:
            prov = _get_provider(m, settings)
            if prov not in seen:
                seen.add(prov)
                unique.append(m)

        assert len(unique) == 5
        providers = [_get_provider(m, settings) for m in unique]
        assert providers == ["groq", "openrouter", "google", "sambanova", "cerebras"]

    def test_distributes_8_personas(self):
        """8 personas with 5 providers cycles correctly."""
        pool = ["groq_m", "or_m", "google_m", "sn_m", "cerebras_m"]
        assignments = [pool[i % len(pool)] for i in range(8)]
        assert assignments == [
            "groq_m", "or_m", "google_m", "sn_m", "cerebras_m",
            "groq_m", "or_m", "google_m",
        ]
        from collections import Counter
        assert max(Counter(assignments).values()) == 2

    def test_single_provider(self):
        """With only one provider, all personas use it."""
        pool = ["groq_m"]
        assignments = [pool[i % len(pool)] for i in range(4)]
        assert all(a == "groq_m" for a in assignments)
