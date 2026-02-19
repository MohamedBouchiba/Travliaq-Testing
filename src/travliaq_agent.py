"""TravliaqAgent: browser-use Agent subclass that prevents early abandonment.

Overrides two internal methods to ensure the persona completes all conversation
phases — especially the final feedback phase — before calling 'done'.

The key insight is that browser-use's _inject_budget_warning (at 75% steps)
actively pushes the LLM to call 'done' early, and _force_done_after_last_step
swaps the action schema to DoneAgentOutput on the last step.  Both must be
counteracted so the agent prioritises feedback submission over early termination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from browser_use import Agent
from browser_use.agent.views import AgentStepInfo
from browser_use.llm.messages import UserMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseTracker — lightweight progress tracker updated by the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PhaseTracker:
    """Track which conversation phase the agent has reached."""

    total_phases: int
    language: str = "fr"
    current_phase_index: int = 0
    feedback_submitted: bool = False
    _phase_names: list[str] = field(default_factory=list)
    # Loop detection: recent action names tracked by orchestrator step callback
    _recent_actions: list[str] = field(default_factory=list)
    _stuck_action_count: int = 0  # increments while stuck; re-injects every 5

    @property
    def feedback_phase_index(self) -> int:
        return self.total_phases - 1  # send_logs is always last

    @property
    def is_feedback_reached(self) -> bool:
        return self.current_phase_index >= self.feedback_phase_index

    @property
    def phases_remaining(self) -> int:
        return max(0, self.total_phases - self.current_phase_index)

    def push_action(self, action_name: str) -> None:
        """Track recent action for loop detection (keep last 6)."""
        self._recent_actions.append(action_name)
        if len(self._recent_actions) > 6:
            self._recent_actions = self._recent_actions[-6:]

    @property
    def is_stuck_in_loop(self) -> bool:
        """True if the last 3+ actions are identical."""
        if len(self._recent_actions) < 3:
            return False
        return len(set(self._recent_actions[-3:])) == 1


# ---------------------------------------------------------------------------
# TravliaqAgent — subclass with anti-abandonment overrides
# ---------------------------------------------------------------------------

class TravliaqAgent(Agent):
    """Agent subclass that overrides budget warning and forced-done behaviour."""

    def __init__(self, *args, phase_tracker: PhaseTracker, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_tracker = phase_tracker

    # -- Override 1: counter the 75% budget warning --------------------------

    async def _inject_budget_warning(self, step_info: AgentStepInfo | None = None) -> None:
        # Let browser-use inject its standard budget warning first
        await super()._inject_budget_warning(step_info)

        if step_info is None:
            return

        steps_used = step_info.step_number + 1
        budget_ratio = steps_used / step_info.max_steps
        pt = self.phase_tracker

        # At >=75% budget — strong override if feedback not yet submitted
        if budget_ratio >= 0.75 and not pt.feedback_submitted:
            if pt.language == "fr":
                msg = (
                    f"⚠️ ANNULE L'AVERTISSEMENT DE BUDGET CI-DESSUS. "
                    f"Tu es à la phase {pt.current_phase_index + 1} sur {pt.total_phases}. "
                    f"Il te reste {pt.phases_remaining} phase(s) obligatoire(s). "
                    f"La phase {pt.total_phases} (envoi du feedback) est OBLIGATOIRE. "
                    f"NE FAIS PAS 'done' avant d'avoir soumis le feedback. "
                    f"PASSE IMMÉDIATEMENT à la phase feedback : scroll en bas du chat, "
                    f"clique sur 'Cliquez ici pour nous aider', écris un résumé de ton "
                    f"expérience, et soumets-le."
                )
            else:
                msg = (
                    f"⚠️ OVERRIDE THE BUDGET WARNING ABOVE. "
                    f"You are on phase {pt.current_phase_index + 1} of {pt.total_phases}. "
                    f"You have {pt.phases_remaining} mandatory phase(s) remaining. "
                    f"Phase {pt.total_phases} (feedback submission) is MANDATORY. "
                    f"Do NOT call 'done' before submitting feedback. "
                    f"SKIP to the feedback phase NOW: scroll to the bottom of the chat, "
                    f"click 'Cliquez ici pour nous aider', write a summary of your "
                    f"experience, and submit it."
                )
            logger.info(f"[TravliaqAgent] Budget override injected at step {steps_used}/{step_info.max_steps}")
            self._message_manager._add_context_message(UserMessage(content=msg))

        # At >=50% budget — gentle progress reminder
        elif budget_ratio >= 0.50 and not pt.is_feedback_reached:
            if pt.language == "fr":
                msg = (
                    f"RAPPEL DE PROGRESSION : Tu es à la phase {pt.current_phase_index + 1} "
                    f"sur {pt.total_phases}. Avance rapidement vers les phases restantes. "
                    f"La phase finale (feedback) est obligatoire — ne l'oublie pas."
                )
            else:
                msg = (
                    f"PROGRESS REMINDER: You are on phase {pt.current_phase_index + 1} "
                    f"of {pt.total_phases}. Move quickly through remaining phases. "
                    f"The final phase (feedback) is mandatory — don't forget it."
                )
            self._message_manager._add_context_message(UserMessage(content=msg))

        # -- Loop detection: break identical action loops ----------------------
        # Re-inject every 5 stuck actions, with escalation at 10+
        if pt.is_stuck_in_loop:
            pt._stuck_action_count += 1
            should_inject = (pt._stuck_action_count == 1 or pt._stuck_action_count % 5 == 0)
            if should_inject:
                repeated = pt._recent_actions[-1] if pt._recent_actions else "unknown"
                if pt._stuck_action_count >= 10:
                    # Escalated: tell agent to skip to next phase
                    if pt.language == "fr":
                        msg = (
                            f"⚠️ BOUCLE CRITIQUE — tu répètes '{repeated}' depuis {pt._stuck_action_count} actions. "
                            f"ABANDONNE complètement cette action. "
                            f"PASSE à la phase suivante immédiatement. "
                            f"CLIQUE sur le champ de texte et TAPE ton prochain message."
                        )
                    else:
                        msg = (
                            f"⚠️ CRITICAL LOOP — you've been repeating '{repeated}' for {pt._stuck_action_count} actions. "
                            f"ABANDON this action completely. "
                            f"SKIP to the next phase immediately. "
                            f"CLICK the text input and TYPE your next message."
                        )
                else:
                    # Standard loop-break message
                    if pt.language == "fr":
                        msg = (
                            f"⚠️ Tu es en BOUCLE — tu répètes l'action '{repeated}' depuis plusieurs étapes. "
                            f"La page EST CHARGÉE. Le chat est visible à gauche avec le champ de texte en bas. "
                            f"ARRÊTE de répéter cette action. CLIQUE sur le champ de texte et TAPE ton message MAINTENANT."
                        )
                    else:
                        msg = (
                            f"⚠️ You are in a LOOP — you've been repeating '{repeated}' for multiple steps. "
                            f"The page IS LOADED. The chat is visible on the left with the text input at the bottom. "
                            f"STOP repeating this action. CLICK the text input and TYPE your message NOW."
                        )
                logger.warning(f"[TravliaqAgent] Loop break injection #{pt._stuck_action_count}: '{repeated}'")
                self._message_manager._add_context_message(UserMessage(content=msg))
        elif not pt.is_stuck_in_loop:
            pt._stuck_action_count = 0  # reset when loop breaks

    # -- Override 2: prevent DoneAgentOutput swap on last step ---------------

    async def _force_done_after_last_step(self, step_info: AgentStepInfo | None = None) -> None:
        if step_info and step_info.is_last_step():
            if self.phase_tracker.feedback_submitted:
                # Feedback done — allow normal forced-done behaviour
                await super()._force_done_after_last_step(step_info)
            else:
                # Feedback NOT done — keep full action set so agent can click
                if self.phase_tracker.language == "fr":
                    msg = (
                        "C'est ta DERNIÈRE action. Tu N'AS PAS encore soumis le feedback. "
                        "Tu DOIS cliquer sur le lien 'Cliquez ici pour nous aider' en bas "
                        "du chat MAINTENANT. Si le lien n'est pas visible, scroll vers le "
                        "bas. C'est PLUS IMPORTANT que de terminer le test normalement."
                    )
                else:
                    msg = (
                        "This is your LAST action. You have NOT submitted feedback yet. "
                        "You MUST click the 'Cliquez ici pour nous aider' link at the "
                        "bottom of the chat NOW. If the link is not visible, scroll down. "
                        "This is MORE IMPORTANT than finishing the test normally."
                    )
                logger.info("[TravliaqAgent] Last step — feedback not submitted, keeping full action set")
                self._message_manager._add_context_message(UserMessage(content=msg))
                # Intentionally NOT calling super() — keep full AgentOutput, not DoneAgentOutput
