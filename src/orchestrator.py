"""Orchestrator for running persona test sessions."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from .agent_factory import create_agent
from .config import Settings, load_yaml_config
from .evaluator import evaluate_run
from .models import RunStatus, TestRunResult
from .persona_loader import PersonaDefinition, load_all_personas, load_persona
from .report_writer import write_report
from .screenshot_manager import save_screenshots

logger = logging.getLogger(__name__)


async def run_single_persona(
    persona: PersonaDefinition,
    settings: Settings,
    yaml_config: dict,
    batch_id: Optional[str] = None,
) -> TestRunResult:
    """Execute a full test run for a single persona."""
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)

    result = TestRunResult(
        run_id=f"{persona.id}-{run_id}",
        batch_id=batch_id,
        persona_id=persona.id,
        persona_name=persona.name,
        persona_language=persona.language,
        started_at=started_at,
        status=RunStatus.RUNNING,
        llm_model_used=settings.azure_openai_deployment,
        config_snapshot={
            "agent": yaml_config.get("agent", {}),
            "browser": yaml_config.get("browser", {}),
        },
    )

    agent = None
    browser = None

    try:
        logger.info(f"[{persona.id}] Creating agent...")
        agent, browser = create_agent(persona, settings, yaml_config)

        max_steps = yaml_config.get("agent", {}).get("max_steps", 60)
        timeout = yaml_config.get("orchestration", {}).get(
            "timeout_per_persona_seconds", 600
        )

        logger.info(f"[{persona.id}] Running agent (max_steps={max_steps}, timeout={timeout}s)...")
        history = await asyncio.wait_for(
            agent.run(max_steps=max_steps),
            timeout=timeout,
        )

        # Extract data from history
        result.total_steps = len(history.model_actions())
        result.agent_thoughts = [str(t) for t in history.model_thoughts()]
        result.conversation_log = history.extracted_content()

        # Save screenshots
        screenshots = history.screenshots()
        if screenshots:
            result.screenshot_paths = save_screenshots(screenshots, persona.id, run_id)
            logger.info(f"[{persona.id}] Saved {len(result.screenshot_paths)} screenshots")

        # Detect phases reached
        result.phases_reached = _detect_phases(history, persona)
        result.phase_furthest = result.phases_reached[-1] if result.phases_reached else "none"
        result.total_messages = _count_messages(history)

        # Detect widgets interacted with
        result.widgets_interacted = _detect_widgets(history)

        result.status = RunStatus.COMPLETED
        logger.info(
            f"[{persona.id}] Completed — {result.total_steps} steps, "
            f"phases: {result.phases_reached}"
        )

    except asyncio.TimeoutError:
        result.status = RunStatus.TIMEOUT
        result.error_message = f"Timed out after {yaml_config.get('orchestration', {}).get('timeout_per_persona_seconds', 600)}s"
        logger.error(f"[{persona.id}] Timeout")

    except Exception as e:
        result.status = RunStatus.FAILED
        result.error_message = str(e)
        logger.exception(f"[{persona.id}] Failed: {e}")

    finally:
        result.finished_at = datetime.now(timezone.utc)
        result.duration_seconds = (result.finished_at - started_at).total_seconds()

        if browser:
            try:
                await browser.close()
            except Exception:
                pass

    # Post-run evaluation
    if result.status == RunStatus.COMPLETED:
        try:
            logger.info(f"[{persona.id}] Running evaluation...")
            evaluation = await evaluate_run(result, persona, settings, yaml_config)
            result.merge_evaluation(evaluation)
        except Exception as e:
            logger.error(f"[{persona.id}] Evaluation failed: {e}")

    # Write JSON report
    try:
        report_path = write_report(result)
        logger.info(f"[{persona.id}] Report written to {report_path}")
    except Exception as e:
        logger.error(f"[{persona.id}] Report write failed: {e}")

    return result


async def run_batch(
    persona_ids: Optional[List[str]],
    settings: Settings,
    yaml_config: dict,
) -> List[TestRunResult]:
    """Run multiple personas sequentially."""
    if persona_ids:
        personas = [load_persona(pid) for pid in persona_ids]
    else:
        personas = load_all_personas()

    batch_id = f"batch-{uuid.uuid4().hex[:8]}"
    results = []

    logger.info(f"[batch:{batch_id}] Running {len(personas)} personas sequentially")

    for persona in personas:
        result = await run_single_persona(persona, settings, yaml_config, batch_id)
        results.append(result)

    return results


def _detect_phases(history, persona: PersonaDefinition) -> List[str]:
    """Detect which conversation phases were reached based on agent actions and content."""
    phases_reached = []
    actions = [str(a) for a in history.model_actions()]
    content = " ".join(str(c) for c in history.extracted_content() if c)
    actions_str = " ".join(actions).lower()
    content_lower = content.lower()

    for goal in persona.conversation_goals:
        phase = goal.phase

        # Greeting — always reached if agent sent any message
        if phase == "greeting" and (actions or content):
            phases_reached.append(phase)
            continue

        # Check if any widget interactions for this phase were performed
        if goal.widget_interactions:
            widget_keywords = []
            for wi in goal.widget_interactions:
                # Extract widget type name (before the colon)
                widget_type = wi.split(":")[0].strip().lower()
                widget_keywords.append(widget_type)
            if any(kw in actions_str or kw in content_lower for kw in widget_keywords):
                phases_reached.append(phase)
                continue

        # Check success indicator
        if goal.success_indicator:
            indicator_lower = goal.success_indicator.lower()
            keywords = [w for w in indicator_lower.split() if len(w) > 3]
            if any(kw in content_lower for kw in keywords):
                phases_reached.append(phase)
                continue

        # Check if phase-related words appear in conversation
        phase_keywords = {
            "preferences": ["préférence", "preference", "style", "intérêt", "interest"],
            "destination": ["destination", "ville", "city", "pays", "country"],
            "dates": ["date", "calendrier", "calendar", "juillet", "mars", "october"],
            "travelers": ["voyageur", "traveler", "adulte", "adult", "enfant", "child"],
            "logistics": ["aéroport", "airport", "vol", "flight", "aller-retour"],
            "accommodation": ["budget", "hôtel", "hotel", "hébergement"],
            "deep_conversation": [],  # Detected by message count
            "completion": ["récapitulatif", "recap", "recherche", "search"],
        }
        keywords = phase_keywords.get(phase, [])
        if keywords and any(kw in content_lower for kw in keywords):
            phases_reached.append(phase)

    return phases_reached


def _count_messages(history) -> int:
    """Count approximate number of messages exchanged."""
    content = history.extracted_content()
    return len([c for c in content if c]) if content else 0


def _detect_widgets(history) -> List[str]:
    """Detect widget types the agent interacted with."""
    widget_types = [
        "datepicker", "daterangepicker", "travelersselector",
        "triptypeconfirm", "preferencestyle", "preferenceinterests",
        "destinationsuggestions", "budgetrangeslider", "cityselector",
        "airportconfirmation",
    ]
    actions_str = " ".join(str(a) for a in history.model_actions()).lower()
    content_str = " ".join(str(c) for c in history.extracted_content() if c).lower()
    combined = actions_str + " " + content_str

    return [wt for wt in widget_types if wt in combined]
