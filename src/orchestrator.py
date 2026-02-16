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


def _log_banner(persona_id: str, message: str) -> None:
    """Print a visible banner in logs."""
    logger.info("=" * 60)
    logger.info(f"[{persona_id}] {message}")
    logger.info("=" * 60)


async def run_single_persona(
    persona: PersonaDefinition,
    settings: Settings,
    yaml_config: dict,
    batch_id: Optional[str] = None,
) -> TestRunResult:
    """Execute a full test run for a single persona."""
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)

    _log_banner(persona.id, f"STARTING RUN — {persona.name} ({persona.language})")
    logger.info(f"[{persona.id}] Run ID: {persona.id}-{run_id}")
    logger.info(f"[{persona.id}] Role: {persona.role[:100]}...")
    logger.info(f"[{persona.id}] Group: {persona.travel_profile.group_type}")
    logger.info(f"[{persona.id}] Budget: {persona.travel_profile.budget_range}")
    logger.info(f"[{persona.id}] Phases planned: {[g.phase for g in persona.conversation_goals]}")
    logger.info(f"[{persona.id}] LLM: {settings.azure_openai_deployment}")

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
        # --- Step 1: Create agent ---
        logger.info(f"[{persona.id}] [1/5] Creating browser-use agent...")
        logger.info(f"[{persona.id}]   Browser headless: {yaml_config.get('browser', {}).get('headless', False)}")
        logger.info(f"[{persona.id}]   Window: {yaml_config.get('browser', {}).get('window_width', 1440)}x{yaml_config.get('browser', {}).get('window_height', 900)}")
        agent, browser = create_agent(persona, settings, yaml_config)
        logger.info(f"[{persona.id}]   Agent created OK")

        max_steps = yaml_config.get("agent", {}).get("max_steps", 60)
        timeout = yaml_config.get("orchestration", {}).get(
            "timeout_per_persona_seconds", 600
        )

        # --- Step 2: Run agent ---
        logger.info(f"[{persona.id}] [2/5] Running agent (max_steps={max_steps}, timeout={timeout}s)...")
        logger.info(f"[{persona.id}]   Target URL: {yaml_config.get('target', {}).get('planner_url_clean', 'N/A')}")
        logger.info(f"[{persona.id}]   Waiting for agent to navigate, chat, and interact with widgets...")

        history = await asyncio.wait_for(
            agent.run(max_steps=max_steps),
            timeout=timeout,
        )

        # --- Step 3: Extract results ---
        logger.info(f"[{persona.id}] [3/5] Agent finished. Extracting results...")

        result.total_steps = len(history.model_actions())
        logger.info(f"[{persona.id}]   Total steps taken: {result.total_steps}")

        result.agent_thoughts = [str(t) for t in history.model_thoughts()]
        logger.info(f"[{persona.id}]   Agent thoughts captured: {len(result.agent_thoughts)}")

        result.conversation_log = history.extracted_content()
        logger.info(f"[{persona.id}]   Conversation entries: {len(result.conversation_log) if result.conversation_log else 0}")

        # Log a sample of what the agent did
        actions = history.action_names()
        if actions:
            logger.info(f"[{persona.id}]   Actions taken: {actions[:15]}{'...' if len(actions) > 15 else ''}")

        urls = history.urls()
        if urls:
            logger.info(f"[{persona.id}]   URLs visited: {urls[:5]}")

        if history.has_errors():
            errors = [str(e) for e in history.errors() if e]
            logger.warning(f"[{persona.id}]   Agent errors: {errors[:5]}")

        # Log first few thoughts for visibility
        if result.agent_thoughts:
            logger.info(f"[{persona.id}]   --- First 3 agent thoughts ---")
            for i, thought in enumerate(result.agent_thoughts[:3]):
                logger.info(f"[{persona.id}]   Thought {i+1}: {thought[:200]}...")

        # Save screenshots
        screenshots = history.screenshots()
        if screenshots:
            result.screenshot_paths = save_screenshots(screenshots, persona.id, run_id)
            logger.info(f"[{persona.id}]   Screenshots saved: {len(result.screenshot_paths)}")
        else:
            logger.info(f"[{persona.id}]   No screenshots captured")

        # Detect phases reached
        result.phases_reached = _detect_phases(history, persona)
        result.phase_furthest = result.phases_reached[-1] if result.phases_reached else "none"
        result.total_messages = _count_messages(history)
        result.widgets_interacted = _detect_widgets(history)

        logger.info(f"[{persona.id}]   Phases reached: {result.phases_reached}")
        logger.info(f"[{persona.id}]   Furthest phase: {result.phase_furthest}")
        logger.info(f"[{persona.id}]   Messages exchanged: {result.total_messages}")
        logger.info(f"[{persona.id}]   Widgets interacted: {result.widgets_interacted}")

        result.status = RunStatus.COMPLETED
        _log_banner(persona.id, f"AGENT COMPLETED — {result.total_steps} steps, {result.total_messages} messages")

    except asyncio.TimeoutError:
        result.status = RunStatus.TIMEOUT
        timeout_val = yaml_config.get('orchestration', {}).get('timeout_per_persona_seconds', 600)
        result.error_message = f"Timed out after {timeout_val}s"
        _log_banner(persona.id, f"TIMEOUT after {timeout_val}s")

    except Exception as e:
        result.status = RunStatus.FAILED
        result.error_message = str(e)
        logger.error(f"[{persona.id}] FAILED with error: {type(e).__name__}: {e}")
        logger.exception(f"[{persona.id}] Full traceback:")

    finally:
        result.finished_at = datetime.now(timezone.utc)
        result.duration_seconds = (result.finished_at - started_at).total_seconds()
        logger.info(f"[{persona.id}]   Duration: {result.duration_seconds:.1f}s")

        if agent:
            try:
                await agent.close()
                logger.info(f"[{persona.id}]   Agent + browser closed OK")
            except Exception as e:
                logger.warning(f"[{persona.id}]   Agent close error: {e}")

    # --- Step 4: Evaluate ---
    if result.status == RunStatus.COMPLETED:
        try:
            logger.info(f"[{persona.id}] [4/5] Running LLM evaluation (9 axes)...")
            evaluation = await evaluate_run(result, persona, settings, yaml_config)
            result.merge_evaluation(evaluation)
            logger.info(f"[{persona.id}]   Overall score: {result.score_overall}/10")
            logger.info(f"[{persona.id}]   Fluidity: {result.scores.fluidity} | Relevance: {result.scores.relevance}")
            logger.info(f"[{persona.id}]   Widget usability: {result.scores.widget_usability} | Memory: {result.scores.conversation_memory}")
            logger.info(f"[{persona.id}]   Strengths: {result.strengths}")
            logger.info(f"[{persona.id}]   Frustrations: {result.frustration_points}")
        except Exception as e:
            logger.error(f"[{persona.id}]   Evaluation FAILED: {type(e).__name__}: {e}")
    else:
        logger.info(f"[{persona.id}] [4/5] Skipping evaluation (status={result.status.value})")

    # --- Step 5: Write report ---
    try:
        logger.info(f"[{persona.id}] [5/5] Writing JSON report...")
        report_path = write_report(result)
        logger.info(f"[{persona.id}]   Report: {report_path}")
    except Exception as e:
        logger.error(f"[{persona.id}]   Report write FAILED: {e}")

    _log_banner(persona.id, f"RUN FINISHED — status={result.status.value}, score={result.score_overall or 'N/A'}")
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

    logger.info("=" * 60)
    logger.info(f"BATCH START — {batch_id}")
    logger.info(f"Personas: {[p.id for p in personas]}")
    logger.info(f"Total: {len(personas)} personas to run sequentially")
    logger.info("=" * 60)

    for i, persona in enumerate(personas, 1):
        logger.info(f"\n--- Batch progress: {i}/{len(personas)} ---")
        result = await run_single_persona(persona, settings, yaml_config, batch_id)
        results.append(result)
        logger.info(f"--- {persona.id}: {result.status.value} (score: {result.score_overall or 'N/A'}) ---\n")

    # Batch summary
    logger.info("=" * 60)
    logger.info(f"BATCH COMPLETE — {batch_id}")
    for r in results:
        logger.info(f"  {r.persona_id}: {r.status.value} | score={r.score_overall or 'N/A'} | phase={r.phase_furthest or 'none'} | {r.duration_seconds:.0f}s")
    completed = [r for r in results if r.score_overall is not None]
    if completed:
        avg = sum(r.score_overall for r in completed) / len(completed)
        logger.info(f"  Average score: {avg:.1f}/10")
    logger.info("=" * 60)

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
            "deep_conversation": [],
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
