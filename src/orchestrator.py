"""Orchestrator for running persona test sessions."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .agent_factory import _get_provider, create_agent, create_llm_for_model
from .config import Settings
from .evaluator import evaluate_run
from .events import DashboardEvent, EventType, get_event_bus
from .loop_detector import LoopDetector
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=10, min=10, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _check_llm_health(settings: Settings) -> bool:
    """Lightweight LLM connectivity check with retry.

    Checks the primary provider first (Groq > OpenRouter > Google).
    """
    if settings.groq_api_key:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True

    if settings.openrouter_api_key and settings.openrouter_model:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://travliaq.com",
                "X-Title": "Travliaq-Testing",
            },
        )
        client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True

    if settings.google_api_key:
        from google import genai
        client = genai.Client(api_key=settings.google_api_key)
        client.models.generate_content(
            model=settings.google_model,
            contents="ping",
            config={"max_output_tokens": 5},
        )
        return True

    raise RuntimeError("No LLM API keys configured")


async def run_single_persona(
    persona: PersonaDefinition,
    settings: Settings,
    yaml_config: dict,
    batch_id: Optional[str] = None,
) -> TestRunResult:
    """Execute a full test run for a single persona."""
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)

    model_chain = settings.build_model_chain()
    primary_model = model_chain[0] if model_chain else "N/A"

    _log_banner(persona.id, f"STARTING RUN — {persona.name} ({persona.language})")
    logger.info(f"[{persona.id}] Run ID: {persona.id}-{run_id}")
    logger.info(f"[{persona.id}] Role: {persona.role[:100]}...")
    logger.info(f"[{persona.id}] Group: {persona.travel_profile.group_type}")
    logger.info(f"[{persona.id}] Budget: {persona.travel_profile.budget_range}")
    logger.info(f"[{persona.id}] Phases planned: {[g.phase for g in persona.conversation_goals]}")
    logger.info(f"[{persona.id}] Model chain ({len(model_chain)}): {' → '.join(model_chain[:4])}{'...' if len(model_chain) > 4 else ''}")

    result = TestRunResult(
        run_id=f"{persona.id}-{run_id}",
        batch_id=batch_id,
        persona_id=persona.id,
        persona_name=persona.name,
        persona_language=persona.language,
        started_at=started_at,
        status=RunStatus.RUNNING,
        llm_model_used=primary_model,
        config_snapshot={
            "agent": yaml_config.get("agent", {}),
            "browser": yaml_config.get("browser", {}),
        },
    )

    agent = None
    browser = None

    bus = get_event_bus()

    try:
        # --- Step 0: LLM health check ---
        logger.info(f"[{persona.id}] [0/5] Checking LLM connectivity...")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.STAGE_HEALTH_CHECK, persona_id=persona.id,
                batch_id=batch_id, stage="0/5",
                data={"message": "Checking LLM connectivity"},
            ))
        try:
            _check_llm_health(settings)
            logger.info(f"[{persona.id}]   LLM health check PASSED")
        except Exception as e:
            logger.error(f"[{persona.id}]   LLM health check FAILED after retries: {e}")
            result.status = RunStatus.FAILED
            result.error_message = f"LLM health check failed: {e}"
            if bus:
                await bus.emit(DashboardEvent(
                    type=EventType.PERSONA_FAILED, persona_id=persona.id,
                    batch_id=batch_id, data={"error": str(e), "stage": "0/5"},
                ))
            return result

        # --- Step 1: Create agent ---
        logger.info(f"[{persona.id}] [1/5] Creating browser-use agent...")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.STAGE_CREATE_AGENT, persona_id=persona.id,
                batch_id=batch_id, stage="1/5",
                data={"message": "Creating browser-use agent"},
            ))
        logger.info(f"[{persona.id}]   Browser headless: {yaml_config.get('browser', {}).get('headless', False)}")
        logger.info(f"[{persona.id}]   Window: {yaml_config.get('browser', {}).get('window_width', 1440)}x{yaml_config.get('browser', {}).get('window_height', 900)}")

        max_steps = yaml_config.get("agent", {}).get("max_steps", 60)
        timeout = yaml_config.get("orchestration", {}).get(
            "timeout_per_persona_seconds", 600
        )

        # Loop detector + step callback
        loop_detector = LoopDetector()

        async def _on_step(browser_state, agent_output, step_number):
            """Step callback: emit AGENT_STEP + loop detection."""
            action_names = []
            if agent_output and hasattr(agent_output, "action") and agent_output.action:
                for act in agent_output.action:
                    try:
                        act_dict = act.model_dump(exclude_unset=True)
                        for key in act_dict:
                            action_names.append(key)
                    except Exception:
                        action_names.append(str(type(act).__name__))

            url = browser_state.url if browser_state and hasattr(browser_state, "url") else None
            thinking = None
            if agent_output and hasattr(agent_output, "thinking") and agent_output.thinking:
                thinking = agent_output.thinking[:200]

            if bus:
                await bus.emit(DashboardEvent(
                    type=EventType.AGENT_STEP, persona_id=persona.id,
                    batch_id=batch_id, stage="2/5",
                    data={
                        "message": f"Step {step_number}/{max_steps}",
                        "step_number": step_number,
                        "max_steps": max_steps,
                        "actions": action_names,
                        "url": url,
                        "thinking": thinking,
                    },
                ))

            for action_name in action_names:
                detection = loop_detector.push(action_name)
                if detection.detected:
                    logger.warning(f"[{persona.id}] LOOP DETECTED at step {step_number}: {detection.pattern}")
                    if bus:
                        await bus.emit(DashboardEvent(
                            type=EventType.LOOP_DETECTED, persona_id=persona.id,
                            batch_id=batch_id, stage="2/5",
                            data={
                                "message": f"Loop detected: {detection.pattern}",
                                "pattern_type": detection.pattern_type,
                                "pattern": detection.pattern,
                                "step_number": step_number,
                                "window": detection.window,
                            },
                        ))

            # --- Phase tracking for TravliaqAgent ---
            _update_phase_tracker(phase_tracker, persona, thinking, action_names)

        agent, browser, fallback_used, phase_tracker = create_agent(persona, settings, yaml_config, step_callback=_on_step)
        logger.info(f"[{persona.id}]   Agent created OK (primary: {primary_model}, fallback: {fallback_used or 'none'})")

        # --- Step 2: Run agent ---
        logger.info(f"[{persona.id}] [2/5] Running agent (max_steps={max_steps}, timeout={timeout}s)...")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.STAGE_RUN_AGENT, persona_id=persona.id,
                batch_id=batch_id, stage="2/5",
                data={"max_steps": max_steps, "timeout": timeout, "message": "Running agent"},
            ))
        logger.info(f"[{persona.id}]   Target URL: {yaml_config.get('target', {}).get('planner_url_clean', 'N/A')}")
        logger.info(f"[{persona.id}]   Waiting for agent to navigate, chat, and interact with widgets...")

        history = await asyncio.wait_for(
            agent.run(max_steps=max_steps, on_step_end=_build_on_step_end(persona.id)),
            timeout=timeout,
        )

        # --- Post-run: detect total model failure (browser-use swallows 404s) ---
        if len(history.model_actions()) == 0 and history.has_errors():
            all_errors = [str(e) for e in history.errors() if e]
            first_error = all_errors[0] if all_errors else ""
            model_kw = ["404", "model", "endpoint", "vision", "image input", "rate limit", "modelprovider", "json_invalid"]
            is_model_failure = any(kw in first_error.lower() for kw in model_kw)
            is_rate_limit = "rate limit" in first_error.lower() or "429" in first_error.lower()

            # If rate-limited, cool down before trying backup models
            if is_rate_limit:
                rl_cooldown = yaml_config.get("orchestration", {}).get(
                    "rate_limit_cooldown_seconds", 60
                )
                logger.info(f"[{persona.id}]   Rate limit detected — cooling down {rl_cooldown}s before backup models...")
                if bus:
                    await bus.emit(DashboardEvent(
                        type=EventType.STAGE_RUN_AGENT, persona_id=persona.id,
                        batch_id=batch_id, stage="2/5",
                        data={"message": f"Rate limited — waiting {rl_cooldown}s before retry"},
                    ))
                await asyncio.sleep(rl_cooldown)

            # Skip models already tried AND models from rate-limited providers
            tried = {primary_model, fallback_used} - {None}
            rate_limited_providers = set()
            if is_rate_limit:
                for tried_model in tried:
                    rate_limited_providers.add(_get_provider(tried_model, settings))
                logger.info(f"[{persona.id}]   Rate-limited providers: {rate_limited_providers}")

            remaining_models = [
                m for m in model_chain
                if m not in tried and _get_provider(m, settings) not in rate_limited_providers
            ]
            if is_model_failure and remaining_models:
                logger.warning(f"[{persona.id}]   Primary + fallback returned 0 actions ({len(all_errors)} errors)")
                logger.warning(f"[{persona.id}]   First error: {first_error[:200]}")
                logger.info(f"[{persona.id}]   {len(remaining_models)} backup model(s) available — starting fallback chain")

                fallback_succeeded = False
                for i, backup_model in enumerate(remaining_models, 1):
                    # Skip if provider was rate-limited by an earlier backup
                    if _get_provider(backup_model, settings) in rate_limited_providers:
                        logger.info(f"[{persona.id}]   Skipping {backup_model} — provider rate-limited")
                        continue
                    logger.info(f"[{persona.id}]   Trying backup model {i}/{len(remaining_models)}: {backup_model}")

                    # Close previous agent/browser
                    if agent:
                        try:
                            await agent.close()
                        except Exception:
                            pass
                        agent = None
                        browser = None

                    # Reset loop detector for clean retry
                    loop_detector = LoopDetector()
                    result.llm_model_used = backup_model

                    if bus:
                        await bus.emit(DashboardEvent(
                            type=EventType.STAGE_RUN_AGENT, persona_id=persona.id,
                            batch_id=batch_id, stage="2/5",
                            data={
                                "max_steps": max_steps, "timeout": timeout,
                                "message": f"Fallback {i}/{len(remaining_models)}: {backup_model}",
                                "fallback": True, "fallback_index": i,
                            },
                        ))

                    agent, browser, _, phase_tracker = create_agent(
                        persona, settings, yaml_config,
                        step_callback=_on_step,
                        model_override=backup_model,
                    )
                    logger.info(f"[{persona.id}]   Backup agent created OK ({backup_model})")

                    history = await asyncio.wait_for(
                        agent.run(max_steps=max_steps, on_step_end=_build_on_step_end(persona.id)),
                        timeout=timeout,
                    )

                    # Check if this backup also failed completely
                    if len(history.model_actions()) == 0 and history.has_errors():
                        backup_errors = [str(e) for e in history.errors() if e]
                        logger.warning(f"[{persona.id}]   Backup {i}/{len(remaining_models)} also failed: {backup_errors[0][:150] if backup_errors else 'unknown'}")
                        # Track rate-limited provider to skip remaining same-provider models
                        backup_err_lower = backup_errors[0].lower() if backup_errors else ""
                        if "rate limit" in backup_err_lower or "429" in backup_err_lower:
                            blocked = _get_provider(backup_model, settings)
                            rate_limited_providers.add(blocked)
                            logger.info(f"[{persona.id}]   Provider '{blocked}' rate-limited — skipping remaining models from it")
                        continue  # try next backup

                    # This backup worked
                    fallback_succeeded = True
                    logger.info(f"[{persona.id}]   Backup model {backup_model} succeeded")
                    break

                if not fallback_succeeded:
                    result.status = RunStatus.FAILED
                    result.error_message = f"All {len(model_chain)} models failed: {first_error[:200]}"
                    logger.error(f"[{persona.id}]   FAILED: all models exhausted")
                    if bus:
                        await bus.emit(DashboardEvent(
                            type=EventType.PERSONA_FAILED, persona_id=persona.id,
                            batch_id=batch_id, data={"error": result.error_message, "stage": "2/5"},
                        ))
                    raise RuntimeError(result.error_message)
            else:
                # Not a model error or no backups — mark as FAILED
                result.status = RunStatus.FAILED
                result.error_message = f"Agent returned 0 actions: {first_error[:300]}"
                logger.error(f"[{persona.id}]   FAILED: 0 actions, error: {first_error[:200]}")
                if bus:
                    await bus.emit(DashboardEvent(
                        type=EventType.PERSONA_FAILED, persona_id=persona.id,
                        batch_id=batch_id, data={"error": result.error_message, "stage": "2/5"},
                    ))
                raise RuntimeError(result.error_message)

        # --- Step 3: Extract results ---
        logger.info(f"[{persona.id}] [3/5] Agent finished. Extracting results...")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.STAGE_EXTRACT_RESULTS, persona_id=persona.id,
                batch_id=batch_id, stage="3/5",
                data={"message": "Extracting results"},
            ))

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
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.PERSONA_TIMEOUT, persona_id=persona.id,
                batch_id=batch_id, data={"error": result.error_message},
            ))

    except Exception as e:
        result.status = RunStatus.FAILED
        result.error_message = str(e)
        logger.error(f"[{persona.id}] FAILED with error: {type(e).__name__}: {e}")
        logger.exception(f"[{persona.id}] Full traceback:")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.PERSONA_FAILED, persona_id=persona.id,
                batch_id=batch_id, data={"error": str(e)},
            ))

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
    should_evaluate = (
        result.status == RunStatus.COMPLETED
        or (result.status == RunStatus.TIMEOUT and result.phases_reached)
    )
    if should_evaluate:
        try:
            label = "partial (timeout)" if result.status == RunStatus.TIMEOUT else "full"
            logger.info(f"[{persona.id}] [4/5] Running {label} LLM evaluation (9 axes)...")
            if bus:
                await bus.emit(DashboardEvent(
                    type=EventType.STAGE_EVALUATE, persona_id=persona.id,
                    batch_id=batch_id, stage="4/5",
                    data={"message": f"Running {label} evaluation"},
                ))
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
        logger.info(f"[{persona.id}] [4/5] Skipping evaluation (status={result.status.value}, phases={result.phases_reached})")

    # --- Step 5: Write report ---
    try:
        logger.info(f"[{persona.id}] [5/5] Writing JSON report...")
        if bus:
            await bus.emit(DashboardEvent(
                type=EventType.STAGE_WRITE_REPORT, persona_id=persona.id,
                batch_id=batch_id, stage="5/5",
                data={"message": "Writing JSON report"},
            ))
        report_path = write_report(result)
        logger.info(f"[{persona.id}]   Report: {report_path}")
    except Exception as e:
        logger.error(f"[{persona.id}]   Report write FAILED: {e}")

    # Emit final persona result event (completed + timeout with partial data)
    if bus and result.status in (RunStatus.COMPLETED, RunStatus.TIMEOUT):
        await bus.emit(DashboardEvent(
            type=EventType.PERSONA_COMPLETED, persona_id=persona.id,
            batch_id=batch_id, data=result.to_json_dict(),
        ))

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

    bus = get_event_bus()
    if bus:
        await bus.emit(DashboardEvent(
            type=EventType.BATCH_STARTED, batch_id=batch_id,
            data={
                "persona_ids": [p.id for p in personas],
                "total": len(personas),
            },
        ))

    cooldown = yaml_config.get("orchestration", {}).get(
        "cooldown_between_personas_seconds", 30
    )

    for i, persona in enumerate(personas, 1):
        logger.info(f"\n--- Batch progress: {i}/{len(personas)} ---")
        result = await run_single_persona(persona, settings, yaml_config, batch_id)
        results.append(result)
        logger.info(f"--- {persona.id}: {result.status.value} (score: {result.score_overall or 'N/A'}) ---\n")

        # Cooldown between personas (skip after last one)
        if i < len(personas) and cooldown > 0:
            logger.info(f"Cooling down {cooldown}s before next persona...")
            await asyncio.sleep(cooldown)

    # Batch summary
    logger.info("=" * 60)
    logger.info(f"BATCH COMPLETE — {batch_id}")
    for r in results:
        logger.info(f"  {r.persona_id}: {r.status.value} | score={r.score_overall or 'N/A'} | phase={r.phase_furthest or 'none'} | {r.duration_seconds:.0f}s")
    completed = [r for r in results if r.score_overall is not None]
    avg = None
    if completed:
        avg = sum(r.score_overall for r in completed) / len(completed)
        logger.info(f"  Average score: {avg:.1f}/10")
    logger.info("=" * 60)

    if bus:
        await bus.emit(DashboardEvent(
            type=EventType.BATCH_COMPLETED, batch_id=batch_id,
            data={
                "total": len(results),
                "completed": len(completed),
                "average_score": round(avg, 1) if avg else None,
                "summary": [
                    {"persona_id": r.persona_id, "status": r.status.value,
                     "score": r.score_overall, "phase": r.phase_furthest,
                     "duration": r.duration_seconds}
                    for r in results
                ],
            },
        ))

    return results


def _build_on_step_end(persona_id: str):
    """Create an on_step_end callback with exponential backoff on failures.

    browser-use has zero internal sleep between consecutive failures. This
    hook fires after each step (including failed ones) and injects a delay
    proportional to the number of consecutive failures, giving rate-limited
    APIs time to recover before the next attempt.
    """
    async def _on_step_end(agent):
        failures = agent.state.consecutive_failures
        if failures > 0:
            delay = min(10 * (2 ** (failures - 1)), 60)  # 10s, 20s, 40s, cap 60s
            logger.info(f"[{persona_id}] Backoff: {failures} consecutive failure(s) — waiting {delay}s")
            await asyncio.sleep(delay)
    return _on_step_end


def _update_phase_tracker(phase_tracker, persona, thinking: str | None, action_names: list[str]) -> None:
    """Update phase_tracker based on the agent's current thinking and actions.

    Uses keyword matching against persona conversation_goals to estimate
    which phase the agent has reached, and detects feedback submission.
    """
    if phase_tracker is None:
        return

    # Track actions for loop detection
    for action_name in action_names:
        phase_tracker.push_action(action_name)

    combined = " ".join(action_names).lower()
    if thinking:
        combined += " " + thinking.lower()

    # Phase keyword map — same approach as _detect_phases but incremental
    phase_keywords = {
        "greeting": ["greeting", "bonjour", "hello", "salut"],
        "preferences": ["préférence", "preference", "style", "intérêt", "interest", "slider"],
        "destination": ["destination", "ville", "city", "pays", "country", "suggestion"],
        "dates": ["date", "calendrier", "calendar", "datepicker"],
        "travelers": ["voyageur", "traveler", "adulte", "adult", "enfant", "child"],
        "logistics": ["aéroport", "airport", "vol", "flight", "aller-retour", "trip_type"],
        "deep_conversation": ["question", "précis", "detail", "expliqu"],
        "completion": ["récapitulatif", "recap", "recherche", "search", "result"],
        "send_logs": ["nous aider", "feedback", "popup", "résumé", "summary"],
    }

    # Walk persona goals and find the highest matching phase index
    for i, goal in enumerate(persona.conversation_goals):
        if i <= phase_tracker.current_phase_index:
            continue  # already past this phase
        keywords = phase_keywords.get(goal.phase, [])
        if keywords and any(kw in combined for kw in keywords):
            phase_tracker.current_phase_index = i

    # Detect feedback submission
    feedback_keywords = ["nous aider", "feedback", "popup", "soumis", "submitted", "submit"]
    if any(kw in combined for kw in feedback_keywords):
        if "click" in combined or "cliqu" in combined or "input" in combined:
            phase_tracker.feedback_submitted = True


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
