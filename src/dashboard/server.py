"""FastAPI dashboard server — SSE + REST + static files."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..events import DashboardEvent, get_event_bus

logger = logging.getLogger(__name__)

app = FastAPI(title="Travliaq Testing Dashboard")

# Paths
STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
SCREENSHOTS_DIR = PROJECT_ROOT / "output" / "screenshots"

# ---------------------------------------------------------------------------
# In-memory batch state — updated by background event listener
# ---------------------------------------------------------------------------
_batch_state: Dict[str, Any] = {
    "batch_id": None,
    "status": "idle",  # idle | running | completed
    "personas": {},  # persona_id -> {status, stage, score, ...}
    "current_persona": None,
    "current_stage": None,
    "started_at": None,
    "events_log": [],  # last 200 events
}

# ---------------------------------------------------------------------------
# Debug log — concise human-readable lines for copy-paste debugging
# ---------------------------------------------------------------------------
_debug_log: List[str] = []
_DEBUG_LOG_MAX = 500


def _reset_state() -> None:
    _batch_state.update({
        "batch_id": None,
        "status": "idle",
        "personas": {},
        "current_persona": None,
        "current_stage": None,
        "started_at": None,
        "events_log": [],
    })
    _debug_log.clear()


def _dbg(line: str) -> None:
    """Append a line to the debug log (ring buffer)."""
    _debug_log.append(line)
    if len(_debug_log) > _DEBUG_LOG_MAX:
        del _debug_log[:len(_debug_log) - _DEBUG_LOG_MAX]


# ---------------------------------------------------------------------------
# Background task: listen to events and update _batch_state
# ---------------------------------------------------------------------------
async def _state_updater() -> None:
    """Subscribe to EventBus and keep _batch_state + _debug_log in sync."""
    bus = get_event_bus()
    while not bus:
        await asyncio.sleep(1.0)
        bus = get_event_bus()
    queue = bus.subscribe()
    while True:
        event: DashboardEvent = await queue.get()
        evt = event.model_dump(mode="json")
        _batch_state["events_log"].append(evt)
        if len(_batch_state["events_log"]) > 200:
            _batch_state["events_log"] = _batch_state["events_log"][-200:]

        etype = event.type.value
        ts = event.timestamp.strftime("%H:%M:%S")
        pid = event.persona_id or ""

        if etype == "batch_started":
            _batch_state["batch_id"] = event.batch_id
            _batch_state["status"] = "running"
            _batch_state["started_at"] = event.timestamp.isoformat()
            for p_id in event.data.get("persona_ids", []):
                _batch_state["personas"][p_id] = {"status": "pending", "stage": None}
            ids = event.data.get("persona_ids", [])
            chain = event.data.get("model_chain", [])
            _dbg(f"[{ts}] BATCH {event.batch_id} | {len(ids)} personas: {', '.join(ids)}")
            if chain:
                _dbg(f"[{ts}]   model chain: {' → '.join(chain)}")

        elif etype == "stage_health_check":
            _batch_state["current_persona"] = pid
            _batch_state["current_stage"] = event.stage
            if pid in _batch_state["personas"]:
                _batch_state["personas"][pid]["status"] = "running"
                _batch_state["personas"][pid]["stage"] = event.stage
            msg = event.data.get("message", "")
            _dbg(f"[{ts}] [{pid}] health check: {msg}")

        elif etype == "stage_create_agent":
            _batch_state["current_persona"] = pid
            _batch_state["current_stage"] = event.stage
            if pid in _batch_state["personas"]:
                _batch_state["personas"][pid]["status"] = "running"
                _batch_state["personas"][pid]["stage"] = event.stage
            primary = event.data.get("primary_model", "?")
            chain = event.data.get("model_chain", [])
            chain_str = f" | chain: {' → '.join(chain)}" if chain else ""
            _dbg(f"[{ts}] [{pid}] creating agent | primary={primary}{chain_str}")

        elif etype == "stage_run_agent":
            _batch_state["current_persona"] = pid
            _batch_state["current_stage"] = event.stage
            if pid in _batch_state["personas"]:
                _batch_state["personas"][pid]["status"] = "running"
                _batch_state["personas"][pid]["stage"] = event.stage
            msg = event.data.get("message", "")
            if event.data.get("fallback"):
                _dbg(f"[{ts}] [{pid}] FALLBACK: {msg}")
            elif "Rate limited" in msg or "waiting" in msg:
                _dbg(f"[{ts}] [{pid}] {msg}")
            elif msg == "Running agent":
                primary = event.data.get("primary_model", "?")
                fallback = event.data.get("fallback_model")
                fb_str = f" | fallback={fallback}" if fallback else ""
                _dbg(f"[{ts}] [{pid}] running agent | model={primary}{fb_str}")

        elif etype.startswith("stage_") and etype not in (
            "stage_health_check", "stage_create_agent", "stage_run_agent",
        ):
            _batch_state["current_persona"] = pid
            _batch_state["current_stage"] = event.stage
            if pid in _batch_state["personas"]:
                p = _batch_state["personas"][pid]
                # Don't override terminal statuses (failed/timeout/completed)
                if p.get("status") not in ("failed", "timeout", "completed"):
                    p["status"] = "running"
                p["stage"] = event.stage

        elif etype == "persona_completed":
            if pid in _batch_state["personas"]:
                d = event.data
                score = d.get("evaluation", {}).get("overall_score")
                status = d.get("status", "completed")
                _batch_state["personas"][pid].update({
                    "status": status,
                    "stage": "5/5",
                    "score": score,
                    "phase_furthest": d.get("execution", {}).get("phase_furthest"),
                    "duration": d.get("timing", {}).get("duration_seconds"),
                    "result_data": d,
                })
                dur = d.get("timing", {}).get("duration_seconds")
                steps = d.get("execution", {}).get("total_steps", "?")
                msgs = d.get("execution", {}).get("total_messages", "?")
                phase = d.get("execution", {}).get("phase_furthest", "?")
                model = d.get("meta", {}).get("llm_model_used", "?")
                strengths = d.get("evaluation", {}).get("strengths", [])
                frustr = d.get("evaluation", {}).get("frustration_points", [])
                _dbg(f"[{ts}] [{pid}] DONE: {status} | {steps} steps | {msgs} msgs | phase={phase} | score={score} | {dur:.0f}s | model={model}")
                if strengths:
                    _dbg(f"[{ts}] [{pid}]   strengths: {'; '.join(str(s) for s in strengths[:3])}")
                if frustr:
                    _dbg(f"[{ts}] [{pid}]   frustrations: {'; '.join(str(f) for f in frustr[:3])}")

        elif etype == "persona_failed":
            if pid in _batch_state["personas"]:
                p = _batch_state["personas"][pid]
                err = event.data.get("error", "")
                p.update({
                    "status": "failed",
                    "stage": event.data.get("stage"),
                    "error": err,
                })
                if not p.get("result_data"):
                    p["result_data"] = {
                        "status": "failed",
                        "error_message": err,
                        "execution": {
                            "phases_reached": [], "phase_furthest": None,
                            "total_steps": p.get("current_step"),
                            "total_messages": 0, "widgets_interacted": [],
                        },
                        "evaluation": None,
                        "timing": {"duration_seconds": None},
                        "logs": {}, "meta": {},
                    }
                _dbg(f"[{ts}] [{pid}] FAILED: {err[:150]}")

        elif etype == "persona_timeout":
            if pid in _batch_state["personas"]:
                p = _batch_state["personas"][pid]
                err = event.data.get("error", "")
                p.update({
                    "status": "timeout",
                    "error": err,
                })
                if not p.get("result_data"):
                    p["result_data"] = {
                        "status": "timeout",
                        "error_message": err,
                        "execution": {
                            "phases_reached": [], "phase_furthest": None,
                            "total_steps": p.get("current_step"),
                            "total_messages": 0, "widgets_interacted": [],
                        },
                        "evaluation": None,
                        "timing": {"duration_seconds": None},
                        "logs": {}, "meta": {},
                    }
                _dbg(f"[{ts}] [{pid}] TIMEOUT: {err[:100]}")

        elif etype == "agent_step":
            if pid in _batch_state["personas"]:
                p = _batch_state["personas"][pid]
                step_num = event.data.get("step_number", 0)
                p["current_step"] = step_num
                p["max_steps"] = event.data.get("max_steps")
                p["current_url"] = event.data.get("url")
                p["current_thinking"] = event.data.get("thinking")
                if "step_history" not in p:
                    p["step_history"] = []
                p["step_history"].append({
                    "step": step_num,
                    "actions": event.data.get("actions"),
                    "url": event.data.get("url"),
                    "thinking": event.data.get("thinking"),
                })
                if len(p["step_history"]) > 20:
                    p["step_history"] = p["step_history"][-20:]

                # Debug log: every 5th step, first 3 steps, or steps with thinking
                actions = event.data.get("actions", [])
                actions_str = ",".join(actions) if actions else "none"
                thinking = event.data.get("thinking", "")
                if step_num <= 3 or step_num % 5 == 0:
                    line = f"[{ts}] [{pid}] step {step_num}: {actions_str}"
                    if thinking:
                        line += f" | {thinking[:80]}"
                    _dbg(line)

        elif etype == "loop_detected":
            if pid in _batch_state["personas"]:
                p = _batch_state["personas"][pid]
                if "loops" not in p:
                    p["loops"] = []
                p["loops"].append({
                    "pattern_type": event.data.get("pattern_type"),
                    "pattern": event.data.get("pattern"),
                    "step_number": event.data.get("step_number"),
                })
                _dbg(f"[{ts}] [{pid}] LOOP: {event.data.get('pattern')} at step {event.data.get('step_number')}")

        elif etype == "batch_completed":
            _batch_state["status"] = "completed"
            _batch_state["current_persona"] = None
            _batch_state["current_stage"] = None
            # Build summary
            summary_parts = []
            for p_id, p_data in _batch_state["personas"].items():
                s = p_data.get("status", "?")
                sc = p_data.get("score")
                sc_str = f"{sc}" if sc else "N/A"
                summary_parts.append(f"{p_id}={s}({sc_str})")
            _dbg(f"[{ts}] BATCH DONE | {' | '.join(summary_parts)}")


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_state_updater())
    # Mount screenshots if directory exists
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/screenshots", StaticFiles(directory=str(SCREENSHOTS_DIR)), name="screenshots")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/stream")
async def event_stream(request: Request) -> StreamingResponse:
    """SSE endpoint — real-time events from the orchestrator."""

    async def generate():
        # Wait for event bus (may not exist yet if no batch running)
        bus = get_event_bus()
        while not bus:
            if await request.is_disconnected():
                return
            yield ": waiting for batch\n\n"
            await asyncio.sleep(2.0)
            bus = get_event_bus()

        queue = bus.subscribe()
        try:
            # Send current state snapshot for late-joining clients
            snapshot = json.dumps(
                {"type": "state_snapshot", "data": _batch_state}, default=str
            )
            yield f"data: {snapshot}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event: DashboardEvent = await asyncio.wait_for(
                        queue.get(), timeout=30.0
                    )
                    yield f"data: {event.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/batch/status")
async def batch_status() -> JSONResponse:
    """Current batch state (polling fallback)."""
    return JSONResponse(_batch_state)


@app.get("/api/results")
async def list_results() -> JSONResponse:
    """List completed result JSON files."""
    if not RESULTS_DIR.exists():
        return JSONResponse([])
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files[:50]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "filename": f.name,
                "persona_id": data.get("persona", {}).get("id"),
                "persona_name": data.get("persona", {}).get("name"),
                "status": data.get("status"),
                "score": data.get("evaluation", {}).get("overall_score"),
                "duration": data.get("timing", {}).get("duration_seconds"),
                "started_at": data.get("timing", {}).get("started_at"),
                "phase_furthest": data.get("execution", {}).get("phase_furthest"),
            })
        except Exception:
            continue
    return JSONResponse(results)


@app.get("/api/results/{filename}")
async def get_result(filename: str) -> JSONResponse:
    """Full result JSON for one run."""
    file_path = RESULTS_DIR / filename
    if not file_path.exists() or not file_path.suffix == ".json":
        return JSONResponse({"error": "not found"}, status_code=404)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/api/debug-log")
async def debug_log() -> PlainTextResponse:
    """Concise human-readable log for debugging. Updates during execution.

    Copy-paste this output to share with a developer for troubleshooting.
    Shows: batch progress, provider switches, errors, phase/score per persona.
    """
    if not _debug_log:
        return PlainTextResponse(
            "No logs yet. Start a batch run with --run-batch.\n",
            media_type="text/plain; charset=utf-8",
        )
    return PlainTextResponse(
        "\n".join(_debug_log) + "\n",
        media_type="text/plain; charset=utf-8",
    )


@app.get("/api/personas")
async def list_personas() -> JSONResponse:
    """List all available persona definitions (full data)."""
    from ..persona_loader import load_all_personas

    personas = load_all_personas()
    return JSONResponse([
        {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "language": p.language,
            "role": p.role,
            "personality_traits": p.personality_traits,
            "conversation_style": {
                "verbosity": p.conversation_style.verbosity,
                "formality": p.conversation_style.formality,
                "asks_questions": p.conversation_style.asks_questions,
                "expresses_frustration": p.conversation_style.expresses_frustration,
                "changes_mind": p.conversation_style.changes_mind,
            },
            "travel_profile": {
                "group_type": p.travel_profile.group_type,
                "travelers": p.travel_profile.travelers,
                "budget_range": p.travel_profile.budget_range,
                "preferred_destinations": p.travel_profile.preferred_destinations,
                "avoided": p.travel_profile.avoided,
                "trip_type": p.travel_profile.trip_type,
                "flexibility": p.travel_profile.flexibility,
                "preferred_month": p.travel_profile.preferred_month,
                "trip_duration": p.travel_profile.trip_duration,
            },
            "conversation_goals": [
                {
                    "phase": g.phase,
                    "goal": g.goal,
                    "example_message": g.example_message,
                    "widget_interactions": g.widget_interactions,
                    "min_messages": g.min_messages,
                    "success_indicator": g.success_indicator,
                }
                for g in p.conversation_goals
            ],
            "evaluation_weight_overrides": p.evaluation_weight_overrides,
        }
        for p in personas
    ])
