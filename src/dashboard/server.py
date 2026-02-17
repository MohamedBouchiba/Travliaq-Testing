"""FastAPI dashboard server — SSE + REST + static files."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


# ---------------------------------------------------------------------------
# Background task: listen to events and update _batch_state
# ---------------------------------------------------------------------------
async def _state_updater() -> None:
    """Subscribe to EventBus and keep _batch_state in sync."""
    bus = get_event_bus()
    if not bus:
        return
    queue = bus.subscribe()
    while True:
        event: DashboardEvent = await queue.get()
        evt = event.model_dump(mode="json")
        _batch_state["events_log"].append(evt)
        if len(_batch_state["events_log"]) > 200:
            _batch_state["events_log"] = _batch_state["events_log"][-200:]

        etype = event.type.value

        if etype == "batch_started":
            _batch_state["batch_id"] = event.batch_id
            _batch_state["status"] = "running"
            _batch_state["started_at"] = event.timestamp.isoformat()
            for pid in event.data.get("persona_ids", []):
                _batch_state["personas"][pid] = {"status": "pending", "stage": None}

        elif etype.startswith("stage_"):
            _batch_state["current_persona"] = event.persona_id
            _batch_state["current_stage"] = event.stage
            if event.persona_id and event.persona_id in _batch_state["personas"]:
                _batch_state["personas"][event.persona_id]["status"] = "running"
                _batch_state["personas"][event.persona_id]["stage"] = event.stage

        elif etype == "persona_completed":
            if event.persona_id and event.persona_id in _batch_state["personas"]:
                score = event.data.get("evaluation", {}).get("overall_score")
                _batch_state["personas"][event.persona_id] = {
                    "status": "completed",
                    "stage": "5/5",
                    "score": score,
                    "phase_furthest": event.data.get("execution", {}).get("phase_furthest"),
                    "duration": event.data.get("timing", {}).get("duration_seconds"),
                }

        elif etype == "persona_failed":
            if event.persona_id and event.persona_id in _batch_state["personas"]:
                _batch_state["personas"][event.persona_id] = {
                    "status": "failed",
                    "stage": event.data.get("stage"),
                    "error": event.data.get("error"),
                }

        elif etype == "persona_timeout":
            if event.persona_id and event.persona_id in _batch_state["personas"]:
                _batch_state["personas"][event.persona_id] = {
                    "status": "timeout",
                    "stage": _batch_state["personas"][event.persona_id].get("stage"),
                    "error": event.data.get("error"),
                }

        elif etype == "batch_completed":
            _batch_state["status"] = "completed"
            _batch_state["current_persona"] = None
            _batch_state["current_stage"] = None


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
    bus = get_event_bus()
    if not bus:
        return JSONResponse({"error": "No active event bus"}, status_code=503)

    queue = bus.subscribe()

    async def generate():
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


@app.get("/api/personas")
async def list_personas() -> JSONResponse:
    """List all available persona definitions."""
    from ..persona_loader import load_all_personas

    personas = load_all_personas()
    return JSONResponse([
        {
            "id": p.id,
            "name": p.name,
            "language": p.language,
            "group_type": p.travel_profile.group_type,
            "budget": p.travel_profile.budget_range,
            "phases": len(p.conversation_goals),
        }
        for p in personas
    ])
