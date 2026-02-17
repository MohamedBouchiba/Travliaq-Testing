"""Event bus for real-time dashboard communication."""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Batch-level
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"

    # Persona run stages (maps to [0/5]...[5/5])
    STAGE_HEALTH_CHECK = "stage_health_check"
    STAGE_CREATE_AGENT = "stage_create_agent"
    STAGE_RUN_AGENT = "stage_run_agent"
    STAGE_EXTRACT_RESULTS = "stage_extract_results"
    STAGE_EVALUATE = "stage_evaluate"
    STAGE_WRITE_REPORT = "stage_write_report"

    # Result events
    PERSONA_COMPLETED = "persona_completed"
    PERSONA_FAILED = "persona_failed"
    PERSONA_TIMEOUT = "persona_timeout"

    # Agent step-level
    AGENT_STEP = "agent_step"
    LOOP_DETECTED = "loop_detected"

    # Extra
    SCREENSHOT_SAVED = "screenshot_saved"


class DashboardEvent(BaseModel):
    """A structured event emitted by the orchestrator for the dashboard."""

    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    persona_id: Optional[str] = None
    batch_id: Optional[str] = None
    stage: Optional[str] = None  # e.g. "0/5", "2/5"
    data: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """In-process pub/sub using asyncio.Queue.

    Multiple SSE clients each get their own queue.
    The bus fans out each event to all subscriber queues.
    """

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def emit(self, event: DashboardEvent) -> None:
        """Fan out event to all subscribers. Non-blocking: drops if queue full."""
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Global singleton — None when dashboard is not running
# ---------------------------------------------------------------------------
_global_bus: Optional[EventBus] = None


def get_event_bus() -> Optional[EventBus]:
    """Return the global EventBus, or None if dashboard is not active."""
    return _global_bus


def set_event_bus(bus: Optional[EventBus]) -> None:
    """Set (or clear) the global EventBus singleton."""
    global _global_bus
    _global_bus = bus
