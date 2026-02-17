"""Data models for test run results."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class EvaluationScores(BaseModel):
    """9-axis UX evaluation scores (0-10 each)."""

    fluidity: float = 0.0
    relevance: float = 0.0
    visual_clarity: float = 0.0
    error_handling: float = 0.0
    conversation_memory: float = 0.0
    widget_usability: float = 0.0
    map_interaction: float = 0.0
    response_speed: float = 0.0
    personality_match: float = 0.0


class TestRunResult(BaseModel):
    """Full result of a single persona test run."""

    # Identification
    run_id: str
    batch_id: Optional[str] = None
    persona_id: str
    persona_name: str
    persona_language: str = "fr"

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Status
    status: RunStatus = RunStatus.RUNNING
    error_message: Optional[str] = None

    # Execution data
    total_steps: Optional[int] = None
    total_messages: Optional[int] = None
    phases_reached: List[str] = Field(default_factory=list)
    phase_furthest: Optional[str] = None
    widgets_interacted: List[str] = Field(default_factory=list)

    # Logs
    conversation_log: Optional[Any] = None
    agent_thoughts: Optional[List[str]] = None
    screenshot_paths: List[str] = Field(default_factory=list)

    # Evaluation
    scores: EvaluationScores = Field(default_factory=EvaluationScores)
    score_justifications: Dict[str, str] = Field(default_factory=dict)
    score_overall: Optional[float] = None
    evaluation_summary: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    frustration_points: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)

    # Metadata
    config_snapshot: Optional[Dict[str, Any]] = None
    llm_model_used: Optional[str] = None

    def merge_evaluation(self, evaluation: dict) -> None:
        """Merge evaluator LLM output into this result."""
        scores_data = evaluation.get("scores", {})
        for axis, data in scores_data.items():
            if hasattr(self.scores, axis):
                if isinstance(data, dict):
                    score_val = data.get("score", 0.0)
                    justification = data.get("justification", "")
                    if justification:
                        self.score_justifications[axis] = justification
                else:
                    score_val = data
                setattr(self.scores, axis, score_val)

        self.score_overall = evaluation.get("overall_score")
        self.evaluation_summary = evaluation.get("evaluation_summary")
        self.strengths = evaluation.get("strengths", [])
        self.frustration_points = evaluation.get("frustration_points", [])
        self.improvement_suggestions = evaluation.get("improvement_suggestions", [])
        self.threats = evaluation.get("threats", [])

    def to_json_dict(self) -> dict:
        """Serialize for JSON file output."""
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "persona": {
                "id": self.persona_id,
                "name": self.persona_name,
                "language": self.persona_language,
            },
            "timing": {
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "duration_seconds": self.duration_seconds,
            },
            "status": self.status.value,
            "error_message": self.error_message,
            "execution": {
                "total_steps": self.total_steps,
                "total_messages": self.total_messages,
                "phases_reached": self.phases_reached,
                "phase_furthest": self.phase_furthest,
                "widgets_interacted": self.widgets_interacted,
            },
            "evaluation": {
                "overall_score": self.score_overall,
                "scores": {
                    axis: {
                        "score": getattr(self.scores, axis),
                        "justification": self.score_justifications.get(axis, ""),
                    }
                    for axis in [
                        "fluidity", "relevance", "visual_clarity", "error_handling",
                        "conversation_memory", "widget_usability", "map_interaction",
                        "response_speed", "personality_match",
                    ]
                },
                "summary": self.evaluation_summary,
                "strengths": self.strengths,
                "frustration_points": self.frustration_points,
                "improvement_suggestions": self.improvement_suggestions,
                "threats": self.threats,
            },
            "logs": {
                "conversation": self.conversation_log,
                "agent_thoughts": self.agent_thoughts,
                "screenshot_paths": self.screenshot_paths,
            },
            "meta": {
                "config_snapshot": self.config_snapshot,
                "llm_model_used": self.llm_model_used,
            },
        }
