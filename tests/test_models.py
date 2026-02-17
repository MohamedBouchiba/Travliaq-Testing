"""Tests for data models."""

from datetime import datetime, timezone

from src.models import EvaluationScores, RunStatus, TestRunResult


def test_run_status_values():
    assert RunStatus.RUNNING == "running"
    assert RunStatus.COMPLETED == "completed"
    assert RunStatus.FAILED == "failed"
    assert RunStatus.TIMEOUT == "timeout"


def test_evaluation_scores_defaults():
    scores = EvaluationScores()
    assert scores.fluidity == 0.0
    assert scores.relevance == 0.0
    assert scores.personality_match == 0.0


def test_test_run_result_defaults():
    result = TestRunResult(
        run_id="test-123",
        persona_id="test",
        persona_name="Test User",
    )
    assert result.status == RunStatus.RUNNING
    assert result.phases_reached == []
    assert result.screenshot_paths == []
    assert result.scores.fluidity == 0.0


def test_merge_evaluation():
    result = TestRunResult(
        run_id="test-123",
        persona_id="test",
        persona_name="Test User",
    )

    evaluation = {
        "scores": {
            "fluidity": {"score": 8.5, "justification": "Smooth flow"},
            "relevance": {"score": 7.0, "justification": "Mostly relevant"},
            "visual_clarity": {"score": 6.5, "justification": "OK"},
            "error_handling": {"score": 8.0, "justification": "Good"},
            "conversation_memory": {"score": 7.5, "justification": "Remembered context"},
            "widget_usability": {"score": 7.0, "justification": "Easy to use"},
            "map_interaction": {"score": 5.0, "justification": "Minimal"},
            "response_speed": {"score": 8.0, "justification": "Fast"},
            "personality_match": {"score": 6.0, "justification": "Generic tone"},
        },
        "overall_score": 7.2,
        "evaluation_summary": "Good experience overall.",
        "strengths": ["Fast responses", "Clear widgets"],
        "frustration_points": ["Map didn't update"],
        "improvement_suggestions": ["Better map integration"],
        "threats": ["Competitor is faster", "Complex UI"],
    }

    result.merge_evaluation(evaluation)

    assert result.scores.fluidity == 8.5
    assert result.scores.relevance == 7.0
    assert result.scores.map_interaction == 5.0
    assert result.score_overall == 7.2
    assert result.evaluation_summary == "Good experience overall."
    assert len(result.strengths) == 2
    assert len(result.frustration_points) == 1
    assert len(result.threats) == 2
    assert "Competitor is faster" in result.threats
    assert result.score_justifications["fluidity"] == "Smooth flow"


def test_to_json_dict():
    result = TestRunResult(
        run_id="test-123",
        batch_id="batch-abc",
        persona_id="family_with_kids",
        persona_name="Sophie Martin",
        persona_language="fr",
        started_at=datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 2, 16, 12, 5, 30, tzinfo=timezone.utc),
        duration_seconds=330.0,
        status=RunStatus.COMPLETED,
        total_steps=45,
        total_messages=12,
        phases_reached=["greeting", "preferences", "destination"],
        phase_furthest="destination",
    )

    data = result.to_json_dict()

    assert data["run_id"] == "test-123"
    assert data["persona"]["id"] == "family_with_kids"
    assert data["persona"]["name"] == "Sophie Martin"
    assert data["timing"]["duration_seconds"] == 330.0
    assert data["status"] == "completed"
    assert data["execution"]["phases_reached"] == ["greeting", "preferences", "destination"]
    assert data["evaluation"]["scores"]["fluidity"]["score"] == 0.0
    assert data["evaluation"]["scores"]["fluidity"]["justification"] == ""
    assert data["evaluation"]["overall_score"] is None
    assert data["evaluation"]["threats"] == []


def test_to_json_dict_with_scores():
    result = TestRunResult(
        run_id="test-456",
        persona_id="test",
        persona_name="Test",
    )
    result.scores.fluidity = 9.0
    result.scores.relevance = 8.0
    result.score_overall = 8.5

    result.threats = ["Risk 1"]

    data = result.to_json_dict()
    assert data["evaluation"]["scores"]["fluidity"]["score"] == 9.0
    assert data["evaluation"]["scores"]["relevance"]["score"] == 8.0
    assert data["evaluation"]["overall_score"] == 8.5
    assert data["evaluation"]["threats"] == ["Risk 1"]
