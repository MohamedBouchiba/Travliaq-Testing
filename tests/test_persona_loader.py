"""Tests for persona loader."""

import json
import tempfile
from pathlib import Path

import pytest

from src.persona_loader import (
    PersonaDefinition,
    load_all_personas,
    load_persona,
)


@pytest.fixture
def sample_persona_data():
    return {
        "id": "test_persona",
        "name": "Test User",
        "age": 30,
        "language": "fr",
        "role": "Un voyageur test",
        "personality_traits": ["patient", "curieux"],
        "conversation_style": {
            "verbosity": "medium",
            "formality": "casual",
            "asks_questions": True,
            "expresses_frustration": False,
            "changes_mind": False,
        },
        "travel_profile": {
            "group_type": "solo",
            "travelers": {"adults": 1, "children": 0, "infants": 0},
            "budget_range": "500-1000 EUR",
            "preferred_destinations": ["plage"],
            "avoided": [],
            "trip_type": "roundtrip",
        },
        "conversation_goals": [
            {
                "phase": "greeting",
                "goal": "Say hello",
                "example_message": "Bonjour !",
            }
        ],
        "evaluation_weight_overrides": {},
    }


@pytest.fixture
def personas_dir(sample_persona_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_persona.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_persona_data, f)
        yield Path(tmpdir)


def test_load_persona_valid(personas_dir):
    persona = load_persona("test_persona", personas_dir)
    assert persona.id == "test_persona"
    assert persona.name == "Test User"
    assert persona.language == "fr"
    assert persona.travel_profile.group_type == "solo"
    assert len(persona.conversation_goals) == 1
    assert persona.conversation_goals[0].phase == "greeting"


def test_load_persona_not_found(personas_dir):
    with pytest.raises(FileNotFoundError):
        load_persona("nonexistent", personas_dir)


def test_load_all_personas(personas_dir, sample_persona_data):
    # Add a second persona
    second = {**sample_persona_data, "id": "second_persona", "name": "Second"}
    with open(personas_dir / "second_persona.json", "w", encoding="utf-8") as f:
        json.dump(second, f)

    personas = load_all_personas(personas_dir)
    assert len(personas) == 2
    ids = {p.id for p in personas}
    assert "test_persona" in ids
    assert "second_persona" in ids


def test_load_all_skips_underscore_files(personas_dir, sample_persona_data):
    # Create a _schema.json file — should be skipped
    with open(personas_dir / "_schema.json", "w", encoding="utf-8") as f:
        json.dump(sample_persona_data, f)

    personas = load_all_personas(personas_dir)
    assert len(personas) == 1  # Only test_persona, not _schema


def test_persona_defaults():
    """Minimal persona with just required fields."""
    persona = PersonaDefinition(
        id="minimal",
        name="Minimal",
        role="Test",
        travel_profile={
            "group_type": "solo",
            "travelers": {"adults": 1},
            "budget_range": "100 EUR",
        },
    )
    assert persona.language == "fr"
    assert persona.conversation_style.verbosity == "medium"
    assert persona.personality_traits == []
    assert persona.conversation_goals == []


def test_load_real_personas():
    """Load all real personas from the project's personas/ directory."""
    real_dir = Path(__file__).parent.parent / "personas"
    if not real_dir.exists():
        pytest.skip("No personas/ directory found")

    personas = load_all_personas(real_dir)
    assert len(personas) >= 4

    for p in personas:
        assert p.id
        assert p.name
        assert p.role
        assert p.travel_profile.group_type
        assert len(p.conversation_goals) >= 1
