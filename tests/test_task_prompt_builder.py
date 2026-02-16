"""Tests for the task prompt builder."""

from pathlib import Path

import pytest

from src.persona_loader import load_persona
from src.task_prompt_builder import build_task_prompt

YAML_CONFIG = {
    "target": {
        "base_url": "https://travliaq.com",
        "planner_url": "https://travliaq.com/planner",
        "planner_url_clean": "https://travliaq.com/planner?noTour=1",
    },
    "browser": {"headless": False},
    "agent": {"max_steps": 60},
}

PERSONAS_DIR = Path(__file__).parent.parent / "personas"


@pytest.fixture
def family_persona():
    return load_persona("family_with_kids", PERSONAS_DIR)


@pytest.fixture
def backpacker_persona():
    return load_persona("budget_backpacker", PERSONAS_DIR)


def test_prompt_contains_three_layers(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)

    # Layer 1: Site knowledge
    assert "CONNAISSANCE DU SITE" in prompt or "SITE KNOWLEDGE" in prompt
    assert "travliaq.com" in prompt
    assert "noTour=1" in prompt

    # Layer 2: Persona
    assert "Sophie Martin" in prompt
    assert "PERSONNAGE" in prompt or "CHARACTER" in prompt

    # Layer 3: Flow
    assert "DÉROULEMENT" in prompt or "CONVERSATION FLOW" in prompt
    assert "Phase 1" in prompt
    assert "GREETING" in prompt


def test_prompt_language_matches_persona_fr(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)
    assert "CONNAISSANCE DU SITE" in prompt
    assert "TON PERSONNAGE" in prompt
    assert "DÉROULEMENT DE LA CONVERSATION" in prompt
    assert "Parle UNIQUEMENT en French" in prompt


def test_prompt_language_matches_persona_en(backpacker_persona):
    prompt = build_task_prompt(backpacker_persona, YAML_CONFIG)
    assert "SITE KNOWLEDGE" in prompt
    assert "YOUR CHARACTER" in prompt
    assert "CONVERSATION FLOW" in prompt
    assert "Speak ONLY in English" in prompt


def test_prompt_includes_widget_guide(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)
    assert "CALENDRIER" in prompt or "CALENDAR" in prompt
    assert "travelersSelector" in prompt or "VOYAGEURS" in prompt
    assert "Confirmer" in prompt or "Confirm" in prompt


def test_prompt_includes_first_message(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)
    assert "Bonjour" in prompt
    assert "vacances en famille" in prompt


def test_prompt_includes_travel_profile(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)
    assert "family" in prompt.lower() or "famille" in prompt.lower()
    assert "2 adults" in prompt or "2 adultes" in prompt.lower() or "2 adults" in prompt.lower()
    assert "1500" in prompt
    assert "juillet" in prompt.lower()


def test_prompt_includes_planner_url(family_persona):
    prompt = build_task_prompt(family_persona, YAML_CONFIG)
    assert "https://travliaq.com/planner?noTour=1" in prompt


def test_all_personas_generate_valid_prompts():
    """Ensure all real personas produce non-empty, well-structured prompts."""
    if not PERSONAS_DIR.exists():
        pytest.skip("No personas/ directory")

    from src.persona_loader import load_all_personas

    personas = load_all_personas(PERSONAS_DIR)
    for p in personas:
        prompt = build_task_prompt(p, YAML_CONFIG)
        assert len(prompt) > 500, f"Prompt for {p.id} is too short"
        assert "travliaq.com" in prompt
        assert p.name in prompt
