"""Load and validate persona JSON files."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationGoal(BaseModel):
    """A single phase in the persona's conversation flow."""

    phase: str
    goal: str
    example_message: Optional[str] = None
    widget_interactions: Optional[List[str]] = None
    min_messages: Optional[int] = None
    success_indicator: Optional[str] = None


class ConversationStyle(BaseModel):
    """How the persona communicates."""

    verbosity: str = "medium"  # low | medium | high
    formality: str = "casual"  # formal | casual | mixed
    asks_questions: bool = True
    expresses_frustration: bool = False
    changes_mind: bool = False


class TravelProfile(BaseModel):
    """The persona's travel preferences."""

    group_type: str
    travelers: Dict[str, int]  # {"adults": 2, "children": 1, "infants": 0}
    budget_range: str
    preferred_destinations: List[str] = Field(default_factory=list)
    avoided: List[str] = Field(default_factory=list)
    trip_type: str = "roundtrip"
    flexibility: str = "flexible_dates"
    preferred_month: Optional[str] = None
    trip_duration: Optional[str] = None


class PersonaDefinition(BaseModel):
    """Full persona definition loaded from JSON."""

    id: str
    name: str
    age: Optional[int] = None
    language: str = "fr"
    role: str
    personality_traits: List[str] = Field(default_factory=list)
    conversation_style: ConversationStyle = Field(default_factory=ConversationStyle)
    travel_profile: TravelProfile
    conversation_goals: List[ConversationGoal] = Field(default_factory=list)
    evaluation_weight_overrides: Dict[str, float] = Field(default_factory=dict)


def _personas_dir() -> Path:
    return Path(__file__).parent.parent / "personas"


def load_persona(persona_id: str, personas_dir: Optional[Path] = None) -> PersonaDefinition:
    """Load a single persona by ID (filename without .json)."""
    base = personas_dir or _personas_dir()
    path = base / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Persona not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PersonaDefinition(**data)


def load_all_personas(personas_dir: Optional[Path] = None) -> List[PersonaDefinition]:
    """Load all persona JSON files from the personas directory."""
    base = personas_dir or _personas_dir()
    personas = []
    for path in sorted(base.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        personas.append(PersonaDefinition(**data))
    return personas
