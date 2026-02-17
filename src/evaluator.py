"""Post-run LLM evaluation — 9-axis UX scoring with subjective report."""

import json
import logging
from typing import Any, Dict

from openai import APIStatusError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import Settings
from .models import TestRunResult
from .persona_loader import PersonaDefinition

logger = logging.getLogger(__name__)

EVALUATION_AXES = [
    "fluidity",
    "relevance",
    "visual_clarity",
    "error_handling",
    "conversation_memory",
    "widget_usability",
    "map_interaction",
    "response_speed",
    "personality_match",
]

EVALUATION_SYSTEM_PROMPT = """Tu es un expert UX spécialisé dans l'évaluation de chatbots de planification de voyage.

Tu vas recevoir :
1. Le profil d'un persona (qui il est, ce qu'il veut)
2. Le log de conversation complet entre le persona et le chatbot
3. Les observations de l'agent pendant la session

Tu dois DEVENIR ce persona et évaluer l'expérience de SON point de vue.

Évalue chaque axe de 0.0 à 10.0 avec une justification courte :

1. fluidity — Fluidité du flux de conversation. Les transitions entre phases sont-elles naturelles ?
2. relevance — Les réponses et suggestions sont-elles pertinentes pour ce persona spécifique ?
3. visual_clarity — L'interface est-elle claire ? Les widgets sont-ils lisibles et bien positionnés ?
4. error_handling — Les erreurs sont-elles gérées gracieusement ? Le bouton Réessayer fonctionne-t-il ?
5. conversation_memory — Le chatbot se souvient-il du contexte précédent (destination, préférences) ?
6. widget_usability — Les widgets sont-ils faciles à trouver et à utiliser ?
7. map_interaction — La carte réagit-elle quand une destination est sélectionnée ?
8. response_speed — Les temps de réponse perçus sont-ils acceptables ?
9. personality_match — Le ton du chatbot s'adapte-t-il au style du persona ?

Ensuite écris un "Ressenti Humain" de 2-3 paragraphes comme si tu ÉTAIS le persona.
Liste les 3 principaux points forts, les 3 principales frustrations, et 3 suggestions d'amélioration.

Réponds UNIQUEMENT en JSON valide avec cette structure exacte :
{
  "scores": {
    "fluidity": {"score": 8.5, "justification": "..."},
    "relevance": {"score": 7.0, "justification": "..."},
    "visual_clarity": {"score": 6.5, "justification": "..."},
    "error_handling": {"score": 8.0, "justification": "..."},
    "conversation_memory": {"score": 7.5, "justification": "..."},
    "widget_usability": {"score": 7.0, "justification": "..."},
    "map_interaction": {"score": 5.0, "justification": "..."},
    "response_speed": {"score": 8.0, "justification": "..."},
    "personality_match": {"score": 6.0, "justification": "..."}
  },
  "overall_score": 7.2,
  "evaluation_summary": "Ressenti humain de 2-3 paragraphes...",
  "strengths": ["point fort 1", "point fort 2", "point fort 3"],
  "frustration_points": ["frustration 1", "frustration 2", "frustration 3"],
  "improvement_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}"""


def _build_eval_user_prompt(
    result: TestRunResult,
    persona: PersonaDefinition,
) -> str:
    """Build the user prompt for the evaluation LLM call."""
    sections = []

    # Persona profile
    sections.append("=== PROFIL DU PERSONA ===")
    sections.append(f"Nom: {persona.name}, {persona.age} ans")
    sections.append(f"Rôle: {persona.role}")
    sections.append(f"Langue: {persona.language}")
    sections.append(f"Traits: {', '.join(persona.personality_traits)}")
    sections.append(f"Groupe: {persona.travel_profile.group_type}")
    sections.append(f"Budget: {persona.travel_profile.budget_range}")
    sections.append(f"Destinations: {', '.join(persona.travel_profile.preferred_destinations)}")

    # Execution summary
    sections.append("\n=== RÉSUMÉ D'EXÉCUTION ===")
    sections.append(f"Statut: {result.status.value}")
    sections.append(f"Durée: {result.duration_seconds:.1f}s" if result.duration_seconds else "Durée: N/A")
    sections.append(f"Nombre de pas: {result.total_steps or 'N/A'}")
    sections.append(f"Phases atteintes: {', '.join(result.phases_reached) or 'aucune'}")
    sections.append(f"Phase la plus avancée: {result.phase_furthest or 'aucune'}")
    sections.append(f"Widgets utilisés: {', '.join(result.widgets_interacted) or 'aucun'}")

    # Conversation log
    sections.append("\n=== LOG DE CONVERSATION ===")
    if result.conversation_log:
        if isinstance(result.conversation_log, list):
            for entry in result.conversation_log:
                sections.append(str(entry))
        else:
            sections.append(str(result.conversation_log))
    else:
        sections.append("(aucun log disponible)")

    # Agent thoughts
    if result.agent_thoughts:
        sections.append("\n=== OBSERVATIONS DE L'AGENT ===")
        for thought in result.agent_thoughts[:20]:  # Limit to 20
            sections.append(f"- {thought}")

    return "\n".join(sections)


def _compute_weighted_overall(
    scores: Dict[str, Any],
    weight_overrides: Dict[str, float],
) -> float:
    """Compute weighted overall score from individual axis scores."""
    total_weight = 0.0
    weighted_sum = 0.0

    for axis in EVALUATION_AXES:
        score_data = scores.get(axis, {})
        score_val = score_data.get("score", 0.0) if isinstance(score_data, dict) else 0.0
        weight = weight_overrides.get(axis, 1.0)
        weighted_sum += score_val * weight
        total_weight += weight

    return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0


async def evaluate_run(
    result: TestRunResult,
    persona: PersonaDefinition,
    settings: Settings,
    yaml_config: dict,
) -> dict:
    """Run the post-conversation LLM evaluation.

    Returns a dict matching the EVALUATION_SYSTEM_PROMPT JSON schema.
    """
    eval_cfg = yaml_config.get("evaluation", {})
    temperature = eval_cfg.get("temperature", 0.3)

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://travliaq.com",
            "X-Title": "Travliaq-Testing",
        },
    )

    user_prompt = _build_eval_user_prompt(result, persona)

    logger.info(f"[{persona.id}] Running evaluation LLM call...")

    messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=45),
        retry=retry_if_exception_type((APIStatusError, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_eval_llm():
        try:
            return client.chat.completions.create(
                model=settings.openrouter_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if "response_format" in str(e).lower() or "json" in str(e).lower():
                logger.warning(f"[eval] json_object mode not supported, retrying without response_format: {e}")
                return client.chat.completions.create(
                    model=settings.openrouter_model,
                    messages=messages,
                    temperature=temperature,
                )
            raise

    response = _call_eval_llm()

    raw_content = response.choices[0].message.content
    evaluation = json.loads(raw_content)

    # Recompute overall with persona-specific weight overrides
    if persona.evaluation_weight_overrides:
        evaluation["overall_score"] = _compute_weighted_overall(
            evaluation.get("scores", {}),
            persona.evaluation_weight_overrides,
        )

    logger.info(
        f"[{persona.id}] Evaluation complete — overall score: {evaluation.get('overall_score', 'N/A')}"
    )

    return evaluation
