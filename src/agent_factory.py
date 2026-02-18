"""Factory to create configured browser-use Agent instances."""

import logging
from pathlib import Path

from browser_use import Browser, ChatGoogle, ChatGroq, ChatOpenAI
from browser_use.browser import BrowserProfile

from .config import Settings
from .persona_loader import PersonaDefinition
from .task_prompt_builder import build_task_prompt
from .travliaq_agent import PhaseTracker, TravliaqAgent

logger = logging.getLogger(__name__)


def _get_provider(model_id: str, settings: Settings | None = None) -> str:
    """Identify LLM provider from model ID."""
    if model_id.startswith("gemini"):
        return "google"
    # Exact match for Groq — avoids routing OpenRouter meta-llama models to Groq
    if settings and model_id == settings.groq_model:
        return "groq"
    # Exact match for SambaNova
    if settings and settings.sambanova_api_key and model_id == settings.sambanova_model:
        return "sambanova"
    if not settings and model_id.startswith(("meta-llama/", "openai/gpt-oss")):
        return "groq"  # legacy fallback without settings context
    return "openrouter"

EXTEND_SYSTEM_MSG_FR = (
    "CRITIQUE: Ceci est un site français de planification de voyage. "
    "Le champ de texte du chat a un placeholder 'Envoyer un message...' — tape ton VRAI message, PAS le placeholder. "
    "Quand tu vois des widgets (calendriers, sliders, boutons), interagis avec eux en cliquant. "
    "Attends que l'indicateur de frappe (3 points animés) disparaisse avant d'envoyer ton message. "
    "Après avoir tapé un message, APPUIE sur Entrée ou CLIQUE sur le bouton Envoyer. Ne laisse jamais un message non envoyé. "
    "La carte à droite va se charger, observe-la mais concentre-toi sur le chat à gauche. "
    "IMPORTANT: Tu dois compléter TOUTES les phases de ta mission. Si tu es bloqué, passe à la phase suivante. "
    "Ne t'arrête JAMAIS avant d'avoir terminé la dernière phase (envoi du feedback)."
)

EXTEND_SYSTEM_MSG_EN = (
    "CRITICAL: This is a French travel planning website. "
    "The chat input textarea has a placeholder 'Send a message...' — type your ACTUAL message, NOT the placeholder text. "
    "When you see widgets (date pickers, sliders, buttons), interact with them by clicking. "
    "Wait for typing indicators (3 animated dots) to finish before sending your next message. "
    "After typing a message, PRESS Enter or CLICK the Send button. Never leave an unsent message. "
    "The map on the right will load, observe it but focus on the left chat panel. "
    "IMPORTANT: You must complete ALL phases of your mission. If stuck, move to the next phase. "
    "NEVER stop before completing the last phase (feedback submission)."
)


def create_llm_for_model(model_id: str, settings: Settings):
    """Create the right LLM provider based on model ID.

    - gemini-* → ChatGoogle (Google AI Studio)
    - meta-llama/* or groq models → ChatGroq
    - anything else → ChatOpenAI via OpenRouter
    """
    if model_id.startswith("gemini"):
        logger.debug(f"Creating ChatGoogle for {model_id}")
        return ChatGoogle(
            model=model_id,
            api_key=settings.google_api_key,
            retry_base_delay=2.0,   # fail fast — let orchestrator handle provider switching
            retry_max_delay=10.0,
            max_retries=1,          # was 3 → 1: don't waste quota retrying dead provider
        )

    if settings.groq_api_key and model_id == settings.groq_model:
        logger.debug(f"Creating ChatGroq for {model_id}")
        return ChatGroq(
            model=model_id,
            api_key=settings.groq_api_key,
            max_retries=1,  # was default 10 — don't burn TPD on retries
        )

    # SambaNova (Maverick — vision-capable, free, 16K context)
    if settings.sambanova_api_key and model_id == settings.sambanova_model:
        logger.debug(f"Creating ChatOpenAI (SambaNova) for {model_id}")
        return ChatOpenAI(
            model=model_id,
            api_key=settings.sambanova_api_key,
            base_url="https://api.sambanova.ai/v1",
            max_retries=1,
            max_completion_tokens=2048,  # 16K context — prompts are ~13K, leave room
        )

    # OpenRouter fallback
    logger.debug(f"Creating ChatOpenAI (OpenRouter) for {model_id}")
    return ChatOpenAI(
        model=model_id,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=1,  # was 3 — fail fast, let orchestrator switch providers
        default_headers={
            "HTTP-Referer": "https://travliaq.com",
            "X-Title": "Travliaq-Testing",
        },
    )


def create_browser(yaml_config: dict) -> Browser:
    """Create a Browser with configured profile.

    WebGL strategy:
    - Headful: let Chromium auto-detect the native GPU (D3D11 on Windows).
      Forcing SwiftShader would waste the hardware GPU.
    - Headless / Docker: Chromium auto-selects SwiftShader. We add
      --enable-unsafe-swiftshader as a safety net.
    """
    browser_cfg = yaml_config["browser"]
    headless = browser_cfg.get("headless", False)

    args = [
        "--enable-webgl",
        "--ignore-gpu-blocklist",
    ]
    if headless:
        args.append("--enable-unsafe-swiftshader")

    profile = BrowserProfile(
        headless=headless,
        chromium_sandbox=False,
        args=args,
        window_size={
            "width": browser_cfg.get("window_width", 1440),
            "height": browser_cfg.get("window_height", 900),
        },
        minimum_wait_page_load_time=browser_cfg.get("minimum_wait_page_load_time", 2),
        wait_for_network_idle_page_load_time=browser_cfg.get(
            "maximum_wait_page_load_time", 15
        ),
    )
    return Browser(browser_profile=profile)


def create_agent(
    persona: PersonaDefinition,
    settings: Settings,
    yaml_config: dict,
    step_callback=None,
    model_override: str | None = None,
    excluded_providers: set[str] | None = None,
) -> tuple[TravliaqAgent, Browser, str | None, PhaseTracker]:
    """Create a browser-use Agent configured for a persona.

    Returns (agent, browser, fallback_model_id, phase_tracker) so the caller
    can close the browser when done and track phase progress.
    """
    chain = settings.build_model_chain()
    if not chain:
        raise RuntimeError("No LLM API keys configured. Set GOOGLE_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY.")

    # Pick primary model
    primary_model = model_override or chain[0]
    llm = create_llm_for_model(primary_model, settings)

    # Agent-level fallback: pick first model from a DIFFERENT provider
    # so that a provider-wide rate limit doesn't cascade to fallback
    primary_provider = _get_provider(primary_model, settings)
    excluded = (excluded_providers or set()) | {primary_provider}
    fallback_llm = None
    fallback_model_id = None
    for candidate in chain:
        if _get_provider(candidate, settings) not in excluded:
            fallback_llm = create_llm_for_model(candidate, settings)
            fallback_model_id = candidate
            break

    task = build_task_prompt(persona, yaml_config)
    browser = create_browser(yaml_config)

    agent_cfg = yaml_config.get("agent", {})

    # Conversation log path
    conv_dir = Path("output/conversations") / persona.id
    conv_dir.mkdir(parents=True, exist_ok=True)

    extend_msg = EXTEND_SYSTEM_MSG_FR if persona.language == "fr" else EXTEND_SYSTEM_MSG_EN

    phase_tracker = PhaseTracker(
        total_phases=len(persona.conversation_goals),
        language=persona.language,
    )

    # Normalize use_vision: YAML gives bool for true/false, string for 'auto'
    use_vision_cfg = agent_cfg.get("use_vision", True)
    if isinstance(use_vision_cfg, str):
        v = use_vision_cfg.lower()
        use_vision_cfg = True if v == "true" else (False if v == "false" else v)

    agent = TravliaqAgent(
        task=task,
        llm=llm,
        fallback_llm=fallback_llm,
        browser=browser,
        max_actions_per_step=agent_cfg.get("max_actions_per_step", 3),
        max_failures=agent_cfg.get("max_failures", 3),
        final_response_after_failure=False,  # stop after exactly max_failures, no grace step
        use_vision=use_vision_cfg,
        save_conversation_path=str(conv_dir),
        extend_system_message=extend_msg,
        register_new_step_callback=step_callback,
        phase_tracker=phase_tracker,
    )

    return agent, browser, fallback_model_id, phase_tracker
