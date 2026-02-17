"""Factory to create configured browser-use Agent instances."""

from pathlib import Path

from browser_use import Agent, Browser, ChatOpenAI
from browser_use.browser import BrowserProfile

from .config import Settings
from .persona_loader import PersonaDefinition
from .task_prompt_builder import build_task_prompt

EXTEND_SYSTEM_MSG_FR = (
    "CRITIQUE: Ceci est un site français de planification de voyage. "
    "Le champ de texte du chat a l'aria-label 'Envoyer un message...' (FR). "
    "Quand tu vois des widgets (calendriers, sliders, boutons), interagis avec eux en cliquant. "
    "Attends que l'indicateur de frappe (3 points animés) disparaisse avant d'envoyer ton message. "
    "Concentre-toi UNIQUEMENT sur le panneau de chat à gauche, ignore la carte à droite."
)

EXTEND_SYSTEM_MSG_EN = (
    "CRITICAL: This is a French travel planning website. "
    "The chat input textarea has aria-label 'Send a message...' (EN) or 'Envoyer un message...' (FR). "
    "When you see widgets (date pickers, sliders, buttons), interact with them by clicking. "
    "Wait for typing indicators (3 animated dots) to finish before sending your next message. "
    "Focus ONLY on the left chat panel, ignore the map on the right."
)


def create_llm(settings: Settings, model_override: str | None = None) -> ChatOpenAI:
    """Create an OpenRouter LLM instance for browser-use."""
    return ChatOpenAI(
        model=model_override or settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
        default_headers={
            "HTTP-Referer": "https://travliaq.com",
            "X-Title": "Travliaq-Testing",
        },
    )


def create_browser(yaml_config: dict) -> Browser:
    """Create a Browser with configured profile."""
    browser_cfg = yaml_config["browser"]
    profile = BrowserProfile(
        headless=browser_cfg.get("headless", False),
        chromium_sandbox=False,
        args=[
            "--disable-gpu",
            "--disable-webgl",
            "--disable-webgl2",
            "--disable-software-rasterizer",
        ],
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
) -> tuple[Agent, Browser]:
    """Create a browser-use Agent configured for a persona.

    Returns (agent, browser) so the caller can close the browser when done.
    """
    task = build_task_prompt(persona, yaml_config)
    llm = create_llm(settings, model_override=model_override)
    browser = create_browser(yaml_config)

    agent_cfg = yaml_config.get("agent", {})

    # Agent-level fallback: browser-use switches on first 429/5xx mid-run
    fallback_llm = None
    primary = model_override or settings.openrouter_model
    for bm in settings.backup_models_list:
        if bm != primary:
            fallback_llm = create_llm(settings, model_override=bm)
            break

    # Conversation log path
    conv_dir = Path("output/conversations") / persona.id
    conv_dir.mkdir(parents=True, exist_ok=True)

    extend_msg = EXTEND_SYSTEM_MSG_FR if persona.language == "fr" else EXTEND_SYSTEM_MSG_EN

    agent = Agent(
        task=task,
        llm=llm,
        fallback_llm=fallback_llm,
        browser=browser,
        max_actions_per_step=agent_cfg.get("max_actions_per_step", 3),
        max_failures=agent_cfg.get("max_failures", 5),
        retry_delay=agent_cfg.get("retry_delay", 5),
        use_vision=agent_cfg.get("use_vision", True),
        save_conversation_path=str(conv_dir),
        extend_system_message=extend_msg,
        register_new_step_callback=step_callback,
    )

    return agent, browser
