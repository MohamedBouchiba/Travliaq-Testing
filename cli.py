"""CLI for Travliaq persona testing — AI-driven UX testing with browser-use."""

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Travliaq Persona Testing — Deploy AI agents to test the travel chatbot UX."""
    pass


@cli.command()
@click.argument("persona_id")
@click.option("--headless/--no-headless", default=False, help="Run browser in headless mode")
def run(persona_id: str, headless: bool):
    """Run a single persona test.

    PERSONA_ID is the filename (without .json) from the personas/ folder.

    Examples:

      python cli.py run family_with_kids

      python cli.py run budget_backpacker --headless
    """
    from src.config import Settings, load_yaml_config
    from src.orchestrator import run_single_persona
    from src.persona_loader import load_persona
    from src.utils import setup_logging

    settings = Settings()
    yaml_config = load_yaml_config()
    setup_logging(settings.log_level)

    if headless:
        yaml_config["browser"]["headless"] = True

    try:
        persona = load_persona(persona_id)
    except FileNotFoundError:
        console.print(f"[bold red]Persona not found:[/] {persona_id}")
        console.print("Use [cyan]python cli.py list-personas[/] to see available personas.")
        sys.exit(1)

    console.print(Panel(
        f"[bold]{persona.name}[/] ({persona.id})\n"
        f"Language: {persona.language} | Group: {persona.travel_profile.group_type}\n"
        f"Role: {persona.role[:100]}...",
        title="Running Persona",
        border_style="green",
    ))

    result = asyncio.run(run_single_persona(persona, settings, yaml_config))
    _print_result(result)


@cli.command()
@click.option("--personas", default=None, help="Comma-separated persona IDs (default: all)")
@click.option("--headless/--no-headless", default=False, help="Run browser in headless mode")
def batch(personas: str | None, headless: bool):
    """Run multiple personas sequentially.

    Examples:

      python cli.py batch

      python cli.py batch --personas family_with_kids,budget_backpacker --headless
    """
    from src.config import Settings, load_yaml_config
    from src.orchestrator import run_batch
    from src.utils import setup_logging

    settings = Settings()
    yaml_config = load_yaml_config()
    setup_logging(settings.log_level)

    if headless:
        yaml_config["browser"]["headless"] = True

    persona_ids = personas.split(",") if personas else None

    console.print(Panel(
        f"Personas: {personas or 'ALL'}\nHeadless: {headless}",
        title="Batch Run",
        border_style="blue",
    ))

    results = asyncio.run(run_batch(persona_ids, settings, yaml_config))
    _print_batch_results(results)


@cli.command("list-personas")
def list_personas():
    """List all available personas."""
    from src.persona_loader import load_all_personas

    personas = load_all_personas()

    table = Table(title="Available Personas")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Lang")
    table.add_column("Group")
    table.add_column("Budget")
    table.add_column("Phases", justify="right")

    for p in personas:
        table.add_row(
            p.id,
            p.name,
            p.language,
            p.travel_profile.group_type,
            p.travel_profile.budget_range[:25],
            str(len(p.conversation_goals)),
        )

    console.print(table)


@cli.command()
@click.option("--port", default=8099, help="Dashboard server port")
@click.option("--host", default="0.0.0.0", help="Dashboard server host")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
@click.option("--run-batch", is_flag=True, help="Start a batch run alongside the dashboard")
@click.option("--personas", default=None, help="Comma-separated persona IDs (with --run-batch)")
@click.option("--headless/--no-headless", default=False, help="Run browser in headless mode (with --run-batch)")
def dashboard(port: int, host: str, no_browser: bool, run_batch: bool, personas: str | None, headless: bool):
    """Start the monitoring dashboard web server.

    Examples:

      python cli.py dashboard

      python cli.py dashboard --run-batch --headless

      python cli.py dashboard --run-batch --personas family_with_kids,budget_backpacker --headless
    """
    import threading
    import webbrowser

    import uvicorn

    from src.config import Settings, load_yaml_config
    from src.events import EventBus, set_event_bus
    from src.dashboard.server import app
    from src.utils import setup_logging

    settings = Settings()
    yaml_config = load_yaml_config()
    setup_logging(settings.log_level)

    # Read port/host from config if available
    dash_cfg = yaml_config.get("dashboard", {})
    port = dash_cfg.get("port", port)
    host = dash_cfg.get("host", host)

    # Set up event bus
    bus = EventBus()
    set_event_bus(bus)

    if not no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    if run_batch:
        from src.orchestrator import run_batch as _run_batch

        if headless:
            yaml_config["browser"]["headless"] = True

        persona_ids = personas.split(",") if personas else None

        @app.on_event("startup")
        async def _start_batch():
            import asyncio
            asyncio.create_task(_run_batch(persona_ids, settings, yaml_config))

    console.print(Panel(
        f"Dashboard: http://localhost:{port}\n"
        f"Batch auto-run: {'yes' if run_batch else 'no (view-only mode)'}",
        title="Travliaq Dashboard",
        border_style="cyan",
    ))

    uvicorn.run(app, host=host, port=port, log_level="warning")


@cli.command("preview-prompt")
@click.argument("persona_id")
def preview_prompt(persona_id: str):
    """Preview the full task prompt that would be sent to the agent.

    Useful for debugging and tuning the prompt.
    """
    from src.config import load_yaml_config
    from src.persona_loader import load_persona
    from src.task_prompt_builder import build_task_prompt

    yaml_config = load_yaml_config()

    try:
        persona = load_persona(persona_id)
    except FileNotFoundError:
        console.print(f"[bold red]Persona not found:[/] {persona_id}")
        sys.exit(1)

    prompt = build_task_prompt(persona, yaml_config)

    console.print(Panel(
        prompt,
        title=f"Task Prompt — {persona.name} ({persona.id})",
        border_style="yellow",
    ))


def _print_result(result):
    """Pretty-print a single run result."""
    from src.models import RunStatus

    status_color = {
        RunStatus.COMPLETED: "green",
        RunStatus.FAILED: "red",
        RunStatus.TIMEOUT: "yellow",
        RunStatus.RUNNING: "blue",
    }
    color = status_color.get(result.status, "white")

    console.print(f"\n[bold {color}]Status:[/] {result.status.value}")
    console.print(f"Duration: {result.duration_seconds:.1f}s" if result.duration_seconds else "Duration: N/A")
    console.print(f"Steps: {result.total_steps or 'N/A'}")
    console.print(f"Phases: {', '.join(result.phases_reached) or 'none'}")
    console.print(f"Furthest: {result.phase_furthest or 'none'}")

    if result.error_message:
        console.print(f"[red]Error: {result.error_message}[/]")

    if result.score_overall is not None:
        # Scores table
        scores_table = Table(title="Evaluation Scores")
        scores_table.add_column("Axis", style="cyan")
        scores_table.add_column("Score", justify="right")

        for axis in [
            "fluidity", "relevance", "visual_clarity", "error_handling",
            "conversation_memory", "widget_usability", "map_interaction",
            "response_speed", "personality_match",
        ]:
            score = getattr(result.scores, axis, 0.0)
            color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            scores_table.add_row(axis, f"[{color}]{score:.1f}[/]")

        scores_table.add_row(
            "[bold]OVERALL[/]",
            f"[bold]{result.score_overall:.1f}[/]",
        )
        console.print(scores_table)

    if result.evaluation_summary:
        console.print(Panel(
            result.evaluation_summary,
            title="Ressenti Humain",
            border_style="magenta",
        ))

    if result.strengths:
        console.print("\n[green]Strengths:[/]")
        for s in result.strengths:
            console.print(f"  + {s}")

    if result.frustration_points:
        console.print("\n[red]Frustrations:[/]")
        for f in result.frustration_points:
            console.print(f"  - {f}")

    if result.improvement_suggestions:
        console.print("\n[yellow]Suggestions:[/]")
        for s in result.improvement_suggestions:
            console.print(f"  > {s}")


def _print_batch_results(results):
    """Pretty-print batch results in a comparison table."""
    table = Table(title="Batch Results")
    table.add_column("Persona", style="cyan")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Furthest Phase")
    table.add_column("Overall Score", justify="right")

    for r in results:
        from src.models import RunStatus
        color = "green" if r.status == RunStatus.COMPLETED else "red" if r.status == RunStatus.FAILED else "yellow"
        duration = f"{r.duration_seconds:.0f}s" if r.duration_seconds else "N/A"
        score = f"{r.score_overall:.1f}" if r.score_overall is not None else "N/A"

        table.add_row(
            r.persona_id,
            f"[{color}]{r.status.value}[/]",
            duration,
            r.phase_furthest or "none",
            score,
        )

    console.print(table)

    # Summary
    completed = [r for r in results if r.score_overall is not None]
    if completed:
        avg_score = sum(r.score_overall for r in completed) / len(completed)
        console.print(f"\n[bold]Average overall score:[/] {avg_score:.1f}/10")


if __name__ == "__main__":
    cli()
