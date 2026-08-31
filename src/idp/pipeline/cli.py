"""Command-line interface.

Usage:
  idp run path/to/file.pdf --schema Invoice --backend ollama
  idp eval --dataset src/idp/eval/datasets/invoices --strategies mock,ollama
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from idp.pipeline.pipeline import Pipeline, run_file, save_result

app = typer.Typer(help="py-idp: general-purpose AI-enabled IDP framework.")
console = Console()


@app.command()
def run(
    path: str = typer.Argument(..., help="Path to a PDF / image / text file to extract from."),
    schema: str = typer.Option("Invoice", help="Schema name: Invoice | Contract | BankStatement"),
    backend: str = typer.Option("auto", help="LLM backend: auto | mock | ollama | openai | anthropic"),
    parser: str = typer.Option("auto", help="Parser: auto | docling | pdfplumber | plain"),
    output: str | None = typer.Option(None, "--output", "-o", help="Optional JSON output path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full IDP pipeline on a single document."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    try:
        result = run_file(path, schema=schema, backend=backend, parser=parser)
    except Exception as e:
        console.print(f"[red]pipeline failed:[/red] {e}")
        raise typer.Exit(code=1)
    _print_result(result)
    if output:
        save_result(result, output)
        console.print(f"[green]saved:[/green] {output}")


@app.command()
def schemas():
    """List the schemas that ship with py-idp."""
    from idp.core.schemas import SCHEMA_REGISTRY
    table = Table(title="Built-in schemas")
    table.add_column("Name", style="cyan")
    table.add_column("Fields (top-level)", style="white")
    for name, model in SCHEMA_REGISTRY.items():
        table.add_row(name, ", ".join(model.model_fields.keys()))
    console.print(table)


@app.command()
def providers():
    """List the LLM providers (international + China) supported by py-idp."""
    from idp.llm.china import list_china_providers
    table = Table(title="China LLM providers (OpenAI-compatible)")
    table.add_column("name", style="cyan")
    table.add_column("env_var", style="yellow")
    table.add_column("default model", style="green")
    table.add_column("vision model", style="magenta")
    table.add_column("notes", style="white")
    for p in list_china_providers():
        table.add_row(
            p["name"],
            p["env_var"],
            p["default_model"] or "—",
            p["vision_model"] or "—",
            p["notes"],
        )
    console.print(table)
    console.print("\nInternational: openai | anthropic | ollama | vllm | lm-studio (any OpenAI-compat)")
    console.print("Mock backends:  mock | mock-random | mock-omits (no API key needed)")


@app.command()
def serve(
    port: int = typer.Option(8501, help="Port for the Streamlit HITL UI."),
):
    """Launch the Streamlit HITL review UI."""
    import subprocess

    cmd = [sys.executable, "-m", "streamlit", "run", "src/idp/hitl/app.py", "--server.port", str(port)]
    console.print(f"[cyan]starting streamlit on :{port}[/cyan]")
    raise SystemExit(subprocess.call(cmd))


@app.command()
def eval(
    dataset: str = typer.Option(..., help="Path to src/idp/eval/datasets/<name>"),
    strategy: str = typer.Option("mock", help="Comma-separated backends to compare: mock,ollama,openai"),
    output: str | None = typer.Option(None, "--output", "-o"),
):
    """Run the eval harness against a labeled dataset."""
    from idp.eval.runner import run_dataset
    strategies = [s.strip() for s in strategy.split(",") if s.strip()]
    res = run_dataset(Path(dataset), strategies=strategies)
    _print_eval(res)
    if output:
        Path(output).write_text(json.dumps(res, indent=2, default=str))
        console.print(f"[green]saved:[/green] {output}")


def _print_result(result):
    table = Table(title=f"py-idp • {Path(result.document.source_path).name}")
    table.add_column("Stage", style="cyan")
    table.add_column("Time (s)", style="green", justify="right")
    for t in result.timings:
        table.add_row(t.name, f"{t.seconds:.3f}")
    console.print(table)

    console.print(f"\n[bold]classification:[/bold] {result.classification}")
    console.print(f"[bold]backend:[/bold]         {result.backend_name} ({result.mode})")
    console.print(f"[bold]schema:[/bold]          {result.schema_name}")
    console.print(f"[bold]validation:[/bold]      {'PASS' if result.validation_passed else 'FAIL'}")
    if result.document.errors:
        console.print(f"[yellow]warnings:[/yellow] {len(result.document.errors)} (see doc.errors)")

    console.print("\n[bold]extraction:[/bold]")
    console.print_json(json.dumps(result.document.extraction, indent=2, default=str))
    if result.confidence:
        console.print("\n[bold]confidence (top-5 lowest first):[/bold]")
        low = sorted(result.confidence.items(), key=lambda kv: kv[1])[:5]
        for k, v in low:
            console.print(f"  {k:<24} {v:.2f}")


def _print_eval(res):
    table = Table(title="eval results")
    table.add_column("strategy", style="cyan")
    table.add_column("schema_valid_rate", style="green", justify="right")
    table.add_column("field_f1", style="green", justify="right")
    table.add_column("avg_sec/doc", style="yellow", justify="right")
    for row in res["rows"]:
        table.add_row(
            row["strategy"],
            f"{row['schema_valid_rate']:.2f}",
            f"{row['field_f1']:.2f}",
            f"{row['avg_sec_per_doc']:.2f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
