# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

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

from idp.pipeline.pipeline import run_file, save_result

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
        raise typer.Exit(code=1) from e
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


@app.command(name="discover-schema")
def discover_schema_cmd(
    path: str = typer.Argument(..., help="Path to the PDF to analyze"),
    hint: str = typer.Option(
        "",
        "--hint",
        "-h",
        help="Natural-language description of fields to extract (e.g. 'extract vendor_name, total_amount, and line items')",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Write JSON Schema to this file (default: stdout)"),
    backend: str = typer.Option(
        "nanonets",
        "--backend",
        "-b",
        help="Multimodal backend to use. Defaults to nanonets; requires IDP_ENABLE_NANONETS=1.",
    ),
):
    """Auto-discover a Pydantic schema from a PDF + hint.

    Given a scanned PDF and a natural-language hint, this command asks
    the configured multimodal backend to propose a JSON Schema, then
    compiles it to a Pydantic class.

    The output is the raw JSON Schema. To get a Pydantic class for use
    with Pipeline(schema=...), call idp.discover_schema(...) from Python.
    """
    from idp.discover import _default_backend_factory

    console.print(f"[cyan]Discovering schema for[/cyan] {path}")
    console.print(f"[cyan]Hint:[/cyan] {hint or '(infer from document)'}")

    try:
        backend_obj = _default_backend_factory(backend)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]backend error:[/red] {e}")
        raise SystemExit(1) from None

    from idp.discover import discover_schema

    try:
        result = discover_schema(path, hint=hint, backend=backend_obj)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]discovery failed:[/red] {e}")
        raise SystemExit(1) from None

    console.print(f"\n[green]Discovered schema[/green] (backend: {result.backend_name}):")
    console.print(f"[bold]class:[/bold] {result.schema_class.__name__}")
    console.print(f"[bold]fields:[/bold] {len(result.schema_class.model_fields)}")
    for name, field in result.schema_class.model_fields.items():
        required = "(required)" if field.is_required() else "(optional)"
        console.print(f"  - {name}: {field.annotation} {required}")

    if output:
        Path(output).write_text(json.dumps(result.json_schema, indent=2))
        console.print(f"\n[green]saved:[/green] {output}")
    else:
        console.print("\n[dim](pass --output FILE to save the JSON Schema)[/dim]")


@app.command(name="discover-template")
def discover_template_cmd(
    sources: list[str] = typer.Argument(
        ...,
        help="Paths to 3-5 sample PDFs / text files of the same document type.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Directory to write the generated <name>.md template into.",
    ),
    schema_name_hint: str | None = typer.Option(
        None,
        "--schema-name",
        "-s",
        help="Override the template name. Default: majority classification vote.",
    ),
    backend: str = typer.Option(
        "mock",
        "--backend",
        "-b",
        help="LLM backend for the underlying pipeline (mock | openai | ollama | ...).",
    ),
    schema: str = typer.Option(
        "Invoice",
        "--schema",
        help="Pydantic schema name used for the per-sample extraction.",
    ),
):
    """Auto-discover a reusable template from 3-5 sample documents.

    Weekend-hack version: clusters the field names found in each
    sample's extraction and writes one template.md. See
    docs/ROADMAP.md B1 for scope.
    """
    from idp.template_discovery import discover_template

    console.print(
        f"[cyan]Discovering template from {len(sources)} sample(s)[/cyan]"
    )
    try:
        template = discover_template(
            samples=sources,
            output_dir=Path(output),
            schema_name_hint=schema_name_hint,
            backend=backend,
            schema=schema,
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        console.print(f"[red]template discovery failed:[/red] {e}")
        raise typer.Exit(code=1) from None

    console.print(f"[green]wrote template:[/green] {template.source_path}")
    console.print(f"[bold]name:[/bold]   {template.name}")
    console.print(f"[bold]schema:[/bold] {template.schema}")
    # Count "## Fields" bullets in the body for a quick summary.
    field_lines = [
        line for line in template.body.splitlines()
        if line.startswith("* **") and "** —" in line
    ]
    console.print(f"[bold]fields:[/bold] {len(field_lines)} detected")


@app.command(name="batch")
def batch(
    sources: list[str] = typer.Argument(
        ...,
        help="Paths to documents, directories (recursively scanned for PDFs/images), or @<file> for a path list.",
    ),
    backend: str = typer.Option("mock", "--backend", "-b", help="LLM backend (mock, openai, ollama, etc.)"),
    schema_name: str = typer.Option("Invoice", "--schema", "-s", help="Pydantic schema name"),
    output: str | None = typer.Option(None, "--output", "-o", help="Per-doc JSONL output file (default: stdout)"),
    report: str | None = typer.Option(None, "--report", "-r", help="Aggregate summary JSON file"),
    dlq: str | None = typer.Option(None, "--dlq", help="Failed-doc JSONL file (dead-letter queue)"),
    checkpoint: str | None = typer.Option(None, "--checkpoint", help="Resume ledger path (idempotent across runs)"),
    limit: int | None = typer.Option(None, "--limit", help="Process at most N docs (for sampling)"),
    progress_every: int = typer.Option(10, "--progress-every", help="Log every N docs (0 = silent)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be processed, don't run"),
):
    """Run a pipeline over many documents and write per-doc results.

    Each SOURCES argument is a path (file or directory) or @<file> for a
    text file with one path per line. Directories are recursively scanned
    for *.pdf, *.png, *.jpg, *.jpeg, *.tiff, *.txt.

    Output JSONL is one record per document with: path, ok, doc_id,
    schema, backend, classification, extraction, confidence, etc.
    A summary report (with totals, latency, error histogram) is written
    if --report is given. Failed docs go to --dlq if given.
    """
    import json
    import time
    from pathlib import Path

    from idp.batch import process_batch

    # 1. Discover paths from the source arguments.
    from idp.cli_sources import collect_paths
    from idp.pipeline.pipeline import Pipeline
    try:
        all_paths = collect_paths(sources)
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from None
    if limit is not None:
        all_paths = all_paths[:limit]

    if dry_run:
        console.print(f"[cyan]dry-run: would process {len(all_paths)} documents:[/cyan]")
        for p in all_paths[:20]:
            console.print(f"  {p}")
        if len(all_paths) > 20:
            console.print(f"  ... and {len(all_paths) - 20} more")
        return

    if not all_paths:
        console.print("[red]error:[/red] no documents found from sources")
        raise typer.Exit(code=1)

    console.print(f"[cyan]found {len(all_paths)} documents; backend={backend} schema={schema_name}[/cyan]")

    # 2. Build the pipeline.
    pipeline = Pipeline(backend=backend, schema=schema_name)

    # 3. Open output files. Using contextlib.ExitStack so we can
    # open 0-3 files based on which CLI flags were passed and
    # still close them all automatically on exit (even on error).
    import contextlib
    out_path = Path(output) if output else None
    dlq_path = Path(dlq) if dlq else None
    report_path = Path(report) if report else None

    # 4. Run the batch.
    started = time.perf_counter()
    counts = {"succeeded": 0, "failed": 0, "skipped": 0}
    latencies: list[float] = []
    error_types: dict[str, int] = {}

    with contextlib.ExitStack() as stack:
        out_fh = stack.enter_context(open(out_path, "w")) if out_path else None
        dlq_fh = stack.enter_context(open(dlq_path, "w")) if dlq_path else None
        try:
            results = process_batch(
                [str(p) for p in all_paths],
                pipeline,
                progress_every=progress_every,
                checkpoint=checkpoint,
            )
            for r in results:
                if r.ok:
                    counts["succeeded"] += 1
                else:
                    counts["failed"] += 1
                    # crude error-type bucketing
                    err = (r.error or "").split(":", 1)[0]
                    error_types[err] = error_types.get(err, 0) + 1
                    if dlq_fh is not None:
                        dlq_fh.write(json.dumps(r.to_dict()) + "\n")
                latencies.append(r.elapsed_seconds)

                line = json.dumps(r.to_dict()) + "\n"
                if out_fh is not None:
                    out_fh.write(line)
                else:
                    # write to stdout (one JSON object per line)
                    console.print(line.rstrip())

                if progress_every and (counts["succeeded"] + counts["failed"]) % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    done = counts["succeeded"] + counts["failed"]
                    console.print(
                        f"[dim]progress: {done}/{len(all_paths)} "
                        f"({100*done/len(all_paths):.0f}%) "
                        f"ok={counts['succeeded']} "
                        f"fail={counts['failed']} "
                        f"elapsed={elapsed:.1f}s[/dim]"
                    )
        finally:
            pass  # ExitStack closes out_fh / dlq_fh on exit

    # The report file is written AFTER the ExitStack exits (so we don't
    # hold the file open while the batch runs). Open it explicitly here.
    total_elapsed = time.perf_counter() - started
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies_sorted else 0.0

    summary = {
        "total": counts["succeeded"] + counts["failed"],
        "succeeded": counts["succeeded"],
        "failed": counts["failed"],
        "elapsed_seconds": round(total_elapsed, 3),
        "throughput_docs_per_sec": round((counts["succeeded"] + counts["failed"]) / total_elapsed, 3) if total_elapsed > 0 else 0,
        "latency_p50_sec": round(p50, 3),
        "latency_p95_sec": round(p95, 3),
        "error_types": error_types,
        "backend": backend,
        "schema": schema_name,
        "checkpoint": checkpoint,
    }

    if report_path is not None:
        report_path.write_text(json.dumps(summary, indent=2))
    else:
        # always print summary to stderr so stdout stays clean JSONL
        console.print(f"[bold green]summary:[/bold green] {json.dumps(summary)}")

    if counts["failed"]:
        raise typer.Exit(code=2)


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


@app.command(name="rl-update")
def rl_update(
    storage: str | None = typer.Option(None, "--storage", help="Path to JsonFileStorage results.jsonl OR sql URL"),
    reviews: str | None = typer.Option(None, "--reviews", help="Path to hand-crafted reviews JSONL"),
    db_url: str | None = typer.Option(None, "--db-url", help="SQL URL (env: IDP_DB_URL); read reviews from this DB"),
    output: str = typer.Option(..., "--output", "-o", help="Where to write policy.json"),
    current: str | None = typer.Option(None, "--current", help="Optional existing policy.json to update incrementally"),
):
    """Update the HITL policy from accumulated human reviews.

    Offline batch: reads all reviewed results, aggregates per-field reward
    signals (+1 if human corrected model output, 0 if accepted), and writes
    a new PolicyConfig JSON with per-field confidence floors and penalties.

    Honest about what this is: a deterministic rule update from human feedback,
    NOT a learned reward model or fine-tuned LLM. The point is to route the
    fields humans keep correcting into HITL more reliably.
    """
    if not storage and not reviews and not db_url:
        console.print("[red]error:[/red] must pass --storage, --reviews, or --db-url")
        raise typer.Exit(code=1)
    from idp.rl.policy import PolicyConfig
    from idp.rl.update import (
        update_policy_from_reviews_file,
        update_policy_from_sql,
        update_policy_from_storage,
    )

    cur = PolicyConfig.load(current) if current else None

    if db_url:
        new_policy = update_policy_from_sql(db_url, output, current=cur)
    elif reviews:
        new_policy = update_policy_from_reviews_file(reviews, output, current=cur)
    else:
        # path-based: heuristic detects if it's a URL
        assert storage is not None  # guaranteed by the not-all-falsy guard above
        if "://" in storage:
            new_policy = update_policy_from_sql(storage, output, current=cur)
        else:
            new_policy = update_policy_from_storage(storage, output, current=cur)

    table = Table(title=f"Updated policy → {output}")
    table.add_column("field", style="cyan")
    table.add_column("high_failure_floor", style="yellow", justify="right")
    table.add_column("penalty", style="yellow", justify="right")
    table.add_column("n_reviews", style="green", justify="right")
    for field_name, floor in new_policy.field_floors.items():
        penalty = new_policy.field_penalties.get(field_name, 0.0)
        table.add_row(field_name, f"{floor:.2f}", f"{penalty:.2f}", "—")
    console.print(table)
    console.print(f"[dim]threshold: {new_policy.high_failure_threshold:.2f}, base floor: {new_policy.base_confidence_floor:.2f}[/dim]")


@app.command(name="rl-eval")
def rl_eval(
    reviews: str | None = typer.Option(None, "--reviews", help="Real reviews JSONL"),
    policy: str = typer.Option(..., "--policy", help="policy.json to evaluate"),
    fixtures: str | None = typer.Option(
        None, "--fixtures",
        help="Eval dataset dir (implies --synthetic mode: reviews are generated from gold truth)"
    ),
    synthetic: bool = typer.Option(
        False, "--synthetic",
        help="Force synthetic mode even when --reviews is provided (useful for smoke-testing the synthetic pipeline)"
    ),
    synthetic_injection_rate: float = typer.Option(0.30, "--injection-rate", help="Fraction of fields to corrupt in --synthetic mode"),
    output: str | None = typer.Option(None, "--output", "-o"),
):
    """Evaluate whether the policy actually does what it claims.

    Modes:
      - Real reviews:    pass --reviews path/to/reviews.jsonl
      - Synthetic reviews: pass --fixtures path/to/dataset (auto-detected) or --synthetic
      - Both --fixtures and --synthetic: synthetic wins (clearest intent)

    Honest about scope: --synthetic generates biased-optimistic reviews from
    gold truth (the gold truth IS what the human corrected TO, so correction
    signals are artificially clean). Real HITL data is what makes these
    numbers trustworthy.

    Reports:
      - hit rate when policy fires (was the flagged field actually corrected?)
      - true-accept rate when policy silent (was the non-flagged field accepted?)
      - per-field breakdown
      - overall review-burden efficiency
    """
    from idp.rl.calibrate import (
        evaluate_policy,
        load_reviews,
        synthetic_reviews_from_gold,
    )
    from idp.rl.policy import PolicyConfig

    pol = PolicyConfig.load(policy)

    if reviews and not fixtures and not synthetic:
        revs = load_reviews(reviews)
        synthetic_mode = False
        notes_extra = "Real reviews (synthetic=false)."
    elif synthetic or fixtures:
        if fixtures is None:
            console.print("[red]error:[/red] --synthetic requires --fixtures <dataset_dir>")
            raise typer.Exit(code=1)
        revs, _ = synthetic_reviews_from_gold(
            fixtures, injection_rate=synthetic_injection_rate, seed=0,
        )
        synthetic_mode = True
        notes_extra = "SYNTHETIC reviews from gold truth."
    else:
        console.print("[red]error:[/red] pass --reviews or --fixtures (with --synthetic)")
        raise typer.Exit(code=1)

    report = evaluate_policy(revs, pol, synthetic=synthetic_mode)
    report.policy_path = policy
    report.notes.append(notes_extra)

    # Print table
    table = Table(title=f"RL calibration — {len(revs)} reviews ({'SYNTHETIC' if synthetic_mode else 'REAL'})")
    table.add_column("field", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("corrected_rate", style="yellow", justify="right")
    table.add_column("fires", style="red", justify="center")
    table.add_column("hit_rate_when_fires", style="green", justify="right")
    table.add_column("true_accept_when_silent", style="green", justify="right")
    for fe in report.field_evals:
        table.add_row(
            fe.field,
            str(fe.n_reviews),
            f"{fe.corrected_rate:.2f}",
            "yes" if fe.policy_fires else "no",
            f"{fe.hit_rate_when_fires:.2f}" if fe.hit_rate_when_fires is not None else "—",
            f"{fe.true_accept_rate_when_silent:.2f}" if fe.true_accept_rate_when_silent is not None else "—",
        )
    console.print(table)
    console.print()
    o = report.overall
    console.print(f"overall hit_rate_when_fires:    [bold green]{o.get('hit_rate_when_fires', 0):.2f}[/bold green]")
    console.print(f"overall true_accept_when_silent:[bold green]{o.get('true_accept_rate_when_silent', 0):.2f}[/bold green]")
    console.print(f"review_efficiency (hit/(hit+false)): [bold]{o.get('review_efficiency', 0):.2f}[/bold]")
    console.print()
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}")

    if output:
        Path(output).write_text(json.dumps(report.as_dict(), indent=2))
        console.print(f"\n[green]saved:[/green] {output}")


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
