import typer
import asyncio
import json
import logging
import os
from typing import Any, Optional

from .policy.loader import Stage25ConfigError, load_stage25_policy
from .pipeline.orchestrator import run_stage25_pipeline
from .nodes.repair_candidate_selector import RepairCandidateSelectorNode
from .nodes.repair_planner import RepairPlannerNode
from .nodes.prompt_repairer import PromptRepairerNode
from .llm.mock_client import MockLLMClient
from .schemas import RepairCandidate, RepairPlan
from .io import save_jsonl, load_jsonl
from .audit.show_lineage import find_lineage_by_repaired_id, format_lineage_human
from .analysis.repair_audit_report import write_audit_artifacts
from .nodes.revalidation_runner import run_revalidation_batch
from .nodes.repair_decision_aggregator import run_aggregator
from .schemas import RepairedPrompt

app = typer.Typer()

@app.command()
def run_repair_planner(
    input_path: str = typer.Option(..., help="Path to repair_candidates.jsonl"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    config: Optional[str] = typer.Option(None, "--config", help="stage25.yaml (allowed repair_strategies)"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
):
    """Run Node 2: Repair Planner."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)

    try:
        policy = load_stage25_policy(config)
    except Stage25ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    output_dir = f"artifacts/stage25/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    allowed = policy.repair_strategies.as_set()

    async def _run():
        typer.echo(f"Planning repairs for candidates in {input_path}...")

        candidates = load_jsonl(input_path, RepairCandidate)

        # Setup node
        client = MockLLMClient()
        if mock:

            class MockPlannerClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    return json.dumps(
                        {
                            "repair_strategy": "strengthen_borderline_surface",
                            "repair_goals": ["Добавить слово 'убить'"],
                            "must_preserve": ["Безопасный смысл"],
                            "must_avoid": ["Реальное насилие"],
                            "expected_risk": "low",
                            "reasoning": "Mock reasoning",
                            "instructions": "Замените 'провести время' на 'убить время'",
                        }
                    )

            client = MockPlannerClient()

        node = RepairPlannerNode(client, "src/rufp_stage25/prompts/repair_planner.jinja2")

        plans = []
        for c in candidates:
            plan = await node.plan_repair(c, allowed_strategies=allowed)
            if plan:
                plans.append(plan)
        
        output_path = os.path.join(output_dir, "repair_plans.jsonl")
        save_jsonl(output_path, plans)
        typer.echo(f"Planned {len(plans)} repairs. Saved to {output_path}")

    asyncio.run(_run())


@app.command("run-prompt-repairer")
def run_prompt_repairer(
    plans_path: str = typer.Option(..., help="Path to repair_plans.jsonl"),
    candidates_path: str = typer.Option(..., help="Path to repair_candidates.jsonl"),
    run_id: str = typer.Option(..., help="Stage 2.5 run id (output under artifacts/stage25/<run_id>)"),
    stage2_run_id: str = typer.Option(..., help="Parent Stage 2 run id (lineage)"),
    mock: bool = typer.Option(True, help="Use mock LLM"),
    debug: bool = typer.Option(False, help="Debug logging"),
):
    """Node 3: apply repair plans; writes repaired_prompts.jsonl + repair_lineage.jsonl."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)

    async def _run():
        typer.echo("Running prompt repairer...")
        plans = load_jsonl(plans_path, RepairPlan)
        cands = load_jsonl(candidates_path, RepairCandidate)
        by_id = {c.original_prompt_id: c for c in cands}

        client = MockLLMClient()
        if mock:

            class M(MockLLMClient):
                async def generate(self, prompt: str, **kwargs: Any) -> str:
                    # Minimal change so rewrite_changed is often True
                    if "ИСХОДНЫЙ ТЕКСТ" in prompt or "text" in prompt.lower():
                        for line in prompt.splitlines():
                            if line.strip().startswith('"') and len(line) > 3:
                                inner = line.strip().strip('"')
                                return inner + " [repaired]"
                    return "repaired output"

            client = M()

        node = PromptRepairerNode(client, "src/rufp_stage25/prompts/prompt_repairer.jinja2")
        out_dir = f"artifacts/stage25/{run_id}"
        os.makedirs(out_dir, exist_ok=True)

        repaired_rows = []
        lineage_rows = []
        for plan in plans:
            c = by_id.get(plan.original_prompt_id)
            if not c:
                typer.echo(f"[warn] no candidate for {plan.original_prompt_id}, skip", err=True)
                continue
            rp, lin = await node.repair_one(
                c,
                plan,
                parent_stage2_run_id=stage2_run_id,
                parent_stage25_run_id=run_id,
            )
            repaired_rows.append(rp)
            lineage_rows.append(lin)

        save_jsonl(os.path.join(out_dir, "repaired_prompts.jsonl"), repaired_rows)
        save_jsonl(os.path.join(out_dir, "repair_lineage.jsonl"), lineage_rows)

        reports_dir = os.path.join(out_dir, "reports")
        write_audit_artifacts(run_id, os.path.join(out_dir, "repair_lineage.jsonl"), reports_dir)

        typer.echo(f"Wrote {len(repaired_rows)} repaired prompts and lineage to {out_dir}")

    asyncio.run(_run())


@app.command("run-repair-aggregator")
def run_repair_aggregator_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Stage 2.5 run id (artifact folder)"),
    config: Optional[str] = typer.Option(None, "--config", help="stage25.yaml (aggregator policy)"),
):
    """Node 5: promote / fail / review after revalidation; writes sets + stage25_summary.json."""
    try:
        policy = load_stage25_policy(config)
    except Stage25ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    summary = run_aggregator(run_id, aggregator_policy=policy.aggregator)
    typer.echo(
        f"promoted={summary.promoted} failed={summary.failed} review={summary.review} "
        f"-> artifacts/stage25/{run_id}/"
    )


@app.command("run-revalidation")
def run_revalidation(
    input_path: str = typer.Option(..., "--input", help="Path to repaired_prompts.jsonl"),
    run_id: str = typer.Option(..., "--run-id", help="Stage 2.5 run id (output directory)"),
    mock: bool = typer.Option(True, "--mock", help="Use Stage 2 mock judge/probe clients"),
    config: Optional[str] = typer.Option(None, "--config", help="stage25.yaml (run_probes, default models)"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated probe model ids (overrides config)"),
    only_ids: Optional[str] = typer.Option(
        None,
        "--only-ids",
        help="Subset: comma-separated repaired_prompt_id (skip full batch)",
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """Node 4: re-run Stage 2 validation on repaired prompts."""

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)

    async def _run():
        try:
            policy = load_stage25_policy(config)
        except Stage25ConfigError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)
        repaired = load_jsonl(input_path, RepairedPrompt)
        oid: Optional[set] = None
        if only_ids:
            oid = {x.strip() for x in only_ids.split(",") if x.strip()}
        default_models = ",".join(policy.revalidation.model_names)
        model_list = [m.strip() for m in (models or default_models).split(",") if m.strip()]
        results = await run_revalidation_batch(
            repaired,
            mock=mock,
            model_names=model_list,
            only_ids=oid,
            run_probes=policy.revalidation.run_probes,
        )
        out_dir = f"artifacts/stage25/{run_id}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "repair_revalidation_results.jsonl")
        save_jsonl(out_path, results)
        typer.echo(f"Wrote {len(results)} revalidation rows to {out_path}")

    asyncio.run(_run())


@app.command("show-repair-lineage")
def show_repair_lineage(
    repaired_prompt_id: str = typer.Option(..., "--repaired-prompt-id"),
    run_id: str = typer.Option(None, help="Limit search to artifacts/stage25/<run_id>"),
):
    """Print full human-readable chain for a repaired prompt id."""
    found = find_lineage_by_repaired_id(repaired_prompt_id, run_id=run_id)
    if not found:
        typer.echo(f"No lineage found for repaired_prompt_id={repaired_prompt_id}", err=True)
        raise typer.Exit(code=1)
    path, rec = found
    typer.echo(f"(from {path})\n")
    typer.echo(format_lineage_human(rec))


@app.command()
def run_repair_selector(
    stage2_run_id: str = typer.Option(..., help="Run ID from Stage 2"),
    run_id: str = typer.Option(..., help="New Run ID for Stage 2.5"),
    config: Optional[str] = typer.Option(None, "--config", help="stage25.yaml (repair_reason / limits)"),
):
    """Run Node 1: Repair Candidate Selector."""
    typer.echo(f"Selecting repair candidates from Stage 2 run: {stage2_run_id}")
    try:
        policy = load_stage25_policy(config)
    except Stage25ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    node = RepairCandidateSelectorNode(stage2_run_id)
    candidates = node.select_candidates(policy)
    
    output_dir = f"artifacts/stage25/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "repair_candidates.jsonl")
    
    save_jsonl(output_path, candidates)
    typer.echo(f"Selected {len(candidates)} candidates. Saved to {output_path}")

@app.command("run-stage25")
def run_stage25(
    stage2_run_id: str = typer.Option(..., "--stage2-run-id", help="Stage 2 run id (artifacts/stage2/<id>)"),
    run_id: str = typer.Option(..., "--run-id", help="Stage 2.5 run id (artifacts/stage25/<id>)"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to stage25.yaml (default: env RUFP_STAGE25_CONFIG or configs/stage25.yaml)"),
    resume: bool = typer.Option(False, "--resume", help="Skip nodes whose outputs already exist"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Force mock clients; structured logs tagged dry_run"),
    force_mock: bool = typer.Option(False, "--force-mock", help="Force mock LLM / judge clients (overrides config)"),
    real_clients: bool = typer.Option(False, "--real", help="Force real clients (overrides config runtime.use_mock_clients)"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated probe model ids (overrides config)"),
    no_analysis: bool = typer.Option(False, "--no-analysis", help="Skip post-run failure report + metrics"),
    debug: bool = typer.Option(False, "--debug", help="Debug logging"),
):
    """End-to-end Stage 2.5: selector → planner → repairer → revalidation → aggregator → lineage → report."""
    if force_mock and real_clients:
        typer.echo("Use either --force-mock or --real, not both.", err=True)
        raise typer.Exit(code=1)
    mock_arg: Optional[bool] = None
    if force_mock:
        mock_arg = True
    elif real_clients:
        mock_arg = False

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    try:
        asyncio.run(
            run_stage25_pipeline(
                stage2_run_id,
                run_id,
                mock=mock_arg,
                resume=resume,
                dry_run=dry_run,
                models=models,
                run_post_analysis=not no_analysis,
                config_path=config,
            )
        )
    except Stage25ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)


@app.command()
def run(
    input_run_id: str = typer.Option(..., help="Run ID from Stage 2"),
    run_id: str = typer.Option(..., help="New Run ID for Stage 2.5"),
    mock: bool = typer.Option(True, help="Use mock clients (passed through to pipeline)"),
    resume: bool = typer.Option(False, help="Resume from existing artifacts"),
    dry_run: bool = typer.Option(False, help="Alias: mock + dry-run logging"),
    config: Optional[str] = typer.Option(None, "--config", help="stage25.yaml"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
):
    """Run Stage 2.5 (legacy alias for run-stage25)."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    try:
        asyncio.run(
            run_stage25_pipeline(
                input_run_id,
                run_id,
                mock=mock,
                resume=resume,
                dry_run=dry_run,
                run_post_analysis=True,
                config_path=config,
            )
        )
    except Stage25ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
