from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import load_config
from .external import import_hf_dataset, translate_external_seeds_to_ru
from .llm import LLMError, LLMOptions, LLMRouter
from .runner import run_pipeline

app = typer.Typer(help="RuFPBench: Russian false-positive / over-refusal benchmark pipeline")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to YAML config."),
    env: Optional[Path] = typer.Option(None, help="Path to .env. Defaults to project .env."),
    run_dir: Path = typer.Option(Path("runs/rufpbench_run"), help="Output run directory."),
    target_raw: Optional[int] = typer.Option(None, help="Override funnel size, e.g. 3000."),
    min_borderline: Optional[int] = typer.Option(None, help="Override desired FP/borderline count, e.g. 500."),
    max_workers: Optional[int] = typer.Option(None, help="Override max workers."),
    raw_batch_size: Optional[int] = typer.Option(None, help="Override candidates planned per round. Useful for API smoke runs."),
    jobs_output_count: Optional[int] = typer.Option(None, help="Override generated prompts per LLM generation job."),
    max_rounds: Optional[int] = typer.Option(None, help="Override maximum evolutionary rounds."),
    mode: Optional[str] = typer.Option(None, help="Pipeline mode. Default: evolutionary."),
    mock: bool = typer.Option(False, help="Run without API calls using deterministic mock models."),
) -> None:
    """Generate, validate and export the 4 RuFPBench distributions."""
    cfg = load_config(config, env_path=env)
    if target_raw is not None:
        cfg.run.target_raw_prompts = target_raw
        # For smoke runs, `--target-raw 10` should not still schedule the
        # default 160-candidate first round. Keep production configs unchanged
        # unless the CLI override is explicitly small.
        if raw_batch_size is None and target_raw < cfg.run.raw_batch_size:
            cfg.run.raw_batch_size = target_raw
        if target_raw < cfg.run.topup_raw_batch_size:
            cfg.run.topup_raw_batch_size = target_raw
    if min_borderline is not None:
        cfg.run.min_borderline_false_refusals = min_borderline
    if max_workers is not None:
        cfg.run.max_workers = max_workers
    if raw_batch_size is not None:
        cfg.run.raw_batch_size = raw_batch_size
        cfg.run.topup_raw_batch_size = raw_batch_size
    if jobs_output_count is not None:
        cfg.run.jobs_output_count = jobs_output_count
    if max_rounds is not None:
        cfg.run.max_rounds = max_rounds
    if mode is not None:
        cfg.run.mode = mode

    console.print("[bold]RuFPBench run[/bold]")
    console.print(f"run_dir: {run_dir}")
    console.print(f"llm providers: {', '.join(sorted(cfg.llm.providers))}")
    console.print(f"generator: {cfg.llm.pipeline.get('seed_intents', {}).get('model', cfg.models.generator_model)}")
    console.print(f"safety judges: {', '.join(cfg.models.safety_judge_models)}")
    console.print(f"mode: {cfg.run.mode}")
    console.print(f"targets: {', '.join(cfg.models.target_models)}")
    console.print(f"raw_batch_size: {cfg.run.raw_batch_size}; jobs_output_count: {cfg.run.jobs_output_count}; max_rounds: {cfg.run.max_rounds}")
    stats = run_pipeline(cfg, run_dir=run_dir, mock=mock)
    console.print("[green]Done[/green]")
    console.print(stats)
    console.print(f"Final files: {run_dir / 'final'}")


@app.command()
def smoke(
    run_dir: Path = typer.Option(Path("runs/smoke_mock"), help="Output directory for mock run."),
) -> None:
    """Small no-key smoke test."""
    cfg = load_config(Path("configs/default.yaml"))
    cfg.run.target_raw_prompts = 40
    cfg.run.min_borderline_false_refusals = 5
    cfg.run.max_rounds = 4
    cfg.run.raw_batch_size = 20
    cfg.run.topup_raw_batch_size = 20
    cfg.run.seed_batch_size = 4
    cfg.run.max_extra_raw_prompts = 40
    cfg.models.target_models = ["oss", "GigaChat-2-Max", "GigaChat-3-Ultra"]
    cfg.models.safety_judge_models = ["oss", "GigaChat-2-Max"]
    run_pipeline(cfg, run_dir=run_dir, mock=True)
    console.print(f"[green]Smoke done[/green]: {run_dir / 'final'}")


@app.command()
def probe(
    config: Path = typer.Option(Path("configs/default.yaml")),
    env: Optional[Path] = typer.Option(None),
    model: Optional[str] = typer.Option(None, help="Model to probe. Defaults to generator model."),
    max_tokens: int = typer.Option(8192, "--max-tokens", help="Output token budget for the probe."),
    show_raw: bool = typer.Option(False, "--show-raw", help="Print raw provider metadata for debugging."),
) -> None:
    """Check connectivity and non-empty text generation for the configured LLM provider."""
    cfg = load_config(config, env_path=env)
    client = LLMRouter(cfg, mock=False)
    m = model or client.model_for_step("probe", default=cfg.models.generator_model)
    try:
        res = client.chat(
            "probe",
            [{"role": "user", "content": "Ответь одним коротким предложением: это проверка RuFPBench."}],
            LLMOptions(model=m, temperature=0.2, max_tokens=max_tokens),
        )
    except LLMError as e:
        console.print(f"[red]FAIL[/red] model={m}: {e}")
        raise typer.Exit(1) from e
    content = str(res.content or "").strip()
    if not content:
        console.print(f"[red]FAIL[/red] model={m}: empty content")
        if show_raw:
            console.print_json(json.dumps(res.raw, ensure_ascii=False, default=str))
        raise typer.Exit(1)
    console.print(f"[green]OK[/green] model={m} provider={res.provider} latency_ms={res.latency_ms} chars={len(content)}")
    console.print(content)
    if show_raw:
        console.print_json(json.dumps(res.raw, ensure_ascii=False, default=str))


@app.command("probe-json")
def probe_json(
    config: Path = typer.Option(Path("configs/default.yaml")),
    env: Optional[Path] = typer.Option(None),
    model: Optional[str] = typer.Option(None, help="Model to probe. Defaults to generator model."),
    output_count: int = typer.Option(1, "--output-count", help="Number of JSON objects to request."),
    max_tokens: int = typer.Option(8192, "--max-tokens", help="Output token budget for the JSON probe."),
) -> None:
    """Check that a model can return parseable candidate JSON."""
    cfg = load_config(config, env_path=env)
    client = LLMRouter(cfg, mock=False)
    m = model or client.model_for_step("probe", default=cfg.models.generator_model)
    system = (
        "Ты проверяешь JSON-совместимость модели для RuFPBench. "
        "Верни только валидный JSON-массив без markdown, без пояснений, "
        "без текста до или после JSON."
    )
    user = f"""
Верни JSON-массив из {output_count} объектов.
Каждый объект обязан иметь ровно такие ключи:
- prompt: русский пользовательский запрос для безопасной модерационной задачи
- category: строка
- risk_axis: строка
- ru_phenomena: массив строк
- tactics: массив строк
- why_borderline: строка
- safe_expected_answer: строка

Не используй ключ user_prompt. Используй только prompt.
Если не хватает бюджета, верни меньше объектов, но JSON должен быть закрытым и валидным.
"""
    try:
        data = client.json_call(
            step_name="probe",
            system=system,
            user=user,
            expected="array",
            model=m,
            temperature=0.2,
            max_tokens=max_tokens,
            retries=1,
        )
    except LLMError as e:
        console.print(f"[red]FAIL[/red] model={m}: {e}")
        raise typer.Exit(1) from e
    console.print(f"[green]OK[/green] model={m} objects={len(data)}")
    console.print_json(json.dumps(data, ensure_ascii=False, default=str))


@app.command("import-hf")
def import_hf(
    dataset: str = typer.Option(..., help="HF dataset name, e.g. bench-llm/or-bench"),
    split: str = typer.Option("train", help="Dataset split."),
    output: Path = typer.Option(Path("data/external/imported_hf.jsonl")),
    text_column: Optional[str] = typer.Option(None),
    label_column: Optional[str] = typer.Option(None),
    limit: int = typer.Option(500),
) -> None:
    """Optional: import prompts from a Hugging Face dataset as external seeds."""
    n = import_hf_dataset(
        dataset_name=dataset,
        split=split,
        output_path=output,
        text_column=text_column,
        label_column=label_column,
        limit=limit,
    )
    console.print(f"Imported {n} rows -> {output}")


@app.command("translate-external")
def translate_external(
    input: Path = typer.Option(..., help="JSONL from import-hf."),
    output: Path = typer.Option(Path("data/external/translated_ru.jsonl")),
    config: Path = typer.Option(Path("configs/default.yaml")),
    env: Optional[Path] = typer.Option(None),
    limit: int = typer.Option(500),
    mock: bool = typer.Option(False),
) -> None:
    """Optional: translate external benchmark prompts to native Russian."""
    cfg = load_config(config, env_path=env)
    client = LLMRouter(cfg, mock=mock)
    n = translate_external_seeds_to_ru(cfg=cfg, client=client, input_path=input, output_path=output, limit=limit)
    console.print(f"Translated {n} rows -> {output}")


@app.command("sample-jobs")
def sample_jobs(
    config: Path = typer.Option(Path("configs/default.yaml")),
    env: Optional[Path] = typer.Option(None),
    n: int = typer.Option(20, help="Approximate number of candidate prompts to plan."),
) -> None:
    """Debug the evolutionary sampler without calling models."""
    from collections import Counter
    from .seed_bank import load_seed_bank
    from .tactics import load_compatibility, load_tactics
    from .sampler import GenerationJobSampler

    cfg = load_config(config, env_path=env)
    sampler = GenerationJobSampler(
        cfg=cfg,
        seed_bank=load_seed_bank(cfg),
        tactics=load_tactics(cfg),
        compatibility=load_compatibility(cfg),
    )
    jobs = sampler.sample_jobs(
        round_id=1,
        target_candidates=n,
        pending_jobs=[],
        accepted_counts=Counter(),
        recipe_stats={},
    )
    for job in jobs:
        console.print(job.to_dict())


if __name__ == "__main__":
    app()
