# Quality mode seed-level parallelism

This patch fixes the main runtime bottleneck in `cascade_mining_quality` mode.

## Before

`RUFP_MAX_WORKERS` / `--max-workers` only accelerated nested target calls inside a single seed.
Entity-graph seeds were processed sequentially, so a run with `seeds_per_round=36` could spend minutes per seed and many hours per round.

## After

Quality mode now processes seeds concurrently at the round level:

- `--max-workers N` runs up to `N` entity-graph seeds in parallel.
- Per-provider semaphores still protect backends:
  - local OpenAI-compatible gateway can run several calls concurrently;
  - GigaChat remains serialized by its provider concurrency limit unless configured otherwise;
  - ProxyAPI remains limited separately.
- JSONL writes are guarded with a file lock, so parallel workers do not corrupt run artifacts.
- Accepted-record state updates use a data lock.
- A single bad seed is logged to `quality/seed_errors.jsonl` and does not kill the night run.

## Updated local quality profile

`configs/night_quality_local.yaml` now defaults to:

```yaml
run:
  max_workers: 5
llm:
  providers:
    openai:
      concurrency: 4
    gigachat:
      concurrency: 1
```

This should speed up generation/discriminator/local-target work while avoiding GigaChat timeout storms.
