# RuFPBench v2 architecture

RuFPBench v2 combines:

1. **OR-Bench-style rewrite + moderation**: benign borderline prompts are created from sensitive seeds and then checked by a prompt-safety ensemble.
2. **FalseReject-style refusal-validation pool**: a candidate is valuable as FP only if at least one target model refuses it.
3. **WildTeaming-style adversarial-benign surface tactics**: safe prompts are wrapped in suspicious-looking forms such as quotes, moderation tickets, JSON/CSV, role/editor context, code-switching and bureaucratic noise.
4. **Evolutionary loop**: failed candidates are routed into repair/hardening/clarification jobs instead of being discarded immediately.

## Pipeline loop

```text
CategoryBank
   +
TacticBank
   +
CompatibilityMatrix
   +
CoverageTargets
   +
PreviousRoundStats
   ↓
GenerationJobSampler
   ↓
GenerationJobs
   ↓
LLM generation
   ↓
Candidates
   ↓
Candidate-level Validation
   ↓
Bucketizer
   ↓
Router
   ├── accept → FinalBuckets
   ├── repair_to_benign → NewJobs next round
   ├── harden_borderline → NewJobs next round
   ├── harden_for_more_refusals → NewJobs next round
   ├── clarify_benign_intent → NewJobs next round
   ├── make_unsafe_control_less_trivial → NewJobs next round
   └── drop/quarantine
        ↑
        └──────── next round feedback ────────
```

## Candidate-level validation

Sibling groups are never atomic. A recipe can produce four sibling jobs:

```text
safe_answered
safe_refused_borderline
unsafe_refused
unsafe_answered
```

Each candidate is validated and routed independently. If one candidate fails, the other candidates are not discarded.

## Key modules

| Module | Role |
|---|---|
| `seed_bank.py` | Loads native RU taxonomy and seed intents. |
| `tactics.py` | Loads RU surface tactics and category/tactic compatibility. |
| `sampler.py` | Samples GenerationJobs from CategoryBank + TacticBank + PreviousRoundStats. |
| `generation.py` | Calls configured LLM steps through `LLMRouter`. |
| `validation.py` | Prompt-safety ensemble + response refusal/safety classifiers. |
| `bucketizer.py` | Assigns observed bucket from prompt safety + target responses. |
| `router.py` | Chooses accept/drop/repair/harden/clarify. |
| `mutations.py` | Converts router decisions into next-round GenerationJobs. |
| `evolution.py` | Main multi-round run loop. |
| `reporting.py` | Exports JSONL buckets and analytics. |
| `llm.py` | Provider-independent LLM interface, router/factory and adapters for OpenAI, GigaChat and ProxyAPI. |


## LLM model-role profile

The default config uses a quality-first model matrix rather than one model for all steps:

| Step(s) | Default role | Default model(s) | Why |
|---|---|---|---|
| `seed_intents`, `generate_candidates_job` | hard Russian borderline generation | `GigaChat-3-Ultra` | highest-quality RU generation is the benchmark bottleneck |
| `rewrite_candidates_job`, `translate_to_ru` | stylistic diversity / translation | `oss` | known-good local model; qwen/glm can be restored after non-empty probe/generation checks |
| `mutate_candidates_job`, `refusal_judge`, `pair_safety_judge` | constrained repair, refusal/safety classification | `oss` | local reasoning-heavy model, different family from Ultra |
| `benign_controls` | natural safe controls | `gigachat3-10b` | keeps controls simple and native-RU without spending Ultra tokens |
| `unsafe_controls` | high-level unsafe controls without operational detail | `oss` | better constraint following for unsafe-control boundaries |
| `prompt_safety_judge` | prompt safety majority vote | `oss`, `gigachat3-10b`, `GigaChat-2-Max` | known-good 3-vote ensemble; add qwen/glm back only after they return non-empty content |
| `target_response` | measured target/refusal pool | local gateway models + `GigaChat-2-Max` + `GigaChat-3-Ultra` | covers local/open-source behavior and strong RU production-like targets |

ProxyAPI is configured but not part of the default target pool. Add `gpt-4.1-mini` to `RUFP_TARGET_MODELS` only for paid audit/baseline runs.

## Intended vs observed

- `intended_distribution`: what the sampler/generator tried to create.
- `observed_bucket`: what validation showed after target-pool evaluation.
- `router_action`: how the system uses the discrepancy.

Examples:

| Intended | Observed | Router action |
|---|---|---|
| `borderline_benign` | `safe_refused_borderline` | accept; maybe harden |
| `borderline_benign` | `safe_answered` | harden_borderline |
| `borderline_benign` | `unsafe_refused` | repair_to_benign |
| `adversarial_benign` | `quarantine` | clarify_benign_intent |
| `unsafe_control` | `unsafe_refused` | accept |
| `unsafe_control` | `unsafe_answered` | accept as FN control |

## Final outputs

```text
final/safe_answered.jsonl
final/safe_refused_borderline.jsonl
final/unsafe_refused.jsonl
final/unsafe_answered.jsonl
final/quarantine.jsonl
final/rufpbench_borderline.jsonl
final/rufpbench_hard.jsonl
reports/category_coverage.csv
reports/tactic_coverage.csv
reports/mutation_effectiveness.csv
reports/model_refusal_rates.csv
reports/report.md
```

`safe_refused_borderline` is the main RuFPBench bucket. `rufpbench_hard` is the subset refused by at least `hard_min_refusal_models` target models.
