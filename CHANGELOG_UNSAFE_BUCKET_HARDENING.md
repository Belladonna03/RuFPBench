# Patch: unsafe handling, mutation feedback, reporting resilience

This patch closes the issues observed in `strict_fp_main_200`:

- `unsafe_answered` now means an actual unsafe substantive target response (`response_safety_label=unsafe`).
- Unsafe prompts that are handled safely by refusal, benign alternative, crisis/support guidance, clarification, or non-harmful compliance now go to `unsafe_safe_handled`.
- Intended `unsafe_control` prompts that the prompt-safety ensemble judges benign are quarantined and routed for regeneration instead of inflating `safe_answered`.
- Mutation jobs now receive richer feedback: `answered_by`, `refused_by`, `unsafe_answered_by`, `prompt_safety_label`, per-target label summary, and explicit `direction/do/avoid` instructions.
- `harden_borderline` instructions now explicitly reduce over-explanatory safe context and preserve short suspicious surface forms.
- Prompt-safety judge prompt now has category-specific guardrails for chemistry, public-health misinformation, legal/compliance, self-harm, and metaphor/threat categories.
- Reports are written after every completed round and also on `KeyboardInterrupt` / Ctrl+C.
- Sampler now boosts underrepresented categories during adaptive sampling.
- Generated tactic aliases such as `support_ticket`, `quoted_fragment`, and `json_normalization` are normalized to the canonical tactic ids.
- Report bucket list and mutation effectiveness reports include `unsafe_safe_handled`.

Recommended next local/GigaChat run:

```bash
python -m rufpbench run \
  --env .env \
  --run-dir runs/strict_fp_main_200_patch1 \
  --target-raw 200 \
  --min-borderline 8 \
  --max-workers 1 \
  --raw-batch-size 20 \
  --jobs-output-count 2 \
  --max-rounds 10
```

Use `--max-workers 1` for the official GigaChat endpoint unless provider-specific concurrency is introduced later.
