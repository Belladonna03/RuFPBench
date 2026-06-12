from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .config import AppConfig
from .schemas import GenerationJob, RecipeStats, SeedIntent, TacticSpec
from .tactics import compatible_tactics
from .utils import stable_id


class GenerationJobSampler:
    def __init__(
        self,
        *,
        cfg: AppConfig,
        seed_bank: list[SeedIntent],
        tactics: dict[str, TacticSpec],
        compatibility: dict[str, Any],
    ):
        self.cfg = cfg
        self.rng = random.Random(cfg.run.random_seed)
        # The seed bank can be clustered by source/category. Shuffle once with
        # the run seed so round-robin still gives reproducible but broader
        # category coverage.
        self.seed_bank = list(seed_bank)
        self.rng.shuffle(self.seed_bank)
        self.tactics = tactics
        self.compatibility = compatibility
        self._seed_cursor = 0

    def sample_jobs(
        self,
        *,
        round_id: int,
        target_candidates: int,
        pending_jobs: list[GenerationJob],
        accepted_counts: Counter[str],
        recipe_stats: dict[str, RecipeStats],
        category_counts: Counter[str] | None = None,
    ) -> list[GenerationJob]:
        jobs: list[GenerationJob] = []
        # Router-created jobs get priority; they are the feedback loop.
        while pending_jobs and sum(j.output_count for j in jobs) < target_candidates:
            jobs.append(pending_jobs.pop(0))

        remaining = max(0, target_candidates - sum(j.output_count for j in jobs))
        if remaining <= 0:
            return jobs

        # Hard floor for controls per new-generation batch. Adaptive FP mining
        # can otherwise push almost all jobs into borderline/adversarial benign,
        # which makes the run look good on FP but weak on safety controls.
        remaining = self._add_control_floor_jobs(
            jobs=jobs,
            round_id=round_id,
            remaining=remaining,
            target_candidates=target_candidates,
            recipe_stats=recipe_stats,
            category_counts=category_counts,
        )
        if remaining <= 0:
            return jobs

        mix = self._adaptive_mix(accepted_counts)
        while remaining > 0:
            target = self._sample_distribution(mix)
            seed = self._sample_seed(recipe_stats=recipe_stats, target_distribution=target, category_counts=category_counts)
            output_count = min(self.cfg.run.jobs_output_count, remaining)
            job = self._make_job(seed=seed, target_distribution=target, round_id=round_id, output_count=output_count)
            jobs.append(job)
            remaining -= output_count
        return jobs

    def _add_control_floor_jobs(
        self,
        *,
        jobs: list[GenerationJob],
        round_id: int,
        remaining: int,
        target_candidates: int,
        recipe_stats: dict[str, RecipeStats],
        category_counts: Counter[str] | None = None,
    ) -> int:
        existing: Counter[str] = Counter()
        for job in jobs:
            existing[job.target_distribution] += max(1, int(job.output_count or 1))

        floors = {
            "benign_control": int(math.ceil(target_candidates * float(self.cfg.run.safe_control_ratio or 0.0))),
            "unsafe_control": int(math.ceil(target_candidates * float(self.cfg.run.unsafe_control_ratio or 0.0))),
        }
        for target_distribution, floor in floors.items():
            need = max(0, floor - existing.get(target_distribution, 0))
            while need > 0 and remaining > 0:
                seed = self._sample_seed(recipe_stats=recipe_stats, target_distribution=target_distribution, category_counts=category_counts)
                output_count = min(self.cfg.run.jobs_output_count, need, remaining)
                jobs.append(self._make_job(
                    seed=seed,
                    target_distribution=target_distribution,
                    round_id=round_id,
                    output_count=output_count,
                ))
                remaining -= output_count
                need -= output_count
        return remaining

    def _adaptive_mix(self, accepted_counts: Counter[str]) -> dict[str, float]:
        mix = dict(self.cfg.run.distribution_mix)
        fp_have = accepted_counts.get("safe_refused_borderline", 0)
        fp_need = self.cfg.run.min_borderline_false_refusals
        # If FP bucket is still low, push more borderline/adversarial benign jobs.
        if fp_have < fp_need:
            mix["borderline_benign"] = mix.get("borderline_benign", 0.0) * 1.25
            mix["adversarial_benign"] = mix.get("adversarial_benign", 0.0) * 1.15
        total = sum(max(0.0, v) for v in mix.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in mix.items()}

    def _sample_distribution(self, mix: dict[str, float]) -> str:
        roll = self.rng.random()
        acc = 0.0
        for k, v in mix.items():
            acc += v
            if roll <= acc:
                return k
        return "borderline_benign"

    def _sample_seed(
        self,
        *,
        recipe_stats: dict[str, RecipeStats],
        target_distribution: str,
        category_counts: Counter[str] | None = None,
    ) -> SeedIntent:
        if not self.seed_bank:
            raise RuntimeError("Seed bank is empty")

        # Round-robin gives reproducibility. When accepted records already show
        # category skew, prefer underrepresented categories before falling back
        # to pure round-robin. This keeps long FP-mining runs from collapsing
        # into a few high-yield themes such as metaphor_threats.
        underrepresented: set[str] = set()
        if self.cfg.run.adaptive_sampling and category_counts:
            all_categories = {s.category for s in self.seed_bank}
            if all_categories:
                total = sum(category_counts.get(c, 0) for c in all_categories)
                avg = total / max(1, len(all_categories))
                threshold = max(1.0, avg * 0.65)
                underrepresented = {c for c in all_categories if category_counts.get(c, 0) <= threshold}

        fallback: SeedIntent | None = None
        for _ in range(min(80, len(self.seed_bank) * 2)):
            seed = self.seed_bank[self._seed_cursor % len(self.seed_bank)]
            self._seed_cursor += 1
            if fallback is None:
                fallback = seed
            if not self.cfg.run.adaptive_sampling:
                return seed
            pseudo_recipe = f"{seed.category}__{target_distribution}"
            st = recipe_stats.get(pseudo_recipe)
            too_unsafe = st is not None and st.attempts >= 20 and st.unsafe_rate >= 0.45
            if too_unsafe:
                continue
            if underrepresented and seed.category not in underrepresented:
                continue
            return seed
        return fallback or self.rng.choice(self.seed_bank)

    def _select_tactics(self, *, seed: SeedIntent, target_distribution: str) -> list[str]:
        if target_distribution == "benign_control":
            candidates = compatible_tactics(
                category=seed.category,
                target_distribution=target_distribution,
                tactics=self.tactics,
                compatibility=self.compatibility,
                strength="weak",
            )
            k = 0 if self.rng.random() < 0.6 else 1
        elif target_distribution == "unsafe_control":
            candidates = compatible_tactics(
                category=seed.category,
                target_distribution="borderline_benign",
                tactics=self.tactics,
                compatibility=self.compatibility,
                strength="weak",
            )
            k = 0 if self.rng.random() < 0.75 else 1
        elif target_distribution == "adversarial_benign":
            candidates = compatible_tactics(
                category=seed.category,
                target_distribution=target_distribution,
                tactics=self.tactics,
                compatibility=self.compatibility,
                strength="all",
            )
            k = self.rng.choice([2, 2, 3])
        else:
            candidates = compatible_tactics(
                category=seed.category,
                target_distribution=target_distribution,
                tactics=self.tactics,
                compatibility=self.compatibility,
                strength="strong",
            )
            k = self.rng.choice([1, 2, 2])
        if not candidates or k <= 0:
            return []
        return self.rng.sample(candidates, k=min(k, len(candidates)))

    def _make_job(self, *, seed: SeedIntent, target_distribution: str, round_id: int, output_count: int) -> GenerationJob:
        tactic_ids = self._select_tactics(seed=seed, target_distribution=target_distribution)
        recipe_id = "__".join([seed.category, target_distribution, *sorted(tactic_ids)])
        sibling_group_id = stable_id("sib", round_id, recipe_id, seed.seed_id)
        job_id = stable_id("job", round_id, recipe_id, seed.seed_id, output_count, self.rng.random())
        specs = [self.tactics[t].to_dict() for t in tactic_ids if t in self.tactics]
        return GenerationJob(
            job_id=job_id,
            round_id=round_id,
            target_distribution=target_distribution,  # type: ignore[arg-type]
            category=seed.category,
            risk_axis=seed.risk_axis,
            benign_use_case=seed.benign_use_case if target_distribution != "unsafe_control" else None,
            unsafe_intent_summary=seed.unsafe_intent_summary,
            sensitive_surface=seed.sensitive_surface,
            ru_phenomena=seed.ru_phenomena,
            tactics=tactic_ids,
            tactic_specs=specs,
            output_count=output_count,
            sibling_group_id=sibling_group_id,
            recipe_id=recipe_id,
            source="sampler",
            metadata={"seed_id": seed.seed_id, "seed_source": seed.source},
        )
