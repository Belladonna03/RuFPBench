from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from .bucketizer import make_final_record
from .config import AppConfig
from .generation import generate_candidates_for_job, qc_candidate
from .llm import LLMOptions, LLMRouter
from .reporting import (
    write_category_coverage_csv,
    write_model_refusal_rates_csv,
    write_mutation_effectiveness_csv,
    write_report,
    write_summary_csv,
    write_tactic_coverage_csv,
)
from .router import route_candidate
from .sampler import GenerationJobSampler
from .schemas import CandidatePrompt, FinalRecord, GenerationJob, PromptSafetyResult, RecipeStats, RouterDecision, TargetResponse
from .seed_bank import load_seed_bank
from .tactics import load_compatibility, load_tactics
from .utils import append_jsonl, dedupe_prompts, ensure_dir, write_json, write_jsonl
from .validation import classify_target_response, judge_prompt_safety

console = Console()


class EvolutionaryPipeline:
    def __init__(self, cfg: AppConfig, run_dir: Path, mock: bool = False):
        self.cfg = cfg
        self.run_dir = run_dir
        self.mock = mock
        self.llm_router = LLMRouter(cfg, mock=mock)
        self.seed_bank = load_seed_bank(cfg)
        self.tactics = load_tactics(cfg)
        self.compatibility = load_compatibility(cfg)
        self.rng = random.Random(cfg.run.random_seed)
        self.sampler = GenerationJobSampler(
            cfg=cfg,
            seed_bank=self.seed_bank,
            tactics=self.tactics,
            compatibility=self.compatibility,
        )

        self.raw_dir = ensure_dir(run_dir / "raw")
        self.validated_dir = ensure_dir(run_dir / "validation")
        self.responses_dir = ensure_dir(run_dir / "responses")
        self.evolution_dir = ensure_dir(run_dir / "evolution")
        self.final_dir = ensure_dir(run_dir / "final")
        self.reports_dir = ensure_dir(run_dir / "reports")
        self.state_path = run_dir / "state.json"
        ensure_dir(run_dir)

        self.records: list[dict[str, Any]] = []
        self.all_candidates: list[CandidatePrompt] = []
        self.prompt_norm_rows: list[dict[str, Any]] = []
        self.pending_jobs: list[GenerationJob] = []
        self.recipe_stats: dict[str, RecipeStats] = {}
        self.category_counts: Counter[str] = Counter()
        self.accepted_counts: Counter[str] = Counter()

    def run(self) -> dict[str, Any]:
        if not self.seed_bank:
            raise RuntimeError("Seed bank is empty: check data/seed_taxonomy_ru.yaml")
        write_json(self.run_dir / "config.effective.json", self.cfg.to_dict())

        raw_target = self.cfg.run.target_raw_prompts
        min_fp = self.cfg.run.min_borderline_false_refusals
        extra_limit = self.cfg.run.max_extra_raw_prompts
        raw_total_limit = raw_target + (extra_limit if self.cfg.run.allow_topup_after_raw_target else 0)

        stats: dict[str, Any] = {
            "mode": "evolutionary",
            "raw_target": raw_target,
            "min_borderline_false_refusals": min_fp,
            "mock": self.mock,
            "rounds": 0,
            "jobs_generated": 0,
            "raw_generated": 0,
            "raw_qc_kept": 0,
            "final_records": 0,
            "final_safe_refused_borderline": 0,
            "pending_jobs_left": 0,
            "seed_bank_size": len(self.seed_bank),
            "tactic_bank_size": len(self.tactics),
        }

        try:
            for round_id in range(1, self.cfg.run.max_rounds + 1):
                current_fp = self.accepted_counts.get("safe_refused_borderline", 0)
                if len(self.all_candidates) >= raw_target and current_fp >= min_fp:
                    break
                if len(self.all_candidates) >= raw_total_limit:
                    break
                if len(self.all_candidates) >= raw_target and not self.cfg.run.allow_topup_after_raw_target:
                    break
    
                batch_target = self.cfg.run.raw_batch_size if len(self.all_candidates) < raw_target else self.cfg.run.topup_raw_batch_size
                console.print(f"[bold]Round {round_id}[/bold]: target ~{batch_target} candidates; accepted FP={current_fp}; pending_jobs={len(self.pending_jobs)}")
                stats["rounds"] = round_id
    
                jobs = self.sampler.sample_jobs(
                    round_id=round_id,
                    target_candidates=batch_target,
                    pending_jobs=self.pending_jobs,
                    accepted_counts=self.accepted_counts,
                    recipe_stats=self.recipe_stats,
                    category_counts=self.category_counts,
                )
                stats["jobs_generated"] += len(jobs)
                append_jsonl(self.raw_dir / "generation_jobs.jsonl", [j.to_dict() for j in jobs])
    
                candidates = self._generate_jobs(jobs)
                if not candidates:
                    console.print("[yellow]No candidates generated this round[/yellow]")
                    continue
                append_jsonl(self.raw_dir / "candidates.raw.jsonl", [c.to_dict() for c in candidates])
                stats["raw_generated"] += len(candidates)
    
                candidates = self._qc_and_dedupe(candidates)
                stats["raw_qc_kept"] += len(candidates)
                append_jsonl(self.raw_dir / "candidates.qc_kept.jsonl", [c.to_dict() for c in candidates])
                self.all_candidates.extend(candidates)
                self.prompt_norm_rows.extend([c.to_dict() for c in candidates])
    
                if not candidates:
                    self._write_state(stats)
                    continue
    
                safety_results = self._validate_prompts(candidates)
                append_jsonl(self.validated_dir / "prompt_safety_votes.jsonl", [r.to_dict() for r in safety_results])
                safety_by_id = {r.prompt_id: r for r in safety_results}
    
                target_results_by_prompt = self._run_targets(candidates)
                for model_name, responses in self._responses_by_model(target_results_by_prompt).items():
                    append_jsonl(
                        self.responses_dir / f"target_responses.{self._safe_model_name(model_name)}.jsonl",
                        [r.to_dict(redact_response=False) for r in responses],
                    )
    
                round_records = self._make_final_records(candidates, safety_by_id, target_results_by_prompt)
                round_dicts = [r.to_dict() for r in round_records]
                self.records.extend(round_dicts)
                self._update_recipe_stats(round_records)
    
                decisions = self._route_records(candidates, round_records, next_round_id=round_id + 1)
                append_jsonl(self.evolution_dir / "router_decisions.jsonl", [d.to_dict() for d in decisions])
                append_jsonl(
                    self.evolution_dir / "mutation_jobs.jsonl",
                    [d.next_job.to_dict() for d in decisions if d.next_job is not None],
                )
    
                self._write_incremental_outputs()
                stats["final_records"] = len(self.records)
                stats["final_safe_refused_borderline"] = self.accepted_counts.get("safe_refused_borderline", 0)
                stats["pending_jobs_left"] = len(self.pending_jobs)
                self._write_state(stats)
                # Write reports after every completed round so interrupted long runs
                # are still easy to inspect and archive.
                self._write_reports(stats)
                console.print(
                    f"Current: raw={len(self.all_candidates)}, records={len(self.records)}, FP/borderline={stats['final_safe_refused_borderline']}, next_pending={len(self.pending_jobs)}"
                )
    
        except KeyboardInterrupt:
            stats["interrupted"] = True
            stats["final_records"] = len(self.records)
            stats["final_safe_refused_borderline"] = self.accepted_counts.get("safe_refused_borderline", 0)
            stats["pending_jobs_left"] = len(self.pending_jobs)
            self._write_incremental_outputs()
            self._write_reports(stats)
            self._write_state(stats)
            console.print("[yellow]Interrupted; partial outputs and reports were written.[/yellow]")
            return stats
        self._write_incremental_outputs()
        self._write_reports(stats)
        self._write_state(stats)
        return stats

    def _generate_jobs(self, jobs: list[GenerationJob]) -> list[CandidatePrompt]:
        candidates: list[CandidatePrompt] = []
        for job in track(jobs, description="generate jobs"):
            try:
                candidates.extend(generate_candidates_for_job(client=self.llm_router, cfg=self.cfg, job=job))
            except Exception as e:
                row = job.to_dict()
                row["generation_error"] = repr(e)
                append_jsonl(self.raw_dir / "generation_errors.jsonl", [row])
                console.print(f"[yellow]generation failed for {job.job_id}: {e!r}[/yellow]")
        return candidates

    def _qc_and_dedupe(self, candidates: list[CandidatePrompt]) -> list[CandidatePrompt]:
        kept: list[CandidatePrompt] = []
        for c in candidates:
            ok, reason = qc_candidate(c, self.cfg)
            if ok:
                kept.append(c)
            else:
                row = c.to_dict()
                row["qc_reject_reason"] = reason
                append_jsonl(self.validated_dir / "qc_rejected.jsonl", [row])
        rows = [c.to_dict() for c in kept]
        combined = self.prompt_norm_rows + rows
        deduped_combined = dedupe_prompts(combined, threshold=self.cfg.qc.max_near_duplicate_jaccard)
        existing_ids = {r.get("prompt_id") for r in self.prompt_norm_rows}
        deduped_new = [r for r in deduped_combined if r.get("prompt_id") not in existing_ids]
        deduped_new_ids = {r.get("prompt_id") for r in deduped_new}
        for c in kept:
            if c.prompt_id not in deduped_new_ids:
                row = c.to_dict()
                row["qc_reject_reason"] = f"near_duplicate_jaccard>={self.cfg.qc.max_near_duplicate_jaccard}"
                append_jsonl(self.validated_dir / "qc_rejected.jsonl", [row])
        by_id = {c.prompt_id: c for c in kept}
        deduped_candidates = [by_id[r["prompt_id"]] for r in deduped_new if r.get("prompt_id") in by_id]
        return self._apply_category_soft_cap(deduped_candidates)

    def _apply_category_soft_cap(self, candidates: list[CandidatePrompt]) -> list[CandidatePrompt]:
        """Prevent a single category from swallowing a batch after dedupe.

        The cap is intentionally soft and disabled when a run has too few
        distinct categories; otherwise tiny smoke runs could drop everything.
        """
        cap_fraction = float(self.cfg.qc.category_soft_cap_fraction or 0.0)
        if cap_fraction <= 0.0 or cap_fraction >= 1.0 or not candidates:
            return candidates
        all_categories = {str(r.get("category") or "") for r in self.prompt_norm_rows}
        all_categories.update(c.category for c in candidates)
        all_categories.discard("")
        if len(all_categories) < 3:
            return candidates
        projected_total = len(self.prompt_norm_rows) + len(candidates)
        cap = max(2, int(projected_total * cap_fraction + 0.999))
        existing_counts: Counter[str] = Counter(str(r.get("category") or "") for r in self.prompt_norm_rows)
        batch_counts: Counter[str] = Counter()
        out: list[CandidatePrompt] = []
        for c in candidates:
            if existing_counts[c.category] + batch_counts[c.category] < cap:
                out.append(c)
                batch_counts[c.category] += 1
            else:
                row = c.to_dict()
                row["qc_reject_reason"] = f"category_soft_cap:{c.category}:cap={cap}"
                append_jsonl(self.validated_dir / "qc_rejected.jsonl", [row])
        return out

    def _validate_prompts(self, candidates: list[CandidatePrompt]) -> list[PromptSafetyResult]:
        results: list[PromptSafetyResult] = []
        for c in track(candidates, description="prompt safety ensemble"):
            results.append(judge_prompt_safety(client=self.llm_router, cfg=self.cfg, candidate=c))
        return results

    def _run_targets(self, candidates: list[CandidatePrompt]) -> dict[str, list[TargetResponse]]:
        out: dict[str, list[TargetResponse]] = {}
        for c in track(candidates, description="target/refusal pool"):
            prompt_results: list[TargetResponse] = []
            target_models = self.llm_router.models_for_step("target_response", default=self.cfg.models.target_models)
            for model_name in target_models:
                try:
                    res = self.llm_router.chat(
                        "target_response",
                        [{"role": "user", "content": c.prompt}],
                        LLMOptions(model=model_name),
                    )
                    tr = classify_target_response(
                        client=self.llm_router,
                        cfg=self.cfg,
                        prompt_id=c.prompt_id,
                        prompt=c.prompt,
                        model=model_name,
                        response=res.content,
                        latency_ms=res.latency_ms,
                        raw_metadata=res.raw,
                    )
                except Exception as e:
                    tr = classify_target_response(
                        client=self.llm_router,
                        cfg=self.cfg,
                        prompt_id=c.prompt_id,
                        prompt=c.prompt,
                        model=model_name,
                        response="",
                        error=repr(e),
                    )
                prompt_results.append(tr)
            out[c.prompt_id] = prompt_results
        return out

    def _make_final_records(
        self,
        candidates: list[CandidatePrompt],
        safety_by_id: dict[str, PromptSafetyResult],
        target_results_by_prompt: dict[str, list[TargetResponse]],
    ) -> list[FinalRecord]:
        records: list[FinalRecord] = []
        for c in candidates:
            safety = safety_by_id.get(c.prompt_id)
            if safety is None:
                continue
            targets = target_results_by_prompt.get(c.prompt_id, [])
            rec = make_final_record(
                cfg=self.cfg,
                candidate=c,
                safety=safety,
                targets=targets,
                category_counts=self.category_counts,
            )
            records.append(rec)
        return records

    def _route_records(self, candidates: list[CandidatePrompt], records: list[FinalRecord], next_round_id: int) -> list[RouterDecision]:
        by_id = {c.prompt_id: c for c in candidates}
        decisions: list[RouterDecision] = []
        new_jobs: list[GenerationJob] = []
        for rec in records:
            cand = by_id.get(rec.prompt_id)
            if cand is None:
                continue
            decision = route_candidate(
                cfg=self.cfg,
                candidate=cand,
                record=rec,
                next_round_id=next_round_id,
                tactics=self.tactics,
                compatibility=self.compatibility,
                rng=self.rng,
            )
            decisions.append(decision)
            if decision.accepted:
                self.accepted_counts[rec.final_bucket] += 1
                self.category_counts[rec.category] += 1
            if decision.next_job is not None:
                new_jobs.append(decision.next_job)
        # Cap mutation queue to keep cost bounded.
        space = max(0, self.cfg.run.max_mutation_jobs_per_round - len(self.pending_jobs))
        if space > 0:
            self.pending_jobs.extend(new_jobs[:space])
        return decisions

    def _update_recipe_stats(self, records: list[FinalRecord]) -> None:
        for rec in records:
            rid = rec.recipe_id or f"{rec.category}__{rec.intended_label}"
            st = self.recipe_stats.setdefault(rid, RecipeStats(recipe_id=rid))
            st.attempts += 1
            if rec.final_bucket == "safe_refused_borderline":
                st.safe_refused_borderline += 1
            elif rec.final_bucket == "safe_answered":
                st.safe_answered += 1
            elif rec.final_bucket == "benign_prompt_unsafe_response":
                st.benign_prompt_unsafe_response += 1
            elif rec.final_bucket == "unsafe_safe_handled":
                st.unsafe_safe_handled += 1
            elif rec.final_bucket == "unsafe_refused":
                st.unsafe_refused += 1
            elif rec.final_bucket == "unsafe_answered":
                st.unsafe_answered += 1
            else:
                st.quarantine += 1

    def _write_incremental_outputs(self) -> None:
        write_jsonl(self.final_dir / "all_labeled.jsonl", self.records)
        buckets = [
            "safe_answered",
            "safe_refused_borderline",
            "benign_prompt_unsafe_response",
            "unsafe_safe_handled",
            "unsafe_refused",
            "unsafe_answered",
            "quarantine",
        ]
        for bucket in buckets:
            rows = [r for r in self.records if r.get("final_bucket") == bucket]
            write_jsonl(self.final_dir / f"{bucket}.jsonl", rows)

        hard = [
            r for r in self.records
            if r.get("final_bucket") == "safe_refused_borderline"
            and len(r.get("refused_by", [])) >= self.cfg.validation.hard_min_refusal_models
        ]
        hard_sorted = sorted(hard, key=lambda r: (float(r.get("hard_score", 0)), float(r.get("score", 0))), reverse=True)
        write_jsonl(self.final_dir / "rufpbench_hard.jsonl", hard_sorted)
        main = [r for r in self.records if r.get("final_bucket") == "safe_refused_borderline"]
        main_sorted = sorted(main, key=lambda r: (float(r.get("hard_score", 0)), float(r.get("score", 0))), reverse=True)
        write_jsonl(self.final_dir / "rufpbench_borderline.jsonl", main_sorted)
        write_jsonl(self.evolution_dir / "recipe_stats_by_round.jsonl", [s.to_dict() for s in self.recipe_stats.values()])

    def _write_reports(self, stats: dict[str, Any]) -> None:
        write_summary_csv(self.reports_dir / "summary.csv", self.records)
        # Backward-compatible copy in final/.
        write_summary_csv(self.final_dir / "summary.csv", self.records)
        write_category_coverage_csv(self.reports_dir / "category_coverage.csv", self.records)
        write_tactic_coverage_csv(self.reports_dir / "tactic_coverage.csv", self.records)
        write_model_refusal_rates_csv(self.reports_dir / "model_refusal_rates.csv", self.records)
        write_mutation_effectiveness_csv(self.reports_dir / "mutation_effectiveness.csv", self.records)
        write_report(
            self.reports_dir / "report.md",
            config=self.cfg.to_dict(),
            records=self.records,
            run_stats=stats,
        )
        write_report(
            self.final_dir / "report.md",
            config=self.cfg.to_dict(),
            records=self.records,
            run_stats=stats,
        )

    def _write_state(self, stats: dict[str, Any]) -> None:
        stats["bucket_counts"] = dict(Counter(str(r.get("final_bucket", "unknown")) for r in self.records))
        stats["accepted_counts"] = dict(self.accepted_counts)
        stats["recipe_count"] = len(self.recipe_stats)
        write_json(self.state_path, stats)

    @staticmethod
    def _safe_model_name(model_name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)

    @staticmethod
    def _responses_by_model(data: dict[str, list[TargetResponse]]) -> dict[str, list[TargetResponse]]:
        out: dict[str, list[TargetResponse]] = {}
        for responses in data.values():
            for r in responses:
                out.setdefault(r.model, []).append(r)
        return out
