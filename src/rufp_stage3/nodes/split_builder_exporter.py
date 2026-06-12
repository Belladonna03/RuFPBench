"""Stage 3 Node 5: leakage-aware splits + multi-format exports + manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from ..io import save_json, save_jsonl
from ..policy.schema import SplitExportConfig, Stage3Policy
from ..release.bundle import write_stage3_release_docs
from ..schemas import DedupClusterRecord, FinalExportRecord, HardSubsetRecord, SplitAssignmentRecord, Stage3CandidateRecord

logger = logging.getLogger(__name__)


class _UF:
    __slots__ = ("p", "rank")

    def __init__(self, items: List[str]) -> None:
        self.p = {x: x for x in items}
        self.rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _item_to_clusters(clusters: List[DedupClusterRecord]) -> Dict[str, str]:
    """Map item_id -> one canonical cluster_id (first seen)."""
    out: Dict[str, str] = {}
    for c in clusters:
        for mid in c.member_item_ids:
            if mid not in out:
                out[mid] = c.cluster_id
    return out


def _build_constraint_groups(
    rows: List[Stage3CandidateRecord],
    dedup_clusters: List[DedupClusterRecord],
    cfg: SplitExportConfig,
) -> Tuple[_UF, Dict[str, str]]:
    ids = [r.item_id for r in rows]
    uf = _UF(ids)
    item_cluster = _item_to_clusters(dedup_clusters)

    if cfg.enforce_single_split_per_dedup_cluster:
        by_c: DefaultDict[str, List[str]] = defaultdict(list)
        for c in dedup_clusters:
            for mid in c.member_item_ids:
                if mid in uf.p:
                    by_c[c.cluster_id].append(mid)
        for _cid, members in by_c.items():
            base = members[0]
            for m in members[1:]:
                uf.union(base, m)

    if cfg.enforce_single_split_per_family:
        by_f: DefaultDict[str, List[str]] = defaultdict(list)
        for r in rows:
            by_f[r.family_id].append(r.item_id)
        for _fid, members in by_f.items():
            base = members[0]
            for m in members[1:]:
                uf.union(base, m)

    return uf, item_cluster


def _group_rows(
    rows: List[Stage3CandidateRecord],
    uf: _UF,
) -> Dict[str, List[Stage3CandidateRecord]]:
    buckets: DefaultDict[str, List[Stage3CandidateRecord]] = defaultdict(list)
    for r in rows:
        buckets[uf.find(r.item_id)].append(r)
    return dict(buckets)


def _group_sort_key(
    grp: List[Stage3CandidateRecord],
) -> Tuple[str, str, str]:
    g = sorted(grp, key=lambda x: x.item_id)
    r0 = g[0]
    sub = r0.subtype or "_none_"
    return (r0.category, sub, r0.item_id)


def _is_repaired_group(grp: List[Stage3CandidateRecord]) -> bool:
    for r in grp:
        if "repaired_accept" in set(r.source_statuses or []):
            return True
    return False


def assign_splits(
    rows: List[Stage3CandidateRecord],
    dedup_clusters: List[DedupClusterRecord],
    policy: Stage3Policy,
) -> List[SplitAssignmentRecord]:
    if not rows:
        return []
    cfg = policy.split_export
    split_cfg = policy.split
    seed = split_cfg.random_seed

    eff = cfg.model_copy(
        update={
            "enforce_single_split_per_family": cfg.enforce_single_split_per_family
            and cfg.family_leakage_policy == "strict",
            "enforce_single_split_per_dedup_cluster": cfg.enforce_single_split_per_dedup_cluster
            and cfg.cluster_leakage_policy == "strict",
        }
    )
    uf, item_cluster = _build_constraint_groups(rows, dedup_clusters, eff)
    groups = _group_rows(rows, uf)
    group_list = sorted(groups.values(), key=_group_sort_key)

    total = len(rows)
    n_dev = int(total * split_cfg.dev_fraction)
    n_test = int(total * split_cfg.test_fraction)
    n_hold = max(0, total - n_dev - n_test)
    targets = {"dev": n_dev, "test": n_test, "review_holdout": n_hold}
    current: Dict[str, int] = {"dev": 0, "test": 0, "review_holdout": 0}

    order = list(group_list)
    if cfg.balance_repaired_across_splits:
        rep = [g for g in order if _is_repaired_group(g)]
        nrep = [g for g in order if not _is_repaired_group(g)]
        mix = cfg.repaired_original_mixing
        if mix == "interleave":
            merged: List[List[Stage3CandidateRecord]] = []
            i, j = 0, 0
            while i < len(rep) or j < len(nrep):
                if j < len(nrep):
                    merged.append(nrep[j])
                    j += 1
                if i < len(rep):
                    merged.append(rep[i])
                    i += 1
            order = merged
        elif mix == "repaired_first":
            order = rep + nrep
        else:
            order = nrep + rep
    if cfg.shuffle_groups_with_seed:
        rng = random.Random(seed)
        rng.shuffle(order)

    assignments: List[SplitAssignmentRecord] = []
    for grp in order:
        sp = max(targets.keys(), key=lambda s: targets[s] - current[s])
        root = min(r.item_id for r in grp)
        for r in sorted(grp, key=lambda x: x.item_id):
            cid = item_cluster.get(r.item_id)
            assignments.append(
                SplitAssignmentRecord(
                    item_id=r.item_id,
                    split=sp,
                    fold_seed=seed,
                    family_id=r.family_id,
                    cluster_id=cid,
                    constraint_group_id=root,
                    metadata={
                        "category": r.category,
                        "subtype": r.subtype,
                        "is_repaired": "repaired_accept" in set(r.source_statuses or []),
                    },
                )
            )
            current[sp] += 1

    assignments.sort(key=lambda x: x.item_id)
    logger.info(
        "split_builder: assigned %d rows dev=%d test=%d holdout=%d groups=%d",
        len(assignments),
        current["dev"],
        current["test"],
        current["review_holdout"],
        len(groups),
    )
    return assignments


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_to_csv_dict(r: Stage3CandidateRecord) -> Dict[str, str]:
    return {
        "item_id": r.item_id,
        "original_prompt_id": r.original_prompt_id,
        "family_id": r.family_id,
        "category": r.category,
        "subtype": r.subtype or "",
        "source_stages": "|".join(r.source_stages or []),
        "source_statuses": "|".join(r.source_statuses or []),
        "prompt_text": (r.prompt_text or "").replace("\n", " ")[:2000],
    }


def write_stage3_split_outputs(
    run_dir: str,
    export_dir: str,
    balanced_pool: List[Stage3CandidateRecord],
    splits: List[SplitAssignmentRecord],
    hard_rows: List[HardSubsetRecord],
    policy: Stage3Policy,
) -> Dict[str, Any]:
    """
    Writes ``stage3_dev_set.jsonl``, ``stage3_test_set.jsonl``, ``stage3_review_holdout.jsonl``,
    ``split_assignment.jsonl``, ``export_manifest.json``, and ``stage3_exports/`` bundle.
    """
    base = Path(run_dir)
    exp = Path(export_dir)
    exp.mkdir(parents=True, exist_ok=True)
    by_id = {r.item_id: r for r in balanced_pool}

    def pool_for(split_name: str) -> List[Stage3CandidateRecord]:
        ids = {s.item_id for s in splits if s.split == split_name}
        return [by_id[i] for i in sorted(ids) if i in by_id]

    save_jsonl(str(base / "stage3_dev_set.jsonl"), pool_for("dev"))
    save_jsonl(str(base / "stage3_test_set.jsonl"), pool_for("test"))
    save_jsonl(str(base / "stage3_review_holdout.jsonl"), pool_for("review_holdout"))

    save_jsonl(str(base / "split_assignment.jsonl"), splits)
    # Backward-compatible alias
    save_jsonl(str(base / "stage3_splits.jsonl"), splits)

    manifest_rows: List[FinalExportRecord] = []
    export_files: List[Dict[str, Any]] = []
    ef = policy.export_formats

    for split_name in ("dev", "test", "review_holdout"):
        rows = pool_for(split_name)
        jpath = exp / f"benchmark_{split_name}.jsonl"
        csv_path = exp / f"benchmark_{split_name}.csv"

        if ef.write_jsonl:
            save_jsonl(str(jpath), rows)
        if ef.write_csv:
            if rows:
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    cols = list(_row_to_csv_dict(rows[0]).keys())
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    for r in rows:
                        w.writerow(_row_to_csv_dict(r))
            else:
                csv_path.write_text("", encoding="utf-8")
        elif ef.write_jsonl:
            csv_path.write_text("", encoding="utf-8")

        primary_name = jpath.name if ef.write_jsonl else csv_path.name
        if ef.write_final_export_jsonl:
            for r in rows:
                h = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
                manifest_rows.append(
                    FinalExportRecord(
                        item_id=r.item_id,
                        split=split_name,
                        export_relpath=primary_name,
                        sha256_text=h,
                    )
                )

        if ef.write_jsonl:
            export_files.append(
                {
                    "path": jpath.name,
                    "format": "jsonl",
                    "rows": len(rows),
                    "sha256": _sha256_file(jpath) if jpath.is_file() and jpath.stat().st_size else None,
                }
            )
        if ef.write_csv:
            export_files.append(
                {
                    "path": csv_path.name,
                    "format": "csv",
                    "rows": len(rows),
                    "sha256": _sha256_file(csv_path) if csv_path.is_file() and csv_path.stat().st_size else None,
                }
            )

    if ef.write_final_export_jsonl:
        save_jsonl(str(exp / "final_export.jsonl"), manifest_rows)
    else:
        (exp / "final_export.jsonl").write_text("", encoding="utf-8")

    # Hard subset (separate export)
    hard_ids = {h.item_id for h in hard_rows}
    hard_pool = [by_id[i] for i in sorted(hard_ids) if i in by_id]
    hard_json = exp / "hard_subset_benchmark.jsonl"
    hard_csv = exp / "hard_subset_benchmark.csv"
    if ef.write_hard_subset_exports:
        save_jsonl(str(hard_json), hard_pool)
        if hard_pool:
            with open(hard_csv, "w", encoding="utf-8", newline="") as f:
                cols = list(_row_to_csv_dict(hard_pool[0]).keys())
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in hard_pool:
                    w.writerow(_row_to_csv_dict(r))
        else:
            hard_csv.write_text("", encoding="utf-8")
        export_files.append(
            {
                "path": hard_json.name,
                "format": "jsonl",
                "rows": len(hard_pool),
                "role": "hard_subset",
                "sha256": _sha256_file(hard_json) if hard_json.is_file() and hard_json.stat().st_size else None,
            }
        )
        export_files.append(
            {
                "path": hard_csv.name,
                "format": "csv",
                "rows": len(hard_pool),
                "role": "hard_subset",
                "sha256": _sha256_file(hard_csv) if hard_csv.is_file() and hard_csv.stat().st_size else None,
            }
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy.split_export.model_dump(),
        "split_fractions": policy.split.model_dump(),
        "counts": {
            "balanced_total": len(balanced_pool),
            "dev": len(pool_for("dev")),
            "test": len(pool_for("test")),
            "review_holdout": len(pool_for("review_holdout")),
            "hard_subset_export_rows": len(hard_pool),
        },
        "constraint_summary": {
            "enforce_family": policy.split_export.enforce_single_split_per_family,
            "enforce_cluster": policy.split_export.enforce_single_split_per_dedup_cluster,
        },
    }
    save_json(str(exp / "metadata.json"), meta)

    run_id = base.name
    release_written = write_stage3_release_docs(
        run_id=run_id,
        run_dir=base,
        export_dir=exp,
        balanced_pool=balanced_pool,
        splits=splits,
        hard_rows=hard_rows,
        policy=policy,
        qc_labels=None,
    )

    readme = exp / "README.txt"
    readme.write_text(
        "Stage 3 exports: benchmark_{dev,test,review_holdout}.jsonl + .csv, "
        "hard_subset_benchmark.jsonl, final_export.jsonl, metadata.json.\n"
        "Release: dataset_card.md, release_checklist.md, final_lineage.jsonl.\n",
        encoding="utf-8",
    )

    export_manifest = {
        "created_at": meta["generated_at"],
        "export_dir": str(exp),
        "files": export_files
        + [
            {"path": "metadata.json", "format": "json", "role": "aggregate"},
            {"path": "final_lineage.jsonl", "format": "jsonl", "role": "lineage"},
            {"path": "dataset_card.md", "format": "markdown", "role": "documentation"},
            {"path": "release_checklist.md", "format": "markdown", "role": "documentation"},
            {"path": "README.txt", "format": "text", "role": "documentation"},
        ],
        "final_export_jsonl": "final_export.jsonl",
        "hard_subset_benchmark": hard_json.name,
        "release_artifacts": release_written,
    }
    save_json(str(base / "export_manifest.json"), export_manifest)

    return export_manifest


def run_splits(
    rows: List[Stage3CandidateRecord],
    policy: Stage3Policy,
    dedup_clusters: Optional[List[DedupClusterRecord]] = None,
) -> List[SplitAssignmentRecord]:
    """Backward-compatible entry: build leakage-aware split assignments."""
    return assign_splits(rows, dedup_clusters or [], policy)
