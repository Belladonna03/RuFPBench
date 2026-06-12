"""Optional audit pack export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "stage3.yaml"


def _cand(item_id: str, *, repaired: bool = False) -> dict:
    return {
        "item_id": item_id,
        "source": "t",
        "prompt_id": item_id,
        "original_prompt_id": item_id,
        "prompt_text": "hello world " * 10,
        "family_id": "f1",
        "category": "c1",
        "subtype": "s1",
        "source_stages": ["stage25" if repaired else "stage2"],
        "source_statuses": ["repaired_accept"] if repaired else ["validated"],
        "probe_profile": {"probe_results": []},
        "lineage_refs": [],
        "generation_route": "",
        "lineage": {},
        "metadata": {},
    }


@pytest.mark.skipif(not CFG.is_file(), reason="configs/stage3.yaml missing")
def test_export_audit_packs_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFP_ARTIFACTS_ROOT", str(tmp_path))
    rid = "audit_test"
    base = tmp_path / "artifacts" / "stage3" / rid
    base.mkdir(parents=True)
    exp = base / "stage3_exports"
    exp.mkdir(parents=True)

    rows = [_cand("a1"), _cand("a2", repaired=True), _cand("a3")]
    (base / "stage3_balanced_pool.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    qc = [
        {"item_id": "a1", "qc_label": "keep", "qc_pass": True, "qc_flags": [], "notes": ""},
        {"item_id": "a2", "qc_label": "keep", "qc_pass": True, "qc_flags": [], "notes": ""},
        {"item_id": "a3", "qc_label": "keep", "qc_pass": True, "qc_flags": [], "notes": ""},
    ]
    (base / "stage3_qc_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qc) + "\n",
        encoding="utf-8",
    )
    hard = [
        {"item_id": "a1", "hard_score": 0.9, "why_hard": "x", "supporting_probe_stats": {}, "diversity_tags": [], "metadata": {}},
    ]
    (base / "stage3_hard_subset.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in hard) + "\n",
        encoding="utf-8",
    )

    from rufp_stage3.policy.loader import load_stage3_policy
    from rufp_stage3.audit.packs import export_stage3_audit_packs

    policy = load_stage3_policy(CFG)
    policy = policy.model_copy(
        update={
            "audit_packs": policy.audit_packs.model_copy(
                update={"random_count": 2, "hard_count": 1, "repaired_count": 1, "random_seed": 1}
            )
        }
    )
    out = export_stage3_audit_packs(rid, policy)
    assert (exp / "audit_pack_random.jsonl").is_file()
    assert (exp / "audit_pack_hard.jsonl").is_file()
    assert (exp / "audit_pack_repaired.jsonl").is_file()
    assert (exp / "audit_packs_manifest.json").is_file()

    line = (exp / "audit_pack_hard.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["item_id"] == "a1"
    assert "prompt_text" in rec and "qc_metadata" in rec
    assert rec["repair_flag"] is False
