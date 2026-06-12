import pytest
import os
import shutil
from src.rufp_stage1.pipeline.orchestrator import run_stage1_pipeline

def test_end_to_end_dry_run():
    # Setup
    input_csv = "sample_base_items.csv"
    output_dir = "artifacts/test_e2e"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    # Run pipeline
    run_stage1_pipeline(
        input_path=input_csv,
        output_dir=output_dir,
        mock=True,
        resume=False,
        debug=False
    )
    
    # Assertions
    assert os.path.exists(os.path.join(output_dir, "family_batches.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "family_routes.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "raw_candidates.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "naturalized_candidates.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "refined_candidates.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "prompt_candidate_bank_raw.jsonl"))
    assert os.path.exists(os.path.join(output_dir, "run_manifest.json"))
    assert os.path.exists(os.path.join(output_dir, "stage1_summary.json"))
