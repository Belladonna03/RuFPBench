import json
import os
from typing import Dict, Any

def generate_failure_report(run_id: str):
    base_dir = f"artifacts/stage2/{run_id}"
    report_dir = os.path.join(base_dir, "reports")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    summary_path = os.path.join(base_dir, "stage2_summary.json")
    if not os.path.exists(summary_path):
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Generate Markdown Report
    report_md = f"""# Stage 2 Failure Report: {run_id}

## Overview
- **Total Processed**: {summary['counts']['total']}
- **Accepted Semantic**: {summary['counts']['accepted_semantic']}
- **Probe Positive (Over-refusals)**: {summary['counts']['probe_positive']}
- **Rejected**: {summary['counts']['reject_set']}
- **Review Queue**: {summary['counts']['review_queue']}

## Key Findings
- **Highest Over-refusal Models**: {json.dumps(summary['probe_positive_rate_by_model'], indent=2)}
- **Top Review Reasons**: {json.dumps(summary['review_reasons'], indent=2)}

## Category Analysis
{json.dumps(summary['stats_by_category'], indent=2)}
"""

    with open(os.path.join(report_dir, "stage2_failure_report.md"), "w") as f:
        f.write(report_md)

    # Save specific metrics
    with open(os.path.join(metrics_dir, "category_funnel.json"), "w") as f:
        json.dump(summary['stats_by_category'], f, indent=2)
    
    with open(os.path.join(metrics_dir, "model_stats.json"), "w") as f:
        json.dump(summary['probe_positive_rate_by_model'], f, indent=2)
