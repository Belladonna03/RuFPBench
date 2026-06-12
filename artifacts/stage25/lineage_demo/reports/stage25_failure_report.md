# Stage 2.5 failure / audit report (`lineage_demo`)

## Summary
- Total lineage records: **10**
- No improvement (repair_changed=false path): **0**

## Repeated failure patterns (reason × decision, count>1)
```json
{
  "too_obvious_safe_or_broken_naturalness|pending_revalidation": 10
}
```

## Unstable categories (high no_improvement rate)
```json
[
  {
    "category": "unknown",
    "count": 10,
    "no_improvement_rate": 0.0,
    "decision_mix": {
      "pending_revalidation": 10
    }
  }
]
```

## By final decision
```json
{
  "pending_revalidation": 10
}
```