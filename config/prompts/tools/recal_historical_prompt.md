### recal_historical_prompt
Reconstruct a prompt file as it existed at a past timestamp by reverse-applying every history entry newer than that timestamp. Use the `timestamp` field returned in `possible_conflicts` / `possible_loops` to view the exact prompt state around a prior edit, not just the one flagged paragraph. Returns `{text, reversed_edits, warnings}` — warnings list history rows whose reverse-apply failed because of drift. Params: `timestamp` (ISO-8601; must match a `run_date` in some phrase history), `prompt_name` (basename, e.g. `worker_full`).
Examples:
- {"timestamp": "2026-04-18T04:03:12+00:00", "prompt_name": "worker_full"}
