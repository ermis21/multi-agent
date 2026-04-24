### dream_submit
Submit **one or more** rewritten prompts in a single atomic batch. The system diffs each submission against the current on-disk file paragraph-by-paragraph and returns every edit annotated with `ok | possible_conflict | possible_loop`, plus a narrative for flagged items. A fresh submission replaces any prior pending batch wholesale — start over if you want to abandon in-progress edits. Do not call `simulate_conversation` yourself; the system auto-runs it when you stop revising. One simulation validates **all** target prompts together, conserving the 3-sim budget across interacting agents.

Params:
- `targets` — list of `{path, new_full_text}` objects. Each `path` is a prompt filename (e.g. `worker_full`); each `new_full_text` is the ENTIRE new prompt for that file as a single string.
- `rationale` — short explanation of the change (applies to the whole batch).

**Multi-target usage.** When a conversation's failure mode spans multiple agents (e.g. a nagging supervisor that also confuses the worker), propose coordinated edits to BOTH prompts in one call — they simulate and finalize together. Typical pattern: `worker_full.md` + `supervisor_full.md` in a single batch.

**Single-target** remains the common case — wrap your one rewrite in a 1-element `targets` list.

<|tool_call|>call: dream_submit, {"targets": [{"path": "worker_full", "new_full_text": "# Worker\n\n..."}], "rationale": "tighten tool-call grammar guidance"}<|tool_call|>
<|tool_call|>call: dream_submit, {"targets": [{"path": "worker_full", "new_full_text": "# Worker\n\n..."}, {"path": "supervisor_full", "new_full_text": "# Supervisor\n\n..."}], "rationale": "supervisor kept nagging on converse-mode research — loosen supervisor checks and tighten worker's mid-research phrasing"}<|tool_call|>
