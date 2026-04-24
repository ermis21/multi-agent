### dream_finalize
Close the pending batch. `keep` commits staged edits to disk and appends history; `drop` discards them. Together the two lists must cover every phrase_id in the batch. Every dreamer turn must end with `dream_finalize` — exiting without it rolls the batch back.

Params:
- `keep` (list of phrase_ids to commit)
- `drop` (list of phrase_ids to discard)
- `rationale` (string; **required** when `keep=[]` and `drop=[]` with no pending batch — explain why no revision is warranted, ≥20 chars)

**Skip path (no revision):** If you truly found nothing worth revising, call `dream_finalize(keep=[], drop=[], rationale="<≥20-char reason>")`. This is an explicit, visible skip — the user sees your rationale in the stream. If you identified ANY concrete issue in the conversation, prefer `dream_submit` with a targeted fix instead of skipping; the user reviews every edit and can reject it, so erring on the side of proposing a fix is cheap.

<|tool_call|>call: dream_finalize, {"keep": ["ph-aaaa", "ph-cccc"], "drop": ["ph-bbbb"]}<|tool_call|>
<|tool_call|>call: dream_finalize, {"keep": [], "drop": [], "rationale": "Conversation ran cleanly; worker used tools correctly, supervisor verdicts were precise, no prompt-attributable issue to fix."}<|tool_call|>
