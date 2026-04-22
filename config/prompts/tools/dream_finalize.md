### dream_finalize
Close the pending batch. `keep` commits staged edits to disk and appends history; `drop` discards them. Together the two lists must cover every phrase_id in the batch. Pass `keep=[]` with all ids in `drop` to abandon the batch entirely. Every dreamer turn on a conversation must end with `dream_finalize` — exiting without it rolls the batch back. Params: `keep` (list of phrase_ids to commit), `drop` (list of phrase_ids to discard).
<|tool_call|>call: dream_finalize, {"keep": ["ph-aaaa", "ph-cccc"], "drop": ["ph-bbbb"]}<|tool_call|>
