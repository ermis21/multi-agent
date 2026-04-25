### edit_revise
Patch a single staged edit in the current pending batch without resubmitting the whole prompt. Use after `dream_submit` returns flagged edits and you want to rewrite one paragraph rather than redo the whole file. Params: `phrase_id` (the `ph-…` id from the pending batch), `new_text` (replacement paragraph), `rationale` (why the revision).
Examples:
- {"phrase_id": "ph-7c29ab1234", "new_text": "Updated paragraph text.", "rationale": "defuse conflict with last week s edit"}
