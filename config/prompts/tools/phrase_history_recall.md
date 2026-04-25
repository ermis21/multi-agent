### phrase_history_recall
Look up the edit history of a tagged phrase. Returns the last `k` history rows (default 3) plus the current pointer text and section path. Use this when a `dream_submit` or `edit_revise` response flags an edit with `possible_conflict` or `possible_loop` and you want more context than the returned excerpt. Params: `phrase_id` (e.g. `ph-7c29ab1234`), optional `k` (int, default 3).
Examples:
- {"phrase_id": "ph-7c29ab1234", "k": 5}
