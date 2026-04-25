### memory_search
Semantic search over stored memories. Returns top-k matches with scores.

Use when: you need recall of prior facts, preferences, or session learnings.
Not when: you want to enumerate recent additions (use `memory_list`) or search the web (use `web_search`).

Examples:
- {"query": "user preferences", "n": 5}
