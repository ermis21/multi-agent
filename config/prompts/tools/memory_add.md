### memory_add
Store a piece of information in long-term memory with optional tags.

Use when: a fact is durable, generalisable across sessions, and not already obvious from the codebase.
Not when: the info is task-local (use the turn) or recoverable from `git log`/code.

Examples:
- {"content": "User prefers concise answers.", "tags": ["preference"]}
