### file_read
Read a file. Prefix path with `project/` to read the system source code (read-only).

Use when: you have an exact path.
Not when: you need to discover paths (use `file_list` or `file_search`) or grep across many files (use `file_search`).

Examples:
- {"path": "notes.txt"}
- {"path": "project/app/agents.py"}
