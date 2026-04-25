### file_search
Recursive glob search for files. `path: "."` fans out across every accessible root; narrow scope with a prefix like `project/` or `workspace/`. Results capped at 200.

Use when: you know the filename pattern but not the location.
Not when: you need file contents (use `file_read`) or want a directory overview (use `directory_tree`).

Examples:
- {"path": ".", "pattern": "*.py"}
- {"path": "project/", "pattern": "*.md"}
