### directory_tree
Recursive directory tree (default depth 3, max 6). `path: "."` returns every accessible root (`project/`, `config/`, `state/`, `cache/`, `workspace/`); use `project/` to walk the source code.

Use when: you want structural overview of a subtree.
Not when: you only need one level (use `file_list`) or are searching by filename pattern (use `file_search`).

Examples:
- {"path": ".", "depth": 2}
- {"path": "project/", "depth": 3}
