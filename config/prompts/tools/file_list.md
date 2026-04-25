### file_list
List files and directories. `path: "."` lists every accessible root (`project/`, `config/`, `state/`, `cache/`, `workspace/`). Use `project/` to reach the source code.

Use when: you want one directory's immediate children.
Not when: you need to walk subdirectories (use `directory_tree`) or match a filename pattern (use `file_search`).

Examples:
- {"path": "."}
- {"path": "project/app"}
