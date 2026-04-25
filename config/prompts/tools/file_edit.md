### file_edit
Replace exactly one occurrence of `old_string` with `new_string` in a workspace file.
Fails if `old_string` is not found or matches more than once — make it specific.
Workspace only — cannot edit files under `project/` (read-only mount).
Examples:
- {"path": "notes.txt", "old_string": "foo", "new_string": "bar"}
