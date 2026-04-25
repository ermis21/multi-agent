### file_write
Write content to a file in the workspace. Creates parent directories if needed.
Workspace only — cannot write under `project/` (read-only mount).
Examples:
- {"path": "relative/path", "content": "text content"}
