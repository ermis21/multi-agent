### execute_command
Execute a shell command in the workspace (alias for shell_exec). Timeout max 120s.
A Python venv is pre-created at `venv/` — use `venv/bin/python` or `venv/bin/pip install`.
> `/Phebe` is READ-ONLY. Any write there (including `sed -i`) will fail. Use `file_edit`/`file_write` for workspace files, or switch to `coding_agent` for project source.
> Always check `exit_code` in the result — non-zero means the command failed even if no exception was raised.
Examples:
- {"command": "venv/bin/pip install requests", "timeout_ms": 30000}
