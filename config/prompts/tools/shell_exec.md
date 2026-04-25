### shell_exec
Execute a bash command in the workspace. Timeout max 120s.
A Python venv is pre-created at `venv/` — use `venv/bin/python` or `venv/bin/pip install`.
> `/Phebe` is READ-ONLY. Any write there (including `sed -i`) will fail. Use `file_edit`/`file_write` for workspace files, or switch to `coding_agent` for project source.
> Always check `exit_code` in the result — non-zero means the command failed even if no exception was raised.

Use when: you need to run a process, script, or pipeline.
Not when: a single file edit suffices (use `file_edit`/`file_write`) or you need to mutate `/project` source (spawn `coding_agent`).

Examples:
- {"command": "venv/bin/python script.py", "timeout_ms": 10000, "cwd": "."}
