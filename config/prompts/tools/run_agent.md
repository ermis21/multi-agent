### run_agent
Delegate a task to a specialized sub-agent. It runs independently with its own tools
and cannot see your conversation. Give it a self-contained task with all needed context.
Examples:
- {"role": "coding_agent", "task": "Add clean_text_for_tts() to /workspace/my_module.py that strips markdown."}
- `role` (required): agent from the "Available sub-agents" list. Parameter key is **role** — not `agent_name`, `agent`, `sub_agent`, or `name`.
- `task` (required): complete instruction with file paths, constraints, expected outcome.
The sub-agent's final response is returned as the tool result.
