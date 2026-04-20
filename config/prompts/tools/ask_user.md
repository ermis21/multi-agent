### ask_user
Ask the user a multiple-choice clarification question. 2-5 options.
The system may answer on the user's behalf if the answer is obvious from context.

Use when:
- The user's intent is ambiguous (could mean A or B)
- You need a preference only the user can provide
- The choice depends on context you don't have

Do NOT ask questions you can answer by reading files, searching memory, or investigating.

<|tool_call|>call: ask_user, {"question": "Which database should I configure?", "options": ["PostgreSQL (recommended for production)", "SQLite (simpler, for development)", "Keep current setup"], "context": "User asked to set up the database. Workspace has both pg and sqlite configs."}<|tool_call|>
