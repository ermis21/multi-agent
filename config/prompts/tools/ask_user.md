### ask_user
Ask the user a multiple-choice clarification question. 2-5 options.
The system may answer on the user's behalf if the answer is obvious from context.

Use when:
- The user's intent is ambiguous (could mean A or B)
- You need a preference only the user can provide
- The choice depends on context you don't have

Do NOT ask questions you can answer by reading files, searching memory, or investigating.

<|tool_call|>call: ask_user, {"question": "Which database should I configure?", "options": ["PostgreSQL (recommended for production)", "SQLite (simpler, for development)", "Keep current setup"], "context": "User asked to set up the database. Workspace has both pg and sqlite configs."}<|tool_call|>

The user always sees two extra choices alongside your A–E options:
- **Other…** — types a free-form reply. Result comes back as `letter: "other"` with their text in `answer`. Treat it as the user's considered answer.
- **Cancel** — dismisses the question to keep chatting. Result comes back as `cancelled: true`. When this happens, STOP immediately: end your turn with `<|end|>` on its own line and do not call further tools. The user's next message will arrive as a fresh user turn.
