### tool_result_recall
Read a chunk of a previously-elided tool result by its handle id. When a tool output exceeds the inline budget the worker sees `[tool_result: …] OK  #rf-XXXXXX` with head+tail preview and an elision marker — use this tool to fetch any middle section you still need. Params: `id` (the `rf-…` handle), `session_id`, optional `offset` (chars) and `limit` (chars, max 16000).

Use when: the head/tail preview is missing the specific span you need.
Not when: the head/tail already shows the answer — recalling a 16k blob just inflates context.

Examples:
- {"id": "rf-7c29ab", "session_id": "discord_123_456", "offset": 0, "limit": 4000}
