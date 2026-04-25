# Tool-call grammar (this model)

Emit each tool call on its own line, in this exact form:

`<|tool_call|>call: TOOL_NAME, {param_json}<|tool_call|>`

The system also accepts the asymmetric form `<|tool_call>...<tool_call|>`,
bare JSON, and fenced JSON, but the symmetric form above is the canonical
target.

Rules:
- One tool call per message line — no prose before or after on the same line.
- `param_json` is a flat JSON object. Do NOT wrap in `{"params": ...}`.
- No markdown fences around the call.

Example:

`<|tool_call|>call: file_read, {"path": "notes.txt"}<|tool_call|>`
