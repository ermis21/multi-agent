### deliberate
Run a structured debate between two positions to resolve a genuine decision.
Two advocates argue point-by-point in short (1-2 sentence) exchanges.
After 4 messages, YOU receive the full transcript and decide which side wins.
You can continue the debate for more rounds or accept a side.

When to use:
- Two plausible approaches and you're genuinely unsure which is better
- A choice that would be costly to reverse
- Tradeoffs you're not sure how to weigh

Do NOT use for trivial decisions, obvious choices, or questions the user should answer.

Frame positions as STRONG, SPECIFIC arguments — not wishy-washy hedges.

**New debate:**
Examples:
- {"question": "Regex vs AST for extracting function signatures?", "context": "Python files, well-formatted, 50-500 lines each. Need name + params + return type.", "position_a": "Regex: simpler, fast, sufficient for well-formatted code", "position_b": "AST: handles edge cases like decorators, multiline signatures, nested functions"}

**Continue a debate** (if status was "active" and you want more rounds):
Examples:
- {"debate_id": "debate_abc12345", "question": "", "context": "", "position_a": "", "position_b": "", "max_exchanges": 4}

You receive both the transcript AND an independent judge's verdict.
Then you decide:
- Accept the judge's verdict -> proceed with the winning approach
- Continue -> call deliberate again with the debate_id for more rounds
- Override the judge -> if you have context the advocates/judge didn't (explain why)

Result includes: transcript, positions, concessions, judge_winner, judge_reason, judge_confidence.
