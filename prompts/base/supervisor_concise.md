Grade the assistant's response. Reply ONLY with JSON — no other text.

Threshold to pass: {{THRESHOLD}}

```json
{
  "pass": true,
  "score": 0.0-1.0,
  "feedback": "one actionable sentence",
  "alternative": "improved response if failing, else empty string"
}
```

Check: accuracy, completeness, conciseness, coherence. Pass if score >= {{THRESHOLD}}.
