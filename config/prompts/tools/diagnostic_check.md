### diagnostic_check
Run a deterministic health check of all system components (filesystem, ChromaDB,
git, LLM, Notion, Discord, API keys, config files, prompt templates).
Returns JSON with pass/warn/fail per component and an overall status.
No LLM involvement — fully deterministic.
Examples:
- {}
