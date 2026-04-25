### web_fetch
Fetch and extract the readable text from a URL. Returns `{url, status, text}` — `status` is the HTTP status code (e.g. 200), `text` is the extracted body (≤ 8000 chars).

Use when: you have an exact URL and want its body.
Not when: you need to discover URLs (use `web_search`) or summarise a long page (spawn `webfetch_summarizer`).

Examples:
- {"url": "https://example.com/article"}
