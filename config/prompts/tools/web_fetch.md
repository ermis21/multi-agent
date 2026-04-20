### web_fetch
Fetch and extract the readable text from a URL. Returns `{url, status, text}` — `status` is the HTTP status code (e.g. 200), `text` is the extracted body (≤ 8000 chars).
<|tool_call|>call: web_fetch, {"url": "https://example.com/article"}<|tool_call|>
