"""
MCP client — bridges mab-api to mab-sandbox.

All tool calls flow through call_tool():
  1. Check if the method is in the agent's allowed_tools list.
  2. Route LOCAL_TOOLS (read_config, write_config) to local functions — no HTTP.
  3. Everything else → POST http://mab-sandbox:9000/mcp

Timeouts:
  File ops (file_read, file_write, file_list): 10s
  All others (shell_exec, git_*, docker_*):    130s
"""

import json
import os

import httpx

from app.config_loader import get_config, patch_config

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://mab-sandbox:9000")

LOCAL_TOOLS: frozenset[str] = frozenset({"read_config", "write_config"})

FAST_TIMEOUT_S = 10
SLOW_TIMEOUT_S = 130
SLOW_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
    "web_search", "web_fetch",
    "memory_add", "memory_search", "memory_list",
    "notion_search", "notion_get_page", "notion_create_page", "notion_update_page",
    "discord_send", "discord_read", "discord_set_nickname", "discord_edit_channel",
    "diagnostic_check",
})

# Shared client — connection pool reused across tool calls in the same process.
# Per-request timeout is passed to each .post() call, overriding the client default.
_client = httpx.AsyncClient(timeout=SLOW_TIMEOUT_S)


async def call_tool(method: str, params: dict, allowed: list[str]) -> dict:
    """
    Execute a tool call.

    Returns a result dict on success.
    Returns {"error": "..."} on permission denial or execution failure.
    Never raises — errors are returned as data so the agent can handle them.
    """
    if method not in allowed:
        return {"error": f"Tool '{method}' is not permitted for this agent role. Allowed: {allowed}"}

    if method == "read_config":
        return {"config": get_config()}

    if method == "write_config":
        try:
            return {"updated": True, "config": patch_config(params)}
        except Exception as e:
            return {"error": f"Config write failed: {e}"}

    timeout = SLOW_TIMEOUT_S if method in SLOW_TOOLS else FAST_TIMEOUT_S
    try:
        r = await _client.post(
            f"{SANDBOX_URL}/mcp",
            json={"method": method, "params": params},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return {"error": f"Tool '{method}' timed out after {timeout}s"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Sandbox HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Sandbox unreachable: {e}"}

    if data.get("error"):
        return {"error": data["error"]}
    return data.get("result", {})


def _extract_tool_call(content: str) -> dict | None:
    """
    Detect a tool call in the model's response.

    The model signals a tool call by responding with raw JSON containing a
    "tool" key. Returns the parsed dict or None if this is a final answer.

    Handles three formats:
      1. Bare JSON:            {"tool": "...", "params": {...}}
      2. Markdown fence:       ```json\n{...}\n```
      3. JSON anywhere in prose: "Let me check:\n{...}"
    """
    stripped = content.strip()

    # 1. Entire response is a JSON tool call
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if "tool" in obj and isinstance(obj.get("params"), dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. JSON inside a markdown code fence
    for fence in ("```json", "```"):
        start = stripped.find(fence)
        if start != -1:
            inner_start = stripped.find("\n", start) + 1
            end = stripped.find("```", inner_start)
            if end != -1:
                try:
                    obj = json.loads(stripped[inner_start:end].strip())
                    if "tool" in obj and isinstance(obj.get("params"), dict):
                        return obj
                except json.JSONDecodeError:
                    pass

    # 3. JSON embedded in prose — scan for any {...} containing "tool"
    depth, start_idx = 0, -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx != -1:
                try:
                    obj = json.loads(stripped[start_idx:i + 1])
                    if "tool" in obj and isinstance(obj.get("params"), dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    return None


def strip_json_fences(raw: str) -> str:
    """
    Strip markdown code fences from a string expected to contain raw JSON.
    Used by the supervisor parser in agents.py.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw
