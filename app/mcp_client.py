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

# Tools handled locally — never sent to sandbox
LOCAL_TOOLS: frozenset[str] = frozenset({"read_config", "write_config"})

# Fast timeout for cheap file operations
FAST_TIMEOUT_S  = 10
# Slow timeout for shell/git/docker operations (subprocess max is 120s)
SLOW_TIMEOUT_S  = 130
SLOW_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
})


async def call_tool(method: str, params: dict, allowed: list[str]) -> dict:
    """
    Execute a tool call.

    Returns a result dict on success.
    Returns {"error": "..."} on permission denial or execution failure.
    Never raises — errors are returned as data so the agent can handle them.
    """
    if method not in allowed:
        return {"error": f"Tool '{method}' is not permitted for this agent role. Allowed: {allowed}"}

    # ── Local tools ──────────────────────────────────────────────────────────
    if method == "read_config":
        return {"config": get_config()}

    if method == "write_config":
        try:
            updated = patch_config(params)
            return {"updated": True, "config": updated}
        except Exception as e:
            return {"error": f"Config write failed: {e}"}

    # ── Sandbox tools ────────────────────────────────────────────────────────
    timeout = SLOW_TIMEOUT_S if method in SLOW_TOOLS else FAST_TIMEOUT_S
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{SANDBOX_URL}/mcp",
                json={"method": method, "params": params},
            )
            r.raise_for_status()
            data = r.json()
    except httpx.TimeoutException:
        return {"error": f"Tool '{method}' timed out after {timeout}s"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Sandbox HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Sandbox unreachable: {e}"}

    # Sandbox returns {"result": ..., "error": ...}
    if data.get("error"):
        return {"error": data["error"]}
    return data.get("result", {})


def _extract_tool_call(content: str) -> dict | None:
    """
    Detect a tool call in the model's response.

    The model signals a tool call by responding with raw JSON containing a
    "tool" key. Returns the parsed dict or None if this is a final answer.
    """
    stripped = content.strip()

    # Fast path: entire response is valid JSON with a "tool" key
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if "tool" in obj and isinstance(obj.get("params"), dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Slow path: JSON block inside a markdown code fence
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

    return None
