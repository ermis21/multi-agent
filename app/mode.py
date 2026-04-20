"""
Mode helpers — plan / build / converse.

Mode affects temperature (plan colder, converse warmer) and which tools the
worker can call. The {{AGENT_MODE}} block renders mode-specific guidance into
the worker prompt; a per-model toggle lets providers that already understand
plan/build/converse skip the long explanation.
"""


def _mode_temperature(cfg: dict, mode: str) -> float:
    """plan = base - delta, build = base, converse = base + delta."""
    base  = cfg["llm"]["temperature"]
    delta = cfg["agent"].get("mode", {}).get("temperature_delta", 0.3)
    offset = {"plan": -delta, "build": 0.0, "converse": delta}.get(mode, 0.0)
    return max(0.0, min(2.0, base + offset))


def _mode_tools(cfg: dict, mode: str, base_tools: list[str]) -> list[str]:
    """Remove mode-specific excluded tools from the base allowed list."""
    excluded = set(cfg["agent"].get("mode", {}).get(mode, {}).get("excluded_tools", []))
    return [t for t in base_tools if t not in excluded]


_MODE_SHORT = {
    "plan": "**Mode: PLAN** — read + analyse only; produce a file-anchored plan, no writes/execution.",
    "build": "**Mode: BUILD** — full tool access; execute end-to-end and verify.",
    "converse": "**Mode: CONVERSE** — conversational; answer directly and terminate with `<|end|>`.",
}

_RESEARCH_RULE = (
    "**Research first**: if the request mentions external information "
    "('from the internet', 'popular', 'latest', 'find examples of', 'research'), "
    "call `web_search` before acting. Do not fabricate candidates."
)


def _mode_context_string(mode: str, cfg: dict | None = None, role_cfg: dict | None = None) -> str:
    """Render the {{AGENT_MODE}} block for a worker prompt.

    When `cfg` is supplied, the block is config-driven: it lists the tools actually
    excluded in this mode and includes a research-first rule. The long form can be
    collapsed to one line by setting `prompts.describe_mode_in_system_prompt: false`,
    or per-model via `models.<name>.describe_mode_in_system_prompt` — useful for
    models (e.g. Claude) that already understand plan/build/converse semantics.
    """
    short = _MODE_SHORT.get(mode, "")
    if cfg is None:
        return short

    toggle = cfg.get("prompts", {}).get("describe_mode_in_system_prompt", True)
    if role_cfg and (model_name := role_cfg.get("model")):
        model_override = cfg.get("models", {}).get(model_name, {}) or {}
        if "describe_mode_in_system_prompt" in model_override:
            toggle = bool(model_override["describe_mode_in_system_prompt"])
    if not toggle:
        return short

    excluded = (cfg.get("agent", {}).get("mode", {}).get(mode, {}) or {}).get("excluded_tools", []) or []
    excluded_list = ", ".join(f"`{t}`" for t in excluded) if excluded else "_(none)_"

    sections = {
        "plan": (
            "**Mode: PLAN** — research-first; produce a file-anchored plan, no writes or execution.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            "- Your final answer MUST name specific file paths (with extensions like `.py`, `.ts`, `.yaml`, `.md`) and the edits to make.\n"
            "- Do NOT try to substitute an excluded tool with a similar one — e.g. `file_edit` cannot replace `file_write` because `file_edit` only modifies existing files and will fail on new paths.\n"
            "- If the task requires excluded tools (writing, executing, committing), produce the plan and ask the user to switch with `/mode build`.\n"
            f"- {_RESEARCH_RULE}"
        ),
        "build": (
            "**Mode: BUILD** — full tool access; execute the task end-to-end and verify results.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            f"- {_RESEARCH_RULE}\n"
            "- Verify results (read the file you wrote, run the test, check the response) before claiming success."
        ),
        "converse": (
            "**Mode: CONVERSE** — conversational; answer directly. For simple greetings or short questions your single reply IS the final answer — end it with `<|end|>`.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            "- Do not emit status-style hedges before the answer.\n"
            f"- {_RESEARCH_RULE}"
        ),
    }
    return sections.get(mode, short)
