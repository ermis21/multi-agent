"""Structured output schemas validated at the supervisor/worker boundary.

Today only the supervisor verdict is shaped here. Add new schemas as more
pipeline boundaries pick up structured outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SupervisorVerdict(BaseModel):
    """Structured grade returned by `_run_supervisor`.

    Forgiving schema — only `pass` and `score` are strictly required (the only
    two fields the loop reads to decide retry vs accept). Every other field is
    optional with a sensible default so a terse but well-shaped supervisor
    response validates cleanly without firing the self-heal retry.

    `pass` is a Python keyword, so it's stored as `pass_` and exposed via the
    `pass` JSON alias; callers consume the verdict via `model_dump(by_alias=True)`
    to keep the legacy dict shape (`{"pass": ..., ...}`).

    Field shapes match the legacy `_fallback` dict in app/supervisor.py to
    avoid downstream changes in loop.py / state.py.
    """

    # `populate_by_name` lets us construct from either `pass` (alias) or `pass_`
    # (field name). `extra="allow"` because the supervisor model occasionally
    # adds notes the schema doesn't anticipate; we don't want to fail those.
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    pass_: bool = Field(alias="pass")
    score: float = Field(ge=0.0, le=1.0)

    feedback: str = ""
    alternative: str = ""
    suggest_spawn: str = ""
    suggest_debate: str = ""

    tool_issues: list[str] = []
    source_gaps: list[str] = []
    research_gaps: list[str] = []
    accuracy_issues: list[str] = []
    completeness_issues: list[str] = []


def format_validation_error(err: ValidationError) -> str:
    """Render a Pydantic ValidationError as a model-friendly correction message.

    Each error becomes a one-line bullet citing the field path, the constraint
    that was violated, and the value that caused it. Used by the supervisor
    self-heal retry to tell the LLM exactly what to fix.
    """
    lines: list[str] = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        msg = e.get("msg", "invalid")
        ctx = e.get("input")
        ctx_str = f" (got: {ctx!r})" if ctx is not None else ""
        lines.append(f"- field `{loc}`: {msg}{ctx_str}")
    return "\n".join(lines)
