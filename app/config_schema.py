"""
Pydantic schema for config/config.yaml.

Closed sections (extra="forbid") reject unknown keys so that typos like
`supervisor_pass_treshold` fail loudly with a difflib suggestion instead of
silently creating a dead key. Loose sections (llm, tools, models) accept
extras because users legitimately extend them with provider- or app-specific
fields.
"""

from __future__ import annotations

import copy
import difflib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config_loader import _deep_merge, _expand_dotted_keys


_CRON_RE = re.compile(r"^\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*$")


class ConfigPatchError(ValueError):
    """Raised when a config patch or full config fails schema validation."""


# ── LLM & models (loose) ───────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = "local"
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    enable_thinking: bool | None = None
    thinking_budget_tokens: int | None = None
    base_url: str | None = None
    url: str | None = None


# ── Prompts ────────────────────────────────────────────────────────────────────

class PromptsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["concise", "full"] = "concise"
    workspace_file_max_chars: int = Field(4000, ge=0)
    config_change_approval_required: bool = True
    describe_mode_in_system_prompt: bool = True


# ── Agent ──────────────────────────────────────────────────────────────────────

class SupervisorModeOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: float | None = Field(None, ge=0.0, le=1.0)
    build: float | None = Field(None, ge=0.0, le=1.0)
    converse: float | None = Field(None, ge=0.0, le=1.0)


class ModeToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    excluded_tools: list[str] = Field(default_factory=list)


class AgentModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: Literal["plan", "build", "converse"] = "converse"
    temperature_delta: float = 0.3
    plan: ModeToolsConfig = Field(default_factory=ModeToolsConfig)
    build: ModeToolsConfig = Field(default_factory=ModeToolsConfig)
    converse: ModeToolsConfig = Field(default_factory=ModeToolsConfig)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supervisor_enabled: bool = True
    max_retries: int = Field(2, ge=0)
    supervisor_pass_threshold: float = Field(0.7, ge=0.0, le=1.0)
    supervisor_mode_overrides: SupervisorModeOverrides = Field(default_factory=SupervisorModeOverrides)
    max_context_turns: int = Field(20, ge=0)
    max_context_messages: int = Field(40, ge=0)
    inflection_mode: Literal["none", "logprobs", "linguistic", "both"] = "none"
    mode: AgentModeConfig = Field(default_factory=AgentModeConfig)


# ── Approval ───────────────────────────────────────────────────────────────────

class ApprovalAutoAllow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class ApprovalBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_allow: ApprovalAutoAllow = Field(default_factory=ApprovalAutoAllow)
    ask_user: list[str] = Field(default_factory=list)
    auto_fail: list[str] = Field(default_factory=list)


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ApprovalBucket = Field(default_factory=ApprovalBucket)
    build: ApprovalBucket = Field(default_factory=ApprovalBucket)
    converse: ApprovalBucket = Field(default_factory=ApprovalBucket)


# ── Scheduled jobs ─────────────────────────────────────────────────────────────

class SoulConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    schedule: str = "0 5 * * *"
    max_chars: int = Field(6000, ge=0)

    @field_validator("schedule")
    @classmethod
    def _cron_shape(cls, v: str) -> str:
        if not _CRON_RE.match(v):
            raise ValueError("schedule must be a 5-field cron expression (min hour dom mon dow)")
        return v


class DiscordModeratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    schedule: str = "0 10 */3 * *"
    conversations_category: str = "Conversations"
    archive_category: str = "Archive"
    inactive_days: int = Field(7, ge=0)
    themed_categories: list[str] = Field(default_factory=list)

    @field_validator("schedule")
    @classmethod
    def _cron_shape(cls, v: str) -> str:
        if not _CRON_RE.match(v):
            raise ValueError("schedule must be a 5-field cron expression (min hour dom mon dow)")
        return v


# ── Misc ───────────────────────────────────────────────────────────────────────

class InflectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entropy_threshold: float = 1.5
    logprob_gap_threshold: float = 0.5
    top_logprobs: int = Field(5, ge=1)
    strong_marker_threshold: int = Field(1, ge=0)
    weak_marker_threshold: int = Field(3, ge=0)
    max_nudges_per_turn: int = Field(2, ge=0)


class DebateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_messages: int = Field(4, ge=1)
    advocate_temperature: float = 0.4
    judge_temperature: float = 0.2
    advocate_model: str = "debate_advocate"
    judge_model: str = "debate_judge"
    use_judge: bool = True
    max_total_messages: int = Field(12, ge=1)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessions_dir: str = "/state/sessions"
    log_supervisor_turns: bool = True
    verbose_tools: bool = True


# ── Context compression (Gemma-aware budgets + elision) ──────────────────────

class ContextBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soul: int = Field(512, ge=0)
    user: int = Field(256, ge=0)
    memory: int = Field(512, ge=0)
    identity: int = Field(128, ge=0)
    tool_docs: int = Field(1500, ge=0)
    skills: int = Field(800, ge=0)
    history: int = Field(6000, ge=0)
    tool_result_inline: int = Field(1500, ge=0)


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    budgets: ContextBudgets = Field(default_factory=ContextBudgets)
    total_soft_cap: int = Field(12000, ge=0)
    tokenizer_backend: Literal["llama", "tiktoken", "heuristic"] = "llama"
    elision_strategy: Literal["head", "tail", "head_tail", "middle"] = "head_tail"


# ── Dream (nightly prompt self-improvement) ──────────────────────────────────

class DreamSimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int = Field(120000, ge=0)
    max_turns_replayed: int = Field(5, ge=1)
    min_turns_to_simulate: int = Field(1, ge=0)


class DreamLoopGuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarity_backend: Literal["fuzzy", "embedding"] = "fuzzy"
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    max_history: int = Field(8, ge=1)
    period_detection_window: int = Field(6, ge=2)


class DreamEmailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    schedule: str = "0 10 * * *"
    to: str = ""
    provider: Literal["gmail", "smtp"] = "gmail"
    fallback_channel_id: str | None = None

    @field_validator("schedule")
    @classmethod
    def _cron_shape(cls, v: str) -> str:
        if not _CRON_RE.match(v):
            raise ValueError("schedule must be a 5-field cron expression (min hour dom mon dow)")
        return v


class DreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    schedule: str = "0 4 * * *"
    stop_on_user_activity: bool = True
    target_prompts: list[str] = Field(default_factory=lambda: ["*"])
    min_tier: Literal["small", "medium", "large", "frontier"] = "large"
    min_context_window: int = Field(200000, ge=0)
    required_capabilities: list[str] = Field(
        default_factory=lambda: [
            "destructive_edit_safe",
            "long_context_reasoning",
            "prompt_self_critique",
        ]
    )
    model: str | None = None
    judge_model: str | None = None
    simulation: DreamSimulationConfig = Field(default_factory=DreamSimulationConfig)
    loop_guard: DreamLoopGuardConfig = Field(default_factory=DreamLoopGuardConfig)
    email: DreamEmailConfig = Field(default_factory=DreamEmailConfig)

    @field_validator("schedule")
    @classmethod
    def _cron_shape(cls, v: str) -> str:
        if not _CRON_RE.match(v):
            raise ValueError("schedule must be a 5-field cron expression (min hour dom mon dow)")
        return v


# ── Tools (loose — users extend with new integrations) ────────────────────────

class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


# ── Root ──────────────────────────────────────────────────────────────────────

class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    models: dict[str, LLMConfig] = Field(default_factory=dict)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    soul: SoulConfig = Field(default_factory=SoulConfig)
    discord_moderator: DiscordModeratorConfig = Field(default_factory=DiscordModeratorConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    inflection: InflectionConfig = Field(default_factory=InflectionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _known_keys_at(model_cls: type[BaseModel], path: tuple[str, ...]) -> list[str]:
    """Walk RootConfig's field tree to the given path and return the sibling keys there."""
    current: Any = model_cls
    for part in path:
        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return []
        fields = current.model_fields
        if part not in fields:
            return []
        annotation = fields[part].annotation
        # Unwrap Optional[...] / Union[None, X]
        origin = getattr(annotation, "__origin__", None)
        if origin is None and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current = annotation
        else:
            args = getattr(annotation, "__args__", ())
            nested = next((a for a in args if isinstance(a, type) and issubclass(a, BaseModel)), None)
            if nested is None:
                return []
            current = nested
    if isinstance(current, type) and issubclass(current, BaseModel):
        return list(current.model_fields.keys())
    return []


def _format_with_suggestions(err: ValidationError, current: dict) -> str:
    """Turn pydantic errors into a single actionable message with 'did you mean' hints."""
    lines: list[str] = []
    for e in err.errors():
        loc = tuple(str(x) for x in e["loc"])
        msg = e["msg"]
        if e["type"] == "extra_forbidden" and loc:
            bad = loc[-1]
            siblings = _known_keys_at(RootConfig, loc[:-1])
            suggestions = difflib.get_close_matches(bad, siblings, n=1, cutoff=0.6)
            path = ".".join(loc)
            if suggestions:
                lines.append(f"unknown key '{path}' — did you mean '{'.'.join(loc[:-1] + (suggestions[0],))}'?")
            else:
                lines.append(f"unknown key '{path}' (valid siblings: {siblings or '<none>'})")
        else:
            path = ".".join(loc) if loc else "<root>"
            lines.append(f"{path}: {msg}")
    return "Config validation failed:\n  - " + "\n  - ".join(lines)


def validate_patch(current: dict, patch: dict) -> dict:
    """
    Deep-merge *patch* into a copy of *current*, validate the merged result against
    RootConfig, and return the merged dict. Raises ConfigPatchError on drift.
    """
    merged = copy.deepcopy(current or {})
    _deep_merge(merged, _expand_dotted_keys(patch or {}))
    try:
        RootConfig.model_validate(merged)
    except ValidationError as e:
        raise ConfigPatchError(_format_with_suggestions(e, current or {})) from e
    return merged


def validate_full(cfg: dict) -> list[str]:
    """
    Validate a full config dict. Returns a list of human-readable issues
    (empty list on clean). Never raises.
    """
    try:
        RootConfig.model_validate(cfg or {})
        return []
    except ValidationError as e:
        formatted = _format_with_suggestions(e, cfg or {})
        # Strip leading "Config validation failed:\n  - " so callers can format as a list
        body = formatted.split("\n", 1)[1] if "\n" in formatted else formatted
        return [line.lstrip("- ").strip() for line in body.splitlines() if line.strip()]
