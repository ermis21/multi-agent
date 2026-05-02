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
    # Optional name of an overlay file under config/prompts/overlays/<name>.md
    # to inject above the rules block via the {{PROMPT_OVERLAY}} marker.
    # Use to teach a weak local model an explicit tool-call grammar.
    prompt_overlay: str | None = None


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
    # Simulate mode is entered by the dream simulator to mark calls as being
    # part of a replay — sandbox handlers then route writes through the
    # overlay. Tools here are excluded regardless of overlay-ability.
    simulate: ModeToolsConfig = Field(default_factory=ModeToolsConfig)


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
    # Dream-replay approval bucket — all writes are overlaid by the sandbox,
    # so no Discord approval ever makes sense under simulate mode.
    simulate: ApprovalBucket = Field(default_factory=ApprovalBucket)


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


class LocalModelsRefreshConfig(BaseModel):
    """Nightly sweep that queries every provider=local endpoint declared in
    `config/model_ranks.yaml`, records observed `model_id` and `context_window`
    to `/state/model_metadata/local_models.json`, and (optionally) patches the
    ranks file in place when catalog metadata has drifted."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    schedule: str = "30 4 * * *"
    auto_patch: bool = True
    timeout_s: float = Field(8.0, ge=0.1)

    @field_validator("schedule")
    @classmethod
    def _cron_shape(cls, v: str) -> str:
        if not _CRON_RE.match(v):
            raise ValueError("schedule must be a 5-field cron expression (min hour dom mon dow)")
        return v


# ── Misc ───────────────────────────────────────────────────────────────────────

class InflectionConfig(BaseModel):
    """Inflection-nudging engine settings (B3).

    Until B3 the `app/inflection.py` module didn't exist on disk and every
    nudge attempt silently failed. This block configures the now-built
    module with three detectable signal sources:

    * `logprobs`         — token-entropy + top-2 logprob gap (uses the
                           logprobs payload from the LLM call when available)
    * `linguistic`       — hedging-marker word counts on the response text
    * `uqlm_whitebox`    — UQLM MinTokenProbability + LengthNormalizedProbability
                           scorers; runs on the SAME logprobs payload, no extra
                           LLM call (cost is just math)

    The legacy `inflection_mode` field is kept for back-compat — old configs
    keep working. New configs should use `engine` which has a wider enum and
    supersedes inflection_mode when both are present.
    """
    model_config = ConfigDict(extra="forbid")

    engine: Literal[
        "off", "logprobs", "linguistic", "both", "uqlm_whitebox"
    ] = "off"
    entropy_threshold: float = 1.5
    logprob_gap_threshold: float = 0.5
    top_logprobs: int = Field(5, ge=1)
    strong_marker_threshold: int = Field(1, ge=0)
    weak_marker_threshold: int = Field(3, ge=0)
    max_nudges_per_turn: int = Field(2, ge=0)
    # UQLM-only: minimum normalized probability below which a nudge fires.
    uqlm_min_norm_prob: float = 0.4


class DebateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # B1: engine selector. "current" = original 2-advocate + judge in app/debate.py;
    # "mad" = MAD-pattern affirmative + negative + moderator + judge fallback in
    # app/debate_mad.py. Default "current" — flip after A/B validation.
    engine: Literal["current", "mad"] = "current"
    checkpoint_messages: int = Field(4, ge=1)
    advocate_temperature: float = 0.4
    judge_temperature: float = 0.2
    advocate_model: str = "debate_advocate"
    judge_model: str = "debate_judge"
    use_judge: bool = True
    max_total_messages: int = Field(12, ge=1)
    # MAD-only: moderator decides per-round if convergence is reached.
    moderator_temperature: float = 0.3
    moderator_model: str = "debate_judge"  # share judge model by default
    # State path for per-model reliability tracking (across-session learning).
    history_path: str = "state/debate_history.jsonl"


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
    # B2 — compactor engine selector. "agent" spawns the legacy session_compactor
    # sub-run (one full LLM round-trip per compaction). "lingua" uses LongLLMLingua
    # to compress the uncompacted tail directly via a small BERT-tier scorer (no
    # LLM call). Both paths must produce text containing `## RUNNING_SUMMARY` —
    # the lingua path wraps its output with that header so downstream
    # `_rebuild_session_context` stays unchanged. Default "agent" — flip after
    # A/B verification on logged sessions.
    compactor_engine: Literal["agent", "lingua"] = "agent"
    # Lingua-only knobs (passed to LongLLMLingua; defaults match the paper).
    lingua_target_token: int = Field(1500, ge=100)
    lingua_rate: float = Field(0.5, ge=0.05, le=1.0)
    # Compactor scheduling helpers (used by app/compactor.py).
    compaction_interval_turns: int = Field(6, ge=1)
    compaction_churn_seconds: float = Field(60.0, ge=0)


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


class DreamCounterfactualConfig(BaseModel):
    """Counterfactual user-simulator settings. Controls the replay's per-turn
    user-turn rewriting and band-based fidelity gating."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    user_sim_model: str = "vpn_local"
    goal_extraction_enabled: bool = True
    max_replay_history_chars: int = Field(3200, ge=0)
    identical_lex_min:   float = Field(0.90, ge=0.0, le=1.0)
    identical_sem_min:   float = Field(0.92, ge=0.0, le=1.0)
    minor_lex_min:       float = Field(0.70, ge=0.0, le=1.0)
    minor_sem_min:       float = Field(0.75, ge=0.0, le=1.0)
    substantial_lex_min: float = Field(0.40, ge=0.0, le=1.0)
    substantial_sem_min: float = Field(0.45, ge=0.0, le=1.0)
    divergent_lex_min:   float = Field(0.15, ge=0.0, le=1.0)
    divergent_sem_min:   float = Field(0.20, ge=0.0, le=1.0)


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
    counterfactual: DreamCounterfactualConfig = Field(default_factory=DreamCounterfactualConfig)
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


class TTSConfig(BaseModel):
    """Discord TTS (A3). `backend` selects which engine the discord container
    loads via `discord/tts_backend.py`. `voice` and per-backend model paths
    are also overridable here so config_agent can tune them at runtime."""
    model_config = ConfigDict(extra="forbid")

    backend: Literal["piper", "kokoro_onnx", "kokoro_torch"] = "piper"
    voice:   str = "af_bella"  # Kokoro default; ignored for Piper (model file is the voice)


class SandboxConfig(BaseModel):
    """Sandbox execution engine (B5). `current` keeps the working subprocess
    path inside phoebe-sandbox; `microsandbox` routes shell_exec into a
    persistent microVM via the embedded microsandbox SDK.

    The microsandbox path is OPT-IN. Activating it requires:
      1. Installing the `microsandbox` PyPI package in sandbox/requirements.txt.
      2. Mounting `/dev/kvm` into the phoebe-sandbox container in docker-compose.yml.
      3. Setting `cfg.sandbox.engine: microsandbox` here.
    Any failure in the microsandbox path falls back to the legacy `current`
    engine with a logged warning so a misconfigured opt-in never breaks
    shell_exec for the agent."""
    model_config = ConfigDict(extra="forbid")

    engine: Literal["current", "microsandbox"] = "current"
    image:  str = "alpine:latest"  # OCI image microsandbox boots from when engine=microsandbox


# ── Pi.dev Remote Control ─────────────────────────────────────────────────────

class RemoteConfig(BaseModel):
    """Pi.dev remote control bridge. Pi connects OUT to Phoebe via WebSocket.
    Phoebe listens on ws_port and multiplexes multiple Pi instances on a single port.
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ws_port: int = Field(8090, ge=1, le=65535)
    ws_host: str = "phoebe-api"
    default_model: str | None = None
    default_thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    session_dir: str = "/state/pi_sessions"
    auto_compaction: bool = True
    auto_retry: bool = True
    max_tool_timeout_s: int = Field(120, ge=1)


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
    local_models_refresh: LocalModelsRefreshConfig = Field(default_factory=LocalModelsRefreshConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    inflection: InflectionConfig = Field(default_factory=InflectionConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)


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
