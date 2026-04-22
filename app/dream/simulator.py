"""Shadow-prompt replay for a pending dream batch.

Given a conversation_sid + its pending batch, the simulator:

  1. Resolves the original model from the session's state.json.
  2. Picks a simulation model — local if the original was local (`model_match=True`),
     else the configured fallback (`model_match=False`).
  3. Materializes a shadow prompts directory as a live copy of `/config/prompts/`,
     then overwrites the batch's target prompt with the proposed text.
  4. Replays the conversation's user turns under the shadow prompt by running the
     original role via `run_agent_role(..., prompts_dir=<shadow>)`. Loads the
     `before` transcript from disk — never re-simulated.
  5. Returns a structured result and advances the pending-batch state machine.

The replay is a single-shot worker call — it does NOT emit `final` turns for the
simulated session, so shadow runs don't poison session history. The simulator
spawns a *child* session id for the replay so its logs land under a separate
folder the runner can clean up.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config_loader import get_config
from app.dream import dream_state, phrase_store
from app.dream.session_iter import SESSIONS_ROOT
from app.model_ranks import _load_catalog

DEFAULT_FALLBACK_LOCAL_MODEL = "vpn_local"
DREAM_SIM_CACHE_ROOT = Path("/cache/dream_sim")


class SimulatorError(RuntimeError):
    """Raised when the simulator cannot proceed (missing batch, missing prompt, …)."""


@dataclass
class TranscriptView:
    """A compressed view of a conversation: user inputs + final assistant replies."""
    transcript: list[dict]
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class SimResult:
    session_id: str
    original_model: str | None
    simulation_model: str
    model_match: bool
    before: TranscriptView
    after: TranscriptView
    simulations_remaining: int
    can_iterate: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_model": self.original_model,
            "simulation_model": self.simulation_model,
            "model_match": self.model_match,
            "before": {"transcript": self.before.transcript,
                       "tool_calls": self.before.tool_calls},
            "after":  {"transcript": self.after.transcript,
                       "tool_calls": self.after.tool_calls},
            "simulations_remaining": self.simulations_remaining,
            "can_iterate": self.can_iterate,
        }


# ── Model selection ──────────────────────────────────────────────────────────

def _resolve_original_model(conv_sid: str) -> str | None:
    """Read `model` from the original session's state.json. Returns None when
    the session has no folder yet or state.json is missing/unreadable."""
    p = SESSIONS_ROOT / conv_sid / "state.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    model = data.get("model")
    return str(model) if model else None


def _model_is_local(model_name: str | None) -> bool:
    """True iff `model_name` has provider=='local' in model_ranks.yaml.

    Returns False for None / unknown models (so the fallback kicks in).
    """
    if not model_name:
        return False
    try:
        cat = _load_catalog()
    except Exception:
        return False
    for entry in cat.models:
        if entry.name == model_name:
            return entry.provider == "local"
    return False


def select_simulation_model(original_model: str | None, cfg: dict) -> tuple[str, bool]:
    """Pick a model name for the replay and whether it matches the original.

    Returns `(simulation_model_name, model_match)`.
    """
    if _model_is_local(original_model):
        assert original_model is not None
        return original_model, True
    sim_cfg = ((cfg or {}).get("dream") or {}).get("simulation") or {}
    fallback = str(sim_cfg.get("fallback_local_model") or DEFAULT_FALLBACK_LOCAL_MODEL)
    return fallback, False


# ── Shadow prompts directory ─────────────────────────────────────────────────

def _copy_prompts_tree(dst: Path) -> None:
    """Mirror the live PROMPTS_DIR tree into `dst` (including tool-doc subdir).

    We copy rather than symlink so the dreamer can never accidentally mutate
    the live tree through the shadow.
    """
    src = phrase_store.PROMPTS_DIR
    if not src.exists():
        raise SimulatorError(f"live prompts dir missing: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def materialize_shadow(batch: dream_state.PendingBatch, root: Path | None = None) -> Path:
    """Write a shadow prompts directory with the batch's candidate text.

    Returns the shadow directory path. Caller is responsible for cleanup —
    the live pipeline uses `run_simulation` which wraps this in a
    `tempfile.TemporaryDirectory`.
    """
    root = root or DREAM_SIM_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    shadow = root / f"shadow-{uuid4().hex[:12]}"
    shadow.mkdir()
    _copy_prompts_tree(shadow)
    target = shadow / f"{batch.target_prompt}.md"
    if not target.exists():
        raise SimulatorError(
            f"batch.target_prompt={batch.target_prompt!r} not found in shadow tree "
            f"(expected {target})"
        )
    target.write_text(batch.data["new_prompt_text"], encoding="utf-8")
    return shadow


# ── Transcript loading ───────────────────────────────────────────────────────

def _truncate_turns(turns: list[dict], max_turns: int) -> list[dict]:
    """First-k + last-k truncation preserving the edges of a long conversation.

    When `len(turns) <= max_turns` return as-is. Otherwise split the budget in
    half; keep the head and the tail with an elision marker between them.
    """
    if max_turns <= 0 or len(turns) <= max_turns:
        return list(turns)
    half = max_turns // 2
    head = turns[:half]
    tail = turns[-(max_turns - half):]
    return head + [{"__elided__": len(turns) - len(head) - len(tail)}] + tail


def load_before_transcript(conv_sid: str, max_turns: int) -> TranscriptView:
    """Read `final` turns from `{sid}/turns.jsonl` and shape them into the
    dreamer-facing `before` view. Never re-simulates."""
    turns_path = SESSIONS_ROOT / conv_sid / "turns.jsonl"
    if not turns_path.exists():
        return TranscriptView(transcript=[], tool_calls=[])
    transcript: list[dict] = []
    with turns_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("role") != "final":
                continue
            for msg in row.get("messages", []) or []:
                if msg.get("role") == "user":
                    transcript.append({"role": "user", "content": msg.get("content", "")})
            transcript.append({
                "role": "assistant",
                "content": row.get("response", ""),
                "timestamp": row.get("timestamp"),
            })
    transcript = _truncate_turns(transcript, max_turns * 2)  # ≈ user+assistant pairs
    return TranscriptView(transcript=transcript, tool_calls=[])


# ── Replay under a shadow prompt ─────────────────────────────────────────────

async def _replay_under_shadow(
    *,
    original_sid: str,
    target_role: str,
    user_messages: list[dict],
    simulation_model_name: str,
    shadow_dir: Path,
    cfg: dict,
) -> TranscriptView:
    """Run the role once with the shadow prompt directory and capture the
    assistant's response."""
    # Lazy import — entrypoints pulls agent/loop which is heavy.
    from app.entrypoints import run_agent_role

    sim_sid = f"{original_sid}__sim_{uuid4().hex[:6]}"
    body = {
        "messages": user_messages,
        "model": simulation_model_name,
    }
    try:
        result = await run_agent_role(
            target_role, body, sim_sid, prompts_dir=shadow_dir,
        )
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_calls = result.get("tool_trace", []) or []
    except Exception as e:
        text = f"[simulation error: {type(e).__name__}: {e}]"
        tool_calls = []
    return TranscriptView(
        transcript=[{"role": "assistant", "content": text,
                     "timestamp": datetime.now(timezone.utc).isoformat()}],
        tool_calls=tool_calls,
    )


# ── Public entry point ───────────────────────────────────────────────────────

def _user_messages_from_before(before: TranscriptView) -> list[dict]:
    """Extract the user turns from the before-transcript so the replay sees
    the same prompts the original agent did."""
    return [t for t in before.transcript if t.get("role") == "user"]


async def run_simulation(
    conversation_sid: str,
    cfg: dict | None = None,
) -> SimResult:
    """Run one shadow replay and advance the pending-batch state machine.

    Preconditions:
      - A pending batch exists for `conversation_sid`.
      - The batch is in a phase that accepts simulations (submit or post_sim).
      - `simulations_run < cap`.

    On return, the caller should persist the batch — `run_simulation` calls
    `dream_state.save_pending(batch)` before returning so state rolls forward
    even if the caller forgets.
    """
    cfg = cfg if cfg is not None else get_config()
    dream_cfg = (cfg or {}).get("dream") or {}
    sim_cfg = dream_cfg.get("simulation") or {}
    cap = int(sim_cfg.get("max_simulations_per_conversation") or 3)
    max_turns = int(sim_cfg.get("max_turns_replayed") or 5)

    batch = dream_state.load_pending(conversation_sid)
    if batch.phase == dream_state.PHASE_FINALIZE_ONLY:
        raise SimulatorError("batch is in finalize_only — further sims rejected")
    if batch.simulations_run >= cap:
        raise SimulatorError(f"simulation cap ({cap}) already reached")

    original_model = _resolve_original_model(conversation_sid)
    sim_model, match = select_simulation_model(original_model, cfg)

    # Determine the role to replay — prefer the original session's role; fall
    # back to "worker" so dream-run of a session with no state.json still works.
    role_to_replay = "worker"
    st_path = SESSIONS_ROOT / conversation_sid / "state.json"
    if st_path.exists():
        try:
            role_to_replay = (
                json.loads(st_path.read_text(encoding="utf-8")).get("agent_role")
                or "worker"
            )
        except (json.JSONDecodeError, OSError):
            pass

    before = load_before_transcript(conversation_sid, max_turns)
    user_msgs = _user_messages_from_before(before)

    with tempfile.TemporaryDirectory(prefix="dream-sim-", dir=str(DREAM_SIM_CACHE_ROOT) if DREAM_SIM_CACHE_ROOT.exists() else None) as tmp:
        shadow = materialize_shadow(batch, root=Path(tmp))
        after = await _replay_under_shadow(
            original_sid=conversation_sid,
            target_role=role_to_replay,
            user_messages=user_msgs,
            simulation_model_name=sim_model,
            shadow_dir=shadow,
            cfg=cfg,
        )

    dream_state.on_simulation_complete(
        batch, model_match=match, simulations_cap=cap,
    )
    dream_state.save_pending(batch)

    remaining = dream_state.simulations_remaining(batch, cap)
    can_iter = dream_state.can_iterate(batch, cap)
    return SimResult(
        session_id=conversation_sid,
        original_model=original_model,
        simulation_model=sim_model,
        model_match=match,
        before=before,
        after=after,
        simulations_remaining=remaining,
        can_iterate=can_iter,
    )
