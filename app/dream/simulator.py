"""Shadow-prompt replay for a pending dream batch — multi-turn edition.

Given a conversation_sid + its pending batch, the simulator:

  1. Resolves the original model from the session's state.json.
  2. Picks a simulation model — local if the original was local (``model_match=True``),
     else the configured fallback (``model_match=False``).
  3. Materializes a shadow prompts directory as a live copy of ``/config/prompts/``,
     then overwrites the batch's target prompt with the proposed text.
  4. Opens a per-replay simulation context (overlay + ephemeral chroma collection
     + contextvar) that the sandbox uses to route writes to an overlay tree so
     the real filesystem / ChromaDB / git is never touched.
  5. Extracts the user's underlying goal (one LLM call), then replays the
     conversation turn-by-turn under the shadow prompt:
       - Compute lex + sem similarity between old_agent[i-1] and new_agent[i-1].
       - If band == IDENTICAL: use the original user turn verbatim (skip LLM).
       - If band == UNRELATED: stick fidelity=low, use verbatim for the rest.
       - Otherwise: spawn ``dream_user_simulator`` to rewrite the user turn
         as a counterfactual reaction to the NEW agent response.
     Each turn runs the target role via ``run_agent_role(...)`` passing the
     accumulated interleaved history and ``mode: simulate``.
  6. Aggregates per-turn bands into a three-level ``fidelity`` and advances
     the pending-batch state machine.
  7. Cleans up the overlay, ephemeral chroma collection, and the per-turn
     sim session folders.

The replay never emits ``final`` turns into the original session's JSONL —
shadow sessions live under their own sim_sid folders which are cleaned up
at the end of the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config_loader import get_config
from app.dream import counterfactual as cf
from app.dream import dream_state, phrase_store, sim_context
from app.dream.session_iter import SESSIONS_ROOT
from app.model_ranks import _load_catalog

logger = logging.getLogger("dream.simulator")

DEFAULT_FALLBACK_LOCAL_MODEL = "vpn_local"
DREAM_SIM_CACHE_ROOT = Path("/cache/dream_sim")
SIM_OVERLAY_ROOT = Path("/cache/dream_sim_overlay")


class SimulatorError(RuntimeError):
    """Raised when the simulator cannot proceed (missing batch, missing prompt, …)."""


# ── Data shapes ─────────────────────────────────────────────────────────────

@dataclass
class TranscriptView:
    """A compressed view of a conversation: user inputs + assistant replies."""
    transcript: list[dict]
    tool_calls: list[dict] = field(default_factory=list)
    per_turn_tool_calls: list[list[dict]] = field(default_factory=list)


@dataclass
class CounterfactualMetrics:
    per_turn: list[dict]
    avg_lex: float
    avg_sem: float
    turns_adjusted: int
    turns_verbatim: int
    max_band: str
    fidelity: str
    cf_aborts: int
    goal: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "per_turn": self.per_turn,
            "avg_lex": self.avg_lex,
            "avg_sem": self.avg_sem,
            "turns_adjusted": self.turns_adjusted,
            "turns_verbatim": self.turns_verbatim,
            "max_band": self.max_band,
            "fidelity": self.fidelity,
            "cf_aborts": self.cf_aborts,
            "goal": self.goal,
        }


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
    counterfactual: CounterfactualMetrics

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_model": self.original_model,
            "simulation_model": self.simulation_model,
            "model_match": self.model_match,
            "before": {"transcript": self.before.transcript,
                       "tool_calls": self.before.tool_calls},
            "after":  {"transcript": self.after.transcript,
                       "tool_calls": self.after.tool_calls,
                       "per_turn_tool_calls": self.after.per_turn_tool_calls},
            "simulations_remaining": self.simulations_remaining,
            "can_iterate": self.can_iterate,
            "counterfactual": self.counterfactual.to_payload(),
        }


# ── Model selection (unchanged from the prior single-shot implementation) ───

def _resolve_original_model(conv_sid: str) -> str | None:
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
    if _model_is_local(original_model):
        assert original_model is not None
        return original_model, True
    sim_cfg = ((cfg or {}).get("dream") or {}).get("simulation") or {}
    fallback = str(sim_cfg.get("fallback_local_model") or DEFAULT_FALLBACK_LOCAL_MODEL)
    return fallback, False


# ── Shadow prompts directory ───────────────────────────────────────────────

def _copy_prompts_tree(dst: Path) -> None:
    src = phrase_store.PROMPTS_DIR
    if not src.exists():
        raise SimulatorError(f"live prompts dir missing: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def materialize_shadow(batch: dream_state.PendingBatch, root: Path | None = None) -> Path:
    """Clone the prompts tree + overlay every target prompt in `batch`.

    Multi-target batches (e.g. edits across both `worker_full.md` and
    `supervisor_full.md`) get ALL their files rewritten in the shadow so
    the replay sees the combined effect — one simulation validates the
    whole batch instead of burning budget per-file.
    """
    root = root or DREAM_SIM_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    shadow = root / f"shadow-{uuid4().hex[:12]}"
    shadow.mkdir()
    _copy_prompts_tree(shadow)
    target_names = batch.target_prompts or ([batch.target_prompt] if batch.target_prompt else [])
    if not target_names:
        raise SimulatorError("batch has no target prompts to overlay")
    new_texts = batch.new_prompt_texts
    for name in target_names:
        target = shadow / f"{name}.md"
        if not target.exists():
            raise SimulatorError(
                f"batch target {name!r} not found in shadow tree (expected {target})"
            )
        body = new_texts.get(name)
        if body is None:
            # Legacy single-target fallback path
            body = batch.data.get("new_prompt_text", "")
        target.write_text(body, encoding="utf-8")
    return shadow


# ── Transcript loading ─────────────────────────────────────────────────────

def _truncate_turns(turns: list[dict], max_turns: int) -> list[dict]:
    """First-k + last-k truncation preserving the edges of a long conversation."""
    if max_turns <= 0 or len(turns) <= max_turns:
        return list(turns)
    half = max_turns // 2
    head = turns[:half]
    tail = turns[-(max_turns - half):]
    return head + [{"__elided__": len(turns) - len(head) - len(tail)}] + tail


def load_before_transcript(conv_sid: str, max_turns: int) -> TranscriptView:
    """Read `final` turns and shape them into the interleaved `before` view.

    The first user turn of each final-turn entry is appended to the transcript,
    then the assistant's response. Multiple user messages within one final
    entry all precede the single assistant response. Never re-simulates.
    """
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
    transcript = _truncate_turns(transcript, max_turns * 2)
    return TranscriptView(transcript=transcript, tool_calls=[])


@dataclass
class _Interleaved:
    """Parallel user / agent turns, zipped 1:1 so replay loops can index both."""
    users: list[str]
    agents: list[str]


def _to_interleaved(before: TranscriptView) -> _Interleaved:
    """Split the interleaved before-transcript into parallel user/agent lists.

    Drops trailing users without a paired assistant response, and elision
    markers inserted by `_truncate_turns`. If a final entry had multiple
    consecutive user messages we concatenate them (separated by `\\n\\n`) so
    the 1:1 zip stays clean — matches how the original agent saw it.
    """
    users: list[str] = []
    agents: list[str] = []
    buffered_user: list[str] = []
    for t in before.transcript:
        if "__elided__" in t:
            continue
        role = t.get("role")
        if role == "user":
            buffered_user.append(t.get("content") or "")
        elif role == "assistant":
            users.append("\n\n".join(buffered_user).strip())
            agents.append(t.get("content") or "")
            buffered_user = []
    # buffered_user at the end = trailing user turn without a reply; drop.
    return _Interleaved(users=users, agents=agents)


# ── Sim session cleanup ────────────────────────────────────────────────────

def _cleanup_sim_sessions(sim_sid: str) -> None:
    """Remove per-turn sub-session folders created during the replay.

    The replay uses session ids of the form ``{sim_sid}_t{i}`` and
    ``{sim_sid}_usersim_t{i}``. These land under ``state/sessions/`` because
    ``_run_worker`` logs per-session state + turns. We drop them here to
    avoid leaking sim runs into the session listing.
    """
    if not SESSIONS_ROOT.exists():
        return
    for child in SESSIONS_ROOT.iterdir():
        name = child.name
        if name.startswith(f"{sim_sid}_t") or name.startswith(f"{sim_sid}_usersim_t") or name == sim_sid:
            try:
                shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass


def _drop_sim_chroma_collection(collection_name: str) -> None:
    """Best-effort: drop the ephemeral sim-scoped chroma collection.

    Failure is logged and swallowed — a left-over collection wastes a handful
    of KB but does not corrupt the real ``memories`` collection.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path="/state/chroma")
        client.delete_collection(collection_name)
    except Exception:  # collection may not have been created
        pass


# ── Goal extraction ────────────────────────────────────────────────────────

async def _extract_goal(
    before: TranscriptView,
    sim_sid: str,
    sim_model: str,
    cfg: dict,
) -> str:
    """One-shot LLM call: "in one sentence, what was the user trying to do?"

    Falls back to the first user turn of the before transcript on failure
    (so the CF user-sim always has SOMETHING to anchor on).
    """
    user_turns = [t for t in before.transcript if t.get("role") == "user"]
    fallback = (user_turns[0].get("content") if user_turns else "") or ""
    cf_cfg = ((cfg or {}).get("dream") or {}).get("counterfactual") or {}
    if not cf_cfg.get("goal_extraction_enabled", True):
        return fallback
    # Keep the call tight — a short prompt on a narrow message set.
    combined = "\n\n---\n\n".join(
        (t.get("content") or "").strip() for t in user_turns[:10]
    )
    if not combined.strip():
        return fallback
    briefing = (
        "Summarise in ONE short sentence what this user was trying to accomplish "
        "across the conversation below. Do not mention the agent; describe only "
        "the user's underlying goal. Reply with the sentence only — no preamble.\n\n"
        f"{combined}"
    )
    try:
        from app.entrypoints import run_agent_role
        result = await run_agent_role(
            "worker",
            {
                "messages": [{"role": "user", "content": briefing}],
                "model": sim_model,
                "mode": "simulate",
            },
            f"{sim_sid}_goal",
        )
        text = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(result, dict) else ""
        )
        text = (text or "").strip()
        return text or fallback
    except Exception as e:
        logger.warning("goal extraction failed: %s", e)
        return fallback


# ── Single replay-turn helper ──────────────────────────────────────────────

async def _run_user_sim(
    *,
    original_user: str,
    old_agent: str,
    new_agent: str,
    goal: str,
    replay_so_far: list[dict],
    similarity: cf.Similarity,
    sim_sid: str,
    i: int,
    user_sim_model: str,
) -> tuple[str, bool, str]:
    """Invoke dream_user_simulator for turn `i`.

    Returns (turn_text, aborted, abort_reason). On abort the caller falls
    through to verbatim.
    """
    from app.entrypoints import run_agent_role
    briefing = cf.build_cf_briefing(
        original_user=original_user,
        old_agent=old_agent,
        new_agent=new_agent,
        goal=goal,
        replay_so_far=replay_so_far,
        similarity=similarity,
    )
    try:
        result = await run_agent_role(
            "dream_user_simulator",
            {
                "messages": [{"role": "user", "content": briefing}],
                "model": user_sim_model,
                "mode": "simulate",
            },
            f"{sim_sid}_usersim_t{i}",
        )
        raw = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(result, dict) else ""
        )
    except Exception as e:
        logger.warning("user_sim call failed at turn %d: %s", i, e)
        return original_user, True, f"user_sim_call_failed:{e!s:.100s}"
    san = cf.sanitize_user_sim_output(raw)
    if san.aborted:
        return original_user, True, san.abort_reason
    return san.text or original_user, False, ""


# ── Multi-turn replay core ─────────────────────────────────────────────────

async def _replay_multi_turn(
    *,
    interleaved: _Interleaved,
    target_role: str,
    sim_model: str,
    user_sim_model: str,
    shadow_dir: Path,
    goal: str,
    sim_sid: str,
    cfg: dict,
) -> tuple[TranscriptView, list[dict], int]:
    """Run the multi-turn replay loop.

    Returns (after_transcript, per_turn_metrics, cf_aborts).
    """
    from app.entrypoints import run_agent_role

    replay_users: list[str] = []
    replay_agents: list[str] = []
    replay_tools: list[list[dict]] = []
    replay_history: list[dict] = []   # interleaved list of dicts used for CF briefings
    per_turn: list[dict] = []
    cf_aborts = 0
    cf_disabled_after_unrelated = False  # once an UNRELATED fires, stop invoking user-sim

    N = len(interleaved.users)
    if N == 0:
        return TranscriptView(transcript=[], tool_calls=[]), [], 0

    for i in range(N):
        # 1. Decide the user turn for this step.
        adjusted = False
        aborted = False
        abort_reason = ""
        sim: cf.Similarity
        if i == 0:
            user_turn = interleaved.users[0]
            sim = cf.Similarity(lex=1.0, sem=1.0, lex_band=cf.Band.IDENTICAL,
                                sem_band=cf.Band.IDENTICAL, band=cf.Band.IDENTICAL)
        else:
            old_a = interleaved.agents[i - 1]
            new_a = replay_agents[i - 1]
            sim = cf.compute_similarity(old_a, new_a, cfg)

            if sim.band == cf.Band.IDENTICAL:
                user_turn = interleaved.users[i]
            elif sim.band == cf.Band.UNRELATED:
                user_turn = interleaved.users[i]   # verbatim on unrelated
                cf_disabled_after_unrelated = True
            elif cf_disabled_after_unrelated:
                user_turn = interleaved.users[i]   # sticky verbatim after first UNRELATED
            else:
                user_turn, aborted, abort_reason = await _run_user_sim(
                    original_user=interleaved.users[i],
                    old_agent=old_a,
                    new_agent=new_a,
                    goal=goal,
                    replay_so_far=list(replay_history),
                    similarity=sim,
                    sim_sid=sim_sid,
                    i=i,
                    user_sim_model=user_sim_model,
                )
                if aborted:
                    cf_aborts += 1
                else:
                    adjusted = user_turn != interleaved.users[i]

        replay_users.append(user_turn)
        replay_history.append({"role": "user", "content": user_turn})

        # 2. Build the message list the agent sees at this turn.
        msg_list: list[dict] = []
        for u, a in zip(replay_users, replay_agents):
            msg_list.append({"role": "user", "content": u})
            msg_list.append({"role": "assistant", "content": a})
        # `replay_agents` is shorter by 1; the current user turn is last.
        msg_list.append({"role": "user", "content": user_turn})

        # 3. Run the agent.
        try:
            result = await run_agent_role(
                target_role,
                {"messages": msg_list, "model": sim_model, "mode": "simulate"},
                f"{sim_sid}_t{i}",
                prompts_dir=shadow_dir,
            )
            text = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(result, dict) else ""
            )
            tool_calls = result.get("tool_trace", []) or []
        except Exception as e:
            text = f"[simulation error: {type(e).__name__}: {e}]"
            tool_calls = []

        replay_agents.append(text)
        replay_tools.append(tool_calls)
        replay_history.append({"role": "assistant", "content": text})

        per_turn.append({
            "i": i,
            "lex": sim.lex,
            "sem": sim.sem,
            "band": sim.band.value,
            "adjusted": adjusted,
            "aborted": aborted,
            "abort_reason": abort_reason,
        })

    # Assemble `after` transcript in the same shape as `before` — interleaved.
    after_transcript: list[dict] = []
    for u, a in zip(replay_users, replay_agents):
        after_transcript.append({"role": "user", "content": u})
        after_transcript.append({
            "role": "assistant",
            "content": a,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    flat_tools = [tc for turn in replay_tools for tc in turn]
    return (
        TranscriptView(
            transcript=after_transcript,
            tool_calls=flat_tools,
            per_turn_tool_calls=replay_tools,
        ),
        per_turn,
        cf_aborts,
    )


# ── Public entry point ─────────────────────────────────────────────────────

async def run_simulation(
    conversation_sid: str,
    cfg: dict | None = None,
) -> SimResult:
    """Run one multi-turn shadow replay and advance the pending-batch state machine.

    Preconditions:
      - A pending batch exists for `conversation_sid`.
      - The batch is in a phase that accepts simulations (submit or post_sim).
      - `simulations_run < cap`.

    Side effects:
      - Writes overlay files under `/cache/dream_sim_overlay/{sim_sid}/fs/`
        (cleaned in the `finally`).
      - Creates an ephemeral chroma collection `sim_{sim_sid}_memories`
        (dropped in the `finally`).
      - Creates per-turn session folders `state/sessions/{sim_sid}_t{i}/` and
        `state/sessions/{sim_sid}_usersim_t{i}/` (cleaned in the `finally`).
      - Calls `dream_state.save_pending(batch)` before returning so state
        rolls forward even if the caller forgets.
    """
    cfg = cfg if cfg is not None else get_config()
    dream_cfg = (cfg or {}).get("dream") or {}
    sim_cfg = dream_cfg.get("simulation") or {}
    cf_cfg = dream_cfg.get("counterfactual") or {}
    cap = int(sim_cfg.get("max_simulations_per_conversation") or 3)
    max_turns = int(sim_cfg.get("max_turns_replayed") or 5)
    user_sim_model = str(cf_cfg.get("user_sim_model") or DEFAULT_FALLBACK_LOCAL_MODEL)

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
    interleaved = _to_interleaved(before)

    sim_sid = f"{conversation_sid}__sim_{uuid4().hex[:6]}"
    overlay_root = SIM_OVERLAY_ROOT / sim_sid / "fs"
    overlay_root.mkdir(parents=True, exist_ok=True)
    sim_mem_collection = f"sim_{sim_sid}_memories"[:60]  # chroma collection name cap

    ctx = sim_context.SimContext(
        sim_sid=sim_sid,
        overlay_root=overlay_root,
        memory_collection=sim_mem_collection,
    )
    ctx_token = sim_context.enter(ctx)

    try:
        with tempfile.TemporaryDirectory(
            prefix="dream-sim-",
            dir=str(DREAM_SIM_CACHE_ROOT) if DREAM_SIM_CACHE_ROOT.exists() else None,
        ) as tmp:
            shadow = materialize_shadow(batch, root=Path(tmp))

            goal = await _extract_goal(before, sim_sid, sim_model, cfg)

            after, per_turn, cf_aborts = await _replay_multi_turn(
                interleaved=interleaved,
                target_role=role_to_replay,
                sim_model=sim_model,
                user_sim_model=user_sim_model,
                shadow_dir=shadow,
                goal=goal,
                sim_sid=sim_sid,
                cfg=cfg,
            )
    finally:
        # Always clean up — a half-applied overlay on disk would bloat /cache.
        sim_context.exit(ctx_token)
        try:
            if overlay_root.parent.exists():
                shutil.rmtree(overlay_root.parent, ignore_errors=True)
        except OSError:
            pass
        _drop_sim_chroma_collection(sim_mem_collection)
        _cleanup_sim_sessions(sim_sid)

    metrics = cf.summarize_metrics(per_turn, cf_aborts=cf_aborts, goal=goal)
    fidelity = metrics["fidelity"]

    # Advance the pending-batch state machine. Extended signature lets the
    # state machine block further iterations on fidelity=low as well.
    dream_state.on_simulation_complete(
        batch, model_match=match, simulations_cap=cap, fidelity=fidelity,
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
        counterfactual=CounterfactualMetrics(
            per_turn=metrics["per_turn"],
            avg_lex=metrics["avg_lex"],
            avg_sem=metrics["avg_sem"],
            turns_adjusted=metrics["turns_adjusted"],
            turns_verbatim=metrics["turns_verbatim"],
            max_band=metrics["max_band"],
            fidelity=metrics["fidelity"],
            cf_aborts=metrics["cf_aborts"],
            goal=metrics["goal"],
        ),
    )


# ── Backwards-compat helpers ───────────────────────────────────────────────

def _user_messages_from_before(before: TranscriptView) -> list[dict]:
    """DEPRECATED — kept for any external callers that may import it.

    The multi-turn replay no longer flattens user turns; use ``_to_interleaved``
    instead. Returns only the user messages as dicts (no role interleaving).
    """
    return [t for t in before.transcript if t.get("role") == "user"]
