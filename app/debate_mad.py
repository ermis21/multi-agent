"""MAD-style debate engine (B1).

Architecture inspired by Skytliang/Multi-Agents-Debate (the original MAD paper):

    affirmative ⇄ negative
                ↑
            moderator    — decides each round if convergence is reached
                ↓
            judge        — fallback when moderator can't decide by max_round

Differences from `app/debate.py` (the legacy "current" engine):
  - Moderator runs per-round, not just at checkpoints, and can short-circuit
    the loop if both sides converge.
  - Judge is a fallback, not the primary verdict source.
  - Per-(model, role) reliability tracked in `state/debate_history.jsonl`
    so a future caller can prefer historically-stronger advocates on close
    calls (the data is collected here; consumers can read it in the future).

Returns the same dict shape as `app/debate.py:_build_result`, so the
`run_debate()` adapter in `app/debate.py` swaps engines transparently.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config_loader import get_config


# ── Prompts ──────────────────────────────────────────────────────────────────

AFFIRMATIVE_SYSTEM = """You are the AFFIRMATIVE in a structured 3-party debate.

YOUR POSITION: {position}

RULES:
- Defend your position with concrete reasoning and evidence.
- Respond ONLY to the most recent point your opponent made.
- Do not repeat earlier arguments.
- If you have nothing left to add, say exactly: NO_NEW_POINTS
- Maximum 4 lines per response."""


NEGATIVE_SYSTEM = """You are the NEGATIVE in a structured 3-party debate.

YOUR POSITION: {position}

RULES:
- Defend your position with concrete reasoning and evidence.
- Respond ONLY to the most recent point your opponent made.
- Do not repeat earlier arguments.
- If you have nothing left to add, say exactly: NO_NEW_POINTS
- Maximum 4 lines per response."""


MODERATOR_SYSTEM = """You are the MODERATOR of a structured debate.

After each round, decide if a clear winner has emerged or if the debate should continue.

OUTPUT EXACTLY THIS FORMAT — no other text:
DECISION: continue | a_wins | b_wins
REASON: <one sentence>
CONFIDENCE: <float 0.0-1.0>

Pick `continue` if the round added new substantive arguments; pick a_wins or b_wins only when one side has clearly stronger arguments AND the other has stopped raising new points (or has conceded)."""


JUDGE_SYSTEM = """You are the JUDGE — final fallback when the moderator could not decide.

Read the full transcript and pick the side whose arguments survived best.

OUTPUT EXACTLY THIS FORMAT — no other text:
WINNER: A or B
REASON: <one sentence — the decisive argument>
CONFIDENCE: <float 0.0 to 1.0>
KEY_POINT: <the single strongest argument from the winning side>"""


# ── In-flight state ──────────────────────────────────────────────────────────

_active: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_content(resp: Any) -> str:
    """Tolerant of both dict-shaped llama responses and the Anthropic SDK shape."""
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return ""
    return str(getattr(resp, "content", "") or "").strip()


def _parse_moderator(text: str) -> dict:
    out = {"decision": "continue", "reason": "", "confidence": 0.5}
    for raw in text.splitlines():
        line = raw.strip()
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("continue", "a_wins", "b_wins"):
                out["decision"] = val
        elif line.upper().startswith("REASON:"):
            out["reason"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                out["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return out


def _parse_judge(text: str) -> dict:
    out = {"winner": "", "reason": "", "confidence": 0.5, "key_point": ""}
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("WINNER:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("A", "B"):
                out["winner"] = val
        elif upper.startswith("REASON:"):
            out["reason"] = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            try:
                out["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif upper.startswith("KEY_POINT:"):
            out["key_point"] = line.split(":", 1)[1].strip()
    return out


def _record_history(history_path: Path, entry: dict) -> None:
    """Append a debate outcome to the history file. Best-effort, never raises."""
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[debate_mad] history write failed: {e}", flush=True)


# ── Main entry ───────────────────────────────────────────────────────────────

async def run_debate_mad(
    question: str,
    context: str,
    position_a: str,
    position_b: str,
    session_id: str = "",
    debate_id: str = "",
    max_exchanges: int = 0,
) -> dict:
    """Run a MAD-style 3-party debate. Returns the same dict shape as
    `app/debate.py:_build_result` so the adapter is transparent."""
    from app.agents import _llm_call

    cfg = get_config()
    debate_cfg = cfg.get("debate", {})
    adv_temp = debate_cfg.get("advocate_temperature", 0.4)
    mod_temp = debate_cfg.get("moderator_temperature", 0.3)
    judge_temp = debate_cfg.get("judge_temperature", 0.2)
    max_total = max_exchanges or debate_cfg.get("max_total_messages", 12)

    adv_role_cfg = {"model": debate_cfg.get("advocate_model", "debate_advocate")}
    mod_role_cfg = {"model": debate_cfg.get("moderator_model", "debate_judge")}
    judge_role_cfg = {"model": debate_cfg.get("judge_model", "debate_judge")}

    history_path = Path(debate_cfg.get("history_path", "state/debate_history.jsonl"))

    # ── Continue existing debate ──
    if debate_id and debate_id in _active:
        state = _active[debate_id]
        if state["concluded"]:
            return _build_result(debate_id, state, state.get("last_verdict", {}))
    else:
        # ── New debate: open with parallel statements ──
        debate_id = debate_id or f"debate_mad_{uuid4().hex[:8]}"

        sys_a = AFFIRMATIVE_SYSTEM.format(position=position_a)
        sys_b = NEGATIVE_SYSTEM.format(position=position_b)
        ctx_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        msgs_a = [
            {"role": "system", "content": sys_a},
            {"role": "user",   "content": f"{ctx_msg}\n\nState your strongest opening argument."},
        ]
        msgs_b = [
            {"role": "system", "content": sys_b},
            {"role": "user",   "content": f"{ctx_msg}\n\nState your strongest opening argument."},
        ]

        resp_a, resp_b = await asyncio.gather(
            _llm_call(msgs_a, cfg, temperature=adv_temp, role_cfg=adv_role_cfg),
            _llm_call(msgs_b, cfg, temperature=adv_temp, role_cfg=adv_role_cfg),
        )
        text_a = _extract_content(resp_a)
        text_b = _extract_content(resp_b)
        msgs_a.append({"role": "assistant", "content": text_a})
        msgs_b.append({"role": "assistant", "content": text_b})

        state = {
            "msgs_a": msgs_a,
            "msgs_b": msgs_b,
            "transcript": [f"[A opening] {text_a}", f"[B opening] {text_b}"],
            "messages_total": 2,
            "position_a": position_a,
            "position_b": position_b,
            "question": question,
            "context": context,
            "concluded": False,
            "ended_by": None,
            "concessions": [],
            "rounds": 1,
            "session_id": session_id,
            "engine": "mad",
        }
        _active[debate_id] = state

    # ── Debate loop ──
    while state["messages_total"] < max_total and not state["concluded"]:
        state["rounds"] += 1
        # Affirmative responds to last negative point
        last_b = state["msgs_b"][-1]["content"]
        state["msgs_a"].append({"role": "user", "content": f"NEGATIVE just said:\n{last_b}\n\nRespond."})
        resp_a = await _llm_call(state["msgs_a"], cfg, temperature=adv_temp, role_cfg=adv_role_cfg)
        text_a = _extract_content(resp_a)
        state["msgs_a"].append({"role": "assistant", "content": text_a})
        state["transcript"].append(f"[A round {state['rounds']}] {text_a}")
        state["messages_total"] += 1
        if "NO_NEW_POINTS" in text_a.upper():
            state["concluded"] = True
            state["ended_by"] = "a_no_new_points"
            break

        # Negative responds to last affirmative point
        state["msgs_b"].append({"role": "user", "content": f"AFFIRMATIVE just said:\n{text_a}\n\nRespond."})
        resp_b = await _llm_call(state["msgs_b"], cfg, temperature=adv_temp, role_cfg=adv_role_cfg)
        text_b = _extract_content(resp_b)
        state["msgs_b"].append({"role": "assistant", "content": text_b})
        state["transcript"].append(f"[B round {state['rounds']}] {text_b}")
        state["messages_total"] += 1
        if "NO_NEW_POINTS" in text_b.upper():
            state["concluded"] = True
            state["ended_by"] = "b_no_new_points"
            break

        # Moderator decides if convergence is reached
        mod_msgs = [
            {"role": "system", "content": MODERATOR_SYSTEM},
            {"role": "user",   "content": (
                f"DEBATE CONTEXT: {context}\n\n"
                f"QUESTION: {question}\n\n"
                f"AFFIRMATIVE position: {position_a}\n"
                f"NEGATIVE position: {position_b}\n\n"
                f"FULL TRANSCRIPT SO FAR:\n" + "\n".join(state["transcript"]) +
                f"\n\nDecide."
            )},
        ]
        resp_mod = await _llm_call(mod_msgs, cfg, temperature=mod_temp, role_cfg=mod_role_cfg)
        mod_text = _extract_content(resp_mod)
        mod = _parse_moderator(mod_text)
        state["transcript"].append(f"[moderator round {state['rounds']}] decision={mod['decision']} confidence={mod['confidence']:.2f} reason={mod['reason']}")
        if mod["decision"] in ("a_wins", "b_wins"):
            state["concluded"] = True
            state["ended_by"] = f"moderator_{mod['decision']}"
            state["last_verdict"] = {
                "winner":     "A" if mod["decision"] == "a_wins" else "B",
                "reason":     mod["reason"],
                "confidence": mod["confidence"],
                "key_point":  "",  # moderator format doesn't extract one
                "source":     "moderator",
            }
            break

    # ── Judge fallback if loop exhausted without verdict ──
    if not state.get("last_verdict") and not state["concluded"]:
        state["concluded"] = True
        state["ended_by"] = "max_reached"
    if not state.get("last_verdict"):
        judge_msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": (
                f"QUESTION: {question}\n\n"
                f"AFFIRMATIVE position: {position_a}\n"
                f"NEGATIVE position: {position_b}\n\n"
                f"TRANSCRIPT:\n" + "\n".join(state["transcript"])
            )},
        ]
        resp_j = await _llm_call(judge_msgs, cfg, temperature=judge_temp, role_cfg=judge_role_cfg)
        judge_text = _extract_content(resp_j)
        verdict = _parse_judge(judge_text)
        verdict["source"] = "judge"
        state["last_verdict"] = verdict

    # ── Record history (per-model reliability) ──
    verdict = state["last_verdict"]
    _record_history(history_path, {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "debate_id":    debate_id,
        "session_id":   session_id,
        "engine":       "mad",
        "advocate_model": adv_role_cfg["model"],
        "moderator_model": mod_role_cfg["model"],
        "judge_model":  judge_role_cfg["model"],
        "winner":       verdict.get("winner", ""),
        "verdict_source": verdict.get("source", ""),
        "confidence":   verdict.get("confidence", 0.5),
        "rounds":       state["rounds"],
        "messages_total": state["messages_total"],
        "ended_by":     state["ended_by"],
    })

    return _build_result(debate_id, state, verdict)


def _build_result(debate_id: str, state: dict, verdict: dict) -> dict:
    """Match the shape of `app/debate.py:_build_result` exactly so the
    adapter call site stays unchanged."""
    transcript_lines = state["transcript"]
    return {
        "debate_id":         debate_id,
        "status":            "concluded" if state["concluded"] else "active",
        "messages_total":    state["messages_total"],
        "position_a":        state["position_a"],
        "position_b":        state["position_b"],
        "transcript":        "\n".join(transcript_lines),
        "last_a":            _last_advocate_text(transcript_lines, "A"),
        "last_b":            _last_advocate_text(transcript_lines, "B"),
        "concessions":       state.get("concessions", []),
        "ended_by":          state.get("ended_by", ""),
        "judge_winner":      verdict.get("winner", ""),
        "judge_reason":      verdict.get("reason", ""),
        "judge_confidence":  verdict.get("confidence", 0.5),
        "judge_key_point":   verdict.get("key_point", ""),
        "engine":            "mad",
        "verdict_source":    verdict.get("source", ""),
    }


def _last_advocate_text(transcript: list[str], side: str) -> str:
    prefix = f"[{side} "
    for line in reversed(transcript):
        if line.startswith(prefix):
            return line.split("] ", 1)[-1] if "] " in line else line
    return ""


def cleanup_debate(debate_id: str) -> None:
    _active.pop(debate_id, None)
