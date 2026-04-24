"""
Structured debate orchestration for the `deliberate` tool.

Two advocate subagents debate point-by-point with strict formatting.
An optional judge provides an independent verdict.
The main agent receives the transcript + verdict and decides.
"""

import asyncio
from uuid import uuid4

from app.config_loader import get_config


ADVOCATE_SYSTEM = """You are Advocate {side} in a structured debate.

YOUR POSITION: {position}

RULES:
- Every response MUST use this exact format:
  POINT: <one sentence — your specific argument>
  EVIDENCE: <one sentence — supporting fact, tradeoff, or example>
- Address ONLY the point your opponent just raised. Do not repeat earlier arguments.
- If your opponent made a valid point you cannot counter, say:
  CONCEDE: <what you concede and why>
- If you have no new arguments left, say:
  NO_NEW_POINTS
- Never hedge or equivocate. Argue from strength.
- Maximum 3 lines total per response."""


JUDGE_SYSTEM = """You are the judge of a structured debate. You see the full transcript.

Determine which position is better supported by the arguments and evidence presented.

Consider:
- Which advocate made more specific, evidence-backed arguments?
- Did either side concede important points?
- Did either side repeat itself or fail to address a counterpoint?

RESPOND WITH ONLY THIS FORMAT — no other text:
WINNER: A or B
REASON: <one sentence — the decisive argument>
CONFIDENCE: <float 0.0 to 1.0>
KEY_POINT: <the single strongest argument from the winning side>"""


_active_debates: dict[str, dict] = {}


async def run_debate(
    question: str,
    context: str,
    position_a: str,
    position_b: str,
    session_id: str = "",
    debate_id: str = "",
    max_exchanges: int = 0,
) -> dict:
    """
    Run or continue a structured debate with judge verdict.

    New debate (debate_id=""):
      1. Both advocates present opening statements (parallel)
      2. Then alternate for checkpoint_messages exchanges
      3. Judge evaluates the transcript (independent verdict)
      4. Return transcript + judge verdict + debate_id

    Continue debate (debate_id=existing):
      1. Resume from where we left off
      2. Run max_exchanges more messages
      3. Judge re-evaluates
      4. Return updated transcript + verdict
    """
    from app.agents import _llm_call

    cfg = get_config()
    debate_cfg = cfg.get("debate", {})
    adv_temp = debate_cfg.get("advocate_temperature", 0.4)
    judge_temp = debate_cfg.get("judge_temperature", 0.2)
    checkpoint_at = max_exchanges or debate_cfg.get("checkpoint_messages", 4)
    max_total = debate_cfg.get("max_total_messages", 12)

    adv_role_cfg = {"model": debate_cfg.get("advocate_model", "debate_advocate")}
    judge_role_cfg = {"model": debate_cfg.get("judge_model", "debate_judge")}

    # New debate — initialize
    if not debate_id or debate_id not in _active_debates:
        debate_id = debate_id or f"debate_{uuid4().hex[:8]}"

        sys_a = ADVOCATE_SYSTEM.format(side="A", position=position_a)
        sys_b = ADVOCATE_SYSTEM.format(side="B", position=position_b)

        context_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        msgs_a = [
            {"role": "system", "content": sys_a},
            {"role": "user", "content": f"{context_msg}\n\nPresent your strongest opening argument."},
        ]
        msgs_b = [
            {"role": "system", "content": sys_b},
            {"role": "user", "content": f"{context_msg}\n\nPresent your strongest opening argument."},
        ]

        resp_a, resp_b = await asyncio.gather(
            _llm_call(msgs_a, cfg, temperature=adv_temp, role_cfg=adv_role_cfg),
            _llm_call(msgs_b, cfg, temperature=adv_temp, role_cfg=adv_role_cfg),
        )

        text_a = _extract_content(resp_a)
        text_b = _extract_content(resp_b)

        msgs_a.append({"role": "assistant", "content": text_a})
        msgs_b.append({"role": "assistant", "content": text_b})

        transcript = [
            f"[A opening] {text_a}",
            f"[B opening] {text_b}",
        ]

        _active_debates[debate_id] = {
            "msgs_a": msgs_a,
            "msgs_b": msgs_b,
            "transcript": transcript,
            "messages_total": 2,
            "position_a": position_a,
            "position_b": position_b,
            "question": question,
            "context": context,
            "concluded": False,
            "concessions": [],
            "ended_by": "",
        }

    state = _active_debates[debate_id]
    if state["concluded"]:
        return _build_result(debate_id, state, state.get("last_verdict", {}))

    msgs_a = state["msgs_a"]
    msgs_b = state["msgs_b"]
    transcript = state["transcript"]
    messages_this_round = 0

    while messages_this_round < checkpoint_at:
        if state["messages_total"] >= max_total:
            state["concluded"] = True
            state["ended_by"] = "max_reached"
            break

        # A's turn — respond to B's last point
        last_b = _last_advocate_text(transcript, "B")
        if last_b:
            msgs_a.append({
                "role": "user",
                "content": f"Opponent's point:\n{last_b}\n\nRespond to this specific point.",
            })
        resp_a = await _llm_call(msgs_a, cfg, temperature=adv_temp, role_cfg=adv_role_cfg)
        text_a = _extract_content(resp_a)
        msgs_a.append({"role": "assistant", "content": text_a})
        transcript.append(f"[A] {text_a}")
        messages_this_round += 1
        state["messages_total"] += 1

        if "NO_NEW_POINTS" in text_a:
            state["concluded"] = True
            state["ended_by"] = "a_no_new_points"
            break
        if "CONCEDE" in text_a:
            state["concessions"].append(f"A conceded: {text_a}")
            state["concluded"] = True
            state["ended_by"] = "a_concede"
            break
        if messages_this_round >= checkpoint_at:
            state["ended_by"] = "checkpoint"
            break
        if state["messages_total"] >= max_total:
            state["concluded"] = True
            state["ended_by"] = "max_reached"
            break

        # B's turn — respond to A's just-made point
        msgs_b.append({
            "role": "user",
            "content": f"Opponent's point:\n{text_a}\n\nRespond to this specific point.",
        })
        resp_b = await _llm_call(msgs_b, cfg, temperature=adv_temp, role_cfg=adv_role_cfg)
        text_b = _extract_content(resp_b)
        msgs_b.append({"role": "assistant", "content": text_b})
        transcript.append(f"[B] {text_b}")
        messages_this_round += 1
        state["messages_total"] += 1

        if "NO_NEW_POINTS" in text_b:
            state["concluded"] = True
            state["ended_by"] = "b_no_new_points"
            break
        if "CONCEDE" in text_b:
            state["concessions"].append(f"B conceded: {text_b}")
            state["concluded"] = True
            state["ended_by"] = "b_concede"
            break

    if not state["ended_by"]:
        state["ended_by"] = "checkpoint"

    use_judge = debate_cfg.get("use_judge", True)
    verdict: dict = {}
    if use_judge:
        transcript_text = "\n\n".join(transcript)
        judge_msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {state['question']}"
                    f"\n\nPOSITION A: {state['position_a']}"
                    f"\nPOSITION B: {state['position_b']}"
                    f"\n\nDEBATE TRANSCRIPT:\n{transcript_text}"
                    f"\n\nWhich position is better supported? Use the required format."
                ),
            },
        ]
        try:
            judge_resp = await _llm_call(
                judge_msgs, cfg, temperature=judge_temp, role_cfg=judge_role_cfg,
            )
            verdict = _parse_judge_verdict(_extract_content(judge_resp))
        except Exception:
            verdict = {
                "winner": "",
                "reason": "judge call failed",
                "confidence": 0.0,
                "key_point": "",
            }

    state["last_verdict"] = verdict
    return _build_result(debate_id, state, verdict)


def _extract_content(resp) -> str:
    """Extract text content from LLM response."""
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(resp)


def _last_advocate_text(transcript: list, side: str) -> str:
    """Get the last message from a specific side."""
    for entry in reversed(transcript):
        if entry.startswith(f"[{side}]") or entry.startswith(f"[{side} opening]"):
            if "] " in entry:
                return entry.split("] ", 1)[1]
            return entry
    return ""


def _parse_judge_verdict(text: str) -> dict:
    """Parse the judge's structured output."""
    verdict = {"winner": "a", "reason": "", "confidence": 0.5, "key_point": ""}
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("WINNER:"):
            w = line.split(":", 1)[1].strip().lower()
            verdict["winner"] = "a" if "a" in w else "b"
        elif line.startswith("REASON:"):
            verdict["reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                verdict["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("KEY_POINT:"):
            verdict["key_point"] = line.split(":", 1)[1].strip()
    return verdict


def _build_result(debate_id: str, state: dict, verdict: dict) -> dict:
    """Build the result dict returned to the main agent — includes judge verdict."""
    return {
        "debate_id": debate_id,
        "status": "concluded" if state["concluded"] else "active",
        "messages_total": state["messages_total"],
        "position_a": state["position_a"],
        "position_b": state["position_b"],
        "transcript": "\n".join(state["transcript"]),
        "last_a": _last_advocate_text(state["transcript"], "A"),
        "last_b": _last_advocate_text(state["transcript"], "B"),
        "concessions": state["concessions"],
        "ended_by": state["ended_by"],
        "judge_winner": verdict.get("winner", ""),
        "judge_reason": verdict.get("reason", ""),
        "judge_confidence": verdict.get("confidence", 0.5),
        "judge_key_point": verdict.get("key_point", ""),
    }


def cleanup_debate(debate_id: str) -> None:
    """Remove a completed debate from memory."""
    _active_debates.pop(debate_id, None)
