"""Discord UI view components.

Extracted from bot_worker.py. Views are pure UI + thin callbacks — their
side-effects reach back into bot_worker (state dicts, helper functions) via
lazy imports to avoid a top-level circular dependency.
"""

import discord


class SpeakView(discord.ui.View):
    """Pause/resume listen button (A3).

    Single-button state machine — only one control visible at a time:

        idle (🔊 Listen) ─click→ playing (⏸ Pause) ⇄ paused (▶ Resume)
                                            │
                                  natural end / vc disconnect
                                            ↓
                                       idle (🔊 Listen)

    Disabled buttons in Discord stay visible (greyed out) — they don't hide.
    A 3-button design with disabled siblings looks broken to users, so we
    use a single Button whose label/emoji/style mutates between states.

    On click while idle:
      - If user is in a voice channel → non-blocking VC playback, button
        becomes ⏸ Pause, poller flips back to 🔊 Listen on natural end.
      - If user is NOT in a VC → file-attachment fallback, button stays as
        🔊 Listen so user can click again after joining a VC.
    """

    _GATEWAY = "http://localhost:4000"
    _IDLE   = ("🔊", "Listen", discord.ButtonStyle.secondary)
    _PAUSE  = ("⏸",  "Pause",  discord.ButtonStyle.primary)
    _RESUME = ("▶",  "Resume", discord.ButtonStyle.success)
    # Which bot identity does voice playback. Worker (Gemma) is default;
    # set DISCORD_VOICE_BOT=mod (or =config) to test if the issue is
    # bot-specific (perms / token region) vs network-layer.
    import os as _os
    _VOICE_BOT = _os.environ.get("DISCORD_VOICE_BOT", "worker").strip() or "worker"

    def __init__(self, text: str, channel_id: int):
        super().__init__(timeout=900)
        self.text                = text
        self.channel_id          = channel_id
        self.voice_channel_id: int | None = None
        self._poll_task: "asyncio.Task | None" = None

    @discord.ui.button(emoji="🔊", label="Listen", style=discord.ButtonStyle.secondary,
                       custom_id="speak_btn")
    async def speak_btn(self, button: discord.ui.Button, interaction: discord.Interaction) -> None:
        # Dispatch on current label — single button cycles through 3 states.
        label = (button.label or "").lower()
        if label == "listen":
            await self._on_listen(button, interaction)
        elif label == "pause":
            await self._on_control(button, interaction, action="pause")
        elif label == "resume":
            await self._on_control(button, interaction, action="resume")
        else:
            # Defensive: unknown state, reset to idle.
            self._morph(button, *self._IDLE)
            await interaction.response.edit_message(view=self)

    # ── State morph helper ──────────────────────────────────────────────

    def _morph(self, button: discord.ui.Button, emoji: str, label: str,
               style: discord.ButtonStyle) -> None:
        button.emoji = emoji
        button.label = label
        button.style = style

    # ── Click handlers ───────────────────────────────────────────────────

    async def _on_listen(self, button: discord.ui.Button, interaction: discord.Interaction) -> None:
        import asyncio
        import bot_worker as bw

        voice_state = interaction.user.voice if interaction.guild else None
        voice_ch    = voice_state.channel if voice_state else None

        if voice_ch is None:
            # No VC → file attachment fallback; button stays as 🔊 Listen.
            await interaction.response.defer()
            await self._wav_fallback(interaction, note=None)
            return

        # VC path — morph button to ⏸ Pause, kick off non-blocking playback.
        # On any failure (UDP block, DAVE rejection, perms), automatically
        # fall back to WAV file attachment so the user still gets the audio
        # even when live VC isn't reachable. Common in early-2026 because:
        # (a) DAVE/E2EE enforcement (Mar 2026) breaks bots that haven't
        # implemented the protocol — py-cord hasn't shipped DAVE yet;
        # (b) some networks block UDP egress to Cloudflare voice IPs.
        self.voice_channel_id = voice_ch.id
        self._morph(button, *self._PAUSE)
        await interaction.response.edit_message(view=self)
        voice_failed_reason = None
        try:
            r = await bw._http.post(
                f"{self._GATEWAY}/discord/speak_voice",
                json={"voice_channel_id": voice_ch.id, "text": self.text,
                      "block": False, "bot": self._VOICE_BOT},
            )
            r.raise_for_status()
            if not r.json().get("ok"):
                voice_failed_reason = r.json().get("error", "unknown")
        except Exception as e:
            voice_failed_reason = str(e)

        if voice_failed_reason:
            # Reset button + cycle to WAV-file fallback.
            self._morph(button, *self._IDLE)
            self.voice_channel_id = None
            if interaction.message:
                try:
                    await interaction.message.edit(view=self)
                except Exception:
                    pass
            await self._wav_fallback(
                interaction,
                note=f"Voice failed ({voice_failed_reason[:80]}); sending as file instead.",
            )
            return

        self._poll_task = asyncio.create_task(self._poll_finish(button, interaction))

    async def _wav_fallback(self, interaction: discord.Interaction, note: str | None = None) -> None:
        """Drop to the WAV-file path. Used both when the user isn't in a VC
        AND when live VC playback fails (UDP block / DAVE / perms)."""
        import bot_worker as bw
        if note:
            try:
                await interaction.followup.send(note, ephemeral=True)
            except Exception:
                pass
        try:
            r = await bw._http.post(
                f"{self._GATEWAY}/discord/speak",
                json={"channel_id": self.channel_id, "text": self.text},
            )
            r.raise_for_status()
            if not r.json().get("ok"):
                await interaction.followup.send(
                    f"WAV fallback also failed: {r.json().get('error', 'unknown')}",
                    ephemeral=True,
                )
        except Exception as e:
            await interaction.followup.send(f"WAV fallback failed: {e}", ephemeral=True)

    async def _on_control(self, button: discord.ui.Button,
                          interaction: discord.Interaction, action: str) -> None:
        """action ∈ {'pause', 'resume'}"""
        import bot_worker as bw
        if not self.voice_channel_id:
            # Stale view — reset to idle so user can click Listen again.
            self._morph(button, *self._IDLE)
            await interaction.response.edit_message(view=self)
            return
        try:
            r = await bw._http.post(
                f"{self._GATEWAY}/discord/playback/{action}",
                json={"voice_channel_id": self.voice_channel_id},
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                await interaction.response.send_message(
                    f"{action.title()} failed: {data.get('error', 'unknown')}", ephemeral=True,
                )
                return
            paused = bool(data.get("paused"))
            # paused → show ▶ Resume; playing → show ⏸ Pause
            self._morph(button, *(self._RESUME if paused else self._PAUSE))
            await interaction.response.edit_message(view=self)
        except Exception as e:
            await interaction.response.send_message(f"{action} failed: {e}", ephemeral=True)

    async def _poll_finish(self, button: discord.ui.Button,
                           interaction: discord.Interaction) -> None:
        """Poll status every 2s; reset button to 🔊 Listen when playback ends."""
        import asyncio
        import bot_worker as bw
        if not self.voice_channel_id:
            return
        for _ in range(300):  # 10 minutes max
            await asyncio.sleep(2.0)
            try:
                r = await bw._http.get(
                    f"{self._GATEWAY}/discord/playback/status",
                    params={"voice_channel_id": self.voice_channel_id},
                )
                if r.status_code == 200 and not r.json().get("active"):
                    break
            except Exception:
                continue
        self._morph(button, *self._IDLE)
        self.voice_channel_id = None
        if interaction.message:
            try:
                await interaction.message.edit(view=self)
            except Exception:
                pass


class CallbackApprovalView(discord.ui.View):
    """Three-button confirmation UI that POSTs back to phoebe-api to unblock call_tool()."""

    def __init__(self, approval_id: str, tool: str, params: dict, session_id: str):
        super().__init__(timeout=600)
        self.approval_id = approval_id
        self.tool        = tool
        self.params      = params
        self.session_id  = session_id
        self.message: discord.Message | None = None

    async def _respond(self, interaction: discord.Interaction, approved: bool, always: bool = False) -> None:
        import bot_worker as bw

        channel = interaction.channel or (self.message.channel if self.message else None)

        # Disable buttons + annotate the original embed so the user sees an
        # immediate, persistent record of the decision. Replaces the old
        # delete-and-pray pattern that left the channel silent for minutes.
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if approved and always:
            status = f"🔒 Always allowing `{self.tool}` in this session — resuming..."
        elif approved:
            status = f"✅ Approved `{self.tool}` — resuming..."
        else:
            status = f"❌ Denied `{self.tool}` — worker will continue."
        try:
            await interaction.response.edit_message(content=status, embed=None, view=self)
        except Exception:
            if self.message:
                try:
                    await self.message.edit(content=status, embed=None, view=self)
                except Exception:
                    pass

        # Guarantee the thinking indicator is active before the HTTP round-trip
        # so the user sees continuous "bot is working" feedback. _start_thinking
        # is idempotent if the indicator is still running from the pre-approval
        # stream.
        if approved and channel:
            bw._start_thinking(self.session_id, channel)

        if always:
            bw._session_always_allow.setdefault(self.session_id, set()).add(self.tool)
            bw._sync_session_state(self.session_id)

        try:
            resp = await bw._http.post(f"{bw.PHOEBE_API_URL}/v1/approval_response", json={
                "approval_id": self.approval_id,
                "approved": approved,
                "always": always,
            })
            resp.raise_for_status()
        except Exception as e:
            print(f"[approval] response failed: {e}", flush=True)

    async def on_timeout(self) -> None:
        import bot_worker as bw
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(content="⏱️ Approval timed out — action cancelled.", view=self)
            except Exception:
                pass
        try:
            await bw._http.post(f"{bw.PHOEBE_API_URL}/v1/approval_response", json={
                "approval_id": self.approval_id,
                "approved": False,
            })
        except Exception:
            pass

    @discord.ui.button(label="Yes, proceed", style=discord.ButtonStyle.green, emoji="✅")
    async def yes(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._respond(interaction, approved=True)

    @discord.ui.button(label="No, cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def no(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._respond(interaction, approved=False)

    @discord.ui.button(label="Always allow", style=discord.ButtonStyle.blurple, emoji="🔒")
    async def always_allow(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._respond(interaction, approved=True, always=True)


class QuestionView(discord.ui.View):
    """Multiple-choice question buttons. Posts answer back to phoebe-api."""

    def __init__(self, question_id: str, options: list[str], session_id: str):
        super().__init__(timeout=300)
        self.question_id = question_id
        self.options = options
        self.session_id = session_id

        letters = "ABCDE"
        styles = [
            discord.ButtonStyle.primary,
            discord.ButtonStyle.secondary,
            discord.ButtonStyle.success,
            discord.ButtonStyle.primary,
            discord.ButtonStyle.secondary,
        ]

        for i, opt in enumerate(options):
            label = f"{letters[i]}: {opt[:75]}"
            btn = discord.ui.Button(
                label=label,
                style=styles[i % len(styles)],
                custom_id=f"q_{question_id}_{letters[i]}",
            )
            btn.callback = self._make_callback(letters[i], opt)
            self.add_item(btn)

    def _make_callback(self, letter: str, option_text: str):
        async def callback(interaction: discord.Interaction):
            import bot_worker as bw
            for item in self.children:
                item.disabled = True  # type: ignore[attr-defined]
            await interaction.response.edit_message(
                content=f"Selected: **{letter}** — {option_text}",
                view=self,
            )
            try:
                await bw._http.post(f"{bw.PHOEBE_API_URL}/v1/question_response", json={
                    "question_id": self.question_id,
                    "answer": letter,
                    "answer_text": option_text,
                })
            except Exception as e:
                print(f"[discord] Failed to send question response: {e}", flush=True)
            self.stop()
        return callback

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]


class PlanReviewView(discord.ui.View):
    """Three-button view shown when the agent produces a plan in plan mode.

    Accept           → store plan, switch to build mode
    Accept+Privileged → store plan + scope paths as auto-allow, switch to build
    Keep Planning    → user sends feedback, agent revises (session continuity handles this)
    """

    def __init__(self, plan_text: str, session_id: str):
        super().__init__(timeout=None)
        self.plan_text  = plan_text
        self.session_id = session_id
        self.message: discord.Message | None = None

    async def _accept(self, interaction: discord.Interaction, privileged: bool = False) -> None:
        import asyncio
        import bot_worker as bw
        await interaction.response.defer()

        bw._session_plans[self.session_id] = self.plan_text

        if privileged:
            scope_paths = bw._parse_plan_scope(self.plan_text)
            bw._session_privileged_paths[self.session_id] = scope_paths
            scope_msg = "\n".join(f"- `{p}`" for p in scope_paths) if scope_paths else "_(none detected)_"
        else:
            bw._session_privileged_paths.pop(self.session_id, None)

        bw._set_mode_for_session(self.session_id, "build")
        bw._save_state()
        bw._sync_session_state(self.session_id)

        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            await self.message.edit(view=self)

        if privileged:
            await interaction.followup.send(
                f"**Plan accepted with privileged access.** Executing plan...\n"
                f"Auto-approved scope:\n{scope_msg}"
            )
        else:
            await interaction.followup.send("**Plan accepted.** Executing plan...")

        channel = interaction.channel
        if channel is not None:
            asyncio.create_task(
                bw._execute_plan(self.session_id, channel),
                name=f"plan_exec_{self.session_id}",
            )

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.green, emoji="\u2705")
    async def accept(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._accept(interaction, privileged=False)

    @discord.ui.button(label="Accept + Privileged", style=discord.ButtonStyle.blurple, emoji="\u26a1")
    async def accept_privileged(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._accept(interaction, privileged=True)

    @discord.ui.button(label="Keep Planning", style=discord.ButtonStyle.secondary, emoji="\ud83d\udcdd")
    async def keep_planning(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            await self.message.edit(view=self)
        await interaction.followup.send(
            "Staying in **plan** mode. Type your feedback and the agent will revise the plan."
        )

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(content="Plan review timed out.", view=self)
            except Exception:
                pass


class InjectionView(discord.ui.View):
    """Four-button popup when a user messages a channel that has a live worker run.

    Immediate    → appended as a user turn before the next LLM call.
    Not urgent   → stapled onto the next tool_result as a [user_note] block.
    Clarify      → like Not urgent, with "this is clarification, not a new task" suffix.
    Queue        → held; delivered as a new turn after the current run finishes.
    """

    def __init__(self, session_id: str, text: str, author_id: int):
        # Matches the worker-stream + approval timeouts (660s) so a user who's
        # thinking through how to route their mid-flight message doesn't lose
        # the prompt out from under them.
        super().__init__(timeout=660)
        self.session_id = session_id
        self.text = text
        self.author_id = author_id
        self.message: discord.Message | None = None

    async def _handle(self, interaction: discord.Interaction, mode: str, label: str) -> None:
        import bot_worker as bw
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("This choice belongs to whoever sent the message.", ephemeral=True)
            return
        await interaction.response.defer()
        result = await bw._post_injection(self.session_id, self.text, mode)
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message is not None:
            try:
                if result is None:
                    await self.message.edit(content=f"❌ Injection failed ({label}).", view=self)
                else:
                    await self.message.edit(content=f"✅ Injected as **{label}**.", view=self)
            except Exception:
                pass

    @discord.ui.button(label="Immediate", style=discord.ButtonStyle.danger, emoji="\u26a1")
    async def btn_immediate(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._handle(interaction, "immediate", "immediate")

    @discord.ui.button(label="Not urgent", style=discord.ButtonStyle.primary, emoji="\U0001f4dd")
    async def btn_not_urgent(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._handle(interaction, "not_urgent", "not urgent")

    @discord.ui.button(label="Clarify", style=discord.ButtonStyle.primary, emoji="\U0001f4ac")
    async def btn_clarify(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._handle(interaction, "clarify", "clarification")

    @discord.ui.button(label="Queue", style=discord.ButtonStyle.secondary, emoji="\U0001f5c3\ufe0f")
    async def btn_queue(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await self._handle(interaction, "queue", "queued")

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                for item in self.children:
                    item.disabled = True  # type: ignore[attr-defined]
                await self.message.edit(
                    content="⏱️ Prompt expired — resend your message to get the options again.",
                    view=self,
                )
            except Exception:
                pass


class DreamEditReviewView(discord.ui.View):
    """Per-edit review for a dream_finalize_review event.

    Accept all  → POST {all: "keep"} and commit the whole batch.
    Reject all  → POST {all: "drop"} and discard the whole batch.
    Select…     → show a multi-select menu of phrase_ids to KEEP; everything
                  unchecked is dropped. For batches > 25 edits (Discord Select
                  cap) we fall back to chunked menus one at a time.

    The trigger flow (`/dream-run` slash in bot_worker) posts this view with
    `dreamer_sid` and the full edit list; the view POSTs to
    /v1/dream/review_response to unblock the dreamer's dream_finalize call.
    """

    _SELECT_CAP = 25  # Discord hard limit on Select options

    def __init__(self, dreamer_sid: str, edits: list[dict]):
        super().__init__(timeout=660)
        self.dreamer_sid = dreamer_sid
        self.edits = edits
        self.message: discord.Message | None = None
        self._resolved = False

    async def _post_decisions(self, decisions: dict[str, str]) -> dict:
        import bot_worker as bw
        try:
            resp = await bw._http.post(
                f"{bw.PHOEBE_API_URL}/v1/dream/review_response",
                json={"dreamer_sid": self.dreamer_sid, "decisions": decisions},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[dream-review] POST failed: {e}", flush=True)
            return {"ok": False, "error": str(e)}

    async def _finish(self, interaction: discord.Interaction, decisions: dict[str, str]) -> None:
        self._resolved = True
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        kept = sum(1 for v in decisions.values() if v == "keep")
        dropped = sum(1 for v in decisions.values() if v == "drop")
        summary = f"Submitted review: **keep={kept}**, **drop={dropped}**"
        try:
            await interaction.response.edit_message(content=summary, view=self)
        except Exception:
            if self.message:
                try:
                    await self.message.edit(content=summary, view=self)
                except Exception:
                    pass
        await self._post_decisions(decisions)
        self.stop()

    @discord.ui.button(label="Accept all", style=discord.ButtonStyle.green, emoji="✅")
    async def accept_all(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        decisions = {e["phrase_id"]: "keep" for e in self.edits}
        await self._finish(interaction, decisions)

    @discord.ui.button(label="Reject all", style=discord.ButtonStyle.red, emoji="❌")
    async def reject_all(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        decisions = {e["phrase_id"]: "drop" for e in self.edits}
        await self._finish(interaction, decisions)

    @discord.ui.button(label="Select…", style=discord.ButtonStyle.blurple, emoji="\U0001f5f3️")
    async def select_subset(self, _button: discord.ui.Button, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        if len(self.edits) <= self._SELECT_CAP:
            select_view = _DreamSelectView(self.dreamer_sid, self.edits, parent=self)
            await interaction.followup.send(
                f"Tick the edits you want to **keep** (unticked = drop):",
                view=select_view,
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                f"⚠ {len(self.edits)} edits exceeds Discord's 25-option select cap. "
                f"Use the CLI (`make dream-run`) for large batches, or Accept/Reject all here.",
                ephemeral=True,
            )

    async def on_timeout(self) -> None:
        if self._resolved or self.message is None:
            return
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        try:
            await self.message.edit(
                content="⏱️ Review timed out — all edits dropped.",
                view=self,
            )
        except Exception:
            pass
        # Server-side review_bus also times out at 660s with the same fallback,
        # so we don't strictly need to POST here — but doing so is idempotent
        # and gives an immediate drop-all if the server timer hasn't fired yet.
        await self._post_decisions({e["phrase_id"]: "drop" for e in self.edits})


class _DreamSelectView(discord.ui.View):
    """Inner multi-select used by DreamEditReviewView's "Select…" button."""

    def __init__(self, dreamer_sid: str, edits: list[dict], *, parent: "DreamEditReviewView"):
        super().__init__(timeout=300)
        self.dreamer_sid = dreamer_sid
        self.edits = edits
        self.parent_view = parent

        opts: list[discord.SelectOption] = []
        for i, e in enumerate(edits, start=1):
            # Prefix the target prompt so Discord's flat dropdown reads as
            # grouped even though SelectOption has no native group header.
            target = (e.get("target_prompt") or "").strip()
            target_prefix = f"[{target}] " if target else ""
            label = f"{i}. {target_prefix}{e.get('kind', '?')} · {(e.get('phrase_id') or '')[:70]}"
            desc_src = (e.get("section_path") or e.get("narrative") or "").replace("\n", " ")
            opts.append(discord.SelectOption(
                label=label[:100],
                value=e["phrase_id"],
                description=desc_src[:100] or None,
            ))
        self.select = discord.ui.Select(
            placeholder="Select edits to KEEP (leave blank to drop all)",
            min_values=0,
            max_values=len(opts),
            options=opts,
        )
        self.select.callback = self._on_select
        self.add_item(self.select)

    async def _on_select(self, interaction: discord.Interaction) -> None:
        kept = set(self.select.values)
        decisions = {
            e["phrase_id"]: ("keep" if e["phrase_id"] in kept else "drop")
            for e in self.edits
        }
        # Mirror disabling on this ephemeral view.
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        try:
            await interaction.response.edit_message(
                content=f"Submitted: keep={len(kept)}, drop={len(self.edits) - len(kept)}",
                view=self,
            )
        except Exception:
            pass
        await self.parent_view._post_decisions(decisions)
        self.parent_view._resolved = True
        # Mirror the parent view's disabled state.
        for item in self.parent_view.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.parent_view.message:
            try:
                await self.parent_view.message.edit(
                    content=f"Review submitted: keep={len(kept)}, drop={len(self.edits) - len(kept)}",
                    view=self.parent_view,
                )
            except Exception:
                pass
        self.parent_view.stop()
        self.stop()
