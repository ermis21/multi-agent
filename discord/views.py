"""Discord UI view components.

Extracted from bot_worker.py. Views are pure UI + thin callbacks — their
side-effects reach back into bot_worker (state dicts, helper functions) via
lazy imports to avoid a top-level circular dependency.
"""

import discord


class SpeakView(discord.ui.View):
    """Single 🔊 button attached to agent text responses for on-demand TTS playback."""

    def __init__(self, text: str, channel_id: int):
        super().__init__(timeout=900)
        self.text       = text
        self.channel_id = channel_id

    @discord.ui.button(emoji="🔊", label="Listen", style=discord.ButtonStyle.secondary)
    async def speak_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        import bot_worker as bw
        button.disabled = True
        await interaction.response.edit_message(view=self)

        voice_state = interaction.user.voice if interaction.guild else None
        voice_ch    = voice_state.channel if voice_state else None

        try:
            if voice_ch:
                tts_resp = await bw._http.post("http://localhost:4000/discord/speak_voice", json={
                    "voice_channel_id": voice_ch.id,
                    "text":             self.text,
                })
            else:
                tts_resp = await bw._http.post("http://localhost:4000/discord/speak", json={
                    "channel_id": self.channel_id,
                    "text":       self.text,
                })
            tts_resp.raise_for_status()
            result = tts_resp.json()
            if not result.get("ok"):
                err = result.get("error", "unknown error")
                await interaction.followup.send(f"Voice generation failed: {err}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"Voice generation failed: {e}", ephemeral=True)


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
    async def yes(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._respond(interaction, approved=True)

    @discord.ui.button(label="No, cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def no(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._respond(interaction, approved=False)

    @discord.ui.button(label="Always allow", style=discord.ButtonStyle.blurple, emoji="🔒")
    async def always_allow(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
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
    async def accept(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._accept(interaction, privileged=False)

    @discord.ui.button(label="Accept + Privileged", style=discord.ButtonStyle.blurple, emoji="\u26a1")
    async def accept_privileged(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._accept(interaction, privileged=True)

    @discord.ui.button(label="Keep Planning", style=discord.ButtonStyle.secondary, emoji="\ud83d\udcdd")
    async def keep_planning(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
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
        super().__init__(timeout=120)
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
    async def btn_immediate(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "immediate", "immediate")

    @discord.ui.button(label="Not urgent", style=discord.ButtonStyle.primary, emoji="\U0001f4dd")
    async def btn_not_urgent(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "not_urgent", "not urgent")

    @discord.ui.button(label="Clarify", style=discord.ButtonStyle.primary, emoji="\U0001f4ac")
    async def btn_clarify(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "clarify", "clarification")

    @discord.ui.button(label="Queue", style=discord.ButtonStyle.secondary, emoji="\U0001f5c3\ufe0f")
    async def btn_queue(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "queue", "queued")

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                for item in self.children:
                    item.disabled = True  # type: ignore[attr-defined]
                await self.message.edit(content="Injection prompt timed out.", view=self)
            except Exception:
                pass
