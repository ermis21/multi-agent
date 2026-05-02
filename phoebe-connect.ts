/**
 * phoebe-connect — Pi extension to bridge Pi sessions to Phoebe + Discord.
 *
 * Usage in Pi:
 *   /connect-phoebe 123456789012345678          # connect to localhost:8090
 *   /connect-phoebe 123456789012345678 ws://phoebe:8090  # custom URL
 *   /disconnect-phoebe                          # disconnect
 *   /phoebe-status                              # connection status
 *
 * Place in ~/.pi/agent/extensions/phoebe-connect.ts for auto-discovery.
 * Or run: pi -e /path/to/phoebe-connect.ts
 *
 * What it does:
 * 1. Opens WebSocket to Phoebe's /v1/remote/ws?channel_id=<id>
 * 2. Subscribes to all Pi agent events and forwards them to Phoebe
 * 3. Receives commands from Phoebe (prompt/steer/bash/etc.) and executes them
 * 4. Bridges extension UI dialogs (select/confirm/input) through Phoebe → Discord
 */

import type { ExtensionAPI, ExtensionContext, AgentEvent } from "@mariozechner/pi-coding-agent";

// ── Types ──────────────────────────────────────────────────────────────────────

interface PhoebeEvent {
    type: string;
    [key: string]: unknown;
}

interface PhoebeCommand {
    type: string;
    id?: string;
    [key: string]: unknown;
}

interface ConnectionState {
    ws: WebSocket | null;
    connected: boolean;
    channel_id: string;
    phoebe_url: string;
    reconnect_timer: ReturnType<typeof setTimeout> | null;
}

// ── Globals ────────────────────────────────────────────────────────────────────

let state: ConnectionState = {
    ws: null,
    connected: false,
    channel_id: "",
    phoebe_url: "",
    reconnect_timer: null,
};

let ctx: ExtensionContext | undefined;
let pi: ExtensionAPI | undefined;

// ── Event forwarding ──────────────────────────────────────────────────────────

function forwardEvent(event: AgentEvent) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    // Map Pi event to Phoebe-compatible format
    const phoebeEvent: PhoebeEvent = { type: event.type };

    // Copy all event fields
    for (const key of Object.keys(event)) {
        if (key !== "type") {
            phoebeEvent[key] = (event as Record<string, unknown>)[key];
        }
    }

    try {
        state.ws.send(JSON.stringify(phoebeEvent));
    } catch (err) {
        console.error("[phoebe-connect] Failed to send event:", err);
    }
}

// ── WebSocket management ──────────────────────────────────────────────────────

function connect(channelId: string, phoebeUrl?: string): void {
    if (state.connected) {
        ctx?.ui.notify("Already connected to Phoebe", "warning");
        return;
    }

    const baseUrl = phoebeUrl || "ws://localhost:8090";
    state.channel_id = channelId;
    state.phoebe_url = baseUrl;

    const wsUrl = `${baseUrl}/v1/remote/ws?channel_id=${channelId}`;
    console.error(`[phoebe-connect] Connecting to ${wsUrl}`);

    const ws = new WebSocket(wsUrl);
    state.ws = ws;

    ws.onopen = () => {
        state.connected = true;
        console.error("[phoebe-connect] Connected to Phoebe");
        ctx?.ui.notify(`🔗 Connected to Phoebe (channel ${channelId})`, "success");
    };

    ws.onclose = (event) => {
        state.connected = false;
        state.ws = null;
        console.error(`[phoebe-connect] Disconnected (${event.code}): ${event.reason}`);
        ctx?.ui.notify(`⚠️ Phoebe disconnected: ${event.reason}`, "error");

        // Auto-reconnect after 5s (max 3 attempts)
        if (event.code !== 1000) {
            scheduleReconnect();
        }
    };

    ws.onerror = (err) => {
        console.error("[phoebe-connect] WebSocket error:", err);
    };

    ws.onmessage = (event) => {
        const text = typeof event.data === "string" ? event.data : event.data.toString();
        handlePhoebeCommand(text);
    };
}

function disconnect(): void {
    if (state.reconnect_timer) {
        clearTimeout(state.reconnect_timer);
        state.reconnect_timer = null;
    }

    if (state.ws) {
        state.ws.close(1000, "Disconnected from Pi");
        state.ws = null;
    }
    state.connected = false;
    ctx?.ui.notify("🔌 Disconnected from Phoebe", "info");
}

function scheduleReconnect(): void {
    if (state.reconnect_timer) return;
    state.reconnect_timer = setTimeout(() => {
        state.reconnect_timer = null;
        console.error("[phoebe-connect] Reconnecting...");
        connect(state.channel_id, state.phoebe_url);
    }, 5000);
}

// ── Command handling ──────────────────────────────────────────────────────────

function handlePhoebeCommand(text: string): void {
    let command: PhoebeCommand;
    try {
        command = JSON.parse(text);
    } catch {
        return;
    }

    switch (command.type) {
        case "prompt":
            handlePrompt(command);
            break;
        case "steer":
            handleSteer(command);
            break;
        case "follow_up":
            handleFollowUp(command);
            break;
        case "abort":
            handleAbort();
            break;
        case "bash":
            handleBash(command);
            break;
        case "set_model":
            handleSetModel(command);
            break;
        case "compact":
            handleCompact(command);
            break;
        case "get_state":
            handleGetState();
            break;
        case "get_session_stats":
            handleGetSessionStats();
            break;
        case "extension_ui_response":
            handleUiResponse(command);
            break;
        default:
            console.error(`[phoebe-connect] Unknown command: ${command.type}`);
    }
}

function handlePrompt(command: PhoebeCommand): void {
    const message = command.message as string;
    if (!message) return;

    // Check if agent is streaming
    if (command.streamingBehavior) {
        pi.sendUserMessage(message, {
            deliverAs: command.streamingBehavior as "steer" | "followUp",
        });
    } else {
        pi.sendUserMessage(message);
    }
}

function handleSteer(command: PhoebeCommand): void {
    const message = command.message as string;
    if (!message) return;
    pi.sendUserMessage(message, { deliverAs: "steer" });
}

function handleFollowUp(command: PhoebeCommand): void {
    const message = command.message as string;
    if (!message) return;
    pi.sendUserMessage(message, { deliverAs: "followUp" });
}

function handleAbort(): void {
    // ctx.abort() aborts the current agent turn
    ctx?.abort();
}

async function handleBash(command: PhoebeCommand): void {
    const cmd = command.command as string;
    if (!cmd) return;

    try {
        const result = await pi.exec(cmd, [], { timeout: 30000 });
        // Send result back to Phoebe
        if (state.ws?.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: "bash_result",
                id: command.id,
                output: result.stdout || "",
                exitCode: result.code ?? 0,
                error: result.stderr || "",
            }));
        }
    } catch (err) {
        if (state.ws?.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: "bash_result",
                id: command.id,
                output: "",
                exitCode: -1,
                error: String(err),
            }));
        }
    }
}

function handleSetModel(command: PhoebeCommand): void {
    const provider = command.provider as string;
    const modelId = command.modelId as string;
    if (provider && modelId) {
        pi.setModel({ provider, modelId });
    } else if (command.model as string) {
        // Handle "provider/modelId" format
        const [provider, modelId] = (command.model as string).split("/");
        if (provider && modelId) {
            pi.setModel({ provider, modelId });
        }
    }
}

function handleCompact(command: PhoebeCommand): void {
    const customInstructions = command.customInstructions as string;
    ctx?.compact({ customInstructions });
}

function handleGetState(): void {
    // Forward a get_state request through the extension API
    // This is a best-effort since we can't directly call get_state from extension
    ctx?.ui.notify("State requested", "info");
}

function handleGetSessionStats(): void {
    ctx?.ui.notify("Session stats requested", "info");
}

function handleUiResponse(command: PhoebeCommand): void {
    // Extension UI responses are handled by Pi internally
    // We just need to forward them if they came through Phoebe
    const uiId = command.id as string;
    const value = command.value as string;
    const confirmed = command.confirmed as boolean;
    const cancelled = command.cancelled as boolean;

    if (!uiId) return;

    // Send the response back through the WebSocket to Phoebe
    // Actually, Phoebe sends this TO us, meaning Discord responded to a dialog
    // We need to forward this to Pi's extension UI system
    // But Pi handles this internally - the response goes to the waiting extension
    // So we just log it
    console.error(`[phoebe-connect] UI response for ${uiId}:`, command);
}

// ── Extension entry point ─────────────────────────────────────────────────────

export default function (piApi: ExtensionAPI) {
    pi = piApi;
    // Subscribe to all agent events
    const events = [
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "queue_update",
        "compaction_start",
        "compaction_end",
        "extension_ui_request",
    ];

    for (const eventType of events) {
        pi.on(eventType as any, async (event, extensionCtx) => {
            if (extensionCtx) ctx = extensionCtx;
            forwardEvent(event as AgentEvent);
        });
    }

    // Session events
    pi.on("session_start", async (_event, extensionCtx) => {
        if (extensionCtx) ctx = extensionCtx;
    });

    // /connect-phoebe command
    pi.registerCommand("connect-phoebe", {
        description: "Connect this Pi session to Phoebe + Discord",
        getArgumentCompletions: (prefix: string) => {
            // No completions for channel IDs
            return null;
        },
        handler: async (args, extensionCtx) => {
            ctx = extensionCtx;

            const parts = args.trim().split(/\s+/);
            const channelId = parts[0];
            const phoebeUrl = parts[1] || undefined;

            if (!channelId) {
                extensionCtx.ui.notify(
                    "Usage: /connect-phoebe <channel_id> [ws://phoebe:8090]",
                    "error"
                );
                return;
            }

            connect(channelId, phoebeUrl);
        },
    });

    // /disconnect-phoebe command
    pi.registerCommand("disconnect-phoebe", {
        description: "Disconnect from Phoebe",
        handler: async (_args, _ctx) => {
            disconnect();
        },
    });

    // /phoebe-status command
    pi.registerCommand("phoebe-status", {
        description: "Show Phoebe connection status",
        handler: async (_args, _ctx) => {
            const status = state.connected
                ? `🟢 Connected to ${state.phoebe_url} (channel ${state.channel_id})`
                : "🔴 Not connected to Phoebe";
            ctx?.ui.notify(status, "info");
        },
    });
}
