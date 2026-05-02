# Pi.dev Remote Control

Attach active Pi.dev coding sessions to the Phoebe Discord UI for real-time monitoring and control.

## Architecture

Two connection modes:

### Mode 1: Pi Extension (recommended)

```
  Pi (your machine)          Phoebe host
  ┌──────────────┐    ┌──────────────────────┐
  │ pi + extension│────│ ws://phoebe:8090     │
  │ /connect-phoebe│───│  /v1/remote/ws       │
  └──────────────┘    │  (FastAPI WS server)  │
                      │  phoebe-discord       │
                      └──────────────────────┘
```

Pi runs with the `phoebe-connect.ts` extension. In Pi's TUI, type `/connect-phoebe <channel_id>` to connect.

### Mode 2: RPC Proxy (headless)

```
  Pi (your machine)          Phoebe host
  ┌──────────────┐    ┌──────────────────────┐
  │ pi --mode rpc │────│ ws://phoebe:8090     │
  │  + rpc-proxy  │────│  /v1/remote/ws       │
  └──────────────┘    │  phoebe-discord       │
                      └──────────────────────┘
```

Pi runs in headless RPC mode, `rpc-proxy.js` bridges stdin/stdout JSONL to WebSocket.

## Quick Start (Extension Mode)

### 1. Install the extension

Copy `phoebe-connect.ts` to your Pi extensions directory:

```bash
mkdir -p ~/.pi/agent/extensions
cp /path/to/Phoebe/phoebe-connect.ts ~/.pi/agent/extensions/
```

Or load it temporarily: `pi -e /path/to/phoebe-connect.ts`

### 2. Connect from Pi

In Pi's TUI, type:

```
/connect-phoebe 123456789012345678          # connect to localhost:8090
/connect-phoebe 123456789012345678 ws://phoebe:8090  # custom URL
```

Replace `123456789012345678` with your Discord channel ID (find in `.env`):
```bash
grep DISCORD_WORKER_CHANNELS .env
```

### 3. Pi extension commands

| Command | Description |
|---------|-------------|
| `/connect-phoebe <channel_id> [url]` | Connect to Phoebe |
| `/disconnect-phoebe` | Disconnect from Phoebe |
| `/phoebe-status` | Show connection status |

### 4. Use in Discord

Once connected, your Discord channel becomes a Pi terminal:

- **Normal messages** → sent as prompts to Pi
- **Messages while Pi is running** → sent as steer commands
- **`/remote status`** → show Pi session state
- **`/remote abort`** → stop current operation
- **`/remote bash "ls"`** → run a shell command
- **`/remote model anthropic/claude-sonnet-4-20250514`** → switch model
- **`/remote detach`** → disconnect Pi

## Quick Start (RPC Proxy Mode)

For headless operation or when you can't install extensions:

```bash
# On your Pi machine, in your project directory:
node /path/to/Phoebe/rpc-proxy.js --url ws://your-phoebe-host:8090/v1/remote/ws?channel_id=1234567890
```

Or with a specific model:

```bash
node /path/to/Phoebe/rpc-proxy.js --url ws://your-phoebe-host:8090/v1/remote/ws?channel_id=1234567890 --pi-args "--model anthropic/claude-sonnet-4-20250514"
```

## Testing Locally

### Extension mode

```bash
# 1. Start Pi with the extension
pi -e /path/to/phoebe-connect.ts

# 2. In Pi's TUI, connect:
/connect-phoebe 1495058700107518191

# 3. Verify the connection from another terminal:
curl http://localhost:8090/v1/remote/status | python3 -m json.tool

# 4. Send a prompt via API:
curl -X POST http://localhost:8090/v1/remote/command \
  -H "Content-Type: application/json" \
  -d '{"channel_id": 1495058700107518191, "command": {"type": "prompt", "message": "hello"}}'

# 5. Check Discord for the response:
curl "http://localhost:4001/discord/read?channel_id=1495058700107518191&limit=5" | python3 -m json.tool
```

### RPC proxy mode

```bash
# 1. Start rpc-proxy connecting to local Phoebe
node rpc-proxy.js --url "ws://localhost:8090/v1/remote/ws?channel_id=YOUR_CHANNEL_ID" --pi-args "--print" &

# 2. Verify the connection
curl http://localhost:8090/v1/remote/status | python3 -m json.tool
curl http://localhost:8090/v1/remote/channel/YOUR_CHANNEL_ID | python3 -m json.tool

# 3. Send a prompt
curl -X POST http://localhost:8090/v1/remote/command \
  -H "Content-Type: application/json" \
  -d '{"channel_id": YOUR_CHANNEL_ID, "command": {"type": "prompt", "message": "hello"}}'

# 4. Check Discord for the response
curl "http://localhost:4001/discord/read?channel_id=YOUR_CHANNEL_ID&limit=5" | python3 -m json.tool

# 5. Cleanup
kill %1
```

Find your Discord channel ID in `.env`:
```bash
grep DISCORD_WORKER_CHANNELS .env
# Example: 1495058700107518191
```

## Discord Slash Commands

| Command | Description |
|---------|-------------|
| `/remote attach [model]` | Show connection URL for a new Pi session |
| `/remote detach` | Detach Pi from this channel |
| `/remote status` | Show Pi session state and stats |
| `/remote steer "…"` | Steer the running agent |
| `/remote follow "…"` | Queue follow-up for after current task |
| `/remote abort` | Abort current operation |
| `/remote bash "cmd"` | Run a shell command in Pi's context |
| `/remote model [preset]` | Switch Pi's model |
| `/remote compact` | Manually compact conversation |
| `/remote new-session` | Start a fresh Pi session |
| `/remote export` | Export session to HTML |
| `/remote help` | Full remote control help |

## Event Rendering

Pi's RPC events are rendered to Discord as follows:

| Pi Event | Discord Render |
|----------|---------------|
| `agent_start` | `-# 🚀 Agent started` |
| `message_update` (`text_delta`) | Live text streaming |
| `message_update` (`thinking_delta`) | `-# 🧠 thinking…` |
| `tool_execution_start` | `-# ⏳ \`toolName\` (args…)` |
| `tool_execution_end` | `-# ✅/❌ \`toolName\` (X lines)` |
| `turn_end` | Final assistant text |
| `agent_end` | `-# ✈️ Agent idle` |
| `compaction_start/end` | `-# 📦 Compacting… / ✅ Compacted` |
| `extension_ui_request` (select/confirm) | Discord buttons with `/remote ui-*` response |
| `queue_update` | `-# 📨 {n} steering / {m} follow-up queued` |

## Extension UI Bridging

When Pi's extensions request user interaction (select, confirm, input, editor), the bridge renders them as Discord messages with instructions:

```
🔘 Allow dangerous command?
  • Allow
  • Block
_Use /remote ui-select <req_id> <option> to respond_
```

Respond with:

```
/remote ui-select abc123 Allow
/remote ui-confirm def456 yes
/remote ui-input ghi789 my-value
/remote ui-editor jkl012 edited-text-here
```

## Multiple Pi Sessions

Phoebe multiplexes multiple Pi instances on a single port. Each Pi connection is bound to a Discord channel:

```
Channel #coding    ← Pi session 1 (claude-sonnet)
Channel #research  ← Pi session 2 (gpt-4o)
Channel #debug     ← Pi session 3 (gemini-pro)
```

Each channel operates independently. Messages in #coding go to Pi session 1, messages in #research go to Pi session 2, etc.

## Session Persistence

Pi saves session state to `session.jsonl` in its working directory. When you reconnect to the same project directory, Pi resumes from where it left off.

## Troubleshooting

**Extension not loading**
- Check extension location: `~/.pi/agent/extensions/phoebe-connect.ts`
- Verify TypeScript compiles: `pi -e /path/to/phoebe-connect.ts`
- Check Pi's stderr for extension errors

**Connection refused**
- Check that Phoebe is running and port 8090 is exposed
- Verify firewall rules allow incoming connections on 8090
- Test: `curl http://localhost:8090/v1/remote/status`

**Pi exits immediately (RPC proxy mode)**
- Check that `pi` is in your PATH
- Verify the WebSocket URL is correct
- Check Pi's stderr for error messages

**Messages not reaching Pi**
- Run `/remote status` to verify the connection is active
- In Pi, run `/phoebe-status` to check extension connection
- Check that you're in the correct channel (the one specified in `channel_id`)

**Events not rendering**
- Check Phoebe logs for bridge errors: `docker compose logs phoebe-api | grep remote`
- Verify the Discord bot has permissions to send messages in the channel

**Discord send failed**
- Check that `phoebe-discord-pycord` is running: `docker compose ps`
- Verify DNS resolution: `docker compose exec phoebe-api python -c "import socket; print(socket.getaddrinfo('phoebe-discord-pycord', 4000))"`

**Extension disconnects repeatedly**
- Check Phoebe's WebSocket logs: `docker compose logs phoebe-api | grep "WebSocket /v1/remote"`
- Verify the channel_id is valid and the bot has access
- Extension auto-reconnects after 5s (check Pi stderr for reconnect messages)
