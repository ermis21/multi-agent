#!/usr/bin/env node
/**
 * rpc-proxy — Bridge between Pi.dev's stdin/stdout JSONL RPC and Phoebe WebSocket.
 *
 * Usage:
 *   rpc-proxy --url ws://phoebe:8090/v1/remote/ws?channel_id=123456
 *
 * This script:
 * 1. Spawns `pi --mode rpc` as a subprocess
 * 2. Connects to Phoebe via WebSocket (built-in Node 24+ WebSocket)
 * 3. Bridges Pi's stdout → WebSocket and WebSocket → Pi's stdin
 *
 * Pi's RPC protocol uses strict JSONL (one JSON object per line, LF delimiter).
 * This proxy preserves that framing over the WebSocket connection.
 *
 * Requires: Node.js 24+ (for global WebSocket), `pi` in PATH
 */

import { spawn } from "node:child_process";
// Node 24+ has global WebSocket (built-in, no npm deps needed)

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let wsUrl = null;
let piArgs = [];
let piCwd = process.cwd();

for (let i = 0; i < args.length; i++) {
    if (args[i] === "--url" && args[i + 1]) {
        wsUrl = args[i + 1];
        i++;
    } else if (args[i] === "--pi-args" && args[i + 1]) {
        piArgs = args[i + 1].split(" ");
        i++;
    } else if (args[i] === "--cwd" && args[i + 1]) {
        piCwd = args[i + 1];
        i++;
    } else if (args[i] === "--help") {
        console.log(`
rpc-proxy — Bridge Pi.dev RPC to Phoebe WebSocket

Usage:
  rpc-proxy --url ws://phoebe:8090/v1/remote/ws?channel_id=123456
  rpc-proxy --url ws://phoebe:8090/v1/remote/ws?channel_id=123456 --pi-args "--model anthropic/claude-sonnet-4-20250514"
  rpc-proxy --url ws://phoebe:8090/v1/remote/ws?channel_id=123456 --cwd /path/to/project

Options:
  --url <ws-url>       WebSocket URL to connect to Phoebe (required)
  --pi-args <args>     Additional args to pass to pi (space-separated)
  --cwd <path>         Working directory for pi
  --help               Show this help
        `.trim());
        process.exit(0);
    }
}

if (!wsUrl) {
    console.error("Error: --url is required");
    console.error("Usage: rpc-proxy --url ws://phoebe:8090/v1/remote/ws?channel_id=123456");
    process.exit(1);
}

// ── Strict JSONL reader ──────────────────────────────────────────────────────
// Splits on LF only, strips trailing CR. Does NOT split on Unicode separators.

function attachJsonlReader(stream, onLine) {
    let buffer = "";
    stream.setEncoding("utf8");
    stream.on("data", (chunk) => {
        buffer += chunk;
        while (true) {
            const idx = buffer.indexOf("\n");
            if (idx === -1) break;
            let line = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            if (line.endsWith("\r")) line = line.slice(0, -1);
            if (line.length > 0) {
                onLine(line);
            }
        }
    });
    stream.on("end", () => {
        if (buffer.length > 0) {
            let line = buffer;
            if (line.endsWith("\r")) line = line.slice(0, -1);
            if (line.length > 0) onLine(line);
        }
    });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
    console.error(`[rpc-proxy] Connecting to Phoebe: ${wsUrl}`);

    // Spawn Pi in RPC mode
    const piSpawnArgs = ["--mode", "rpc", ...piArgs];
    const pi = spawn("pi", piSpawnArgs, {
        cwd: piCwd,
        stdio: ["pipe", "pipe", "inherit"], // stdin, stdout, stderr
    });

    pi.on("error", (err) => {
        console.error(`[rpc-proxy] Failed to spawn pi: ${err.message}`);
        process.exit(1);
    });

    let piExited = false;
    pi.on("close", (code) => {
        if (!piExited) {
            piExited = true;
            console.error(`[rpc-proxy] Pi exited with code ${code}`);
            ws.close(1000, "Pi process exited");
        }
    });

    // Connect to Phoebe WebSocket
    const ws = new WebSocket(wsUrl);

    ws.onerror = (err) => {
        console.error(`[rpc-proxy] WebSocket error: ${err.message || err}`);
    };

    ws.onclose = (event) => {
        console.error(`[rpc-proxy] WebSocket closed (${event.code}): ${event.reason}`);
        if (!piExited) {
            pi.kill("SIGTERM");
        }
    };

    ws.onopen = () => {
        console.error("[rpc-proxy] WebSocket connected to Phoebe");
    };

    // Bridge: Pi stdout → WebSocket
    attachJsonlReader(pi.stdout, (line) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(line);
        }
    });

    // Bridge: WebSocket → Pi stdin
    ws.onmessage = (event) => {
        const text = typeof event.data === "string" ? event.data : event.data.toString();
        // Ensure each message ends with LF for JSONL framing
        const line = text.endsWith("\n") ? text : text + "\n";
        try {
            pi.stdin.write(line);
        } catch (err) {
            console.error(`[rpc-proxy] Failed to write to pi stdin: ${err.message}`);
        }
    };

    // Handle graceful shutdown
    const shutdown = (signal) => {
        console.error(`[rpc-proxy] ${signal} received, shutting down`);
        if (!piExited) {
            pi.kill("SIGTERM");
        }
        ws.close(1000, "rpc-proxy shutting down");
        process.exit(0);
    };

    process.on("SIGINT", () => shutdown("SIGINT"));
    process.on("SIGTERM", () => shutdown("SIGTERM"));

    // Keep process alive
    await new Promise(() => {});
}

main().catch((err) => {
    console.error(`[rpc-proxy] Fatal error: ${err.message}`);
    process.exit(1);
});
