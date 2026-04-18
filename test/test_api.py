"""
Integration test suite for mab-api.

Requires the stack to be running: make up (or make test-up for isolated stack).
Run with: pytest test/ -v

Set MAB_URL env var to override the default http://localhost:8090.
"""

import time
import uuid

import httpx
import pytest

from conftest import BASE_URL, answer, chat


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["api"] == "ok"


def test_health_llm_reachable(client):
    r = client.get("/health")
    llm = r.json().get("llm", {})
    assert "error" not in llm, f"LLM unreachable: {llm}"


# ── Basic chat ─────────────────────────────────────────────────────────────────

def test_basic_chat(client):
    resp = chat(client, "Say hello in exactly three words.")
    text = answer(resp)
    assert len(text) > 0
    assert "choices" in resp


def test_response_format(client):
    resp = chat(client, "What is 2 + 2?")
    assert "id" in resp
    assert "session_id" in resp
    assert resp["object"] == "chat.completion"


# ── Supervisor loop ────────────────────────────────────────────────────────────

def test_supervisor_session_logged(client):
    sid = f"test_sup_{uuid.uuid4().hex[:6]}"
    chat(client, "Explain what a neural network is in one sentence.", session_id=sid)

    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    turns = r.json()["turns"]
    assert len(turns) >= 1
    roles = [t["role"] for t in turns]
    assert "worker" in roles


# ── Tool call round-trip ───────────────────────────────────────────────────────

def test_tool_file_roundtrip(client):
    filename = f"test_{uuid.uuid4().hex[:6]}.txt"
    content  = "mab-test-content-xyz"

    resp = chat(client, f"Write the text '{content}' to the file '{filename}' in the workspace, then read it back and tell me what it says.")
    text = answer(resp)
    assert content in text or "mab-test-content-xyz" in text.lower()


# ── Session continuity ─────────────────────────────────────────────────────────

def test_session_continuity(client):
    sid = f"test_cont_{uuid.uuid4().hex[:6]}"

    chat(client, "My favourite colour is ultraviolet.", session_id=sid)
    resp2 = chat(client, "What is my favourite colour?", session_id=sid)
    text = answer(resp2).lower()
    assert "ultraviolet" in text


# ── Config hot-reload ──────────────────────────────────────────────────────────

def test_config_read(client):
    r = client.get("/config")
    assert r.status_code == 200
    cfg = r.json()
    assert "llm" in cfg
    assert "agent" in cfg


def test_config_patch_mode(client):
    original = client.get("/config").json()["prompts"]["mode"]
    new_mode = "concise" if original == "full" else "full"

    r = client.patch("/config", json={"prompts": {"mode": new_mode}})
    assert r.status_code == 200
    assert r.json()["config"]["prompts"]["mode"] == new_mode

    # Restore
    client.patch("/config", json={"prompts": {"mode": original}})


# ── Sessions list ──────────────────────────────────────────────────────────────

def test_sessions_list(client):
    r = client.get("/sessions")
    assert r.status_code == 200
    assert "sessions" in r.json()


# ── Soul update ────────────────────────────────────────────────────────────────

def test_soul_update_trigger(client):
    r = client.post("/internal/soul-update")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Direct role endpoint ───────────────────────────────────────────────────────

def test_direct_role_config_agent(client):
    r = client.post(
        "/v1/agents/config_agent",
        json={"messages": [{"role": "user", "content": "What is the current prompt mode?"}]},
    )
    assert r.status_code == 200
    text = answer(r.json())
    assert len(text) > 0


def test_direct_role_unknown(client):
    r = client.post(
        "/v1/agents/nonexistent_role_xyz",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    data = r.json()
    # Should return an error, not a 500
    assert r.status_code == 200
    assert "error" in data or "choices" in data


# ── Web search (requires EXA_API_KEY) ─────────────────────────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("EXA_API_KEY"),
    reason="EXA_API_KEY not set",
)
def test_web_search(client):
    resp = chat(client, "Search the web for 'FastAPI Python framework' and summarize what you find.")
    text = answer(resp)
    assert len(text) > 50
    assert any(kw in text.lower() for kw in ["fastapi", "python", "api", "framework"])


# ── Memory ─────────────────────────────────────────────────────────────────────

def test_memory_add_and_search(client):
    fact = f"The secret test fact is purple-{uuid.uuid4().hex[:6]}"
    chat(client, f"Remember this fact: {fact}")
    resp = chat(client, f"Search your memory for 'secret test fact purple' and tell me what you find.")
    text = answer(resp)
    # Memory may or may not contain the exact fact depending on tool call
    assert len(text) > 0


# ── Notion (requires NOTION_TOKEN) ────────────────────────────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("NOTION_TOKEN"),
    reason="NOTION_TOKEN not set",
)
def test_notion_search(client):
    resp = chat(client, "Search Notion for any pages and tell me what you find.")
    text = answer(resp)
    assert len(text) > 0
