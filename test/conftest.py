"""Shared fixtures for the mab integration test suite."""

import os
import pytest
import httpx

BASE_URL = os.environ.get("MAB_URL", "http://localhost:8090")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=180) as c:
        yield c


def chat(client: httpx.Client, message: str, session_id: str | None = None) -> dict:
    """Helper: send a chat message and return the parsed response."""
    body = {"messages": [{"role": "user", "content": message}]}
    if session_id:
        body["session_id"] = session_id
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()


def answer(response: dict) -> str:
    """Extract the assistant content from a chat response."""
    return response["choices"][0]["message"]["content"]
