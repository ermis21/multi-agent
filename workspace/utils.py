import os
import re

MAX_MSG_LEN = int(os.environ.get("DISCORD_MAX_MESSAGE_LENGTH", "1900"))

# If set, only these user IDs may interact with the bots.
# Parsed once at import time so every message check is an O(1) set lookup.
_raw = os.environ.get("DISCORD_ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: frozenset[int] = frozenset(
    int(uid) for uid in _raw.split(",") if uid.strip()
)


def is_allowed(user_id: int) -> bool:
    """Return True if this user is permitted to use the bot.

    When DISCORD_ALLOWED_USER_IDS is empty, everyone is allowed.
    When it is set, only listed IDs pass.
    """
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def split_message(text: str) -> list[str]:
    """Split text into Discord-safe chunks, breaking at newlines where possible."""
    chunks = []
    while len(text) > MAX_MSG_LEN:
        split_at = text.rfind("\n", 0, MAX_MSG_LEN)
        if split_at == -1:
            split_at = MAX_MSG_LEN
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def clean_text_for_tts(text: str) -> str:
    """Remove Markdown formatting for cleaner TTS playback."""
    # Remove bold, italics, and strikethrough
    text = re.sub(r"(\*\*|__|[*_]|~~)", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"", text)
    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Remove URLs/Links ([text](url) -> text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"", text)
    return text.strip()
