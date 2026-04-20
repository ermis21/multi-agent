### tts_speak
Generate a spoken audio response and send it as a WAV file to a Discord channel.
Use this when the user asks you to "speak", "say out loud", or "respond with voice".
`channel_id` is required. Keep `text` concise — long responses may take a few seconds to synthesise.
<|tool_call|>call: tts_speak, {"channel_id": 123456789, "text": "Hello! Here is your answer."}<|tool_call|>
