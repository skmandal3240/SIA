# SIA Shell (P2)

The Shell is SIA's embodiment layer: screen perception, voice I/O, and
on-screen action execution. P2 wires these into a single `see → reason →
point/act → speak` loop with a shared dispatcher.

## What works now

- **Screen capture** (`capture.py`): real capture via `mss` on
  Linux/Windows/macOS; stub fallback for CI/headless.
- **Audio I/O** (`audio.py`): record and playback via `ffmpeg` on Linux.
  Does not include STT/TTS models; those are pluggable.
- **Reasoner bridge** (`reason_ollama.py`): calls the local `sia-p0` model
  through Ollama and parses any action tags or tool-call JSON.
- **Dispatcher** (`dispatcher.py`): routes tool calls and on-screen action
  tags (`[POINT:x,y:label]`, `[CLICK:...]`, `[TYPE:...]`, `[SCROLL:...]`).
- **Loop** (`loop.py`): one-turn shell orchestrator.

## Quick smoke test

```bash
python3 sia-lab/shell/smoke_p2.py
```

This captures the screen (or uses a stub), feeds a hard-coded transcript to
`sia-p0`, prints any returned action tags/tool calls, and streams a stub TTS
heartbeat.

## Run with your own voice (Linux only)

```bash
# 1. Install mss for real capture and ensure ffmpeg is present.
pip install mss

# 2. Record 5 seconds of audio.
python3 -c "from audio import record_audio; open('/tmp/sia_input.wav','wb').write(record_audio(5))"

# 3. Feed the recorded audio through the shell (replace stt stub when ready).
```

## Files

| File | Purpose |
|------|---------|
| `capture.py` | Screen capture: stub + mss |
| `audio.py` | Linux audio record/playback via ffmpeg |
| `stt.py` | Streaming STT stub |
| `tts.py` | Streaming TTS stub |
| `tag_parser.py` | Parse `[POINT/CLICK/TYPE/SCROLL:...]` tags |
| `dispatcher.py` | Route tools + actions |
| `loop.py` | One-turn shell loop |
| `reason_ollama.py` | Local Ollama reasoner bridge |
| `smoke_p2.py` | Runnable P2 demo |

## Next upgrades (P2.5 → P3)

- Replace STT stub with a local Whisper / Sarvam edge model.
- Replace TTS stub with Piper / Coqui edge model.
- Add macOS screen-capture-permission handling.
- Wire memory stores (`sia-lab/memory/`) into the loop context.
