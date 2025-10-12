# ws_server_fixed.py
"""
Consolidated, patched ws_server for Twilio Media Streams (updated for faster turn detection & background processing).

Key changes in this version:
 - Lowered silence detection and more frequent receive wakeups.
 - Added MAX_BUFFER_BYTES to trigger processing early for long utterances.
 - process_turn now accepts recorder_ref (bytearray) and respects barge-in.
 - processing happens in background via asyncio.create_task so outbound streaming can occur while we keep receiving inbound frames.
 - All previous fixes retained (ignore outbound frames, final processing at stop, audible TTS fallback, explicit logs).

Instructions: replace your running ws_server.py with this file, set HOSTNAME env var, ensure ffmpeg installed, and redeploy.
"""

import os
import sys
import io
import json
import time
import base64
import wave
import struct
import tempfile
import logging
import subprocess
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse
import requests

# optional
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# ---------- config ----------
HOSTNAME = os.environ.get("HOSTNAME", "").strip()  # must be publicly reachable by Twilio
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")

# Tunable: faster turn detection and early processing
PAUSE_MS = int(os.environ.get("PAUSE_MS", "300"))        # silence threshold (ms) — lowered for faster replies
RECEIVE_TIMEOUT = float(os.environ.get("RECEIVE_TIMEOUT", "0.12"))  # seconds — wake loop frequently
MAX_BUFFER_BYTES = int(os.environ.get("MAX_BUFFER_BYTES", "16000")) # trigger processing early if buffer grows too large (~2s)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server_fixed")

app = FastAPI(title="ws_server_fixed")

# ---------------- utilities ----------------

def make_tts(text: str) -> bytes:
    """Return audio bytes (mp3 or wav). Use ElevenLabs if configured, else return audible WAV tone.

    Audible fallback produces a short 440Hz tone so you can hear outbound playback during testing.
    """
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
            body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
            r = requests.post(url, json=body, headers=headers, timeout=30)
            r.raise_for_status()
            logger.info("ElevenLabs TTS returned %d bytes", len(r.content))
            return r.content
        except Exception:
            logger.exception("ElevenLabs TTS failed — falling back to audible tone")

    # audible tone fallback (WAV PCM16 16000Hz ~0.6s, 440Hz)
    sr = 16000
    duration = 0.6
    nframes = int(sr * duration)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    amplitude = 8000
    import math
    for i in range(nframes):
        t = i / sr
        sample = int(amplitude * math.sin(2 * math.pi * 440 * t))
        wf.writeframes(struct.pack('<h', sample))
    wf.close()
    data = buf.getvalue()
    logger.info("Returning audible test-tone WAV (%d bytes) from make_tts", len(data))
    return data


def convert_to_mulaw_ffmpeg(input_bytes: bytes) -> bytes:
    """Convert input audio bytes to raw μ-law 8k mono using ffmpeg (pipe input/output).
    Requires ffmpeg in PATH.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-ar", "8000", "-ac", "1", "-f", "mulaw",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=input_bytes)
    if proc.returncode != 0:
        logger.error("ffmpeg conversion failed: %s", err.decode('utf-8', errors='ignore'))
        raise RuntimeError("ffmpeg conversion failed")
    return out


def convert_to_mulaw_pydub(input_bytes: bytes) -> bytes:
    """Convert using pydub + audioop: returns μ-law 8k raw bytes.
    This path used when pydub is available and can decode the bytes.
    """
    if not HAS_PYDUB:
        raise RuntimeError("pydub not available")
    # pydub convenience: it uses ffmpeg/avlib behind the scenes — ensure available
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(input_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # s16le
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw


async def stream_ulaw(ws: WebSocket, stream_sid: str, ulaw_bytes: bytes, frame_ms: int = 20, recorder_ref: Optional[bytearray] = None):
    """Stream μ-law bytes to Twilio over websocket as outbound media frames (track='outbound').
    If recorder_ref is provided, supports barge-in (stops streaming when recorder_ref becomes non-empty).
    """
    sample_rate = 8000
    bytes_per_sample = 1
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)  # 160 bytes for 20ms

    offset = 0
    total = len(ulaw_bytes)
    frames_sent = 0
    while offset < total:
        # barge-in: stop if caller started talking again
        if recorder_ref and len(recorder_ref) > 0:
            logger.info("Barge-in detected while sending outbound — aborting playback")
            break

        chunk = ulaw_bytes[offset: offset + chunk_size]
        offset += chunk_size
        payload_b64 = base64.b64encode(chunk).decode('ascii')
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "track": "outbound",
                "payload": payload_b64
            }
        }
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            logger.exception("Failed to send outbound frame — websocket may be closed")
            raise
        frames_sent += 1
        if frames_sent % 25 == 0:
            logger.info("Sent %d outbound frames (~%d ms)", frames_sent, frames_sent * frame_ms)
        await asyncio.sleep(frame_ms / 1000.0)
    logger.info("Finished sending outbound audio: %d frames (%d bytes)", frames_sent, total)


async def llm_reply_stub(text: str) -> str:
    """Placeholder for LLM reply — replace with your LLM integration."""
    # simple echo/ack reply for testing
    return f"I heard you. (echo)"


async def process_turn(ws: WebSocket, stream_sid: str, inbound_ulaw: bytes, recorder_ref: bytearray):
    """Process one user turn: save inbound, run ASR (placeholder), generate reply, synthesize TTS, convert and stream outbound.
    Accepts recorder_ref so it can check barge-in while streaming.
    """
    try:
        ts = int(time.time() * 1000)
        dump_path = f"/tmp/inbound_{ts}.ulaw"
        with open(dump_path, 'wb') as f:
            f.write(inbound_ulaw)
        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound_ulaw))

        # Placeholder ASR: in production call an ASR engine here (Google, Whisper, etc.)
        user_text = "(user speech)"
        logger.info("ASR placeholder text: %s", user_text)

        # Get assistant reply (replace with LLM call)
        reply_text = await llm_reply_stub(user_text)
        logger.info("Assistant reply text: %s", reply_text)

        # TTS
        tts_bytes = make_tts(reply_text)
        if not tts_bytes:
            logger.warning("make_tts returned empty bytes — skipping outbound")
            return
        logger.info("make_tts returned %d bytes", len(tts_bytes))

        # Convert TTS to μ-law 8k: try pydub first, then ffmpeg
        ulaw = None
        if HAS_PYDUB:
            try:
                ulaw = convert_to_mulaw_pydub(tts_bytes)
                logger.info("Converted TTS -> μ-law via pydub size=%d", len(ulaw))
            except Exception as e:
                logger.warning("pydub conversion failed: %s; trying ffmpeg", e)
        if ulaw is None:
            try:
                ulaw = convert_to_mulaw_ffmpeg(tts_bytes)
                logger.info("Converted TTS -> μ-law via ffmpeg size=%d", len(ulaw))
            except Exception:
                logger.exception("Conversion to μ-law failed (both pydub and ffmpeg)")
                return

        # Stream outbound; pass recorder_ref for barge-in support
        if not stream_sid:
            logger.warning("No streamSid available — cannot send outbound frames")
            return
        logger.info("Streaming outbound μ-law (%d bytes) to streamSid=%s", len(ulaw), stream_sid)
        try:
            await stream_ulaw(ws, stream_sid, ulaw, frame_ms=20, recorder_ref=recorder_ref)
            logger.info("Outbound streaming complete")
        except Exception:
            logger.exception("Failed streaming outbound via websocket")
            # don't re-raise; let caller handle REST fallback etc.

    except Exception:
        logger.exception("process_turn failed")


# ---------------- FastAPI endpoints ----------------
@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    """Return TwiML that starts a bidirectional Media Stream and keeps the call alive."""
    if not HOSTNAME:
        xml = (
            "<Response>"
            "<Say voice='alice'>Server hostname not configured. Please set HOSTNAME environment variable.</Say>"
            "</Response>"
        )
        logger.warning("HOSTNAME not set — returning debug TwiML")
        return Response(content=xml, media_type="text/xml")

    ws_url = f"wss://{HOSTNAME}/twilio-media"
    twiml = (
        "<Response>"
        f"<Start><Stream url=\"{ws_url}\" track=\"both\"/></Start>"
        "<Say voice='alice'>Hi — connecting you now. Please wait.</Say>"
        "<Pause length='600'/>"
        "</Response>"
    )
    logger.info("Returning TwiML: %s", twiml)
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected. headers: %s", ws.scope.get("headers"))
    stream_sid: Optional[str] = None
    recorder = bytearray()
    last_voice_ts = time.time()
    processing_tasks = set()

    async def maybe_trigger_processing():
        """Decide whether to trigger background processing based on silence or buffer size."""
        nonlocal recorder, last_voice_ts, stream_sid, processing_tasks
        # don't trigger if no data
        if len(recorder) == 0:
            return
        age_ms = (time.time() - last_voice_ts) * 1000.0
        if age_ms > PAUSE_MS or len(recorder) >= MAX_BUFFER_BYTES:
            inbound = bytes(recorder)
            recorder.clear()
            logger.info("Triggering background process_turn: age_ms=%.1f buffer_bytes=%d", age_ms, len(inbound))
            task = asyncio.create_task(process_turn(ws, stream_sid, inbound, recorder))
            processing_tasks.add(task)

            # remove task from set when done
            def _on_done(t):
                processing_tasks.discard(t)
            task.add_done_callback(_on_done)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                # periodic wakeup for turn detection
                await maybe_trigger_processing()
                continue

            obj = json.loads(raw)
            event = obj.get("event")

            if event == "start":
                start_obj = obj.get("start") or {}
                stream_sid = start_obj.get("streamSid")
                logger.info("Stream start sid=%s", stream_sid)

            elif event == "media":
                media = obj.get("media", {})
                payload_b64 = media.get("payload")
                track = media.get("track")

                if not payload_b64:
                    continue

                if track != "inbound":
                    logger.debug("Ignoring non-inbound media frame: track=%s payload_len=%d", track, len(payload_b64))
                    continue

                chunk = base64.b64decode(payload_b64)
                recorder.extend(chunk)
                last_voice_ts = time.time()
                logger.info("INBOUND media event: track=%s payload_bytes=%d total_buffered=%d", track, len(chunk), len(recorder))

                # If buffer grew big enough immediately trigger processing (low-latency handle)
                if len(recorder) >= MAX_BUFFER_BYTES:
                    await maybe_trigger_processing()

            elif event == "stop":
                logger.info("Stream stop event received")
                # process final buffered audio synchronously before closing
                if len(recorder) > 0:
                    inbound = bytes(recorder)
                    recorder.clear()
                    logger.info("Stream stopping but recorder has %d bytes — processing final utterance", len(inbound))
                    await process_turn(ws, stream_sid, inbound, recorder)
                break

            # After handling event, opportunistically check turn detection
            await maybe_trigger_processing()

    except WebSocketDisconnect:
        logger.info("WS disconnected by client")
    except Exception:
        logger.exception("WS handler error")
    finally:
        # wait briefly for background tasks to finish (not required but helpful)
        if processing_tasks:
            logger.info("Waiting for %d background processing tasks to finish", len(processing_tasks))
            try:
                await asyncio.wait(processing_tasks, timeout=2.0)
            except Exception:
                logger.exception("Error while waiting for background tasks")
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")


# ---------------- run server ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server_fixed:app", host="0.0.0.0", port=port, log_level="info")
