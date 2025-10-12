# ws_server_fixed.py
# FORCED REST PLAY version — every assistant reply will be delivered via Twilio REST <Play>
# This guarantees playback into the call (if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are set)
# Behavior changes vs earlier versions:
# - After producing TTS bytes, the server will always upload/serve the WAV and instruct Twilio to <Play> it
#   into the active call via REST API. This bypasses the websocket outbound streaming path.
# - Useful when Media Stream websocket is unreliable or closed prematurely.

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

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException
import requests

# optional pydub
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# ---------- config ----------
HOSTNAME = os.environ.get("HOSTNAME", "").strip()
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

PAUSE_MS = int(os.environ.get("PAUSE_MS", "300"))
RECEIVE_TIMEOUT = float(os.environ.get("RECEIVE_TIMEOUT", "0.12"))
MAX_BUFFER_BYTES = int(os.environ.get("MAX_BUFFER_BYTES", "16000"))

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server_fixed_forced_rest")

app = FastAPI(title="ws_server_fixed_forced_rest")

# ---------------- utilities ----------------

def make_tts(text: str) -> bytes:
    """Try ElevenLabs TTS; on failure return audible WAV tone. Logs response body for diagnostics."""
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY.strip()}
            body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
            r = requests.post(url, json=body, headers=headers, timeout=30)
            if 200 <= r.status_code < 300:
                logger.info("ElevenLabs TTS OK (%d bytes)", len(r.content))
                return r.content
            else:
                logger.error("ElevenLabs TTS failed status=%d body=%s", r.status_code, r.text)
        except Exception:
            logger.exception("ElevenLabs TTS exception — falling back to audible tone")

    # audible tone fallback WAV (PCM16 16000Hz ~0.6s, 440Hz)
    sr = 16000
    duration = 0.6
    nframes = int(sr * duration)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    import math
    amplitude = 12000
    for i in range(nframes):
        t = i / sr
        sample = int(amplitude * math.sin(2 * math.pi * 440 * t))
        wf.writeframes(struct.pack('<h', sample))
    wf.close()
    data = buf.getvalue()
    logger.info("Returning audible test-tone WAV (%d bytes) from make_tts", len(data))
    return data


def convert_to_mulaw_ffmpeg(input_bytes: bytes) -> bytes:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ar", "8000", "-ac", "1", "-f", "mulaw", "pipe:1"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=input_bytes)
    if proc.returncode != 0:
        logger.error("ffmpeg conversion failed: %s", err.decode('utf-8', errors='ignore'))
        raise RuntimeError("ffmpeg conversion failed")
    return out


def convert_to_mulaw_pydub(input_bytes: bytes) -> bytes:
    if not HAS_PYDUB:
        raise RuntimeError("pydub not available")
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(input_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw


def twilio_play_via_rest(call_sid: str, tts_bytes: bytes) -> bool:
    """Save tts_bytes to /tmp and POST TwiML <Play> to call via Twilio REST. Requires TWILIO_* env vars."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        logger.warning("TWILIO_ACCOUNT_SID/AUTH_TOKEN not set; cannot REST Play")
        return False
    if not HOSTNAME:
        logger.warning("HOSTNAME not set; cannot expose TTS file for Twilio")
        return False
    try:
        ts = int(time.time() * 1000)
        fname = f"tts_{call_sid}_{ts}.wav"
        path = f"/tmp/{fname}"
        with open(path, "wb") as f:
            f.write(tts_bytes)
        url = f"https://{HOSTNAME}/tts/{fname}"
        twiml = f"<Response><Play>{url}</Play></Response>"
        api_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json"
        logger.info("Attempting Twilio REST Play: POST %s (play %s)", api_url, url)
        r = requests.post(api_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), data={"Twiml": twiml}, timeout=10)
        if r.status_code // 100 == 2:
            logger.info("Twilio REST Play accepted (status=%d)", r.status_code)
            return True
        else:
            logger.warning("Twilio REST Play failed: status=%d body=%s", r.status_code, r.text)
            return False
    except Exception:
        logger.exception("Twilio REST Play fallback failed")
        return False

# Serve TTS files for Twilio to fetch
from fastapi.responses import FileResponse

@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    logger.info("Serving TTS file %s", path)
    return FileResponse(path, media_type="audio/wav")

# health
@app.get("/health")
async def health():
    return {"ok": True, "hostname": HOSTNAME}

# ---------------- processing (FORCE REST PLAY) ----------------

async def llm_reply_stub(text: str) -> str:
    return "I heard you. (echo)"

async def process_turn_force_rest(ws: WebSocket, stream_sid: str, call_sid: Optional[str], inbound_ulaw: bytes, recorder_ref: bytearray):
    """Process one user turn and force Twilio REST Play for playback into the call."""
    try:
        ts = int(time.time() * 1000)
        dump_path = f"/tmp/inbound_{ts}.ulaw"
        with open(dump_path, "wb") as f:
            f.write(inbound_ulaw)
        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound_ulaw))

        user_text = "(user speech)"
        logger.info("ASR placeholder text: %s", user_text)

        reply_text = await llm_reply_stub(user_text)
        logger.info("Assistant reply text: %s", reply_text)

        # Synthesize TTS (mp3/wav bytes)
        tts_bytes = make_tts(reply_text)
        if not tts_bytes:
            logger.warning("make_tts returned empty bytes; skipping REST Play")
            return
        logger.info("make_tts returned %d bytes", len(tts_bytes))

        # Force REST Play into call
        if not call_sid:
            logger.warning("No call_sid available; cannot force REST Play")
            return
        ok = twilio_play_via_rest(call_sid, tts_bytes)
        if ok:
            logger.info("Twilio REST Play forced successfully for callSid=%s", call_sid)
        else:
            logger.warning("Twilio REST Play forced but failed for callSid=%s", call_sid)

    except Exception:
        logger.exception("process_turn_force_rest failed")

# ---------------- TwiML / websocket handler ----------------
@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    if not HOSTNAME:
        xml = "<Response><Say voice='alice'>Server hostname not configured. Please set HOSTNAME environment variable.</Say></Response>"
        logger.warning("HOSTNAME missing; returning debug TwiML")
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
    logger.info("WS connected. headers=%s", ws.scope.get("headers"))
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    recorder = bytearray()
    last_voice_ts = time.time()
    processing_tasks = set()

    async def maybe_trigger_processing():
        nonlocal recorder, last_voice_ts, stream_sid, call_sid, processing_tasks
        if len(recorder) == 0:
            return
        age_ms = (time.time() - last_voice_ts) * 1000.0
        if age_ms > PAUSE_MS or len(recorder) >= MAX_BUFFER_BYTES:
            inbound = bytes(recorder)
            recorder.clear()
            logger.info("Triggering background process_turn_force_rest: age_ms=%.1f buffer_bytes=%d", age_ms, len(inbound))
            task = asyncio.create_task(process_turn_force_rest(ws, stream_sid, call_sid, inbound, recorder))
            processing_tasks.add(task)
            def _on_done(t): processing_tasks.discard(t)
            task.add_done_callback(_on_done)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                await maybe_trigger_processing()
                continue

            obj = json.loads(raw)
            event = obj.get("event")

            if event == "start":
                start_obj = obj.get("start") or {}
                stream_sid = start_obj.get("streamSid")
                call_sid = start_obj.get("callSid") or start_obj.get("call_sid") or call_sid
                logger.info("Stream start sid=%s callSid=%s", stream_sid, call_sid)

            elif event == "media":
                media = obj.get("media", {}) or {}
                payload_b64 = media.get("payload")
                track = media.get("track")
                if not payload_b64:
                    continue
                if track != "inbound":
                    logger.debug("Ignoring non-inbound media frame: track=%s", track)
                    continue
                chunk = base64.b64decode(payload_b64)
                recorder.extend(chunk)
                last_voice_ts = time.time()
                logger.info("INBOUND media event: track=%s payload_bytes=%d total_buffered=%d", track, len(chunk), len(recorder))
                if len(recorder) >= MAX_BUFFER_BYTES:
                    await maybe_trigger_processing()

            elif event == "stop":
                logger.info("Stream stop event received")
                if len(recorder) > 0:
                    inbound = bytes(recorder)
                    recorder.clear()
                    logger.info("Stream stopping but recorder has %d bytes — processing final utterance", len(inbound))
                    await process_turn_force_rest(ws, stream_sid, call_sid, inbound, recorder)
                break

            await maybe_trigger_processing()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception:
        logger.exception("WS handler error")
    finally:
        if processing_tasks:
            logger.info("Waiting for %d background tasks", len(processing_tasks))
            try:
                await asyncio.wait(processing_tasks, timeout=2.0)
            except Exception:
                logger.exception("Error waiting for tasks")
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")

# run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server_fixed_forced_rest:app", host="0.0.0.0", port=port, log_level="info")
