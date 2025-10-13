"""
Updated ws_server.py
- Keeps TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN naming consistent
- Adds play_tts_with_wait (poll + SDK update) with fallback to REST <Play>
- Forces REST Play when SDK update unavailable/failed
- Provides /twiml_stream TwiML entrypoint and /twilio-media websocket handler
- Serves generated TTS files at /tts/{fname}

Drop-in intended replacement for your existing ws_server.py.
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

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
import requests

# optional pydub for conversion
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# Twilio SDK
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
except Exception:
    TwilioClient = None
    TwilioRestException = Exception

# ---------- configuration ----------
HOSTNAME = os.environ.get("HOSTNAME", "").strip()
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# speech tuning
PAUSE_MS = int(os.environ.get("PAUSE_MS", "700"))
RECEIVE_TIMEOUT = float(os.environ.get("RECEIVE_TIMEOUT", "0.12"))
MAX_BUFFER_BYTES = int(os.environ.get("MAX_BUFFER_BYTES", "16000"))

# logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server_updated")

app = FastAPI(title="ws_server_updated")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    logger.warning("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not set in environment")

# Twilio SDK client (optional)
if TwilioClient and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception:
        twilio_client = None
        logger.exception("Failed to instantiate Twilio Client")
else:
    twilio_client = None

# ----------------- utilities -----------------

def make_tts(text: str) -> bytes:
    """Return audio bytes (preferably wav or mp3). Uses ElevenLabs when configured.
    On failure returns a short audible WAV tone as fallback.
    """
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
            body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
            r = requests.post(url, json=body, headers=headers, timeout=30)
            if 200 <= r.status_code < 300:
                logger.info("ElevenLabs TTS OK (%d bytes)", len(r.content))
                return r.content
            else:
                logger.error("ElevenLabs TTS failed status=%d body=%s", r.status_code, r.text)
        except Exception:
            logger.exception("ElevenLabs TTS exception")

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


def mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes: bytes) -> bytes:
    """Convert mp3/wav (bytes) to headerless μ-law 8k mono (1 byte per sample).
    Uses pydub when available, otherwise falls back to ffmpeg.
    """
    if not HAS_PYDUB:
        return convert_to_mulaw_ffmpeg(tts_bytes)
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(tts_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw


# ------------ Twilio REST play fallback (save file + POST TwiML) -------------

def twilio_play_via_rest(call_sid: str, tts_bytes: bytes) -> bool:
    """Save tts_bytes to /tmp and POST TwiML <Play> to call via Twilio REST (requests.post).
    Requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and HOSTNAME to be configured.
    """
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


# --------------- Twilio SDK wrapper: poll and update (retry) ----------------

def play_tts_with_wait(call_sid: str, tts_url: str, wait_timeout: float = 5.0, poll_interval: float = 0.5) -> bool:
    """Poll call status via Twilio SDK until in-progress then attempt update(twiml=...).
    Returns True if SDK update succeeded; False otherwise.
    """
    if not twilio_client:
        logger.debug("Twilio SDK client not available; skipping play_tts_with_wait")
        return False

    deadline = time.time() + wait_timeout
    last_status = None
    try:
        while time.time() < deadline:
            try:
                call = twilio_client.calls(call_sid).fetch()
            except TwilioRestException as e:
                logger.debug("Twilio fetch failed while polling for in-progress: %s", e)
                last_status = None
                break
            except Exception as e:
                logger.exception("Unexpected error fetching call %s while waiting: %s", call_sid, e)
                break

            status = getattr(call, "status", None)
            last_status = status
            logger.info("play_tts_with_wait: call %s current status=%s (deadline in %.1fs)", call_sid, status, max(0.0, deadline - time.time()))

            if status == "in-progress":
                twiml = f"<Response><Play>{tts_url}</Play></Response>"
                try:
                    twilio_client.calls(call_sid).update(twiml=twiml)
                    logger.info("play_tts_with_wait: SDK update succeeded for call %s", call_sid)
                    return True
                except TwilioRestException as e:
                    logger.warning("play_tts_with_wait: SDK update failed for call %s: status=%s code=%s msg=%s", call_sid, getattr(e, "status", None), getattr(e, "code", None), getattr(e, "msg", str(e)))
                    logger.debug("Twilio exception details:", exc_info=True)
                    return False
                except Exception as e:
                    logger.exception("play_tts_with_wait: unexpected exception updating call %s: %s", call_sid, e)
                    return False

            time.sleep(poll_interval)

    except Exception:
        logger.exception("Error in play_tts_with_wait polling loop")
        return False

    logger.info("play_tts_with_wait: timed out waiting for in-progress (last_status=%s)", last_status)
    return False


# ---------------- processing flow ----------------

async def process_turn_force_rest(ws: WebSocket, stream_sid: str, call_sid: Optional[str], inbound_ulaw: bytes, recorder_ref: bytearray):
    """Process one user turn and attempt to play back TTS into the live call.
    Tries SDK update (with wait) then falls back to REST Play.
    """
    try:
        ts = int(time.time() * 1000)
        dump_path = f"/tmp/inbound_{ts}.ulaw"
        with open(dump_path, "wb") as f:
            f.write(inbound_ulaw)
        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound_ulaw))

        user_text = "(user speech)"
        logger.info("ASR placeholder text: %s", user_text)

        # TODO: replace with your LLM/ASR integration
        reply_text = "I heard you. (echo)"
        logger.info("Assistant reply text: %s", reply_text)

        tts_bytes = make_tts(reply_text)
        if not tts_bytes:
            logger.warning("make_tts returned empty bytes; skipping REST Play")
            return

        # Save to an accessible URL path for Twilio to fetch
        ts2 = int(time.time() * 1000)
        fname = f"tts_{call_sid}_{ts2}.wav"
        path = f"/tmp/{fname}"
        with open(path, "wb") as f:
            f.write(tts_bytes)
        tts_url = f"https://{HOSTNAME}/tts/{fname}"

        # 1) Try SDK update with polling/wait
        if call_sid:
            ok = play_tts_with_wait(call_sid, tts_url, wait_timeout=5.0, poll_interval=0.5)
            if ok:
                logger.info("play_tts_with_wait succeeded for callSid=%s", call_sid)
                return
            else:
                logger.warning("play_tts_with_wait failed or timed out for callSid=%s — falling back to REST POST", call_sid)

        # 2) Fallback: POST TwiML via REST (requests)
        fallback_ok = twilio_play_via_rest(call_sid or "", tts_bytes)
        if fallback_ok:
            logger.info("Twilio REST Play forced successfully for callSid=%s", call_sid)
        else:
            logger.warning("Twilio REST Play forced but failed for callSid=%s", call_sid)

    except Exception:
        logger.exception("process_turn_force_rest failed")


# ---------------- Serve TTS files ----------------
@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    logger.info("Serving TTS file %s", path)
    return FileResponse(path, media_type="audio/wav")


# ---------------- Websocket: Twilio Media Streams ----------------
PAUSE_MS = int(os.environ.get("PAUSE_MS", "700"))

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    awaiting_tts = False
    last_voice_ts = time.time()
    recorder = bytearray()

    async def maybe_trigger_processing(processing_tasks: set):
        nonlocal recorder, last_voice_ts, stream_sid, call_sid
        if len(recorder) == 0:
            return
        age_ms = (time.time() - last_voice_ts) * 1000.0
        if age_ms > PAUSE_MS or len(recorder) >= MAX_BUFFER_BYTES:
            inbound = bytes(recorder)
            recorder.clear()
            logger.info("Triggering background process_turn_force_rest: age_ms=%.1f buffer_bytes=%d", age_ms, len(inbound))
            task = asyncio.create_task(process_turn_force_rest(ws, stream_sid, call_sid, inbound, recorder))
            processing_tasks.add(task)
            def _on_done(t):
                processing_tasks.discard(t)
            task.add_done_callback(_on_done)

    processing_tasks = set()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                await maybe_trigger_processing(processing_tasks)
                continue

            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                start_obj = msg.get("start") or {}
                stream_sid = start_obj.get("streamSid")
                call_sid = start_obj.get("callSid") or start_obj.get("call_sid") or call_sid
                logger.info("Stream start sid=%s callSid=%s", stream_sid, call_sid)

            elif event == "media":
                media = msg.get("media", {}) or {}
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
                    await maybe_trigger_processing(processing_tasks)

            elif event == "mark":
                mark_name = (msg.get("mark") or {}).get("name")
                logger.info("Got mark ack: %s", mark_name)

            elif event == "stop":
                logger.info("Stream stop event received")
                if len(recorder) > 0:
                    inbound = bytes(recorder)
                    recorder.clear()
                    logger.info("Stream stopping but recorder has %d bytes — processing final utterance", len(inbound))
                    await process_turn_force_rest(ws, stream_sid, call_sid, inbound, recorder)
                break

            await maybe_trigger_processing(processing_tasks)

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


# ---------------- TwiML entrypoint ----------------
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
        "<Say voice='alice'>Hi — connecting you now.</Say>"
        "</Response>"
    )
    logger.info("Returning TwiML Start/Stream -> %s", ws_url)
    return Response(content=twiml, media_type="text/xml")


# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"ok": True, "hostname": HOSTNAME}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server_updated:app", host="0.0.0.0", port=port, log_level="info")
