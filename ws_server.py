# ws_server.py
# Patched: adds /twiml_stream TwiML webhook to start media streams
# Keep TWILIO_ACCOUNT_SID name stable as requested.

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

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException, Form
import requests

# optional pydub
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# ---------- config ----------
HOSTNAME = os.environ.get("HOSTNAME", "").strip()  # must be your public hostname (no https prefix)
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

PAUSE_MS = int(os.environ.get("PAUSE_MS", "300"))
RECEIVE_TIMEOUT = float(os.environ.get("RECEIVE_TIMEOUT", "0.12"))
MAX_BUFFER_BYTES = int(os.environ.get("MAX_BUFFER_BYTES", "16000"))

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server_patched")

app = FastAPI(title="ws_server_patched")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    logger.warning("TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not set in environment")

# Twilio client (SDK)
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception:
    twilio_client = None

# ---------------- New: TwiML webhook route ----------------
# Twilio will POST to this path to fetch TwiML. The TwiML returned instructs Twilio
# to Start a Media Stream which connects to our wss endpoint /twilio-media.
@app.post("/twiml_stream")
async def twiml_stream(request: Request):
    """
    Twilio Voice webhook that returns TwiML to start a media stream.
    Twilio expects application/xml TwiML back.
    """
    try:
        form = await request.form()
        # Twilio sends CallSid and other details in form fields
        call_sid = form.get("CallSid") or form.get("callSid") or None
        from_num = form.get("From")
        to_num = form.get("To")
        logger.info("Received Twilio webhook /twiml_stream callSid=%s From=%s To=%s form_keys=%s",
                    call_sid, from_num, to_num, list(form.keys()))
    except Exception:
        # If content-type isn't form-encoded, just log and continue
        call_sid = None
        logger.exception("Error reading form in /twiml_stream")

    if not HOSTNAME:
        # return helpful TwiML so Twilio shows the error in its webhook inspector
        logger.warning("/twiml_stream called but HOSTNAME not configured")
        twiml_err = '<?xml version="1.0" encoding="UTF-8"?><Response><Say>Server not configured with HOSTNAME</Say></Response>'
        return Response(content=twiml_err, media_type="application/xml")

    # Build the websocket URL Twilio should connect to for media (wss)
    # Note: Twilio requires a secure WebSocket (wss://)
    ws_url = f"wss://{HOSTNAME}/twilio-media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{ws_url}" />
  </Start>
</Response>"""
    logger.info("Returning TwiML Start/Stream -> %s", ws_url)
    return Response(content=twiml, media_type="application/xml")


@app.get("/twiml_stream")
async def twiml_stream_get():
    # simple GET to check endpoint in a browser
    if not HOSTNAME:
        return {"ok": False, "reason": "HOSTNAME not set"}
    return {"ok": True, "start_stream_to": f"wss://{HOSTNAME}/twilio-media"}

# ---------------- Twilio SDK wrapper with poll/retry ----------------
def play_tts_with_wait(call_sid: str, tts_url: str, wait_timeout: float = 5.0, poll_interval: float = 0.5) -> bool:
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
                    logger.warning("play_tts_with_wait: SDK update failed for call %s: status=%s code=%s msg=%s",
                                   call_sid, getattr(e, "status", None), getattr(e, "code", None), getattr(e, "msg", str(e)))
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

# ---------------- Existing fallback: save file and POST TwiML via REST ----------------
def twilio_play_via_rest(call_sid: str, tts_bytes: bytes) -> bool:
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

# ---------------- small utilities (TTS, conversion) ----------------
def make_tts(text: str) -> bytes:
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
            logger.exception("ElevenLabs TTS exception — falling back to empty bytes")
    return b""

def mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes: bytes) -> bytes:
    if not HAS_PYDUB:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ar", "8000", "-ac", "1", "-f", "mulaw", "pipe:1"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(input=tts_bytes)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg conversion failed: %s" % err.decode('utf-8', errors='ignore'))
        return out
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(tts_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw

# ---------------- processing flow: use wrapper then fallback ----------------
async def process_turn_force_rest(ws: WebSocket, stream_sid: str, call_sid: Optional[str], inbound_ulaw: bytes, recorder_ref: bytearray):
    try:
        ts = int(time.time() * 1000)
        dump_path = f"/tmp/inbound_{ts}.ulaw"
        with open(dump_path, "wb") as f:
            f.write(inbound_ulaw)
        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound_ulaw))

        user_text = "(user speech)"
        logger.info("ASR placeholder text: %s", user_text)

        reply_text = "I heard you. (echo)"
        logger.info("Assistant reply text: %s", reply_text)

        tts_bytes = make_tts(reply_text)
        if not tts_bytes:
            logger.warning("make_tts returned empty bytes; skipping REST Play")
            return

        ts2 = int(time.time() * 1000)
        fname = f"tts_{call_sid}_{ts2}.wav"
        path = f"/tmp/{fname}"
        with open(path, "wb") as f:
            f.write(tts_bytes)
        tts_url = f"https://{HOSTNAME}/tts/{fname}"

        if call_sid:
            ok = play_tts_with_wait(call_sid, tts_url, wait_timeout=5.0, poll_interval=0.5)
            if ok:
                logger.info("play_tts_with_wait succeeded for callSid=%s", call_sid)
                return
            else:
                logger.warning("play_tts_with_wait failed or timed out for callSid=%s — falling back to REST POST", call_sid)

        fallback_ok = twilio_play_via_rest(call_sid or "", tts_bytes)
        if fallback_ok:
            logger.info("Twilio REST Play forced successfully for callSid=%s", call_sid)
        else:
            logger.warning("Twilio REST Play forced but failed for callSid=%s", call_sid)

    except Exception:
        logger.exception("process_turn_force_rest failed")

# ---------------- Serve TTS files ----------------
from fastapi.responses import FileResponse

@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    logger.info("Serving TTS file %s", path)
    return FileResponse(path, media_type="audio/wav")

# Health
@app.get("/health")
async def health():
    return {"ok": True, "hostname": HOSTNAME}

# Minimal WS handler (keeps the same forced-rest processing trigger points)
PAUSE_MS = int(os.environ.get("PAUSE_MS", "700"))

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server:app", host="0.0.0.0", port=port, log_level="info")
