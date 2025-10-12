# ws_server_fixed.py
# Robust Twilio Media Stream server with REST Play fallback when WS streaming fails.
# - Detects turns, generates TTS, streams μ-law frames to Twilio (track="outbound").
# - If websocket closes while streaming, falls back to Twilio REST Play (requires TWILIO_* env vars).
# - Emits mark events "play-start"/"play-end" for easier debugging in Twilio Console.
# - Keep HOSTNAME, ELEVEN_API_KEY, ELEVEN_VOICE set as before.
#
# Set these env vars for REST fallback:
#   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

import os, sys, io, json, time, base64, wave, struct, tempfile, logging, subprocess, asyncio
from typing import Optional
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException
import requests

# optional
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
logger = logging.getLogger("ws_server_fixed")

app = FastAPI(title="ws_server_fixed")

# ---------------- utilities ----------------

def make_tts(text: str) -> bytes:
    """Try ElevenLabs TTS; if fails, return audible test tone. Logs non-2xx body for diagnostics."""
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVEN_API_KEY.strip(),
            }
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
    amplitude = 12000  # bumped a bit for audibility
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

async def send_mark_safe(ws: WebSocket, stream_sid: str, name: str):
    """Send a Twilio 'mark' event if ws open; swallow errors."""
    if not ws:
        return
    try:
        await ws.send_text(json.dumps({"event":"mark","streamSid":stream_sid,"mark":{"name":name}}))
    except Exception:
        logger.debug("Could not send mark '%s' — WS may be closed", name)

async def stream_ulaw(ws: WebSocket, stream_sid: str, ulaw_bytes: bytes, frame_ms: int = 20, recorder_ref: Optional[bytearray] = None):
    """
    Stream μ-law bytes to Twilio over websocket as outbound frames.
    Raises WebSocketDisconnect or RuntimeError if WS closed while streaming.
    """
    sample_rate = 8000
    bytes_per_sample = 1
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)

    # send mark start if possible
    await send_mark_safe(ws, stream_sid, "play-start")

    offset = 0
    total = len(ulaw_bytes)
    frames_sent = 0
    while offset < total:
        if recorder_ref and len(recorder_ref) > 0:
            logger.info("Barge-in detected while sending outbound — aborting playback")
            break

        chunk = ulaw_bytes[offset: offset + chunk_size]
        offset += chunk_size
        payload_b64 = base64.b64encode(chunk).decode('ascii')
        msg = {"event": "media", "streamSid": stream_sid, "media": {"track": "outbound", "payload": payload_b64}}

        try:
            await ws.send_text(json.dumps(msg))
        except WebSocketDisconnect:
            logger.warning("WebSocketDisconnect while sending outbound frame")
            # ensure mark end attempt (best-effort)
            try:
                await send_mark_safe(ws, stream_sid, "play-end")
            except Exception:
                pass
            # re-raise to let caller handle REST fallback
            raise
        except Exception:
            logger.exception("Failed to send outbound frame — websocket may be closed")
            # re-raise so caller will attempt fallback
            raise

        frames_sent += 1
        if frames_sent % 25 == 0:
            logger.info("Sent %d outbound frames (~%d ms)", frames_sent, frames_sent * frame_ms)
        await asyncio.sleep(frame_ms / 1000.0)

    # send mark end if possible
    await send_mark_safe(ws, stream_sid, "play-end")
    logger.info("Finished sending outbound audio: %d frames (%d bytes)", frames_sent, total)

def twilio_play_via_rest(call_sid: str, tts_bytes: bytes) -> bool:
    """Save tts_bytes to /tmp and POST TwiML <Play> to call via Twilio REST."""
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
@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    from fastapi.responses import FileResponse
    logger.info("Serving TTS file %s", path)
    return FileResponse(path, media_type="audio/wav")

# health
@app.get("/health")
async def health():
    return {"ok": True, "hostname": HOSTNAME}

# ---------------- turn processing ----------------

async def llm_reply_stub(text: str) -> str:
    return "I heard you. (echo)"

async def process_turn(ws: WebSocket, stream_sid: str, call_sid: Optional[str], inbound_ulaw: bytes, recorder_ref: bytearray):
    """
    Process the user turn: save inbound, ASR placeholder, LLM reply, TTS, convert, try WS streaming.
    If WS streaming fails, attempt REST Play fallback using call_sid (if configured).
    """
    try:
        ts = int(time.time() * 1000)
        dump_path = f"/tmp/inbound_{ts}.ulaw"
        with open(dump_path, "wb") as f:
            f.write(inbound_ulaw)
        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound_ulaw))

        # ASR placeholder
        user_text = "(user speech)"
        logger.info("ASR placeholder text: %s", user_text)

        # LLM reply
        reply_text = await llm_reply_stub(user_text)
        logger.info("Assistant reply text: %s", reply_text)

        # TTS
        tts_bytes = make_tts(reply_text)
        if not tts_bytes:
            logger.warning("make_tts returned empty bytes; skipping outbound")
            return
        logger.info("make_tts returned %d bytes", len(tts_bytes))

        # Convert to μ-law
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
                logger.exception("Conversion to μ-law failed")
                ulaw = None

        if ulaw is None:
            logger.warning("No μ-law audio available; skipping outbound")
            return

        if stream_sid:
            try:
                logger.info("Streaming outbound μ-law (%d bytes) to streamSid=%s via websocket", len(ulaw), stream_sid)
                await stream_ulaw(ws, stream_sid, ulaw, frame_ms=20, recorder_ref=recorder_ref)
                logger.info("Outbound streaming complete (websocket)")
                return
            except Exception:
                logger.exception("WebSocket streaming failed — will attempt REST fallback if available")

        # WS not available or streaming failed, fallback to REST Play
        if call_sid:
            ok = twilio_play_via_rest(call_sid, tts_bytes)
            if ok:
                logger.info("Played reply into call via Twilio REST Play fallback")
                return
            else:
                logger.warning("REST Play fallback failed or not configured")
        else:
            logger.warning("No call_sid available; cannot REST Play")

    except Exception:
        logger.exception("process_turn failed")

# ---------------- TwiML / websocket handler ----------------

@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    if not HOSTNAME:
        xml = "<Response><Say voice='alice'>Server hostname is not configured. Please set HOSTNAME environment variable.</Say></Response>"
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
            logger.info("Triggering background process_turn: age_ms=%.1f buffer_bytes=%d", age_ms, len(inbound))
            task = asyncio.create_task(process_turn(ws, stream_sid, call_sid, inbound, recorder))
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
                # Twilio may provide callSid in the start metadata
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
                    await process_turn(ws, stream_sid, call_sid, inbound, recorder)
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
    uvicorn.run("ws_server_fixed:app", host="0.0.0.0", port=port, log_level="info")
