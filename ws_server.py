# ws_server.py
# Twilio Media Streams bidirectional handler — robust version
# - periodic timeout to allow silence detection even when no new WS messages arrive
# - fallback to Twilio REST Play when WS closes during outbound streaming
# - saves callSid from start event
# - audible TTS fallback for testing
#
# Required env vars:
#   HOSTNAME            (public hostname reachable by Twilio, e.g. openai-twilio...onrender.com)
# Optional env vars for real TTS:
#   ELEVEN_API_KEY, ELEVEN_VOICE
# Optional env vars for REST fallback:
#   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
#
# Ensure ffmpeg is installed in the container (or pydub + ffmpeg support)
# Dockerfile note: apt-get install -y ffmpeg  (Debian/Ubuntu) or apk add ffmpeg (Alpine)

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

# pydub optional
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# ----- config -----
HOSTNAME = os.environ.get("HOSTNAME", "").strip()
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

PAUSE_MS = int(os.environ.get("PAUSE_MS", "500"))  # ms silence for end-of-turn detection
RECEIVE_TIMEOUT = float(os.environ.get("RECEIVE_TIMEOUT", "0.25"))  # seconds to wake up receive loop

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server")

app = FastAPI(title="twilio-media-server")

# ---------- utilities ----------

def make_tts(text: str) -> bytes:
    """ElevenLabs if configured, else audible test-tone WAV (PCM16 16000Hz)."""
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
            logger.exception("ElevenLabs TTS failed; falling back to audible tone")

    # audible tone fallback WAV (PCM16, 16000Hz, ~0.6s, 440Hz)
    sr = 16000
    duration = 0.6
    nframes = int(sr * duration)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    import math
    amplitude = 8000
    for i in range(nframes):
        t = i / sr
        sample = int(amplitude * math.sin(2 * math.pi * 440 * t))
        wf.writeframes(struct.pack('<h', sample))
    wf.close()
    data = buf.getvalue()
    logger.info("Returning audible test-tone WAV (%d bytes) from make_tts", len(data))
    return data

def convert_to_mulaw_ffmpeg(input_bytes: bytes) -> bytes:
    """Use ffmpeg to convert audio to μ-law 8k raw bytes."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ar", "8000", "-ac", "1", "-f", "mulaw", "pipe:1"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=input_bytes)
    if proc.returncode != 0:
        logger.error("ffmpeg conversion failed: %s", err.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg conversion failed")
    return out

def convert_to_mulaw_pydub(input_bytes: bytes) -> bytes:
    """Convert with pydub+audioop to μ-law 8k raw bytes (requires pydub & ffmpeg)."""
    if not HAS_PYDUB:
        raise RuntimeError("pydub not available")
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(input_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # s16le
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw

async def stream_ulaw(ws: WebSocket, stream_sid: str, ulaw_bytes: bytes, frame_ms: int = 20):
    """Send μ-law frames over websocket as outbound media frames (track='outbound')."""
    sample_rate = 8000
    bytes_per_sample = 1
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    frames_sent = 0
    total = len(ulaw_bytes)
    while offset < total:
        chunk = ulaw_bytes[offset: offset + chunk_size]
        offset += chunk_size
        payload_b64 = base64.b64encode(chunk).decode("ascii")
        msg = {"event": "media", "streamSid": stream_sid, "media": {"track": "outbound", "payload": payload_b64}}
        await ws.send_text(json.dumps(msg))
        frames_sent += 1
        if frames_sent % 25 == 0:
            logger.info("Sent %d outbound frames (~%d ms)", frames_sent, frames_sent*frame_ms)
        await asyncio.sleep(frame_ms / 1000.0)
    logger.info("Finished sending outbound audio: %d frames (%d bytes)", frames_sent, total)

def twilio_play_via_rest(call_sid: str, tts_bytes: bytes) -> bool:
    """Fallback: save tts_bytes to /tmp and tell Twilio via REST to Play it into the call.
       Requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and HOSTNAME reachable by Twilio.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
        logger.warning("Twilio REST credentials not set; cannot perform REST Play fallback")
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
        # Build TwiML to play the hosted file
        twiml = f"<Response><Play>{url}</Play></Response>"
        api_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json"
        logger.info("Attempting Twilio REST Play: POST %s (play %s)", api_url, url)
        r = requests.post(api_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), data={"Twiml": twiml}, timeout=10)
        if r.status_code // 100 == 2:
            logger.info("Twilio REST Play request accepted (status=%d)", r.status_code)
            return True
        else:
            logger.warning("Twilio REST Play failed: status=%d body=%s", r.status_code, r.text)
            return False
    except Exception:
        logger.exception("Twilio REST Play fallback failed")
        return False

# Route to serve transient TTS files for Twilio to fetch
@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    path = f"/tmp/{fname}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    # Use FileResponse from fastapi
    from fastapi.responses import FileResponse
    logger.info("Serving TTS file %s", path)
    return FileResponse(path, media_type="audio/wav")

# Simple healthcheck
@app.get("/health")
async def health():
    return {"ok": True, "hostname": HOSTNAME}

# ----- main realtime handler -----

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected. headers=%s", ws.scope.get("headers"))
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    recorder = bytearray()
    last_voice_ts = time.time()
    processing = False

    async def try_detect_and_process():
        nonlocal processing, recorder, stream_sid, call_sid
        if processing or len(recorder) == 0:
            return
        age_ms = (time.time() - last_voice_ts) * 1000.0
        if age_ms > PAUSE_MS:
            processing = True
            inbound = bytes(recorder)
            recorder.clear()
            logger.info("Detected end-of-turn: inbound bytes=%d — processing", len(inbound))
            try:
                # process the turn (ASR placeholder -> TTS -> convert -> stream)
                # use the same process_turn logic inline (to have ws scope)
                # Save inbound to disk for debugging
                ts = int(time.time() * 1000)
                dump_path = f"/tmp/inbound_{ts}.ulaw"
                with open(dump_path, "wb") as f:
                    f.write(inbound)
                logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound))

                # placeholder ASR
                user_text = "(user speech)"
                logger.info("ASR placeholder: %s", user_text)

                # LLM reply placeholder
                reply_text = f"I heard you. (echo)"
                logger.info("Assistant reply text: %s", reply_text)

                # TTS
                tts_bytes = make_tts(reply_text)
                if not tts_bytes:
                    logger.warning("make_tts returned empty; skipping outbound")
                    return

                logger.info("make_tts returned %d bytes", len(tts_bytes))

                # Convert to μ-law 8k
                ulaw = None
                if HAS_PYDUB:
                    try:
                        ulaw = convert_to_mulaw_pydub(tts_bytes)
                        logger.info("Converted TTS -> μ-law via pydub size=%d", len(ulaw))
                    except Exception as e:
                        logger.warning("pydub conversion failed: %s; attempting ffmpeg", e)
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

                # Attempt to stream via WS first
                if stream_sid:
                    try:
                        logger.info("Streaming outbound μ-law (%d bytes) to streamSid=%s via websocket", len(ulaw), stream_sid)
                        await stream_ulaw(ws, stream_sid, ulaw, frame_ms=20)
                        logger.info("Outbound streaming complete (websocket)")
                        return
                    except WebSocketDisconnect:
                        logger.warning("WebSocket disconnected while streaming outbound")
                    except Exception:
                        logger.exception("Error streaming outbound over WS; will try REST fallback if available")

                # If we reach here, WS streaming failed — attempt REST fallback
                if call_sid:
                    logger.info("Attempting Twilio REST Play fallback for callSid=%s", call_sid)
                    ok = twilio_play_via_rest(call_sid, tts_bytes)
                    if ok:
                        logger.info("Twilio REST Play fallback succeeded")
                    else:
                        logger.warning("Twilio REST Play fallback failed or not configured")
                else:
                    logger.warning("No callSid available; cannot attempt Twilio REST Play fallback")

            finally:
                processing = False

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                # No message for RECEIVE_TIMEOUT seconds — run detection
                await try_detect_and_process()
                continue

            # Got a message — process it
            obj = json.loads(raw)
            event = obj.get("event")

            if event == "start":
                start = obj.get("start", {}) or {}
                stream_sid = start.get("streamSid")
                # callSid is available in the start object — save it for REST fallback
                call_sid = start.get("callSid") or start.get("call_sid") or call_sid
                logger.info("Stream start sid=%s callSid=%s", stream_sid, call_sid)

            elif event == "media":
                media = obj.get("media", {}) or {}
                payload_b64 = media.get("payload")
                track = media.get("track")
                if not payload_b64:
                    logger.debug("media event with no payload")
                    continue

                # IMPORTANT: only append inbound frames (caller audio). Ignore outbound playback frames.
                if track != "inbound":
                    logger.debug("Ignoring non-inbound media frame: track=%s payload_bytes=%d", track, len(payload_b64))
                    continue

                chunk = base64.b64decode(payload_b64)
                recorder.extend(chunk)
                last_voice_ts = time.time()
                logger.info("INBOUND media event: track=%s payload_bytes=%d total_buffered=%d", track, len(chunk), len(recorder))

            elif event == "mark":
                mark = obj.get("mark", {})
                logger.info("Mark event received: %s", mark)

            elif event == "stop":
                logger.info("Stream stop event received")
                # process leftover buffer once before breaking
                if len(recorder) > 0 and not processing:
                    logger.info("Stream stopping but recorder has %d bytes — processing final utterance", len(recorder))
                    # run processing synchronously here to attempt to reply before closing
                    processing = True
                    inbound = bytes(recorder)
                    recorder.clear()
                    try:
                        # call the same detection/processing routine directly
                        # to reuse same logic we write inbound to file and produce TTS -> stream/REST
                        ts = int(time.time() * 1000)
                        dump_path = f"/tmp/inbound_{ts}.ulaw"
                        with open(dump_path, "wb") as f:
                            f.write(inbound)
                        logger.info("Saved inbound μ-law to %s (bytes=%d)", dump_path, len(inbound))

                        user_text = "(user speech)"
                        reply_text = "I heard you. (echo)"
                        tts_bytes = make_tts(reply_text)
                        if not tts_bytes:
                            logger.warning("make_tts empty at stream stop")
                        else:
                            # convert
                            ulaw = None
                            if HAS_PYDUB:
                                try:
                                    ulaw = convert_to_mulaw_pydub(tts_bytes)
                                    logger.info("Converted at stop via pydub size=%d", len(ulaw))
                                except Exception:
                                    logger.warning("pydub conversion failed at stop; trying ffmpeg")
                            if ulaw is None:
                                try:
                                    ulaw = convert_to_mulaw_ffmpeg(tts_bytes)
                                    logger.info("Converted at stop via ffmpeg size=%d", len(ulaw))
                                except Exception:
                                    logger.exception("Conversion failed at stop")
                                    ulaw = None

                            if ulaw and stream_sid:
                                try:
                                    await stream_ulaw(ws, stream_sid, ulaw, frame_ms=20)
                                    logger.info("Outbound streamed at stop")
                                except Exception:
                                    logger.exception("Failed to stream at stop; trying REST fallback")
                                    if call_sid:
                                        twilio_play_via_rest(call_sid, tts_bytes)
                            elif tts_bytes and call_sid:
                                twilio_play_via_rest(call_sid, tts_bytes)
                    finally:
                        processing = False
                break

            # after handling incoming event, attempt quick detection (to allow fast replies)
            await try_detect_and_process()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception:
        logger.exception("Websocket handler error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")

# twiml_stream endpoint (unchanged conceptually)
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

# run if executed directly (reads PORT)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server:app", host="0.0.0.0", port=port, log_level="info")
