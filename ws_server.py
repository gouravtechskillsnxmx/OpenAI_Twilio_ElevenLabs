# ws_server.py — diagnostic + audible fallback patch
import os, sys, io, json, time, base64, wave, struct, tempfile, logging, subprocess, asyncio
from typing import Optional
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse
import requests

# optional dependencies
try:
    from pydub import AudioSegment
    import audioop
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

HOSTNAME = os.environ.get("HOSTNAME", "").strip()
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server_diag")

app = FastAPI()

# silence detection (ms)
PAUSE_MS = 900  # slightly longer to be safer (0.9s)

def make_tts(text: str) -> bytes:
    """
    Use ElevenLabs if configured. Otherwise return a short *audible* tone WAV (not silent)
    so outbound path can be validated by ear.
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
            logger.exception("ElevenLabs TTS failed — falling back to audible test tone")

    # Audible tone fallback (WAV PCM16 16000Hz, ~0.6s, 440Hz)
    sr = 16000
    duration = 0.6
    nframes = int(sr * duration)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    amplitude = 8000
    for i in range(nframes):
        t = i / sr
        sample = int(amplitude * (0.6 * 1.0) * __import__('math').sin(2 * __import__('math').pi * 440 * t))
        wf.writeframes(struct.pack('<h', sample))
    wf.close()
    data = buf.getvalue()
    logger.info("Returning audible test-tone WAV (%d bytes) from make_tts", len(data))
    return data

def convert_to_mulaw_ffmpeg(in_bytes: bytes) -> bytes:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ar", "8000", "-ac", "1", "-f", "mulaw", "pipe:1"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=in_bytes)
    if proc.returncode != 0:
        logger.error("ffmpeg conversion error: %s", err.decode('utf8', errors='ignore'))
        raise RuntimeError("ffmpeg failed")
    return out

def convert_to_mulaw_pydub(in_bytes: bytes) -> bytes:
    if not HAS_PYDUB:
        raise RuntimeError("pydub not available")
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(in_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm, 2)
    return ulaw

async def stream_ulaw(ws: WebSocket, stream_sid: str, ulaw_bytes: bytes, frame_ms: int = 20):
    sample_rate = 8000
    bytes_per_sample = 1
    chunk = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    frames = 0
    while offset < len(ulaw_bytes):
        block = ulaw_bytes[offset:offset+chunk]
        offset += chunk
        await ws.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"track": "outbound", "payload": base64.b64encode(block).decode("ascii")}
        }))
        frames += 1
        if frames % 25 == 0:
            logger.info("Sent %d outbound frames (~%d ms)", frames, frames*frame_ms)
        await asyncio.sleep(frame_ms/1000.0)
    logger.info("Completed streaming outbound: %d frames (%d bytes)", frames, len(ulaw_bytes))

@app.api_route("/twiml_stream", methods=["GET","POST"])
async def twiml_stream(request: Request):
    if not HOSTNAME:
        xml = "<Response><Say voice='alice'>HOSTNAME not configured. Set HOSTNAME env var.</Say></Response>"
        logger.warning("HOSTNAME missing; returning debug TwiML")
        return Response(content=xml, media_type="text/xml")
    ws_url = f"wss://{HOSTNAME}/twilio-media"
    xml = (
        "<Response>"
        f"<Start><Stream url=\"{ws_url}\" track=\"both\"/></Start>"
        "<Say voice='alice'>Hi — connecting you now. Please wait.</Say>"
        "<Pause length='600'/>"
        "</Response>"
    )
    logger.info("Returning TwiML: %s", xml)
    return Response(content=xml, media_type="text/xml")

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected. headers=%s", ws.scope.get("headers"))
    stream_sid: Optional[str] = None
    recorder = bytearray()
    last_voice_ts = time.time()
    awaiting = False

    try:
        while True:
            raw = await ws.receive_text()
            obj = json.loads(raw)
            event = obj.get("event")
            # LOG raw event for debugging
            logger.debug("Raw WS event: %s", json.dumps(obj)[:800])

            if event == "start":
                stream_sid = (obj.get("start") or {}).get("streamSid")
                logger.info("Stream start sid=%s", stream_sid)

            elif event == "media":
                media = obj.get("media", {})
                payload = media.get("payload")
                track = media.get("track")
                if payload:
                    chunk = base64.b64decode(payload)
                    recorder.extend(chunk)
                    last_voice_ts = time.time()
                    logger.info("INBOUND media event: track=%s payload_bytes=%d total_buffered=%d", track, len(chunk), len(recorder))
                else:
                    logger.info("media event with no payload")

            elif event == "stop":
                logger.info("Stream stop event received")
                break

            # TURN detection
            now = time.time()
            if (now - last_voice_ts) * 1000.0 > PAUSE_MS and len(recorder) > 0 and not awaiting:
                awaiting = True
                try:
                    inbound_ulaw = bytes(recorder)
                    recorder.clear()
                    ts = int(time.time()*1000)
                    dump_path = f"/tmp/inbound_{ts}.ulaw"
                    with open(dump_path, "wb") as f:
                        f.write(inbound_ulaw)
                    logger.info("Detected end-of-turn: inbound bytes=%d dumped=%s", len(inbound_ulaw), dump_path)

                    # PLACEHOLDER ASR -> for now we set a placeholder text
                    user_text = "(user speech detected)" 
                    logger.info("Detected user_text (placeholder): %s", user_text)

                    # Generate assistant reply text (replace with LLM)
                    reply_text = f"I heard you. (placeholder reply to: {user_text})"
                    logger.info("Assistant reply text: %s", reply_text)

                    # TTS
                    tts = make_tts(reply_text)
                    if not tts:
                        logger.warning("make_tts returned empty bytes; skipping outbound playback")
                        awaiting = False
                        continue
                    logger.info("make_tts returned %d bytes", len(tts))

                    # Convert to μ-law 8k (try pydub first, ffmpeg fallback)
                    try:
                        ulaw = convert_to_mulaw_pydub(tts) if HAS_PYDUB else convert_to_mulaw_ffmpeg(tts)
                        logger.info("convert_to_mulaw_pydub succeeded size=%d", len(ulaw))
                    except Exception as e:
                        logger.warning("pydub conversion failed or not available: %s; trying ffmpeg", str(e))
                        try:
                            ulaw = convert_to_mulaw_ffmpeg(tts)
                            logger.info("ffmpeg conversion succeeded size=%d", len(ulaw))
                        except Exception:
                            logger.exception("Both conversion methods failed; skipping playback")
                            awaiting = False
                            continue

                    # Stream outbound μ-law frames marked as outbound
                    if not stream_sid:
                        logger.warning("No streamSid; cannot send outbound audio")
                        awaiting = False
                        continue

                    logger.info("Streaming outbound audio: %d bytes to streamSid=%s", len(ulaw), stream_sid)
                    await stream_ulaw(ws, stream_sid, ulaw, frame_ms=20)
                    logger.info("Outbound streaming complete")

                except Exception:
                    logger.exception("TURN processing failed")
                finally:
                    awaiting = False

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")
