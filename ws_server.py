# ws_server.py  — realtime (Twilio Media Streams, bidirectional)

import os, sys, json, base64, logging, asyncio, tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from requests.auth import HTTPBasicAuth
import requests

# audio utils
from pydub import AudioSegment

# ---- Optional: OpenAI + ElevenLabs (same envs you already had) ----
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

OPENAI_KEY   = os.environ.get("OPENAI_KEY")
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE   = os.environ.get("ELEVEN_VOICE")
HOSTNAME       = os.environ.get("HOSTNAME", "")

openai_client = None
if OPENAI_KEY and OpenAIClient:
    try:
        openai_client = OpenAIClient(api_key=OPENAI_KEY)
    except Exception:
        pass

logger = logging.getLogger("realtime")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Voice Bot — Realtime")

# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME, "realtime": True}

# =========================================================
#  A. TwiML entrypoint that starts a real-time stream
# =========================================================
@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    """
    Twilio hits this when the call starts.
    We return <Connect><Stream> to open a WS to /twilio-media.
    """
    ws = f"wss://{HOSTNAME}/twilio-media" if HOSTNAME else "wss://YOUR_HOST/twilio-media"
    vr = VoiceResponse()
    with Connect() as c:
        c.stream(url=ws)
    vr.append(c)
    # A short prompt; barge-in will happen automatically once stream is up.
    vr.say("Hi! I'm listening. You can talk over me at any time.", voice="alice")
    return Response(content=str(vr), media_type="text/xml")

# =========================================================
#  B. Realtime WS endpoint for Twilio Media Streams
#     - receives inbound audio frames (base64 PCMU)
#     - runs a tiny VAD-like chunker (pause => treat as utterance)
#     - calls LLM to generate reply
#     - synthesizes TTS and streams back as base64 PCMU in small chunks
# =========================================================

# speech tuning
PAUSE_MS = 600               # treat ~600ms silence as end of turn
CHUNK_MS = 200               # send back audio in ~200ms chunks
FRAME_BYTES = 1600           # 8kHz, 20ms for mulaw ~160 bytes; with base64 overhead; we’ll buffer/convert

def pcm16_to_mulaw_bytes(raw_pcm_16_mono_8k: bytes) -> bytes:
    """Convert signed 16-bit PCM (s16le, 8kHz mono) -> 8-bit μ-law (G.711)."""
    # pydub can do this via AudioSegment export with format="mulaw"
    seg = AudioSegment(
        raw_pcm_16_mono_8k, frame_rate=8000, sample_width=2, channels=1
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulaw")
    seg.export(tmp.name, format="mulaw")  # 8k mu-law
    ulaw = open(tmp.name, "rb").read()
    return ulaw

def wav_or_mp3_bytes_to_ulaw(raw_bytes: bytes) -> bytes:
    """Utility for converting an mp3/wav TTS to mulaw 8k mono."""
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(raw_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    # export to mulaw
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulaw")
    seg.export(tmp.name, format="mulaw")
    return open(tmp.name, "rb").read()

def make_tts(text: str) -> bytes:
    """Return audio bytes (wav or mp3). Uses ElevenLabs if configured, else OpenAI TTS, else Twilio <Say> fallback."""
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_API_KEY,
        }
        body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
        r = requests.post(url, json=body, headers=headers, timeout=30)
        r.raise_for_status()
        return r.content  # mp3
    # (Optional) add OpenAI TTS if you use it. Otherwise, we’ll just speak with Say (fallback).
    # Return empty to signal fallback.
    return b""

async def llm_reply(text: str) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role":"system","content":"You are a helpful, concise, interruptible voice assistant."},
                    {"role":"user","content": text}
                ],
                max_tokens=200
            )
            msg = resp.choices[0].message
            return (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or "Okay."
        except Exception:
            logger.exception("LLM call failed")
    return "Okay."

import io, time
from collections import deque

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")

    # buffers
    last_voice_ts = time.time()
    recorder = bytearray()        # raw mulaw bytes from Twilio (we’ll convert to PCM16 if needed)
    awaiting_tts = False

    try:
        while True:
            msg = await ws.receive_text()
            obj = json.loads(msg)
            event = obj.get("event")

            if event == "start":
                logger.info("Stream start: %s", obj.get("start", {}))

            elif event == "media":
                # Twilio sends base64-encoded audio/x-mulaw;rate=8000;channels=1;packet=20ms
                payload = obj.get("media") or obj.get("payload") or {}
                b64 = payload.get("payload") or payload.get("data")
                if not b64:
                    continue
                ulaw = base64.b64decode(b64)
                recorder.extend(ulaw)
                last_voice_ts = time.time()

                # If caller speaks while we’re sending TTS, we “barge-in”: drop pending TTS.
                # (In practice you’d gate this with a flag and stop sending.)
                # Here, just note it; send loop below will check.

            elif event == "mark":
                pass

            elif event == "stop":
                logger.info("Stream stop")
                break

            # ---- TURN DETECTION (simple pause-based) ----
            now = time.time()
            if (now - last_voice_ts) > (PAUSE_MS / 1000.0) and len(recorder) > 0 and not awaiting_tts:
                # treat it as a completed user utterance
                awaiting_tts = True

                # Convert captured μ-law -> PCM16 WAV (via pydub) then to text
                try:
                    # write ulaw to temp, convert to wav s16le 8k mono
                    ulaw_bytes = bytes(recorder)
                    recorder.clear()

                    # decode ulaw using pydub
                    seg = AudioSegment(
                        ulaw_bytes, frame_rate=8000, sample_width=1, channels=1
                    ).set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    seg.export(tmp_wav.name, format="wav")

                    # Transcribe (Whisper)
                    text = ""
                    if openai_client:
                        with open(tmp_wav.name, "rb") as f:
                            tr = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
                        text = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "") or ""
                    else:
                        text = ""  # if you want offline ASR, plug it here

                    logger.info("USER SAID: %s", text)

                    # Hangup keywords
                    if any(k in text.lower() for k in ["hang up", "goodbye", "stop call", "disconnect", "bye"]):
                        await ws.send_text(json.dumps({"event":"mark", "name":"goodbye"}))
                        break

                    # Get reply from LLM
                    reply = await llm_reply(text)
                    logger.info("BOT REPLY: %s", reply)

                    # Generate TTS (mp3)
                    tts_bytes = make_tts(reply)
                    if not tts_bytes:
                        # If no TTS engine, we can instruct Twilio to <Say> via TwiML update (not realtime).
                        # For true realtime, configure ElevenLabs or similar TTS.
                        logger.warning("No TTS engine configured — skipping audio send")
                        awaiting_tts = False
                        continue

                    # Convert mp3 -> 8kHz μ-law
                    seg_tts = AudioSegment.from_file_using_temporary_files(io.BytesIO(tts_bytes))
                    seg_tts = seg_tts.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    tmp_ulaw = tempfile.NamedTemporaryFile(delete=False, suffix=".ulaw")
                    seg_tts.export(tmp_ulaw.name, format="mulaw")
                    ulaw_tts = open(tmp_ulaw.name, "rb").read()

                    # Stream back in ~200ms chunks
                    chunk_bytes = int(0.2 * 8000)  # 1600 samples @8k; but μ-law is 1 byte/sample => 1600 bytes ≈ 200ms
                    for i in range(0, len(ulaw_tts), chunk_bytes):
                        # If user started speaking again (recorder has new audio), barge-in: stop TTS immediately
                        if len(recorder) > 0:
                            logger.info("BARGE-IN detected — stopping TTS playback")
                            break
                        chunk = ulaw_tts[i:i+chunk_bytes]
                        b64 = base64.b64encode(chunk).decode("ascii")
                        frame = {
                            "event": "media",
                            "media": {
                                "payload": b64
                            }
                        }
                        await ws.send_text(json.dumps(frame))
                        await asyncio.sleep(0.18)  # ~just under 200ms wall-time

                    awaiting_tts = False

                except Exception:
                    logger.exception("TURN pipeline failed")
                    awaiting_tts = False

    except WebSocketDisconnect:
        logger.info("WS disconnect")
    except Exception:
        logger.exception("WS error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")
