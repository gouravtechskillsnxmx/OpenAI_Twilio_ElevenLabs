# ws_server.py  — realtime (Twilio Media Streams, bidirectional) — UPDATED

import os, sys, json, base64, logging, asyncio, tempfile, io, time, audioop
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from requests.auth import HTTPBasicAuth
import requests

# audio utils
from pydub import AudioSegment
# Add these imports near top of your file
import asyncio
import base64
import json
import subprocess
from fastapi import WebSocket
from typing import Optional



# ---- Optional: OpenAI + ElevenLabs (same envs you already had) ----
try:
    # Placeholder for your OpenAI client import, if any
    from openai import OpenAI as OpenAIClient  # adjust if different
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
        openai_client = None

logger = logging.getLogger("realtime")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Voice Bot — Realtime")


# --- Utility: convert any audio bytes to raw mu-law 8k mono bytes via ffmpeg ---
def convert_to_mulaw_8k(raw_audio_bytes: bytes) -> bytes:
    """
    Uses ffmpeg subprocess: input bytes piped to ffmpeg stdin, outputs raw mulaw 8k mono.
    Returns bytes (audio/x-mulaw, 8000 Hz, 1 channel).
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",            # read input from stdin
        "-ar", "8000",             # sample rate
        "-ac", "1",                # mono
        "-f", "mulaw",             # mu-law raw
        "pipe:1"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=raw_audio_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode('utf-8', errors='ignore')}")
    return out

# --- Utility: chunk & base64 encode mu-law bytes into Twilio media messages ---
async def stream_mulaw_to_twilio(ws: WebSocket, mulaw_bytes: bytes, frame_ms: int = 20):
    """
    Send mulaw_bytes to Twilio over websocket `ws` as outbound media frames.
    Uses frame_ms to compute chunk size (20 ms => 160 bytes for 8k mu-law).
    """
    sample_rate = 8000
    bytes_per_sample = 1  # mu-law is 8-bit per sample
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)  # e.g., 160 bytes

    # Send frames sequentially with ~frame_ms interval
    offset = 0
    total = len(mulaw_bytes)
    while offset < total:
        chunk = mulaw_bytes[offset: offset + chunk_size]
        offset += chunk_size

        # base64 encode
        payload_b64 = base64.b64encode(chunk).decode("ascii")

        msg = {
            "event": "media",
            "media": {
                "track": "outbound",      # important: outbound track
                "payload": payload_b64
            }
        }
        await ws.send_text(json.dumps(msg))
        # pace so audio plays at approx real-time
        await asyncio.sleep(frame_ms / 1000.0)

    # Optionally send an 'event' to indicate end-of-audio; Twilio doesn't mandate a special "eof" for media messages,
    # but you can optionally send a custom event or simply stop sending frames.
    # Example custom end event (not required):
    # await ws.send_text(json.dumps({"event": "media_end"}))
# --- Example handler snippet: called when websocket receives Twilio messages ---
async def handle_twilio_ws_message(ws: WebSocket, message_json: dict):
    """
    Call this from your websocket message loop.
    When 'start' event arrives, you can optionally synthesize response (TTS) and stream back.
    When 'media' inbound arrives, you should collect/recognize it (if you want ASR).
    """
    event = message_json.get("event")
    if event == "start":
        # Twilio started the stream. You can note streamSid/callSid here.
        stream_sid = message_json.get("streamSid")
        call_sid = message_json.get("callSid")
        # Example: send a quick TTS reply ("Hello, how can I assist?")
        # IMPORTANT: if you want to reply after receiving user speech, do ASR first then reply.
        reply_text = "Hello, this is an automated response. How can I help?"
        # Synthesize TTS (placeholder - implement your make_tts)
        tts_bytes = make_tts(reply_text)               # returns wav/mp3 bytes (your existing function)
        # Convert to mu-law 8k
        mu_bytes = convert_to_mulaw_8k(tts_bytes)
        # Stream to Twilio
        # run streaming in background so you can continue handling incoming media
        asyncio.create_task(stream_mulaw_to_twilio(ws, mu_bytes))

    elif event == "media":
        # inbound media payload from Twilio (base64 mu-law)
        media = message_json.get("media", {})
        payload = media.get("payload")
        track = media.get("track")
        if payload and track == "inbound":
            # decode inbound mu-law payload and feed to your ASR pipeline
            inbound_bytes = base64.b64decode(payload)
            # append to buffer or call your ASR worker...
            # e.g., append to queue: inbound_queue.put(inbound_bytes)
            pass

    elif event == "stop":
        # Twilio stopped the stream (call ended or stream stopped)
        # cleanup any background tasks, buffers, etc.
        pass


# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME, "realtime": True}

# =========================================================
#  A. TwiML entrypoint that starts a real-time stream
# =========================================================
@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    """Twilio hits this when the call starts.
    We return <Connect><Stream> to open a WS to /twilio-media.
    """
    ws = f"wss://{HOSTNAME}/twilio-media" if HOSTNAME else "wss://YOUR_HOST/twilio-media"
  
    vr = VoiceResponse()
    # A short prompt; barge-in will happen automatically once stream is up.
    vr.say("Hi! I'm listening. You can talk over me at any time.", voice="alice")
    with Connect() as c:
        # If you need both tracks, add track="both_tracks"
        c.stream(url=ws)
    vr.append(c)

    return Response(content=str(vr), media_type="text/xml")

# =========================================================
#  B. Realtime WS endpoint for Twilio Media Streams
#     - receives inbound audio frames (base64 PCMU)
#     - runs a tiny VAD-like chunker (pause => treat as utterance)
#     - calls LLM to generate reply
#     - synthesizes TTS and streams back as base64 PCMU in small chunks
# =========================================================

# speech tuning
PAUSE_MS = 700  # silence gap to consider end-of-utterance

def mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes: bytes) -> bytes:
    """Convert MP3/WAV bytes to *raw* μ-law 8k mono (no headers) for Twilio."""
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(tts_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # s16le
    pcm16 = seg.raw_data  # bytes, 16-bit little-endian
    ulaw = audioop.lin2ulaw(pcm16, 2)  # 1 byte/sample, headerless
    return ulaw

def make_tts(text: str) -> bytes:
    """Return audio bytes (wav or mp3). Uses ElevenLabs if configured, else OpenAI TTS, else fallback.\
    Empty bytes means 'use <Say> or ignore'."""
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
    # (Optional) add OpenAI TTS here if desired. Otherwise return empty for fallback.
    return b""

async def llm_reply(text: str) -> str:
    if openai_client:
        try:
            # Replace with your actual chat/completions call
            return "Here is a helpful reply to: " + text
        except Exception:
            logger.exception("LLM error")
    return f"You said: {text}"

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")
    stream_sid: Optional[str] = None
    awaiting_tts = False
    last_voice_ts = time.time()
    recorder = bytearray()  # holds inbound μ-law audio for the current turn

    try:
        while True:
            raw = await ws.receive_text()
            obj = json.loads(raw)
            event = obj.get("event")

            if event == "start":
                stream_sid = (obj.get("start") or {}).get("streamSid")
                logger.info("Stream start sid=%s", stream_sid)

            elif event == "media":
                media = obj.get("media", {})
                payload_b64 = media.get("payload", "")
                if not payload_b64:
                    continue
                # payload is μ-law/8000 mono
                chunk = base64.b64decode(payload_b64)
                recorder.extend(chunk)
                last_voice_ts = time.time()

            elif event == "mark":
                # Twilio tells us playback finished for a previously-sent chunk set
                mark_name = (obj.get("mark") or {}).get("name")
                logger.info("Got mark ack: %s", mark_name)

            elif event == "stop":
                logger.info("Stream stop")
                break

            # ---- TURN DETECTION (simple pause-based) ----
            now = time.time()
            if (now - last_voice_ts) > (PAUSE_MS / 1000.0) and len(recorder) > 0 and not awaiting_tts:
                # Treat it as a completed user utterance
                awaiting_tts = True

                try:
                    # Decode μ-law -> PCM16 WAV using pydub (for ASR), then transcribe (placeholder)
                    ulaw_bytes = bytes(recorder)
                    recorder.clear()

                    seg = AudioSegment(
                        ulaw_bytes, frame_rate=8000, sample_width=1, channels=1
                    ).set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    seg.export(tmp_wav.name, format="wav")  # temporary file for ASR

                    # TODO: plug Whisper/ASR here; placeholder just sets text to "..."
                    text = ""  # If you use Whisper, run it on tmp_wav.name
                    if not text.strip():
                        text = "(heard you, generating a response)"

                    reply = await llm_reply(text)

                    # TTS
                    tts_bytes = make_tts(reply)
                    if not tts_bytes:
                        # Fallback: send a 'say' equivalent via your telephony flow, or skip
                        awaiting_tts = False
                        continue

                    # Convert to raw μ-law (headerless) for Twilio
                    ulaw_tts = mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes)

                    if not stream_sid:
                        logger.warning("No streamSid yet; cannot send audio to Twilio")
                        awaiting_tts = False
                        continue

                    # Stream back in ~200ms chunks
                    chunk_bytes = int(0.2 * 8000)  # 1600 bytes ~200ms at 8k μ-law
                    for i in range(0, len(ulaw_tts), chunk_bytes):
                        # BARGE-IN: if user starts talking again, clear Twilio playback buffer and stop
                        if len(recorder) > 0:
                            try:
                                await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
                                logger.info("BARGE-IN: sent clear to Twilio")
                            except Exception:
                                logger.exception("Failed to send clear")
                            break

                        chunk = ulaw_tts[i:i+chunk_bytes]
                        b64 = base64.b64encode(chunk).decode("ascii")
                        frame = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                # "track": "outbound",  # optional
                                "payload": b64
                            }
                        }
                        await ws.send_text(json.dumps(frame))
                        await asyncio.sleep(0.18)  # slightly under real-time

                    # Ask Twilio to notify when playback of this block has completed
                    try:
                        mark_name = f"reply-{int(time.time()*1000)}"
                        await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": mark_name}}))
                    except Exception:
                        logger.exception("Failed to send mark event")

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