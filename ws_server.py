# ws_server.py — consolidated stable version
import os
import sys
import io
import json
import time
import base64
import wave
import struct
import logging
import tempfile
import subprocess
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse
import requests
from pydub import AudioSegment
import audioop

# ----- Configuration / env -----
HOSTNAME = os.environ.get("HOSTNAME", "").strip()  # must be public domain reachable by Twilio (no scheme)
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server")

app = FastAPI(title="AI Voice — Realtime (stable)")

# ----- Utilities -----
def make_tts(text: str) -> bytes:
    """
    Return audio bytes (mp3/wav). Use ElevenLabs if configured, otherwise return a short debug WAV.
    """
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVEN_API_KEY,
            }
            body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
            r = requests.post(url, json=body, headers=headers, timeout=30)
            r.raise_for_status()
            logger.info("ElevenLabs TTS returned %d bytes", len(r.content))
            return r.content
        except Exception:
            logger.exception("ElevenLabs TTS call failed — falling back to debug WAV")

    # Debug fallback: return a short silent WAV to validate outbound streaming path
    try:
        duration_s = 0.6
        sr = 16000
        nframes = int(duration_s * sr)
        buf = io.BytesIO()
        wf = wave.open(buf, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # write silence
        for _ in range(nframes):
            wf.writeframes(struct.pack('<h', 0))
        wf.close()
        data = buf.getvalue()
        logger.info("Returning debug silent WAV (%d bytes) from make_tts", len(data))
        return data
    except Exception:
        logger.exception("Failed to produce debug WAV")
        return b""

def convert_to_mulaw_8k_ffmpeg(input_bytes: bytes) -> bytes:
    """
    Convert input audio bytes (wav or mp3) to raw μ-law 8k mono using ffmpeg.
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
        logger.error("ffmpeg conversion failed: %s", err.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg conversion failed")
    return out

def convert_to_mulaw_8k_pydub(input_bytes: bytes) -> bytes:
    """
    Convert using pydub + audioop as fallback (no external ffmpeg required if pydub can read bytes).
    Produces μ-law 8k raw bytes.
    """
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(input_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # s16le
    pcm = seg.raw_data
    # convert s16le PCM to mu-law
    ulaw = audioop.lin2ulaw(pcm, 2)
    return ulaw

async def stream_mulaw_chunks(ws: WebSocket, stream_sid: str, mulaw_bytes: bytes, frame_ms: int = 20):
    """
    Stream μ-law bytes to Twilio via websocket as outbound media frames (track='outbound').
    """
    sample_rate = 8000
    bytes_per_sample = 1  # μ-law 8-bit
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)  # e.g., 160 bytes for 20ms

    offset = 0
    total = len(mulaw_bytes)
    frames_sent = 0
    while offset < total:
        chunk = mulaw_bytes[offset: offset + chunk_size]
        offset += chunk_size
        payload_b64 = base64.b64encode(chunk).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "track": "outbound",   # CRITICAL: Twilio needs this to play into the call
                "payload": payload_b64
            }
        }
        await ws.send_text(json.dumps(msg))
        frames_sent += 1
        # pace for real-time playback
        await asyncio.sleep(frame_ms / 1000.0)
    logger.info("stream_mulaw_chunks finished: %d frames, %d bytes", frames_sent, total)

async def llm_reply_stub(text: str) -> str:
    """
    Placeholder LLM reply. Replace with your real LLM call (OpenAI, local model, etc).
    """
    return "I heard you. This is an example reply from the assistant."

# ----- FastAPI endpoints -----
@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    """
    Return TwiML that starts a bidirectional Media Stream and keeps the call alive while the websocket is active.
    HOSTNAME must be set to a publicly reachable domain (no scheme).
    """
    if not HOSTNAME:
        xml = (
            "<Response>"
            "<Say voice='alice'>Server hostname is not configured. Please set HOSTNAME environment variable.</Say>"
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

PAUSE_MS = 600  # ms pause to detect end of utterance (turn-taking)

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected. scope headers: %s", ws.scope.get("headers"))
    stream_sid: Optional[str] = None
    recorder = bytearray()
    last_voice_ts = time.time()
    awaiting_tts = False

    try:
        while True:
            raw = await ws.receive_text()
            obj = json.loads(raw)
            event = obj.get("event")
            # handle start, media, stop, other events
            if event == "start":
                start_obj = obj.get("start") or {}
                stream_sid = start_obj.get("streamSid")
                logger.info("Stream start sid=%s", stream_sid)

            elif event == "media":
                media = obj.get("media", {})
                payload = media.get("payload")
                track = media.get("track")
                if payload:
                    chunk = base64.b64decode(payload)
                    # Twilio sends inbound as μ-law 8k typically
                    recorder.extend(chunk)
                    last_voice_ts = time.time()

            elif event == "stop":
                logger.info("Stream stop event received")
                break

            # TURN detection: if silence gap and we have recorded audio, process it
            now = time.time()
            if (now - last_voice_ts) * 1000.0 > PAUSE_MS and len(recorder) > 0 and not awaiting_tts:
                awaiting_tts = True
                try:
                    inbound_ulaw = bytes(recorder)
                    recorder.clear()

                    # Convert inbound μ-law 8k to WAV for ASR or storage
                    try:
                        seg = AudioSegment(
                            # pydub expects raw_data plus parameters
                            data=inbound_ulaw,
                            sample_width=1, frame_rate=8000, channels=1
                        )
                        # convert to 16-bit PCM WAV temporary file for ASR usage if needed
                        wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        seg = seg.set_frame_rate(16000).set_sample_width(2).set_channels(1)
                        seg.export(wav_tmp.name, format="wav")
                        wav_tmp.close()
                    except Exception:
                        logger.exception("Failed to export inbound segment to WAV (non-fatal)")

                    # Placeholder ASR/text extraction:
                    user_text = "(user speech)"
                    logger.info("Detected utterance (placeholder text): %s", user_text)

                    # Get assistant reply (replace with your LLM/assistant call)
                    reply_text = await llm_reply_stub(user_text)
                    logger.info("Assistant reply text: %s", reply_text)

                    # Synthesize TTS
                    tts_bytes = make_tts(reply_text)
                    if not tts_bytes:
                        logger.warning("make_tts returned empty bytes; skipping playback")
                        awaiting_tts = False
                        continue

                    # Convert TTS bytes to μ-law 8k
                    try:
                        ulaw_tts = convert_to_mulaw_8k_pydub(tts_bytes)
                    except Exception:
                        try:
                            ulaw_tts = convert_to_mulaw_8k_ffmpeg(tts_bytes)
                        except Exception:
                            logger.exception("Failed to convert TTS to μ-law 8k")
                            awaiting_tts = False
                            continue

                    if not stream_sid:
                        logger.warning("No streamSid available; cannot send outbound audio")
                        awaiting_tts = False
                        continue

                    logger.info("Streaming outbound audio: %d bytes to streamSid=%s", len(ulaw_tts), stream_sid)

                    # Stream outbound frames; allow barge-in: stop streaming if new inbound arrives
                    sample_rate = 8000
                    chunk_bytes = int(sample_rate * 0.02)  # 20 ms frames = 160 bytes
                    frames_sent = 0
                    offset = 0
                    while offset < len(ulaw_tts):
                        # barge-in detection — if caller started speaking again, stop playback
                        if len(recorder) > 0:
                            logger.info("Barge-in detected: inbound audio arrived while playing outbound. Stopping playback.")
                            break

                        chunk = ulaw_tts[offset:offset + chunk_bytes]
                        offset += chunk_bytes
                        payload_b64 = base64.b64encode(chunk).decode("ascii")
                        frame = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "track": "outbound",
                                "payload": payload_b64
                            }
                        }
                        await ws.send_text(json.dumps(frame))
                        frames_sent += 1
                        if frames_sent % 25 == 0:
                            logger.info("Sent %d frames (~%d ms) of outbound audio", frames_sent, frames_sent*20)
                        await asyncio.sleep(0.02)
                    logger.info("Finished outbound streaming: %d frames, %d bytes", frames_sent, len(ulaw_tts))

                    # Optional: mark event so Twilio can correlate
                    try:
                        mark = {"event": "mark", "streamSid": stream_sid, "mark": {"name": f"reply-{int(time.time()*1000)}"}}
                        await ws.send_text(json.dumps(mark))
                    except Exception:
                        logger.exception("Failed to send mark event")

                except Exception:
                    logger.exception("TURN handling failed")
                finally:
                    awaiting_tts = False

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("WS closed")

# Run via `python ws_server.py` — reads PORT env var
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run("ws_server:app", host="0.0.0.0", port=port, log_level="info")
