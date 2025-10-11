# ws_server_patched.py  — realtime (Twilio Media Streams, bidirectional) — patch

import os, sys, json, base64, logging, asyncio, tempfile, io, time, audioop, subprocess
from typing import Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
import requests
from pydub import AudioSegment

# env
OPENAI_KEY = os.environ.get("OPENAI_KEY")
ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")
HOSTNAME = os.environ.get("HOSTNAME", "")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("realtime")

app = FastAPI(title="AI Voice Bot — Realtime (patched)")

# ----- utilities -----

def convert_to_mulaw_8k(raw_audio_bytes: bytes) -> bytes:
    """Use ffmpeg to convert any input audio bytes (mp3/wav) to raw mu-law 8k mono (headerless).
    Requires ffmpeg available in PATH.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-ar", "8000",
        "-ac", "1",
        "-f", "mulaw",
        "pipe:1",
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=raw_audio_bytes)
    if proc.returncode != 0:
        logger.error("ffmpeg conversion failed: %s", err.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg failed")
    return out


def mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes: bytes) -> bytes:
    # fallback implementation using pydub + audioop (keeps everything in-process)
    seg = AudioSegment.from_file_using_temporary_files(io.BytesIO(tts_bytes))
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)  # s16le
    pcm16 = seg.raw_data
    ulaw = audioop.lin2ulaw(pcm16, 2)
    return ulaw


def make_tts(text: str) -> bytes:
    """Return audio bytes (mp3 or wav). If ElevenLabs configured use it; otherwise return empty and log.
    You can replace fallback with your preferred TTS.
    """
    if ELEVEN_API_KEY and ELEVEN_VOICE:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
            body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
            r = requests.post(url, json=body, headers=headers, timeout=30)
            r.raise_for_status()
            if not r.content:
                logger.warning("ElevenLabs returned empty audio for text=%r", text)
            else:
                logger.info("TTS length=%d bytes", len(r.content))
            return r.content
        except Exception:
            logger.exception("ElevenLabs TTS failed")
            return b""

    # DEBUG / DEV fallback: produce a short silent WAV so you can still test playback path
    # This creates 0.3s of silence WAV (wav header + silence). Useful for testing without keys.
    try:
        import wave, struct
        duration_s = 0.3
        sr = 16000
        nframes = int(duration_s * sr)
        buf = io.BytesIO()
        wf = wave.open(buf, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for _ in range(nframes):
            wf.writeframes(struct.pack('<h', 0))
        wf.close()
        data = buf.getvalue()
        logger.info("Returning debug silent wav (%d bytes) as TTS fallback", len(data))
        return data
    except Exception:
        logger.exception("Local TTS fallback failed")
        return b""


async def stream_mulaw_chunks(ws: WebSocket, stream_sid: str, mulaw_bytes: bytes, frame_ms: int = 20):
    """Send mulaw_bytes to Twilio over websocket `ws` as outbound media frames.
    Each frame is ~frame_ms long. Adds "track":"outbound" so Twilio plays it.
    """
    sample_rate = 8000
    bytes_per_sample = 1
    chunk_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)

    offset = 0
    total = len(mulaw_bytes)
    sent = 0
    while offset < total:
        chunk = mulaw_bytes[offset:offset + chunk_size]
        offset += chunk_size
        payload_b64 = base64.b64encode(chunk).decode('ascii')
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "track": "outbound",
                "payload": payload_b64,
            }
        }
        await ws.send_text(json.dumps(msg))
        sent += 1
        # real-time pacing
        await asyncio.sleep(frame_ms / 1000.0)
    logger.info("Finished sending outbound audio: %d frames, %d bytes", sent, total)


async def llm_reply(text: str) -> str:
    # placeholder; integrate your real LLM client here
    return f"Assistant reply to: {text}"


# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME, "realtime": True}


# =========================================================
#  A. TwiML entrypoint that starts a real-time stream
# =========================================================from fastapi import Request, Response

@app.api_route("/twiml_stream", methods=["GET", "POST"])
async def twiml_stream(request: Request):
    """
    Recommended TwiML: explicit <Start><Stream url="wss://.../twilio-media" track="both"/></Start>
    then a Say and a long Pause so the call stays open while the websocket handles media.
    """
    if not HOSTNAME:
        xml = (
            "<Response>"
            "<Say voice='alice'>Server hostname is not configured. Please set HOSTNAME environment variable.</Say>"
            "</Response>"
        )
        logger.warning("HOSTNAME not set; returning debug TwiML.")
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

# =========================================================
#  B. Realtime WS endpoint for Twilio Media Streams
# =========================================================
PAUSE_MS = 700

@app.websocket("/twilio-media")
async def twilio_media(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")
    stream_sid: Optional[str] = None
    awaiting_tts = False
    last_voice_ts = time.time()
    recorder = bytearray()

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
                chunk = base64.b64decode(payload_b64)
                recorder.extend(chunk)
                last_voice_ts = time.time()

            elif event == "mark":
                mark_name = (obj.get("mark") or {}).get("name")
                logger.info("Got mark ack: %s", mark_name)

            elif event == "stop":
                logger.info("Stream stop")
                break

            # TURN detection: silence gap indicates end of user utterance
            now = time.time()
            if (now - last_voice_ts) > (PAUSE_MS / 1000.0) and len(recorder) > 0 and not awaiting_tts:
                awaiting_tts = True
                try:
                    ulaw_bytes = bytes(recorder)
                    recorder.clear()

                    # convert inbound μ-law to WAV for ASR or processing
                    seg = AudioSegment(ulaw_bytes, frame_rate=8000, sample_width=1, channels=1)
                    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    seg.export(tmp_wav.name, format="wav")

                    # TODO: replace with your ASR; placeholder text
                    text = "(user speech detected)"
                    reply = await llm_reply(text)

                    # synthesize TTS
                    tts_bytes = make_tts(reply)
                    if not tts_bytes:
                        logger.warning("Empty TTS bytes; skipping outbound playback")
                        awaiting_tts = False
                        continue

                    # convert to raw μ-law 8k
                    try:
                        ulaw_tts = mp3_or_wav_bytes_to_raw_mulaw_8k(tts_bytes)
                    except Exception:
                        # fallback to ffmpeg conversion path
                        ulaw_tts = convert_to_mulaw_8k(tts_bytes)

                    if not stream_sid:
                        logger.warning("No streamSid; cannot send audio")
                        awaiting_tts = False
                        continue

                    # Stream outbound in ~20ms frames (160 bytes per frame)
                    await stream_mulaw_chunks(ws, stream_sid, ulaw_tts, frame_ms=20)

                    # send a mark so Twilio can ack playback (optional)
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
