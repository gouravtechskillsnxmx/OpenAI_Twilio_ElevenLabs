# ws_server.py
import os
import sys
import time
import json
import logging
import tempfile
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import html

from fastapi import FastAPI, Request, Response, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from requests.auth import HTTPBasicAuth
import requests
import boto3

# Optional new OpenAI client wrapper
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# Twilio REST client
from twilio.rest import Client as TwilioClient

# memory module & API routes (if present in repo)
try:
    from memory import write_fact  # optional
    from memory_api import router as memory_router
except Exception:
    memory_router = None

# ---------------- CONFIG / ENV ----------------
HOLD_STORE_DIR = "/tmp/hold_store"
Path(HOLD_STORE_DIR).mkdir(parents=True, exist_ok=True)

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "+15312303465")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT")  # external agent
AGENT_KEY = os.environ.get("AGENT_KEY")

ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

HOSTNAME = os.environ.get("HOSTNAME", "")
TTS_PRESIGNED_EXPIRES = int(os.environ.get("TTS_PRESIGNED_EXPIRES", "3600"))

REDIS_URL = os.environ.get("REDIS_URL")

# ---------------- logging & app ----------------
logger = logging.getLogger("ai-sales-agent")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Sales Agent")
if memory_router:
    app.include_router(memory_router)

twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None

openai_client = None
if OPENAI_AL_KEY and OpenAIClient is not None:
    try:
        openai_client = OpenAIClient(api_key=OPENAI_KEY)
    except Exception:
        logger.exception("Failed to init OpenAIClient")

# boto3 S3 client
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
else:
    s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------- hold_store: Redis preferred, else in-memory + file fallback ----------------
_hold_in_memory: Dict[str, dict] = {}

redis_client = None
if REDIS_URL:
    try:
        import redis  # pip package
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning("Redis init failed; continuing without Redis: %s", e)
        redis_client = None

class HoldStore:
    @staticmethod
    def set_ready(convo_id: str, payload: dict, expire: int = 3600) -> bool:
        try:
            if redis_client:
                try:
                    redis_client.set(f"hold:{convo_id}", json.dumps(payload), ex=expire)
                    logger.info("Redis: set hold:%s", convo_id)
                    return True
                except Exception:
                    logger.exception("Redis set failed for hold:%s -- falling back", convo_id)

            _hold_in_memory[convo_id] = payload
            try:
                p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
                p.write_text(json.dumps(payload))
                logger.info("File fallback: wrote %s", str(p))
            except Exception:
                logger.exception("File fallback write failed for hold:%s", convo_id)
            return False
        except Exception:
            logger.exception("Unexpected error in HoldStore.set_ready for %s", convo_id)
            return False

    @staticmethod
    def get_ready(convo_id: str) -> Optional[dict]:
        # 1) Redis
        try:
            if redis_client:
                try:
                    v = redis_client.get(f"hold:{convo_id}")
                    if v:
                        data = json.loads(v)
                        redis_client.delete(f"hold:{convo_id}")
                        logger.info("Redis: got and cleared hold:%s", convo_id)
                        return data
                except Exception:
                    logger.exception("Redis get failed for hold:%s", convo_id)
        except Exception:
            pass

        # 2) In-memory
        try:
            if convo_id in _hold_in_memory:
                data = _hold_in_memory.pop(convo_id)
                logger.info("In-memory: popped hold for %s", convo_id)
                p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
                if p.exists():
                    p.unlink()
                return data
        except Exception:
            pass

        # 3) File fallback
        try:
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            if p.exists():
                raw = p.read_text()
                data = json.loads(raw)
                p.unlink()
                logger.info("File fallback: read & cleared %s", str(p))
                return data
        except Exception:
            pass

        return None

    @staticmethod
    def clear(convo_id: str):
        try:
            if redis_client:
                redis_client.delete(f"hold:{convo_id}")
            _hold_in_memory.pop(convo_id, None)
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            if p.exists():
                p.unlink()
        except Exception:
            pass

hold_store = HoldStore()

# ---------------- helpers ----------------
def recording_callback_url() -> str:
    return f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"

def build_download_url(recording_url: str) -> str:
    if not recording_url:
        return recording_url
    lower = recording_url.lower()
    for ext in (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"):
        if lower.endswith(ext):
            return recording_url
    if "api.twilio.com" in lower:
        return recording_url + ".mp3"
    return recording_url

def _unescape_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        u2 = html.unescape(u)
        if u2.startswith('"') and u2.endswith('"'):
            u2 = u2[1:-1]
        return u2
    except Exception:
        return u

# ---------------- TTS helpers ----------------
def create_tts_elevenlabs(text: str) -> bytes:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE:
        raise RuntimeError("ElevenLabs not configured (ELEVEN_API_KEY/ELEVEN_VOICE).")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
    body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
    r = requests.post(url, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content

def create_and_upload_tts(text: str, expires_in: int = TTS_PRESIGNED_EXPIRES) -> str:
    audio_bytes = create_tts_elevenlabs(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    key = f"tts/{os.path.basename(tmp.name)}"
    s3.upload_file(tmp.name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
    presigned = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_in)
    os.unlink(tmp.name)  # clean up
    return presigned

# ---------------- agent integration ----------------
def call_agent_and_get_reply(convo_id: str, user_text: str, timeout: int = 15) -> Dict[str, Any]:
    if AGENT_ENDPOINT:
        try:
            headers = {"Content-Type": "application/json"}
            if AGENT_KEY:
                headers["Authorization"] = f"Bearer {AGENT_KEY}"
            payload = {"convo_id": convo_id, "text": user_text}
            logger.info("Calling external agent at %s (convo=%s)", AGENT_ENDPOINT, convo_id)
            r = requests.post(AGENT_ENDPOINT, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            reply_text = j.get("reply_text") or j.get("reply") or j.get("text") or j.get("replyText") or ""
            memory_writes = j.get("memory_writes") or j.get("memoryWrites") or []
            if not isinstance(memory_writes, list):
                memory_writes = []
            return {"reply_text": reply_text, "memory_writes": memory_writes}
        except Exception as e:
            logger.exception("Agent call failed: %s", e)

    if openai_client:
        try:
            logger.info("Falling back to OpenAI for convo=%s", convo_id)
            chat_resp = openai_client.chat.completions.create(
                model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "256"))
            )
            content = ""
            try:
                choice = chat_resp.choices[0]
                content =(choice.message.content if hasattr(choice, "message") and choice.message else "") or str(chat_resp)
            except Exception:
                content = str(chat_resp)
            return {"reply_text": content.strip(), "memory_writes": []}
        except Exception:
            logger.exception("OpenAI fallback failed")
            return {"reply_text": "Sorry, I'm having trouble right now.", "memory_writes": []}

    logger.warning("No agent or OpenAI available for convo=%s", convo_id)
    return {"reply_text": f"Echo: {user_text}", "memory_writes": []}

# ---------------- transcription ----------------
def transcribe_with_openai(file_path: str) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI not configured.")
    with open(file_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    return (getattr(resp, "text", "") or "").strip()

# ---------------- TwiML endpoints ----------------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    resp = VoiceResponse()
    resp.say("Hello, this is our AI assistant. Please say something after the beep.", voice="alice")
    resp.record(max_length=30, action=recording_callback_url(), play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    payload = dict(form)
    recording_url = payload.get("RecordingUrl")
    call_sid = payload.get("CallSid")
    from_number = payload.get("From")
    logger.info("Incoming recording: CallSid=%s From=%s RecordingUrl=%s", call_sid, from_number, recording_url)

    if not recording_url or not call_sid:
        resp = VoiceResponse()
        resp.say("We couldn't process your recording. Please try again later.", voice="alice")
        return Response(content=str(resp), media_type="text/xml", status_code=200)

    background_tasks.add_task(process_recording_background, call_sid, recording_url, from_number)

    # Redirect to /hold to keep call alive
    resp = VoiceResponse()
    hold_url = str(request.url_for("hold", convo_id=call_sid))
    resp.redirect(hold_url)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- background pipeline ----------------
async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    logger.info("[%s] Background processing started", call_sid)
    try:
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in download_url else None

        # Download
        r = requests.get(download_url, auth=auth, timeout=30)
        r.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp३")
        tmp.write(r.content)
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] Recording saved to %s", call_sid, file_path)

        # Transcribe
        transcript = ""
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] Transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] Transcription failed", call_sid)
        finally:
            os.unlink(file_path)

        # Agent
        agent_out = call_agent_and_get_reply(call_sid, transcript or " ")
        reply_text = agent_out.get("reply_text", "Sorry, I didn't understand.")
        memory_writes = agent_out.get("memory_writes", [])
        logger.info("[%s] Reply: %s", call_sid, reply_text[:200])

        # Memory writes
        for mw in memory_writes:
            try:
                if callable(write_fact):
                    write_fact(mw)
            except Exception:
                logger.exception("Memory write failed")

        # TTS
        tts_url = None
        try:
            tts_url = create_and_upload_tts(reply_text)
            logger.info("[%s] TTS uploaded: %s", call_sid, tts_url)
        except Exception:
            logger.exception("[%s] TTS failed, using <Say>", call_sid)

        # Save to hold_store
        hold_store.set_ready(call_sid, {"tts_url": tts_url, "reply_text": reply_text})
        logger.info("[%s] Hold ready", call_sid)

    except Exception as e:
        logger.exception("[%s] Background error: %s", call_sid, e)
        hold_store.set_ready(call_sid, {"tts_url": nNone, "reply_text": "Sorry, something went wrong."})

# ---------------- /hold endpoint ----------------
@app.get("/hold")
@app.post("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    try:
        ready = hold_store.get_ready(convo_id)
        resp = VoiceResponse()

        if ready:
            tts_url = _unescape_url(ready.get("tts_url"))
            if tts_url:
                resp.play(tts_url)
            else:
                resp.say(ready.get("reply_text", "Sorry, I couldn't respond."), voice="alice")
            resp.record(max_length=30, action=recording_callback_url(), play_beep=True, timeout=2)
            return Response(content=str(resp), media_type="text/xml")

        # Not ready: hold loop
        base = str(request.base_url).rstrip("/")
        hold_path = str(request.url_for("hold", convo_id=convo_id))
        redirect_url = f"{base}{hold_path}"

        resp.say("Please hold while I prepare your response.", voice="alice")
        resp.pause(length=8)
        resp.redirect(redirect_url)
        return Response(content=str(resp), media_type="text/xml")

    except Exception as e:
        logger.exception("Hold handler error: %s", e)
        resp = VoiceResponse()
        resp.say("An error occurred. We'll call you back.", voice="alice")
        return Response(content=str(resp), media_type="text/xml")

# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME or "local"}

@app.get("/debug/ping")
async def debug_ping():
    return {"ok": True, "service": "agent-server-ms", "ts": time.time()}