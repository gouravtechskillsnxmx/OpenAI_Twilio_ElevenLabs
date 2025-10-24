# ws_server.py
import os
import sys
import time
import json
import logging
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
import html

from fastapi import FastAPI, Request, Response, BackgroundTasks, Query
from twilio.twiml.voice_response import VoiceResponse
from requests.auth import HTTPBasicAuth
import requests
import boto3

# Optional OpenAI client
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# Twilio client
from twilio.rest import Client as TwilioClient

# Memory (optional)
try:
    from memory import write_fact
    from memory_api import router as memory_router
except Exception:
    memory_router = None

# ---------------- CONFIG ----------------
HOLD_STORE_DIR = "/tmp/hold_store"
Path(HOLD_STORE_DIR).mkdir(parents=True, exist_ok=True)

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "+15312303465")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT")
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

# ---------------- LOGGING & APP ----------------
logger = logging.getLogger("ai-sales-agent")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Sales Agent")
if memory_router:
    app.include_router(memory_router)

twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

openai_client = None
if OPENAI_KEY and OpenAIClient:
    try:
        openai_client = OpenAIClient(api_key=OPENAI_KEY)
    except Exception:
        logger.exception("Failed to init OpenAI client")

# S3 client
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
) if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY else boto3.client("s3", region_name=AWS_REGION)

# ---------------- HOLD STORE ----------------
_hold_in_memory: Dict[str, dict] = {}
redis_client = None
if REDIS_URL:
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning("Redis failed to init: %s", e)

class HoldStore:
    @staticmethod
    def set_ready(convo_id: str, payload: dict, expire: int = 3600) -> bool:
        """
        Set hold result. Returns True if stored in Redis successfully.
        Falls back to in-memory + file storage on error.
        """
        try:
            if redis_client:
                try:
                    redis_client.set(f"hold:{convo_id}", json.dumps(payload), ex=expire)
                    logger.info("Redis: set hold:%s", convo_id)
                    return True
                except Exception:
                    logger.exception("Redis set failed; falling back")
            # fallback to memory + file
            _hold_in_memory[convo_id] = payload
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            p.write_text(json.dumps(payload))
            logger.info("File fallback: wrote %s", p)
            return False
        except Exception as e:
            logger.exception("HoldStore.set_ready error: %s", e)
            return False

    @staticmethod
    def get_ready(convo_id: str) -> Optional[dict]:
        """
        Read & remove hold payload (Redis first, then memory, then file).
        """
        # Redis
        if redis_client:
            try:
                v = redis_client.get(f"hold:{convo_id}")
                if v:
                    data = json.loads(v)
                    try:
                        redis_client.delete(f"hold:{convo_id}")
                    except Exception:
                        pass
                    logger.info("Redis: got hold:%s", convo_id)
                    return data
            except Exception:
                logger.exception("Redis get failed; trying fallback")

        # In-memory
        if convo_id in _hold_in_memory:
            data = _hold_in_memory.pop(convo_id)
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
            logger.info("In-memory: popped hold:%s", convo_id)
            return data

        # File fallback
        p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                try:
                    p.unlink()
                except Exception:
                    pass
                logger.info("File: read hold:%s", convo_id)
                return data
            except Exception:
                logger.exception("File fallback read failed for %s", p)

        return None

    @staticmethod
    def clear(convo_id: str):
        try:
            if redis_client:
                try:
                    redis_client.delete(f"hold:{convo_id}")
                except Exception:
                    pass
            _hold_in_memory.pop(convo_id, None)
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

hold_store = HoldStore()

# ---------------- HELPERS ----------------
def recording_callback_url() -> str:
    return f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"

def build_download_url(url: str) -> str:
    if not url:
        return url
    lower = url.lower()
    for ext in (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"):
        if lower.endswith(ext):
            return url
    return url + ".mp3" if "api.twilio.com" in lower else url

def _unescape_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        u2 = html.unescape(u)
        return u2[1:-1] if u2.startswith('"') and u2.endswith('"') else u2
    except Exception:
        return u

def is_url_playable(url: Optional[str], timeout=(3, 8)) -> bool:
    """
    Quick ranged GET check to see if a remote audio URL is reachable and serves range requests.
    Accepts 200 or 206 as playable. Returns False on any error.
    """
    if not url:
        return False
    try:
        headers = {"Range": "bytes=0-0"}
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
        status = getattr(r, "status_code", None)
        try:
            r.close()
        except Exception:
            pass
        if status in (200, 206):
            return True
        logger.warning("is_url_playable -> status %s for %s", status, url)
        return False
    except Exception as e:
        logger.warning("is_url_playable error for %s: %s", url, e)
        return False

# ---------------- TTS ----------------
def create_tts_elevenlabs(text: str) -> bytes:
    """
    Call ElevenLabs and return audio bytes. If ElevenLabs is not configured, raises.
    """
    if not ELEVEN_API_KEY or not ELEVEN_VOICE:
        raise RuntimeError("ElevenLabs not configured")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"}
    data = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
    r = requests.post(url, json=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content

def create_and_upload_tts(text: str, expires_in: int = TTS_PRESIGNED_EXPIRES) -> Optional[str]:
    """
    Generate audio bytes via ElevenLabs (if configured) then upload to S3 and return presigned URL.
    If ElevenLabs is not available or returns 401/other error, return None to indicate fallback to text.
    """
    try:
        audio_bytes = create_tts_elevenlabs(text)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        logger.error("ElevenLabs TTS failed (status=%s). Falling back to text. Error: %s", status, e)
        return None
    except Exception:
        logger.exception("Unexpected error creating TTS; falling back to text-only")
        return None

    # Upload to S3
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        key = f"tts/{os.path.basename(tmp.name)}"
        try:
            s3.upload_file(tmp.name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
        except Exception:
            logger.exception("S3 upload failed for %s/%s", S3_BUCKET, key)
            # Best to remove temp file
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise
        # generate presigned URL
        presigned = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_in)
        return presigned
    finally:
        # best-effort cleanup of tmp file
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ---------------- AGENT ----------------
def call_agent_and_get_reply(convo_id: str, user_text: str, timeout: int = 15) -> Dict[str, Any]:
    if AGENT_ENDPOINT:
        try:
            headers = {"Content-Type": "application/json"}
            if AGENT_KEY:
                headers["Authorization"] = f"Bearer {AGENT_KEY}"
            payload = {"convo_id": convo_id, "text": user_text}
            r = requests.post(AGENT_ENDPOINT, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            reply = j.get("reply_text") or j.get("reply") or j.get("text") or ""
            memory = j.get("memory_writes") or j.get("memoryWrites") or []
            return {"reply_text": reply, "memory_writes": memory if isinstance(memory, list) else []}
        except Exception as e:
            logger.exception("Agent failed: %s", e)

    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=256
            )
            content = resp.choices[0].message.content if getattr(resp, "choices", None) else ""
            return {"reply_text": content.strip(), "memory_writes": []}
        except Exception:
            logger.exception("OpenAI failed")

    return {"reply_text": f"Echo: {user_text}", "memory_writes": []}

# ---------------- TRANSCRIPTION ----------------
def transcribe_with_openai(file_path: str) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI not configured")
    with open(file_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    return (getattr(resp, "text", "") or "").strip()

# ---------------- ENDPOINTS ----------------
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

    logger.info("Recording: CallSid=%s From=%s", call_sid, from_number)

    if not recording_url or not call_sid:
        resp = VoiceResponse()
        resp.say("Error processing recording.", voice="alice")
        return Response(content=str(resp), media_type="text/xml", status_code=200)

    # launch background
    background_tasks.add_task(process_recording_background, call_sid, recording_url, from_number)

    # Keep the caller on hold while we prepare the response.
    # We redirect to /hold; Twilio will fetch /hold and stay in a pause/redirect loop until
    # the background sets the hold payload.
    hold_url = f"{request.base_url}hold?convo_id={call_sid}"
    resp = VoiceResponse()
    resp.say("Please hold while I prepare your response.", voice="alice")
    resp.pause(length=5)
    resp.redirect(hold_url)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- BACKGROUND ----------------
async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    logger.info("[%s] Background start - download_url=%s", call_sid, recording_url)
    try:
        url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and "api.twilio.com" in url else None
        try:
            r = requests.get(url, auth=auth, timeout=30)
            r.raise_for_status()
        except Exception as e:
            logger.exception("[%s] Failed to download recording: %s", call_sid, e)
            # set safe fallback and return
            hold_store.set_ready(call_sid, {"tts_url": None, "reply_text": "Sorry, we couldn't retrieve your recording right now."})
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] saved recording to %s", call_sid, file_path)

        transcript = ""
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] STT/transcription failed: %s", call_sid, e)
            transcript = ""

        # cleanup audio file
        try:
            os.unlink(file_path)
        except Exception:
            pass

        # Agent call - expects structured dict
        try:
            agent_out = call_agent_and_get_reply(call_sid, transcript or " ")
            reply_text = (agent_out.get("reply_text") if isinstance(agent_out, dict) else str(agent_out)) or ""
            memory_writes = agent_out.get("memory_writes") if isinstance(agent_out, dict) else []
        except Exception as e:
            logger.exception("[%s] agent call failed: %s", call_sid, e)
            reply_text = "Sorry, I'm having trouble right now."
            memory_writes = []

        logger.info("[%s] assistant reply (truncated): %s", call_sid, (reply_text[:300] + "...") if len(reply_text) > 300 else reply_text)

        # If agent returned memory writes, persist them (best-effort)
        if memory_writes and isinstance(memory_writes, list):
            for mw in memory_writes:
                try:
                    if callable(write_fact):
                        write_fact(mw)
                except Exception:
                    logger.exception("[%s] failed to write memory write: %s", call_sid, mw)

        # TTS via ElevenLabs and S3 upload. If TTS not available/failed, tts_url will be None.
        tts_url = None
        try:
            candidate = create_and_upload_tts(reply_text)
            # unescape (in case S3 presigned URL contains &amp;)
            candidate_unescaped = _unescape_url(candidate) if candidate else None
            if candidate_unescaped and is_url_playable(candidate_unescaped):
                tts_url = candidate_unescaped
                logger.info("[%s] tts_url: %s", call_sid, tts_url)
            else:
                if candidate:
                    logger.warning("[%s] Created TTS but URL not playable; falling back to text reply", call_sid)
                tts_url = None
        except Exception as e:
            logger.exception("[%s] TTS generation/upload failed: %s", call_sid, e)
            tts_url = None

        # persist hold payload
        try:
            hold_store.set_ready(call_sid, {"tts_url": tts_url, "reply_text": reply_text})
            logger.info("[%s] Hold ready", call_sid)
        except Exception:
            logger.exception("[%s] Failed to set hold ready", call_sid)

        # If call has ended by now, create fallback outbound call to play response (best-effort).
        try:
            if twilio_client and from_number:
                call = twilio_client.calls(call_sid).fetch()
                status = getattr(call, "status", "").lower()
                if status not in ("in-progress", "queued", "ringing"):
                    # call ended, create outbound fallback to play the reply
                    twiml = None
                    if tts_url:
                        twiml = f"<Response><Play>{tts_url}</Play></Response>"
                    else:
                        safe_text = (reply_text or "Hello. I have a response for you.").replace("&", " and ")
                        twiml = f"<Response><Say>{safe_text}</Say></Response>"
                    try:
                        created = twilio_client.calls.create(to=from_number, from_=TWILIO_FROM, twiml=twiml)
                        logger.info("[%s] Created outbound fallback call %s", call_sid, getattr(created, "sid", "unknown"))
                    except Exception:
                        logger.exception("[%s] Failed creating fallback outbound call", call_sid)
        except Exception:
            # non-fatal
            pass

    except Exception as e:
        logger.exception("[%s] Unexpected pipeline error: %s", call_sid, e)
        hold_store.set_ready(call_sid, {"tts_url": None, "reply_text": "Sorry, something went wrong."})

# ---------------- HOLD ----------------
@app.get("/hold")
@app.post("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    """
    Twilio will repeatedly request /hold while the background prepares the response.
    When ready, we either <Play> the presigned S3 URL (if reachable) or <Say> the reply_text.
    """
    try:
        ready = hold_store.get_ready(convo_id)
        resp = VoiceResponse()

        if ready:
            tts_url = _unescape_url(ready.get("tts_url"))
            if tts_url and is_url_playable(tts_url):
                resp.play(tts_url)
            else:
                # Fallback to speak text (avoid empty text)
                txt = ready.get("reply_text", "")
                if not txt:
                    txt = "Sorry, I don't have an answer right now."
                resp.say(txt, voice="alice")
            # After playing/saying the reply, prompt for more (records again).
            resp.record(max_length=30, action=recording_callback_url(), play_beep=True, timeout=2)
            return Response(content=str(resp), media_type="text/xml")

        # Not ready -> keep caller on hold and redirect back to /hold so Twilio polls again.
        base = str(request.base_url).rstrip("/")
        redirect_url = f"{base}/hold?convo_id={convo_id}"

        resp.say("Please hold while I prepare your response.", voice="alice")
        # pause for a bit, then redirect back to /hold. Twilio will repeat until background sets the hold.
        resp.pause(length=8)
        resp.redirect(redirect_url)
        return Response(content=str(resp), media_type="text/xml")

    except Exception as e:
        logger.exception("Hold error: %s", e)
        resp = VoiceResponse()
        resp.say("An error occurred.", voice="alice")
        return Response(content=str(resp), media_type="text/xml")

# ---------------- HEALTH ----------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/debug/ping")
async def ping():
    return {"ok": True, "ts": time.time()}
