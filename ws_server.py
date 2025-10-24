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
if OPENAI_KEY and OpenAIClient is not None:
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
    """
    Robust hold store. Uses Redis if available. Otherwise uses per-process memory and a file fallback
    under HOLD_STORE_DIR so other processes on the same host can read the ready payload.
    """

    @staticmethod
    def set_ready(convo_id: str, payload: dict, expire: int = 3600) -> bool:
        """
        Set ready payload. Returns True if stored in Redis (preferred), False if fallback used.
        """
        try:
            if redis_client:
                try:
                    # store JSON string, set expiry
                    redis_client.set(f"hold:{convo_id}", json.dumps(payload), ex=expire)
                    logger.info("Redis: set hold:%s", convo_id)
                    return True
                except Exception:
                    logger.exception("Redis set failed for hold:%s -- falling back", convo_id)

            # fallback: in-memory and file (durable on same host)
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
        """
        Try to fetch the ready payload:
          1) Redis (and delete key so it's one-shot)
          2) In-memory (pop so it's one-shot)
          3) File fallback: /tmp/hold_store/<convo_id>.json read & delete
        Returns parsed dict or None.
        """
        # 1) Redis
        try:
            if redis_client:
                try:
                    v = redis_client.get(f"hold:{convo_id}")
                    if v:
                        try:
                            data = json.loads(v)
                        except Exception:
                            logger.exception("Failed parsing Redis hold value for %s", convo_id)
                            data = None
                        # try delete key (one-shot)
                        try:
                            redis_client.delete(f"hold:{convo_id}")
                        except Exception:
                            pass
                        if data is not None:
                            logger.info("Redis: got and cleared hold:%s", convo_id)
                            return data
                    else:
                        logger.debug("Redis: no hold:%s", convo_id)
                except Exception:
                    logger.exception("Redis get failed for hold:%s", convo_id)
        except Exception:
            logger.exception("Unexpected error checking Redis for hold:%s", convo_id)

        # 2) In-memory (same-process)
        try:
            if convo_id in _hold_in_memory:
                data = _hold_in_memory.pop(convo_id)
                logger.info("In-memory: popped hold for %s", convo_id)
                # cleanup file if present
                try:
                    p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
                return data
        except Exception:
            logger.exception("In-memory fallback failed for %s", convo_id)

        # 3) File fallback (works across processes on same host)
        try:
            p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
            if p.exists():
                try:
                    raw = p.read_text()
                    data = json.loads(raw)
                    # one-shot: remove file once read
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    logger.info("File fallback: read & cleared %s", str(p))
                    return data
                except Exception:
                    logger.exception("Failed parsing/reading file fallback for %s", convo_id)
        except Exception:
            logger.exception("File fallback check failed for %s", convo_id)

        return None

    @staticmethod
    def clear(convo_id: str):
        try:
            if redis_client:
                try:
                    redis_client.delete(f"hold:{convo_id}")
                except Exception:
                    logger.exception("Redis delete failed for hold:%s", convo_id)
            _hold_in_memory.pop(convo_id, None)
            try:
                p = Path(HOLD_STORE_DIR) / f"{convo_id}.json"
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        except Exception:
            logger.exception("Unexpected error clearing hold for %s", convo_id)

hold_store = HoldStore()

# ---------------- helpers ----------------
def recording_callback_url() -> str:
    if HOSTNAME:
        return f"https://{HOSTNAME}/recording"
    return "/recording"

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
    tmp_name = tmp.name
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    key = f"tts/{os.path.basename(tmp_name)}"
    s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
    presigned = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_in)
    return presigned

# ---------------- agent integration ----------------
def call_agent_and_get_reply(convo_id: str, user_text: str, timeout: int = 15) -> Dict[str, Any]:
    # Attempt external agent
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
        except requests.exceptions.RequestException as e:
            logger.exception("Agent endpoint request failed: %s", e)
        except ValueError:
            logger.exception("Agent endpoint returned invalid JSON")
        except Exception:
            logger.exception("Unexpected error calling agent endpoint")

    # Fallback to OpenAI (if configured)
    if openai_client:
        try:
            logger.info("Falling back to OpenAI directly for convo=%s", convo_id)
            chat_resp = openai_client.chat.completions.create(
                model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": "You are a helpful voice assistant."}, {"role": "user", "content": user_text}],
                max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "256"))
            )
            # extract message content robustly
            content = ""
            try:
                choice = chat_resp.choices[0]
                if hasattr(choice, "message") and choice.message:
                    content = choice.message.get("content") if isinstance(choice.message, dict) else getattr(choice.message, "content", "")
                else:
                    content = getattr(choice, "text", "") or str(chat_resp)
            except Exception:
                content = str(chat_resp)
            return {"reply_text": (content or "").strip(), "memory_writes": []}
        except Exception:
            logger.exception("OpenAI chat fallback failed")
            return {"reply_text": "Sorry, I'm having trouble right now.", "memory_writes": []}

    # Last resort echo
    logger.warning("No agent and no OpenAI client available for convo=%s", convo_id)
    return {"reply_text": f"Echo: {user_text}", "memory_writes": []}

# ---------------- transcription ----------------
def transcribe_with_openai(file_path: str) -> str:
    if openai_client is None:
        raise RuntimeError("OpenAI not configured (OPENAI_KEY missing or client not available).")
    with open(file_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None) or str(resp)
    return (text or "").strip()

# ---------------- TwiML endpoints ----------------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    resp = VoiceResponse()
    resp.say("Hello, this is our AI assistant. Please say something after the beep.", voice="alice")
    action = recording_callback_url()
    resp.record(max_length=30, action=action, play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    payload = dict(form)
    recording_url = payload.get("RecordingUrl")
    call_sid = payload.get("CallSid")
    from_number = payload.get("From")
    logger.info("Incoming recording webhook: CallSid=%s From=%s RecordingUrl=%s", call_sid, from_number, recording_url)

    if not recording_url or not call_sid:
        logger.warning("Missing RecordingUrl or CallSid in request payload: %s", payload)
        # return TwiML to avoid Twilio "application error"
        resp = VoiceResponse()
        resp.say("We couldn't process your recording. Please try again later.", voice="alice")
        return Response(content=str(resp), media_type="text/xml", status_code=200)

    # Start background processing
    background_tasks.add_task(process_recording_background, call_sid, recording_url, from_number)

    # Return TwiML to redirect to /hold (keeps call active)
    resp = VoiceResponse()
    hold_url = request.url_for("hold") + f"?convo_id={call_sid}"
    resp.redirect(hold_url)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- background pipeline ----------------
async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    logger.info("[%s] background start - download_url=%s", call_sid, recording_url)
    try:
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in (download_url or "")) else None

        # 1) Download recording
        try:
            r = requests.get(download_url, auth=auth, timeout=30)
            r.raise_for_status()
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            logger.error("[%s] Download HTTP error %s: %s", call_sid, status, getattr(he.response, "text", str(he))[:400])
            resp = VoiceResponse()
            resp.say("Sorry, we couldn't get your audio right now. Please try again later.", voice="alice")
            _safe_update_or_call(call_sid, from_number, str(resp))
            return
        except Exception as e:
            logger.exception("[%s] Download failed: %s", call_sid, e)
            resp = VoiceResponse()
            resp.say("Sorry, we couldn't get your audio right now. Please try again later.", voice="alice")
            _safe_update_or_call(call_sid, from_number, str(resp))
            return

        # save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] saved recording to %s", call_sid, file_path)

        # transcribe
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] transcribe failed: %s", call_sid, e)
            transcript = ""

        # agent
        try:
            agent_out = call_agent_and_get_reply(call_sid, transcript)
            reply_text = (agent_out.get("reply_text") if isinstance(agent_out, dict) else str(agent_out)) or ""
            memory_writes = agent_out.get("memory_writes") if isinstance(agent_out, dict) else []
        except Exception as e:
            logger.exception("[%s] agent call failed: %s", call_sid, e)
            reply_text = "Sorry, I'm having trouble right now."
            memory_writes = []

        logger.info("[%s] assistant reply (truncated): %s", call_sid, (reply_text[:300] + "...") if len(reply_text) > 300 else reply_text)

        # persist memory writes
        if memory_writes and isinstance(memory_writes, list):
            for mw in memory_writes:
                try:
                    if callable(write_fact):
                        write_fact(mw)
                except Exception:
                    logger.exception("[%s] memory write failed: %s", call_sid, mw)

        # generate TTS and upload & set hold_store ready payload
        tts_url = None
        try:
            tts_url = create_and_upload_tts(reply_text)
            logger.info("[%s] TTS generated and uploaded: %s", call_sid, tts_url)
        except Exception:
            logger.exception("[%s] TTS generation/upload failed; falling back to Say", call_sid)

        payload = {"tts_url": tts_url, "reply_text": reply_text}
        hold_store.set_ready(call_sid, payload)
        logger.info("[%s] set hold ready", call_sid)
        return
    except Exception as e:
        logger.exception("[%s] Unexpected error in background pipeline: %s", call_sid, e)

# ---------------- safe_update_or_call (updated) ----------------
def _safe_update_or_call(call_sid: str, from_number: Optional[str], twiml: str):
    """
    Safely update an in-progress Twilio call or, if ended, create a new outbound call.
    Unescape HTML entities in extracted Play URL (e.g. &amp; -> &) before HTTP checks.
    """
    try:
        play_url = None
        try:
            lower = twiml.lower()
            if "<play>" in lower and "</play>" in lower:
                start = lower.index("<play>") + len("<play>")
                end = lower.index("</play>", start)
                play_url = twiml[start:end].strip()
                import html as _html
                play_url = _html.unescape(play_url)
                if play_url.startswith('"') and play_url.endswith('"'):
                    play_url = play_url[1:-1]
                logger.info("[%s] extracted/unescaped Play URL for verification: %s", call_sid, play_url)
        except Exception:
            play_url = None

        url_ok = True
        if play_url:
            try:
                headers = {"Range": "bytes=0-0"}
                rchk = requests.get(play_url, headers=headers, timeout=(3, 8), stream=True)
                status = getattr(rchk, "status_code", None)
                if status in (200, 206):
                    url_ok = True
                    logger.info("[%s] ranged-GET %s -> %s", call_sid, play_url, status)
                else:
                    url_ok = False
                    logger.warning("[%s] ranged-GET %s -> %s (not OK)", call_sid, play_url, status)
                try:
                    rchk.close()
                except Exception:
                    pass
            except Exception:
                url_ok = False
                logger.exception("[%s] Error during ranged GET for Play URL %s", call_sid, play_url)

        if play_url and not url_ok:
            logger.warning("[%s] Play URL unreachable; using friendly fallback TwiML.", call_sid)
            twiml = "<Response><Say>Sorry — I couldn't generate your voice response right now. We'll call you back shortly.</Say></Response>"

        # Fetch call status
        call = None
        try:
            if twilio_client:
                call = twilio_client.calls(call_sid).fetch()
                logger.info("[%s] fetched call status=%s", call_sid, getattr(call, "status", None))
        except Exception:
            logger.exception("[%s] Failed to fetch call status", call_sid)

        # If call is in-progress (or ringing/queued), update it
        if call and getattr(call, "status", "").lower() in ("in-progress", "ringing", "queued"):
            try:
                twilio_client.calls(call_sid).update(twiml=twiml)
                logger.info("[%s] Successfully updated live call.", call_sid)
                return
            except Exception:
                logger.exception("[%s] Failed to update live call; will attempt outbound fallback.", call_sid)

        # If cannot update live call, create outbound call. Use twiml (which may be Play or friendly Say).
        if from_number and TWILIO_FROM:
            try:
                created = twilio_client.calls.create(
                    to=from_number,
                    from_=TWILIO_FROM,
                    twiml=twiml
                )
                logger.info("[%s] Created fallback outbound call %s", call_sid, created.sid)
                return
            except Exception:
                logger.exception("[%s] Failed to create fallback outbound call", call_sid)

        logger.warning("[%s] No viable path to update or create outbound call; giving up.", call_sid)

    except Exception as e:
        logger.exception("[%s] safe_update_or_call unexpected error: %s", call_sid, e)

# ---------------- /hold endpoint (poll until ready) ----------------
# Replace your existing /hold handler with this function.
@app.get("/hold")
@app.post("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    """
    Polls hold_store for readiness.
      - If ready: return Play(tts_url) + Record (final TwiML), then clear store.
      - If not ready: Say a friendly message, Pause, then Redirect back to /hold (absolute URL).
    This handler always returns valid TwiML (no 500) so Twilio won't play "application error".
    """
    try:
        ready = None
        try:
            ready = hold_store.get_ready(convo_id)
        except Exception:
            logger.exception("Hold endpoint: error reading hold_store for %s", convo_id)
            ready = None

        resp = VoiceResponse()

        if ready:
            # If we have a ready payload, prefer the TTS URL; else fallback to reply_text via Say
            tts_url = _unescape_url(ready.get("tts_url")) if isinstance(ready, dict) else None
            if tts_url:
                resp.play(tts_url)
            else:
                reply = ready.get("reply_text") if isinstance(ready, dict) else None
                if reply:
                    resp.say(reply, voice="alice")
                else:
                    resp.say("Sorry, I couldn't generate your audio. We'll call you back shortly.", voice="alice")

            # After delivering, record follow-up and then end
            resp.record(max_length=30, action=recording_callback_url(), play_beep=True, timeout=2)

            # Best-effort clear (get_ready already pops it in most implementations)
            try:
                hold_store.clear(convo_id)
            except Exception:
                logger.exception("Failed clearing hold_store for %s", convo_id)

            return Response(content=str(resp), media_type="text/xml")

        # Not ready: tell the caller we're working, pause, then redirect back to the absolute /hold URL.
        # Build absolute redirect URL using request.base_url + request.url_for
        try:
            base = str(request.base_url).rstrip("/")
            hold_path = str(request.url_for("hold"))
            # Ensure single leading slash on hold_path
            if not hold_path.startswith("/"):
                hold_path = "/" + hold_path
            redirect_url = f"{base}{hold_path}?convo_id={convo_id}"
        except Exception:
            # Fallback: use HOSTNAME env if available or a relative path
            if HOSTNAME:
                redirect_url = f"https://{HOSTNAME}/hold?convo_id={convo_id}"
            else:
                redirect_url = f"/hold?convo_id={convo_id}"

        # Friendly message before redirect (reduces chance Twilio treats rapid re-requests as an error)
        resp.say("Please hold while I prepare your response. This may take a few seconds.", voice="alice")
        # Pause longer to avoid too-tight polling
        resp.pause(length=8)
        resp.redirect(redirect_url)
        return Response(content=str(resp), media_type="text/xml")

    except Exception as e:
        logger.exception("Unhandled error in /hold handler for %s: %s", convo_id, e)
        # Always return valid TwiML rather than raising 500
        resp = VoiceResponse()
        resp.say("Sorry — an application error has occurred. We'll call you back shortly.", voice="alice")
        return Response(content=str(resp), media_type="text/xml")

# ---------------- health/debug ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME or ""}

@app.get("/debug/ping")
async def debug_ping():
    return {"ok": True, "service": "agent-server-ms", "ts": time.time()}

# End of file