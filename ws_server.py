# ws_server.py
import os
import sys
import time
import json
import logging
import tempfile
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from requests.auth import HTTPBasicAuth
import requests
import boto3

# OpenAI new client
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# Twilio REST client
from twilio.rest import Client as TwilioClient

# memory module & API routes (from files you already added)
# Ensure db.py, models.py, memory.py, memory_api.py are in the repo
from memory import write_fact  # function to write facts to Postgres + audit
from memory_api import router as memory_router

# ---------------- CONFIG / ENV ----------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM","+15312303465")  # optional for calling out
OPENAI_KEY = os.environ.get("OPENAI_KEY")
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT")  # your established ChatGPT agent endpoint (optional)
AGENT_KEY = os.environ.get("AGENT_KEY")

ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

HOSTNAME = os.environ.get("HOSTNAME", "")  # public hostname used to generate callback URLs
TTS_PRESIGNED_EXPIRES = int(os.environ.get("TTS_PRESIGNED_EXPIRES", "3600"))

# Optional: throttling / simple config
CALLS_PER_MINUTE_PER_NUMBER = int(os.environ.get("CALLS_PER_MINUTE_PER_NUMBER", "5"))

# ---------------- clients & setup ----------------
logger = logging.getLogger("ai-sales-agent")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Sales Agent")

# include memory API (read/write/forget/audit endpoints)
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

# hold_store helper — uses Redis if REDIS_URL env var present, else in-memory (single-process only).
import os
import json
from typing import Optional

REDIS_URL = os.environ.get("REDIS_URL")

if REDIS_URL:
    import redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    class HoldStoreRedis:
        @staticmethod
        def set_ready(convo_id: str, payload: dict, expire: int = 3600):
            redis_client.set(f"hold:{convo_id}", json.dumps(payload), ex=expire)
        @staticmethod
        def get_ready(convo_id: str) -> Optional[dict]:
            v = redis_client.get(f"hold:{convo_id}")
            return json.loads(v) if v else None
        @staticmethod
        def clear(convo_id: str):
            redis_client.delete(f"hold:{convo_id}")

    hold_store = HoldStoreRedis()
else:
    # WARNING: in-memory only works for single-process, non-production
    _HOLD_MEM = {}
    class HoldStoreMem:
        @staticmethod
        def set_ready(convo_id: str, payload: dict, expire: int = 3600):
            _HOLD_MEM[convo_id] = payload
        @staticmethod
        def get_ready(convo_id: str) -> Optional[dict]:
            return _HOLD_MEM.get(convo_id)
        @staticmethod
        def clear(convo_id: str):
            _HOLD_MEM.pop(convo_id, None)
    hold_store = HoldStoreMem()


# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME}

# ---------------- helpers ----------------

from fastapi import FastAPI
app = app  # assume your FastAPI app is named app

@app.get("/debug/ping")
async def debug_ping():
    return {"ok": True, "service": "agent-server-ms", "ts": __import__("time").time()}

def recording_callback_url() -> str:
    if HOSTNAME:
        return f"https://{HOSTNAME}/recording"
    return "/recording"

def build_download_url(recording_url: str) -> str:
    """
    If Twilio recording URL doesn't end with extension, append .mp3 for Twilio.
    If the provided URL already has audio extension, use as is.
    """
    if not recording_url:
        return recording_url
    lower = recording_url.lower()
    for ext in (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"):
        if lower.endswith(ext):
            return recording_url
    if "api.twilio.com" in lower:
        return recording_url + ".mp3"
    return recording_url

def create_tts_elevenlabs(text: str) -> bytes:
    """
    Synchronous ElevenLabs TTS; returns audio bytes (mp3).
    Adapt headers/body if ElevenLabs changes their API.
    """
    if not ELEVEN_API_KEY or not ELEVEN_VOICE:
        raise RuntimeError("ElevenLabs not configured (ELEVEN_API_KEY/ELEVEN_VOICE).")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY,
    }
    body = {"text": text, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
    r = requests.post(url, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content

def create_and_upload_tts(text: str, expires_in: int = TTS_PRESIGNED_EXPIRES) -> str:
    """
    1) Generate audio bytes via ElevenLabs
    2) Upload to S3 (no ACL - bucket owner enforced support)
    3) Return presigned GET URL
    """
    audio_bytes = create_tts_elevenlabs(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()

    key = f"tts/{os.path.basename(tmp_name)}"
    try:
        s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
    except Exception:
        logger.exception("S3 upload failed for %s/%s", S3_BUCKET, key)
        raise
    presigned = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_in)
    return presigned

# ---------------- agent integration ----------------
def call_agent_and_get_reply(convo_id: str, user_text: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Robust agent caller:
      - If AGENT_ENDPOINT is set, POST to it with Authorization: Bearer AGENT_KEY (if AGENT_KEY set).
      - Accepts multiple possible shapes from agent:
          { "reply": "..."} OR {"reply_text":"..."} OR {"text":"..."} OR {"reply_text":"...", "memory_writes":[...]}
      - Returns dict: {"reply_text": str, "memory_writes": list}
      - On network/agent error, falls back to calling OpenAI directly if openai_client is configured.
    """
    # 1) Prefer external agent microservice
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
            # normalize reply_text
            reply_text = j.get("reply_text") or j.get("reply") or j.get("text") or j.get("replyText") or ""
            memory_writes = j.get("memory_writes") or j.get("memoryWrites") or []
            # ensure types
            if not isinstance(memory_writes, list):
                memory_writes = []
            return {"reply_text": reply_text, "memory_writes": memory_writes}
        except requests.exceptions.RequestException as e:
            logger.exception("Agent endpoint request failed: %s", e)
        except ValueError:
            logger.exception("Agent endpoint returned invalid JSON")
        except Exception:
            logger.exception("Unexpected error calling agent endpoint")

    # 2) Fallback to local OpenAI client if available
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
                # prefer message content
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

    # 3) Last resort echo
    logger.warning("No agent and no OpenAI client available for convo=%s", convo_id)
    return {"reply_text": f"Echo: {user_text}", "memory_writes": []}

# ---------------- transcription ----------------
def transcribe_with_openai(file_path: str) -> str:
    """
    Uses OpenAI's audio transcription (whisper) with the new OpenAI client if available.
    """
    if openai_client is None:
        raise RuntimeError("OpenAI not configured (OPENAI_KEY missing or client not available).")
    with open(file_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
    # resp might be object-like or dict-like
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None) or str(resp)
    return (text or "").strip()

# ---------------- TwiML endpoint ----------------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    resp = VoiceResponse()
    resp.say("Hello, this is our AI assistant. Please say something after the beep.", voice="alice")
    action = recording_callback_url()
    # short record for responsive loop
    resp.record(max_length=30, action=action, play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- recording webhook ----------------
from fastapi import Request, Response
from twilio.twiml.voice_response import VoiceResponse
from urllib.parse import urlencode

from fastapi import Query
from twilio.twiml.voice_response import VoiceResponse

from fastapi import Response

@app.post("/hold")
@app.get("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    """
    If TTS is ready for convo_id -> return Play+Record (final TwiML).
    If not ready -> pause then redirect back to /hold (loop).
    """
    ready = hold_store.get_ready(convo_id)
    if ready:
        tts_url = ready.get("tts_url")
        final_twiml = ready.get("twiml")
        resp = VoiceResponse()
        if tts_url:
            resp.play(tts_url)
            resp.record(
                max_length=20,
                action=f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording",
                play_beep=True,
                timeout=2,
            )
        elif final_twiml:
            return Response(content=final_twiml, media_type="text/xml")
        else:
            resp.say("Sorry — something went wrong preparing the reply.", voice="alice")
        hold_store.clear(convo_id)
        return Response(content=str(resp), media_type="text/xml")

    # Not ready — pause + redirect loop
    resp = VoiceResponse()
    resp.pause(length=5)
    redirect_url = f"https://{HOSTNAME}/hold?convo_id={convo_id}" if HOSTNAME else f"/hold?convo_id={convo_id}"
    resp.redirect(redirect_url)
    return Response(content=str(resp), media_type="text/xml")

from fastapi import Request, Response
from twilio.twiml.voice_response import VoiceResponse
from urllib.parse import urlencode

@app.post("/recording")
async def recording(request: Request):
    """
    Twilio Record action handler:
    - Immediately respond with TwiML that redirects caller to /hold?convo_id=...
    - Prevents Twilio from playing 'application error' and keeps caller alive.
    """
    form = await request.form()
    convo_id = form.get("CallSid") or form.get("convo_id")
    params = {"convo_id": convo_id}
    redirect_url = f"https://{HOSTNAME}/hold?{urlencode(params)}" if HOSTNAME else f"/hold?{urlencode(params)}"

    resp = VoiceResponse()
    resp.say("Thanks — please hold while I prepare your response.", voice="alice")
    resp.redirect(redirect_url)
    return Response(content=str(resp), media_type="text/xml")


# ---------------- background pipeline ----------------
import os, tempfile, requests, logging
from requests.auth import HTTPBasicAuth
from typing import Optional


async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    """
    Background pipeline (updated TwiML construction to avoid malformed XML/application error):
    - identical flow to your previous function, but builds TwiML using VoiceResponse()
      to ensure valid XML and proper escaping for <Say>/<Play>.
    """
    logger.info("[%s] background start - download_url=%s", call_sid, recording_url)

    try:
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (
            TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in (download_url or "")
        ) else None

        # 1) Download recording
        try:
            r = requests.get(download_url, auth=auth, timeout=30)
            r.raise_for_status()
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            logger.error("[%s] Download HTTP error %s: %s", call_sid, status,
                         getattr(he.response, "text", str(he))[:400])
            # Friendly fallback TwiML
            resp = VoiceResponse()
            resp.say("Sorry, we couldn't get your audio right now. Please try again later.", voice="alice")
            # Use the safe update/helper to update or create outbound call
            _safe_update_or_call(call_sid, from_number, str(resp))
            return
        except Exception as e:
            logger.exception("[%s] Download failed: %s", call_sid, e)
            return

        # 2) Save audio to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] saved recording to %s", call_sid, file_path)

        # 3) Transcribe
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] STT/transcription failed: %s", call_sid, e)
            transcript = ""

        # 4) Agent call
        try:
            agent_out = call_agent_and_get_reply(call_sid, transcript)
            reply_text = (agent_out.get("reply_text") if isinstance(agent_out, dict) else str(agent_out)) or ""
            memory_writes = agent_out.get("memory_writes") if isinstance(agent_out, dict) else []
        except Exception as e:
            logger.exception("[%s] agent call failed: %s", call_sid, e)
            reply_text = "Sorry, I'm having trouble right now."
            memory_writes = []

        logger.info("[%s] assistant reply (truncated): %s", call_sid, (reply_text[:300] + "...") if len(reply_text) > 300 else reply_text)

        # 5) Persist memory writes (unchanged)
        if memory_writes and isinstance(memory_writes, list):
            for mw in memory_writes:
                try:
                    fact_key = mw.get("fact_key") if isinstance(mw, dict) else None
                    content = mw.get("content") if isinstance(mw, dict) else {"value": mw}
                    created_by = f"voice:{from_number or call_sid}"
                    written = write_fact(content, created_by=created_by, fact_key=fact_key)
                    logger.info("[%s] wrote memory fact id=%s key=%s", call_sid, written.get("id"), fact_key)
                except Exception:
                    logger.exception("[%s] failed to write memory: %s", call_sid, mw)

        # 6) TTS & upload
        tts_url = None
        try:
            tts_url = create_and_upload_tts(reply_text)
            # after tts_url is available, set hold_store so /hold will serve it
            hold_store.set_ready(call_sid, {"tts_url": tts_url})
            logger.info("[%s] tts_url: %s", call_sid, tts_url)
        except Exception as e:
            logger.exception("[%s] TTS/upload failed: %s", call_sid, e)
            # Fallback: Say the reply_text (safely) then Record
            resp = VoiceResponse()
            # Limit length to avoid Twilio truncation / invalid TwiML
            safe_text = (reply_text or "Sorry, I'm having trouble right now.").strip()
            # truncate to 300 chars for Say to keep TwiML safe
            if len(safe_text) > 300:
                safe_text = safe_text[:297] + "..."
            resp.say(safe_text, voice="alice")
            resp.record(max_length=6, action=recording_callback_url(), play_beep=True, timeout=2)
            _safe_update_or_call(call_sid, from_number, str(resp))
            return

        # 7) Build TwiML safely with VoiceResponse: Play presigned URL then Record
        resp = VoiceResponse()
        # Play the presigned tts_url (already validated elsewhere). Using resp.play ensures proper escaping.
        resp.play(tts_url)
        resp.record(max_length=6, action=recording_callback_url(), play_beep=True, timeout=2)
        twiml_str = str(resp)

        logger.info("[%s] updating Twilio call with Play+Record", call_sid)
        _safe_update_or_call(call_sid, from_number, twiml_str)

    except Exception as e:
        logger.exception("[%s] Unexpected pipeline error: %s", call_sid, e)


def _safe_update_or_call(call_sid: str, from_number: Optional[str], twiml: str):
    """
    Safely update an in-progress Twilio call or, if ended, create a new outbound call.
    - Unescape HTML entities in extracted Play URL (fix &amp; -> &).
    - Validate Play URL via a small ranged GET (accept 200 or 206).
    - If Play URL invalid, swap to a friendly <Say> twiml (no technical wording).
    - When creating outbound call, wrap TwiML safely with VoiceResponse and include status_callback.
    """
    try:
        play_url = None
        url_ok = True

        # Extract <Play> content (conservative string search) and unescape HTML entities
        try:
            lower = twiml.lower()
            if "<play>" in lower and "</play>" in lower:
                start = lower.index("<play>") + len("<play>")
                end = lower.index("</play>", start)
                play_url = twiml[start:end].strip()
                import html as _html
                play_url = _html.unescape(play_url)
                # trim any accidental quotes
                if play_url.startswith('"') and play_url.endswith('"'):
                    play_url = play_url[1:-1]
                logger.info("[%s] extracted/unescaped Play URL for verification: %s", call_sid, play_url)
        except Exception:
            play_url = None

        # Verify Play URL quickly with a small ranged GET (works when HEAD may be blocked)
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

        # If Play URL exists but is not reachable, replace twiml with a friendly fallback Say
        if play_url and not url_ok:
            logger.warning("[%s] Play URL unreachable; using friendly fallback TwiML.", call_sid)
            twiml = "<Response><Say>Sorry — I couldn't generate your voice response right now. We'll call you back shortly.</Say></Response>"

        # Try fetching call status to decide if redirect allowed
        call = None
        try:
            if twilio_client:
                call = twilio_client.calls(call_sid).fetch()
                logger.info("[%s] fetched call status=%s", call_sid, getattr(call, "status", None))
        except Exception as e:
            # log details but continue to fallback behavior
            logger.exception("[%s] Failed to fetch call status: %s", call_sid, e)

        # If call is in-progress (or ringing/queued), attempt to update it
        if call and getattr(call, "status", "").lower() in ("in-progress", "ringing", "queued"):
            try:
                twilio_client.calls(call_sid).update(twiml=twiml)
                logger.info("[%s] Successfully updated live call.", call_sid)
                return
            except Exception as e:
                logger.exception("[%s] Failed to update live call; will attempt outbound fallback. Error: %s", call_sid, e)

        # If cannot update live call or call is completed, create an outbound fallback call (if we have a number)
        if from_number and TWILIO_FROM:
            try:
                # small delay to avoid racing Twilio teardown/recreate edge cases
                import time as _time
                _time.sleep(0.5)

                # status_callback so we can inspect Twilio lifecycle (helpful for debugging)
                status_cb = f"https://{HOSTNAME}/call_status" if HOSTNAME else None

                # Build safe VoiceResponse for the outbound call:
                resp = VoiceResponse()
                # If original twiml had a Play and that URL was validated OK, include it.
                if play_url and url_ok:
                    # optional short message first then play
                    resp.say("Here's the information you asked for.", voice="alice")
                    resp.play(play_url)
                else:
                    # Use friendly Say fallback so Twilio won't try to fetch a broken/forbidden URL
                    # Try to extract a short textual reply from twiml if possible (very simple attempt)
                    short_text = None
                    try:
                        # look for a <Say> in the twiml and use that text (unescape if found)
                        low = twiml.lower()
                        if "<say>" in low and "</say>" in low:
                            s = low.index("<say>") + len("<say>")
                            e = low.index("</say>", s)
                            short_text = twiml[s:e].strip()
                            import html as _html
                            short_text = _html.unescape(short_text)
                    except Exception:
                        short_text = None

                    if short_text:
                        # truncate to safe length
                        st = short_text.strip()
                        if len(st) > 300:
                            st = st[:297] + "..."
                        resp.say(st, voice="alice")
                    else:
                        resp.say("Sorry — I couldn't generate your voice response right now. We'll call you back shortly.", voice="alice")

                # Create outbound call with the constructed safe TwiML
                created = twilio_client.calls.create(
                    to=from_number,
                    from_=TWILIO_FROM,
                    twiml=str(resp),
                    status_callback=status_cb,
                    status_callback_event=["initiated", "ringing", "answered", "completed"] if status_cb else None
                )
                logger.info("[%s] Created fallback outbound call %s", call_sid, getattr(created, "sid", created))
                return
            except Exception as e:
                logger.exception("[%s] Failed to create fallback outbound call: %s", call_sid, e)

        logger.warning("[%s] No viable path to update or create outbound call; giving up.", call_sid)

    except Exception as e:
        logger.exception("[%s] safe_update_or_call unexpected error: %s", call_sid, e)

# ---------------- optional outbound call endpoint ----------------
@app.post("/call_outbound")
async def call_outbound(request: Request):
    """
    Trigger an outbound call from Twilio => to_number.
    Must set TWILIO_FROM env. Useful to avoid dialing Twilio number from an international mobile.
    body form: to_number=+91...."
    """
    form = await request.form()
    to_number = form.get("to_number")
    if not to_number:
        return JSONResponse({"error": "to_number required"}, status_code=400)
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
        return JSONResponse({"error": "twilio credentials or TWILIO_FROM missing"}, status_code=500)
    twiml_url = f"https://{HOSTNAME}/twiml" if HOSTNAME else None
    try:
        call = twilio_client.calls.create(to=to_number, from_=TWILIO_FROM, url=twiml_url)
        return {"call_sid": call.sid}
    except Exception as e:
        logger.exception("Failed to create outbound call to %s: %s", to_number, e)
        return JSONResponse({"error": "call failed"}, status_code=500)
