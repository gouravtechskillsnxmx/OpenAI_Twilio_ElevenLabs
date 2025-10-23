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
    resp.record(max_length=6, action=action, play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- recording webhook ----------------
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
        return JSONResponse({"error": "missing RecordingUrl or CallSid"}, status_code=400)

    # ACK quickly and process in background
    background_tasks.add_task(process_recording_background, call_sid, recording_url, from_number)
    return Response(status_code=204)

# ---------------- background pipeline ----------------
import os, tempfile, requests, logging
from requests.auth import HTTPBasicAuth
from typing import Optional


async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    """
    Background pipeline:
    1) Download recording (with Twilio basic auth if Twilio URL)
    2) Transcribe via OpenAI
    3) Call agent (or fallback)
    4) Handle memory writes
    5) Generate TTS, upload to S3
    6) Safely update Twilio call or fallback to outbound call
    """
    logger.info("[%s] background start - download_url=%s", call_sid, recording_url)

    try:
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (
                                              
            TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in download_url
        ) else None

        # 1️⃣ Download recording
        try:
            r = requests.get(download_url, auth=auth, timeout=30)
            r.raise_for_status()
        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            logger.error("[%s] Download HTTP error %s: %s", call_sid, status, getattr(he.response, "text", str(he))[:400])
                            
            fallback_twiml = "<Response><Say>Sorry, we couldn't get your audio right now. Please try again later.</Say></Response>"
            _safe_update_or_call(call_sid, from_number, fallback_twiml)
            return
        except Exception as e:
            logger.exception("[%s] Download failed: %s", call_sid, e)
            return

        # 2️⃣ Save temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] saved recording to %s", call_sid, file_path)

        # 3️⃣ Transcribe
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] STT/transcription failed: %s", call_sid, e)
            transcript = ""

        # 4️⃣ Agent call
        try:
            agent_out = call_agent_and_get_reply(call_sid, transcript)
                              
            reply_text = (agent_out.get("reply_text") if isinstance(agent_out, dict) else str(agent_out)) or ""
            memory_writes = agent_out.get("memory_writes") if isinstance(agent_out, dict) else []
        except Exception as e:
            logger.exception("[%s] agent call failed: %s", call_sid, e)
            reply_text = "Sorry, I'm having trouble right now."
            memory_writes = []

        logger.info("[%s] assistant reply: %s", call_sid, reply_text[:300] + ("..." if len(reply_text) > 300 else ""))

        # 5️⃣ Write memory facts if any
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

        # 6️⃣ TTS & upload
        try:
            tts_url = create_and_upload_tts(reply_text)
            logger.info("[%s] tts_url: %s", call_sid, tts_url)
        except Exception as e:
            logger.exception("[%s] TTS/upload failed: %s", call_sid, e)
                                    
            twiml = f"<Response><Say>{reply_text}</Say><Record maxLength='6' action='https://{HOSTNAME}/recording' playBeep='true' timeout='2'/></Response>"
                
                                                                
                                                 
                                  
                                                                 
                                                                                                                                   
            _safe_update_or_call(call_sid, from_number, twiml)
                             
                                                                                
            return

        # 7️⃣ Build TwiML: Play + Record again
        record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"
        twiml = f"""<Response>
            <Play>{tts_url}</Play>
            <Record maxLength="6" action="{record_action}" playBeep="true" timeout="2" />
        </Response>"""

        logger.info("[%s] updating Twilio call with Play+Record", call_sid)
        _safe_update_or_call(call_sid, from_number, twiml)
                             
    except Exception as e:
        logger.exception("[%s] Unexpected pipeline error: %s", call_sid, e)


def _safe_update_or_call(call_sid: str, from_number: Optional[str], twiml: str):
    """
    Safely update an in-progress Twilio call or, if ended, create a new outbound call.
    Prevents Twilio 400/21220 errors ("Call is not in-progress. Cannot redirect.")
    This version also validates any <Play> URL is reachable before attempting to redirect,
    and replaces the TwiML with a friendly fallback if the TTS URL is not fetchable.
    """
    try:
        # If twiml contains a <Play> tag, extract the URL and verify it's reachable by Twilio
        play_url = None
        try:
            lower = twiml.lower()
            if "<play>" in lower and "</play>" in lower:
                # very small, conservative extraction (not XML parser to keep it simple)
                start = lower.index("<play>") + len("<play>")
                end = lower.index("</play>", start)
                play_url = twiml[start:end].strip()
                logger.info("[%s] extracted Play URL for verification: %s", call_sid, play_url)
        except Exception:
            play_url = None

        head_ok = True
        if play_url:
            try:
                # HEAD to verify resource is reachable (short timeouts)
                h = requests.head(play_url, timeout=(3, 8))
                head_ok = (h.status_code == 200)
                logger.info("[%s] HEAD %s -> %s", call_sid, play_url, h.status_code)
            except Exception:
                head_ok = False
                logger.exception("[%s] Error while HEADing Play URL %s", call_sid, play_url)

        # If the Play URL is present but not reachable, replace twiml with a friendly fallback message.
        if play_url and not head_ok:
            logger.warning("[%s] Play URL not reachable; using friendly fallback TwiML instead of technical error.", call_sid)
            twiml = "<Response><Say>Sorry — I couldn't prepare your voice response right now. We'll call you back shortly.</Say></Response>"

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

        # If call isn't in-progress, attempt to create outbound call that plays the twiml (which may be Play or friendly Say)
        if from_number and TWILIO_FROM:
            try:
                new_call = twilio_client.calls.create(
                    to=from_number,
                    from_=TWILIO_FROM,
                    twiml=twiml,
                )
                logger.info("[%s] Created fallback outbound call %s", call_sid, new_call.sid)
                return
            except Exception:
                logger.exception("[%s] Failed to create fallback outbound call", call_sid)

        logger.warning("[%s] No viable path to update or outbound-call; giving up.", call_sid)

    except Exception as e:
        logger.exception("[%s] safe_update_or_call failed: %s", call_sid, e)

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
