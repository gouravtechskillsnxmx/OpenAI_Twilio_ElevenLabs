# ws_server.py — Twilio TwiML + recording webhook + hold loop (FastAPI)

import os
import asyncio
import logging
import time
import urllib.parse
from typing import Optional, Dict

from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

# ---------- Config & Env ----------
try:
    from dotenv import load_dotenv  # optional convenience when running locally
    load_dotenv()
except Exception:
    pass

# NOTE: Per your setup, the correct env var is "OpenAI_Key".
# We also fall back to "OPENAI_API_KEY" for compatibility.
OPENAI_API_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE = os.getenv("ELEVEN_VOICE", "Xb7hH8MSUJpSbSDYk0k2")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM", "+15312303465")

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://openai-twilio-elevenlabs-realtime.onrender.com",
).rstrip("/")

# ---------- FastAPI ----------
app = FastAPI()

# ---------- In-memory state (de-bounce & turns) ----------
_state: Dict[str, Dict] = {}  # call_sid -> {"last_rec_sid": str, "last_at": float, "turn": int}
_state_lock = asyncio.Lock()

MIN_DURATION_SECONDS = 2            # ignore recordings shorter than this
RECORDING_COOLDOWN_SECONDS = 2.0    # ignore recordings that arrive too quickly

async def should_ignore_recording(call_sid: str,
                                  rec_sid: Optional[str],
                                  rec_duration: Optional[str]) -> bool:
    """
    Returns True if this recording should be ignored (too short, duplicate, too soon).
    """
    now = time.time()
    async with _state_lock:
        st = _state.get(call_sid, {"last_rec_sid": None, "last_at": 0.0, "turn": 0})

        # Too short?
        if rec_duration is not None:
            try:
                dur = int(rec_duration)
                if dur < MIN_DURATION_SECONDS:
                    logger.info("[%s] Ignoring recording: duration=%ss < %ss",
                                call_sid, dur, MIN_DURATION_SECONDS)
                    return True
            except Exception:
                pass

        # Duplicate RecordingSid?
        if rec_sid and st.get("last_rec_sid") == rec_sid:
            logger.info("[%s] Ignoring duplicate RecordingSid=%s", call_sid, rec_sid)
            return True

        # Too soon?
        if now - float(st.get("last_at", 0.0)) < RECORDING_COOLDOWN_SECONDS:
            logger.info("[%s] Ignoring recording: cooldown %.1fs not elapsed",
                        call_sid, RECORDING_COOLDOWN_SECONDS)
            return True

        # Acceptable — tentatively update last seen info
        if rec_sid:
            st["last_rec_sid"] = rec_sid
        st["last_at"] = now
        _state[call_sid] = st
        return False

# ---------- TwiML helpers ----------
def twiml(resp: VoiceResponse) -> Response:
    """
    Convert a twilio VoiceResponse into a FastAPI Response with proper media type.
    """
    return Response(content=str(resp), media_type="text/xml")

def recording_callback_url(request: Request) -> str:
    """
    Absolute callback URL for /recording on this service.
    """
    return f"{str(request.base_url).rstrip('/')}/recording"

# ---------- Simple in-memory hold store ----------
_hold: Dict[str, Dict] = {}
_hold_lock = asyncio.Lock()

# ---------- Friendly root / health ----------
@app.get("/")
async def root():
    return PlainTextResponse("ok: /twiml for Twilio, /health for health", status_code=200)

@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

# ---------- Initial TwiML (GET and POST) ----------
@app.get("/twiml")
@app.post("/twiml")
async def twiml_entry(request: Request):
    """
    Greet the caller, then record and POST the result to /recording.
    """
    try:
        vr = VoiceResponse()
        vr.say("Hello! Please leave your message after the beep.", voice="alice")
        vr.record(
            max_length=30,
            play_beep=True,
            timeout=3,
            action=recording_callback_url(request),
        )
        return twiml(vr)
    except Exception as e:
        logger.exception("twiml_entry error: %s", e)
        vr = VoiceResponse()
        vr.say("An application error has occurred. Goodbye.", voice="alice")
        vr.hangup()
        return twiml(vr)

# ---------- Recording webhook (accept POST and GET; plus OPTIONS to avoid 405) ----------
@app.options("/recording")
async def recording_options():
    # Some proxies pre-flight or probe; just say OK
    return PlainTextResponse("", status_code=200)

@app.post("/recording")
@app.get("/recording")
async def recording_webhook(
    request: Request,
    # Form (Twilio default for POST)
    CallSid: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    RecordingUrl: Optional[str] = Form(None),
    RecordingSid: Optional[str] = Form(None),
    RecordingDuration: Optional[str] = Form(None),
    # Query fallbacks (Twilio sometimes does GET with query params)
    q_CallSid: Optional[str] = Query(None, alias="CallSid"),
    q_From: Optional[str] = Query(None, alias="From"),
    q_RecordingUrl: Optional[str] = Query(None, alias="RecordingUrl"),
    q_RecordingSid: Optional[str] = Query(None, alias="RecordingSid"),
    q_RecordingDuration: Optional[str] = Query(None, alias="RecordingDuration"),
):
    method = request.method.upper()

    call_sid = CallSid or q_CallSid
    from_num = From or q_From
    rec_url  = RecordingUrl or q_RecordingUrl
    rec_sid  = RecordingSid or q_RecordingSid
    rec_dur  = RecordingDuration or q_RecordingDuration

    if not call_sid or not rec_url:
        logger.warning("Recording webhook(%s): missing CallSid or RecordingUrl", method)
        vr = VoiceResponse()
        vr.say("We did not get your message. Please try again.", voice="alice")
        vr.record(
            max_length=30,
            play_beep=True,
            timeout=3,
            action=recording_callback_url(request),
        )
        return twiml(vr)

    logger.info("Recording webhook(%s): CallSid=%s From=%s RecordingUrl=%s",
                method, call_sid, from_num, rec_url)

    # De-bounce noisy loops
    if await should_ignore_recording(call_sid, rec_sid, rec_dur):
        # Quietly re-arm the recorder without speaking again
        vr = VoiceResponse()
        vr.pause(length=1)
        vr.record(
            max_length=30,
            play_beep=True,
            timeout=3,
            action=recording_callback_url(request),
        )
        return twiml(vr)

    # Background pipeline (replace with your real STT->Agent->TTS)
    asyncio.create_task(process_recording_background(call_sid, rec_url, from_num))

    # Put caller into the hold loop
    vr = VoiceResponse()
    vr.say("Got it. Please hold while I prepare your response.", voice="alice")
    hold_url = f"{str(request.base_url).rstrip('/')}/hold?convo_id={urllib.parse.quote_plus(call_sid)}"
    vr.redirect(hold_url)
    return twiml(vr)

# ---------- Manual test helper to preload hold ----------
@app.post("/_test_set_hold")
async def test_set_hold(
    convo_id: str = Query(...),
    reply_text: str = Query("Hello from test"),
    tts_url: Optional[str] = Query(None),
):
    async with _hold_lock:
        _hold[convo_id] = {"reply_text": reply_text, "tts_url": tts_url}
    return {"ok": True, "convo_id": convo_id, "payload": _hold[convo_id]}

# ---------- Hold loop ----------
@app.get("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    """
    Twilio hits this repeatedly via <Redirect/> while your background task works.
    When a payload is available, we deliver it ONCE and immediately re-arm <Record>
    so the user can speak their next turn without hearing the greeting again.
    """
    try:
        # Pop once so we don't repeat
        async with _hold_lock:
            payload = _hold.pop(convo_id, None)

        if payload:
            reply_text = payload.get("reply_text") or "Here is your reply."
            tts_url = payload.get("tts_url")

            # Track the turn (for visibility/debug)
            async with _state_lock:
                st = _state.get(convo_id, {"turn": 0, "last_at": time.time()})
                st["turn"] = int(st.get("turn", 0)) + 1
                st["last_at"] = time.time()
                _state[convo_id] = st
                logger.info("[%s] Delivering reply (turn=%s)", convo_id, st["turn"])

            vr = VoiceResponse()
            if tts_url:
                vr.play(tts_url)
            else:
                vr.say(reply_text, voice="alice")

            # Re-arm recording for the next user turn (no extra greeting)
            vr.record(
                max_length=30,
                play_beep=True,
                timeout=3,
                action=recording_callback_url(request),
            )
            return twiml(vr)

        # Nothing ready yet — short wait + redirect back to /hold
        vr = VoiceResponse()
        vr.pause(length=2)
        vr.redirect(f"{str(request.base_url).rstrip('/')}/hold?convo_id={urllib.parse.quote_plus(convo_id)}")
        return twiml(vr)

    except Exception as e:
        logger.exception("Hold error: %s", e)
        vr = VoiceResponse()
        vr.say("An error occurred.", voice="alice")
        return twiml(vr)

# ---------- Background pipeline (stub) ----------
import os, time, asyncio, tempfile, logging, requests
from typing import Optional
from requests.auth import HTTPBasicAuth

logger = logging.getLogger("ws_server")

# assume these exist elsewhere in your file:
# - build_download_url(url: str) -> str
# - download_bytes_with_retry(url: str, auth=None, timeout=20, attempts=3, backoff=0.6, min_size_bytes=1024) -> bytes
# - create_tts_elevenlabs(text: str) -> Optional[bytes]   (or return None to fall back to <Say>)
# - upload_bytes_to_s3(b: bytes, filename: str) -> str    (returns https URL)  [optional for TTS]
# - _hold: Dict[str, Dict[str, Optional[str]]]
# - _hold_lock: asyncio.Lock()

import openai
import boto3
from urllib.parse import urlparse

# Add these at the top (after imports)
from openai import OpenAI
import boto3
import io

# Initialize clients at top
client = OpenAI(api_key=OPENAI_API_KEY)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
) if AWS_ACCESS_KEY_ID else None

async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    logger.info("[%s] Background start", call_sid)
    try:
        # 1. Download
        url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if "api.twilio.com" in url else None
        audio_bytes = download_bytes_with_retry(url, auth=auth)
        logger.info("[%s] Downloaded %d bytes", call_sid, len(audio_bytes))

        # Save temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_bytes)
        tmp.close()
        audio_path = tmp.name

        # 2. Transcribe with OpenAI v1.0+
        transcript = ""
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcript = transcription.text.strip()
            logger.info("[%s] Transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.error("[%s] Transcription failed: %s", call_sid, e)
        finally:
            os.unlink(audio_path)

        # 3. Generate reply
        reply_text = "I didn't catch that."
        if transcript:
            try:
                chat_completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant. Keep replies short."},
                        {"role": "user", "content": transcript}
                    ],
                    max_tokens=150
                )
                reply_text = chat_completion.choices[0].message.content.strip()
                logger.info("[%s] GPT reply: %s", call_sid, reply_text)
            except Exception as e:
                logger.error("[%s] GPT failed: %s", call_sid, e)
                reply_text = "I'm having trouble thinking right now."

        # 4. Optional TTS
        tts_url = None
        if ELEVEN_API_KEY and ELEVEN_VOICE and s3_client and AWS_S3_BUCKET:
            try:
                tts_resp = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}",
                    headers={"xi-api-key": ELEVEN_API_KEY},
                    json={"text": reply_text}
                )
                tts_resp.raise_for_status()
                key = f"tts/{call_sid}.mp3"
                s3_client.upload_fileobj(
                    io.BytesIO(tts_resp.content),
                    AWS_S3_BUCKET,
                    key,
                    ExtraArgs={"ContentType": "audio/mpeg"}
                )
                tts_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
                logger.info("[%s] TTS uploaded: %s", call_sid, tts_url)
            except Exception as e:
                logger.warning("[%s] TTS failed: %s", call_sid, e)

        # 5. Save to hold
        async with _hold_lock:
            _hold[call_sid] = {"reply_text": reply_text, "tts_url": tts_url}
        logger.info("[%s] Hold ready", call_sid)

    except Exception as e:
        logger.exception("[%s] Background error: %s", call_sid, e)
        async with _hold_lock:
            _hold[call_sid] = {"reply_text": "Sorry, something went wrong.", "tts_url": None}

import re
from requests.auth import HTTPBasicAuth
_TWILIO_REC_RE = re.compile(r"https://api\.twilio\.com/2010-04-01/Accounts/[^/]+/Recordings/(RE[a-zA-Z0-9]+)(?:\.(mp3|wav))?$")

def build_download_url(url: str) -> str:
    """
    - If this is a Twilio Recording resource URL without an extension, force .mp3
    - Otherwise return as-is
    """
    m = _TWILIO_REC_RE.match(url)
    if m and not m.group(2):
        rid = m.group(1)
        fixed = _TWILIO_REC_RE.sub(lambda _: f"{_.group(0)}.mp3", url)
        logger.info("Normalized Twilio RecordingUrl -> %s", fixed)
        return fixed
    return url

def download_bytes_with_retry(url: str, auth=None, timeout=20, attempts=3, backoff=0.6, min_size_bytes: int = 1024) -> bytes:
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            # HEAD for quick visibility
            try:
                rh = requests.head(url, auth=auth, timeout=timeout, allow_redirects=True)
                logger.info("HEAD %s -> %s (len=%s, type=%s)", url, rh.status_code, rh.headers.get("Content-Length"), rh.headers.get("Content-Type"))
            except Exception as he:
                logger.warning("HEAD failed for %s: %s", url, he)

            r = requests.get(url, auth=auth, timeout=timeout, allow_redirects=True)
            logger.info("GET %s -> %s (type=%s)", url, r.status_code, r.headers.get("Content-Type"))
            r.raise_for_status()
            content = r.content

            if len(content) < min_size_bytes:
                logger.warning("Downloaded content too small (%d bytes) from %s (first 120 bytes: %r)", len(content), url, content[:120])
                raise RuntimeError(f"Downloaded content too small ({len(content)} bytes)")

            return content
        except Exception as e:
            last_exc = e
            logger.error("Download attempt %d/%d failed for %s: %s", i, attempts, url, e)
            if i < attempts:
                time.sleep(backoff * i)
    raise last_exc or RuntimeError("download failed")
