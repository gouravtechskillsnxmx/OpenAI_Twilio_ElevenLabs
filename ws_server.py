# ws_server.py
import os
import sys
import time
import logging
import tempfile
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import requests
from requests.auth import HTTPBasicAuth

from memory_api import router as memory_router
app.include_router(memory_router)

# OpenAI new client
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

import boto3
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse

# ---------------- CONFIG / ENV ----------------
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")  # your Twilio number (E.164) used for outbound if needed
OPENAI_KEY = os.environ.get("OPENAI_KEY")
# If you have a dedicated ChatGPT agent endpoint, set AGENT_ENDPOINT and AGENT_KEY
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT")  # optional: your existing agent endpoint URL
AGENT_KEY = os.environ.get("AGENT_KEY")            # optional: auth for that endpoint

ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE")  # your custom voice id

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# AWS creds via env (aws cli style) or IAM role attached to the Render service
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

HOSTNAME = os.environ.get("HOSTNAME", "")  # e.g. fl-ai-sales-agent3.onrender.com
TTS_PRESIGNED_EXPIRES = int(os.environ.get("TTS_PRESIGNED_EXPIRES", "600"))  # seconds

# optional: simple cost cap / throttle params
CALLS_PER_MINUTE_PER_NUMBER = int(os.environ.get("CALLS_PER_MINUTE_PER_NUMBER", "3"))

# ---------------- clients ----------------
twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

openai_client = None
if OPENAI_KEY and OpenAIClient is not None:
    openai_client = OpenAIClient(api_key=OPENAI_KEY)

# S3 client
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
else:
    # allow boto3 to use instance/role credentials
    s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------- logging ----------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-sales-agent")

app = FastAPI()

# ---------------- health ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "hostname": HOSTNAME}

# ---------------- helper: build recording callback URL ----------------
def recording_callback_url():
    host = HOSTNAME or os.environ.get("REQUEST_HOST")
    if host:
        return f"https://{host}/recording"
    return None

# ---------------- TwiML endpoint ----------------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    """
    TwiML to start conversation: short record and POST to /recording.
    Twilio will make a Request to the returned action (we should ensure it's HTTPS).
    """
    resp = VoiceResponse()
    resp.say("Hello. Connecting you to our assistant. Please say something after the beep.", voice="alice")
    action = recording_callback_url() or (str(request.base_url).rstrip("/") + "/recording")
    # keep recording short for responsive loop (tune as needed)
    resp.record(max_length=6, action=action, play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

# ---------------- recording hook (Twilio posts after each Record) ----------------
@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    payload = dict(form)
    recording_url = payload.get("RecordingUrl")
    call_sid = payload.get("CallSid")
    from_number = payload.get("From")
    logger.info("Incoming recording webhook: CallSid=%s From=%s RecordingUrl=%s", call_sid, from_number, recording_url)

    if not recording_url or not call_sid:
        logger.warning("Missing RecordingUrl or CallSid in request: %s", payload)
        return JSONResponse({"error": "missing RecordingUrl or CallSid"}, status_code=400)

    # ACK quickly and process in background
    background_tasks.add_task(process_recording_background, call_sid, recording_url)
    return Response(status_code=204)

# ---------------- processing pipeline ----------------
def build_download_url(recording_url: str) -> str:
    # If recording_url already looks like it ends with audio ext, use as-is
    lower = (recording_url or "").lower()
    for ext in (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"):
        if lower.endswith(ext):
            return recording_url
    # Twilio recordings normally require appending .mp3; do that only for api.twilio.com pattern
    if "api.twilio.com" in (recording_url or ""):
        return recording_url + ".mp3"
    return recording_url

def transcribe_with_openai(file_path: str) -> str:
    """
    Uses new OpenAI Python client if available, otherwise raises.
    If you have a different agent endpoint, swap this out.
    """
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_KEY and ensure openai library >=1.0 installed.")
    # choose an available model; change to model you have access to
    model = "whisper-1"
    with open(file_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(model=model, file=f)
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
    return (text or "").strip()

def call_agent_and_get_reply(convo_id: str, user_text: str) -> str:
    """
    Hook into your existing ChatGPT agent here. Two options:
    - If you have AGENT_ENDPOINT (a custom API that your agent exposes), POST the transcript there and return response text.
    - Otherwise, as fallback we call OpenAI ChatCompletion (not recommended if you have your own agent).
    Replace this function to call your established agent (AGENT_ENDPOINT/AGENT_KEY).
    """
    if AGENT_ENDPOINT and AGENT_KEY:
        try:
            r = requests.post(AGENT_ENDPOINT, json={"convo_id": convo_id, "text": user_text}, headers={"Authorization": f"Bearer {AGENT_KEY}"}, timeout=15)
            r.raise_for_status()
            j = r.json()
            return j.get("reply_text") or j.get("text") or str(j)
        except Exception as e:
            logger.exception("Agent endpoint error: %s", e)

    # Fallback: simple echo / OpenAI ChatCompletion (if OPENAI_KEY available)
    if openai_client:
        # simple Chat API usage via openai_client
        chat_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # choose a model you have access to; change if needed
            messages=[{"role": "system", "content": "You are a helpful voice assistant."}, {"role": "user", "content": user_text}],
        )
        # extract reply
        content = ""
        try:
            content = chat_resp.choices[0].message["content"]
        except Exception:
            content = str(chat_resp)
        return content

    return f"Echo: {user_text}"

def create_tts_elevenlabs(text: str) -> bytes:
    """
    Create audio bytes from ElevenLabs (synchronous). Returns raw audio bytes (mp3).
    Replace endpoint if ElevenLabs API changes.
    """
    if not ELEVEN_API_KEY or not ELEVEN_VOICE:
        raise RuntimeError("ElevenLabs not configured. Set ELEVEN_API_KEY and ELEVEN_VOICE.")
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
    Create TTS via ElevenLabs, upload bytes to S3 (no ACL), and return a presigned GET URL for Twilio to play.
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

async def process_recording_background(call_sid: str, recording_url: str):
    """
    1) download (authenticated) the recording
    2) transcribe -> transcript
    3) call agent -> assistant_text
    4) create TTS -> presigned S3 URL
    5) update Twilio call to Play that URL and record again
    """
    logger.info("[%s] processing recording: %s", call_sid, recording_url)
    try:
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in download_url) else None
        r = requests.get(download_url, auth=auth, timeout=30)
        if r.status_code != 200:
            logger.error("[%s] download returned status %s body=%s", call_sid, r.status_code, r.text[:400])
            # send polite fallback to caller if needed
            fallback_twiml = "<Response><Say>Sorry, we couldn't get your audio. Please try again later.</Say></Response>"
            try:
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback_twiml)
            except Exception:
                logger.exception("[%s] twilio update fallback failed", call_sid)
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name

        # STT
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception as e:
            logger.exception("[%s] STT failed: %s", call_sid, e)
            transcript = ""

        # agent
        assistant_text = call_agent_and_get_reply(call_sid, transcript)
        logger.info("[%s] assistant_text: %s", call_sid, assistant_text[:300])

        # TTS
        try:
            tts_url = create_and_upload_tts(assistant_text)
            logger.info("[%s] tts_url: %s", call_sid, tts_url)
        except Exception as e:
            logger.exception("[%s] TTS/upload failed: %s", call_sid, e)
            # fallback to Twilio TTS
            twiml = f"<Response><Say>{assistant_text}</Say><Record maxLength='6' action='https://{HOSTNAME}/recording' playBeep='true' timeout='2'/></Response>"
            try:
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=twiml)
            except Exception:
                logger.exception("[%s] twilio update fallback failed", call_sid)
            return

        # prepare twiml to play the tts_url and record again
        record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"
        twiml = f"""<Response>
            <Play>{tts_url}</Play>
            <Record maxLength="6" action="{record_action}" playBeep="true" timeout="2" />
        </Response>"""
        logger.info("[%s] updating Twilio with twiml (Play+Record)", call_sid)
        try:
            if twilio_client:
                twilio_client.calls(call_sid).update(twiml=twiml)
        except Exception:
            logger.exception("[%s] twilio update failed for Play+Record", call_sid)

    except Exception as e:
        logger.exception("[%s] Unexpected pipeline error: %s", call_sid, e)
