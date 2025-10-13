"""
ws_server_enhanced.py - Original logic preserved, with extra detailed diagnostics added.
This file is intended to help investigate calls that drop before ElevenLabs audio is played.
It adds:
 - timestamps and monotonic timers around important operations
 - iteration counters and elapsed times in the polling loop
 - file write checks (fsync) and permission logging
 - process, thread, and asyncio task identifiers
 - more detailed dumps of Twilio SDK responses (repr) and attribute reads guarded by try/except
 - logging of environment variables (masked) and host information
 - increased verbosity for websocket inbound media events
 - a safe-to-drop-in replacement for the original file (logic preserved)
 
DO NOT change logic here without the user's explicit permission.
"""

import os
import asyncio
import logging
import json
import time
import traceback
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client as TwilioClient

# --- Configuration ----------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")  # keep this name stable
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://openai-twilio-elevenlabs.onrender.com")
TTS_DIR = Path(os.getenv("TTS_DIR", "/tmp"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "0.5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("POLL_TIMEOUT_SECONDS", "5.0"))

# --- Validate env vars -----------------------------------------------------
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables must be set.")

# Twilio client
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Logging ----------------------------------------------------------------
# Use a more verbose formatter that includes thread and process info
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s")
logger = logging.getLogger("ws_server")
logger.setLevel(logging.DEBUG)

# Add a console handler that prints JSON-style context for easier parsing in logs, while keeping human readable fallback
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s")
console.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console)

# --- FastAPI app ------------------------------------------------------------
app = FastAPI(title="Twilio Media + TTS Server (enhanced logging)")

# Mount a static directory for /tts files (so Twilio can fetch them via HTTP)
TTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tts_static", StaticFiles(directory=str(TTS_DIR)), name="tts_static")


# --- Small helpers ---------------------------------------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def monotonic_ms():
    return int(time.monotonic() * 1000)

def mask_secret(s):
    if not s:
        return None
    if len(s) <= 8:
        return "****"
    return s[:4] + "..." + s[-4:]

def task_name_or_none():
    try:
        return asyncio.current_task().get_name()
    except Exception:
        return None

# --- Helpers ----------------------------------------------------------------
async def call_get_status(call_sid: str) -> Optional[str]:
    """
    Fetch call resource and return its .status string. Returns None if fetch fails.
    """
    start = monotonic_ms()
    logger.debug("call_get_status: entering call_sid=%s at=%s task=%s", call_sid, now_iso(), task_name_or_none())
    try:
        logger.info("BEGIN Twilio API Request (fetch call) call_sid=%s start_ms=%d", call_sid, start)
        call = twilio_client.calls(call_sid).fetch()
        elapsed = monotonic_ms() - start
        logger.info("END Twilio API Request (fetch call) call_sid=%s elapsed_ms=%d repr_len=%d", call_sid, elapsed, len(repr(call)))
        # Log important call attributes for debugging
        try:
            status = getattr(call, "status", None)
            direction = getattr(call, "direction", None)
            from_ = getattr(call, "from_", getattr(call, "from", None))
            to = getattr(call, "to", None)
            logger.debug("call_get_status: attrs: status=%s direction=%s from=%s to=%s", status, direction, from_, to)
        except Exception:
            logger.exception("call_get_status: failed to read some call attributes for %s", call_sid)
        return getattr(call, "status", None)
    except Exception as ex:
        logger.exception("Exception fetching call status for %s: %s", call_sid, ex)
        # Include traceback to help remote debugging
        logger.debug("call_get_status: traceback: %s", traceback.format_exc())
        return None


async def play_tts_with_wait(call_sid: str, tts_url: str, timeout_s: float = POLL_TIMEOUT_SECONDS, poll_interval: float = POLL_INTERVAL_SECONDS) -> bool:
    """
    Wait up to timeout_s for the call to become 'in-progress'. If it becomes in-progress,
    attempt to redirect Twilio by calling twilio.calls(call_sid).update(twiml=...).
    Returns True if update was issued (and Twilio returned a 2xx), False otherwise.
    """
    start_time = monotonic_ms()
    logger.debug("play_tts_with_wait: start call_sid=%s tts_url=%s timeout_s=%s poll_interval=%s at=%s task=%s", call_sid, tts_url, timeout_s, poll_interval, now_iso(), task_name_or_none())
    deadline = datetime.utcnow() + timedelta(seconds=timeout_s)
    last_status = None
    loop_count = 0

    logger.info("play_tts_with_wait: call %s waiting for in-progress until=%s (timeout_s=%.1f)", call_sid, deadline.isoformat(), timeout_s)

    while datetime.utcnow() < deadline:
        loop_count += 1
        loop_start = monotonic_ms()
        try:
            status = await call_get_status(call_sid)
        except Exception as ex:
            logger.exception("play_tts_with_wait: unexpected exception while checking status for %s: %s", call_sid, ex)
            logger.debug("play_tts_with_wait: traceback: %s", traceback.format_exc())
            status = None

        elapsed_loop = monotonic_ms() - loop_start
        logger.debug("play_tts_with_wait: loop=%d elapsed_ms=%d status=%s", loop_count, elapsed_loop, status)

        if status:
            last_status = status
            logger.info("play_tts_with_wait: call=%s loop=%d status=%s deadline_in_ms=%d", call_sid, loop_count, status, int((deadline - datetime.utcnow()).total_seconds() * 1000))
            # Extra diagnostics: if status is not in-progress, log commonly-seen statuses
            if status != "in-progress":
                logger.debug("play_tts_with_wait: call=%s not yet in-progress (common: queued, ringing, in-progress, completed). current=%s", call_sid, status)
            if status == "in-progress":
                # Construct TwiML to play the TTS URL
                twiml = f"<Response><Play>{tts_url}</Play></Response>"
                try:
                    logger.info("play_tts_with_wait: attempting calls(%s).update(twiml=...) at=%s", call_sid, now_iso())
                    resp = twilio_client.calls(call_sid).update(twiml=twiml)
                    # If the call is redirected successfully, Twilio returns a Call instance (200)
                    logger.info("play_tts_with_wait: update returned repr=%s", repr(resp))
                    # Try to log some attributes safely
                    try:
                        resp_sid = getattr(resp, 'sid', None)
                        resp_status = getattr(resp, 'status', None)
                        resp_to = getattr(resp, 'to', None)
                        resp_from = getattr(resp, 'from_', getattr(resp, 'from', None))
                        logger.debug("play_tts_with_wait: update response attrs sid=%s status=%s to=%s from=%s", resp_sid, resp_status, resp_to, resp_from)
                    except Exception:
                        logger.exception("play_tts_with_wait: failed to read response attributes after update for %s", call_sid)
                    total_elapsed = monotonic_ms() - start_time
                    logger.info("play_tts_with_wait: update success for %s total_elapsed_ms=%d loops=%d", call_sid, total_elapsed, loop_count)
                    return True
                except Exception as ex:
                    # This is the most important failure point to debug when Twilio returns 21220 or similar
                    logger.exception("play_tts_with_wait: failed to update call %s while in-progress: %s", call_sid, ex)
                    logger.debug("play_tts_with_wait: update exception traceback: %s", traceback.format_exc())
                    # Log the current process/thread/task and environ as context
                    try:
                        logger.debug("play_tts_with_wait: context pid=%d tid=%d task=%s env_base_url=%s", os.getpid(), threading.get_ident(), task_name_or_none(), mask_secret(BASE_URL))
                        logger.debug("play_tts_with_wait: masked_twilio_sid=%s masked_token=%s", mask_secret(TWILIO_ACCOUNT_SID), mask_secret(TWILIO_AUTH_TOKEN))
                    except Exception:
                        logger.exception("play_tts_with_wait: failed to log extra context")
                    return False
        else:
            logger.warning("play_tts_with_wait: could not read status for call %s on loop %d, will retry (deadline in %.1fs)", call_sid, loop_count, (deadline - datetime.utcnow()).total_seconds())
        # Sleep with jitter to avoid tight hammering and to be easier to correlate in logs
        try:
            sleep_ms = int(poll_interval * 1000)
            logger.debug("play_tts_with_wait: sleeping for %d ms (loop=%d)", sleep_ms, loop_count)
            await asyncio.sleep(poll_interval)
        except Exception:
            logger.exception("play_tts_with_wait: sleep interrupted")
    # timed out waiting
    logger.warning("play_tts_with_wait: timed out waiting for in-progress call=%s last_status=%s loops=%d", call_sid, last_status, loop_count)
    # Final fallback: attempt a REST update once more (this will often fail with 21220 if the call is not in-progress)
    twiml = f"<Response><Play>{tts_url}</Play></Response>"
    try:
        logger.info("play_tts_with_wait: falling back to final calls(%s).update(twiml=...) attempt at=%s", call_sid, now_iso())
        resp = twilio_client.calls(call_sid).update(twiml=twiml)
        logger.info("play_tts_with_wait: final update response repr=%s", repr(resp))
        return True
    except Exception as ex:
        logger.exception("play_tts_with_wait: final REST update failed for call %s: %s", call_sid, ex)
        logger.debug("play_tts_with_wait: final update exception traceback: %s", traceback.format_exc())
        return False


def tts_file_url_for(call_sid: str) -> (str, str):
    """
    Create a deterministic filename for TTS for this call (the real TTS generator
    should write files into TTS_DIR with the same name).
    """
    ts = int(datetime.utcnow().timestamp() * 1000)
    fname = f"tts_{call_sid}_{ts}.wav"
    url = f"{BASE_URL}/tts/{fname}"
    logger.debug("tts_file_url_for: call_sid=%s -> url=%s fname=%s", call_sid, url, fname)
    return url, fname


# --- Routes -----------------------------------------------------------------
@app.post("/twiml_stream")
async def twiml_stream(request: Request):
    """
    Twilio will POST to this URL when the call starts. We reply with TwiML that starts
    a <Stream> to our WebSocket endpoint: /twilio-media
    """
    logger.debug("twiml_stream: received request headers=%s", dict(request.headers))
    form = await request.form()
    call_sid = form.get("CallSid") or form.get("CallSid".lower()) or form.get("CallSid".upper())
    logger.info("Received Twilio webhook /twiml_stream callSid=%s From=%s To=%s form_keys=%s pid=%d", call_sid, form.get("From"), form.get("To"), list(form.keys()), os.getpid())
    # Additional diagnostics
    try:
        # Redact any keys that look sensitive
        safe_form = {}
        for k, v in form.items():
            sk = k
            sv = v
            if 'token' in k.lower() or 'auth' in k.lower() or 'secret' in k.lower():
                sv = 'REDACTED'
            safe_form[sk] = sv
        logger.debug("twiml_stream: full form data (redacted)=%s", json.dumps(safe_form))
    except Exception:
        logger.exception("twiml_stream: failed to log form data for call %s", call_sid)

    # Twilio expects an XML TwiML response. Use Start/Stream element.
    host = request.headers.get('host') or ''
    stream_url = f"wss://{host}/twilio-media"
    twiml = f\"\"\"<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{stream_url}"/>
  </Start>
</Response>\"\"\"
    logger.info("twiml_stream: Returning TwiML Start/Stream -> %s for call %s twiml_len=%d", stream_url, call_sid, len(twiml))
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio-media")
async def twilio_media_ws(ws: WebSocket):
    """
    Basic Twilio Media WebSocket handler. This receives media fragments from Twilio.
    The Twilio Media WebSocket uses JSON messages for events (start, media, stop).
    We process basic events: 'start', 'media', 'stop'.
    """
    await ws.accept()
    client = ws.client
    logger.info("WebSocket accepted client=%s remote=%s pid=%d tid=%d", getattr(client, 'host', None), getattr(client, 'port', None), os.getpid(), threading.get_ident())
    msg_count = 0
    try:
        while True:
            msg = await ws.receive_text()
            msg_count += 1
            logger.debug("WS received raw length=%d first200=%s", len(msg), msg[:200])
            # Twilio sends JSON text messages; do a lightweight parse
            try:
                obj = json.loads(msg)
            except Exception:
                logger.exception("Failed to json.loads websocket message on msg_count=%d", msg_count)
                continue

            event = obj.get("event")
            if event == "start":
                logger.info("INBOUND stream started (msg_count=%d): %s", msg_count, json.dumps(obj.get("start", {})))
            elif event == "media":
                payload = obj.get("media", {}).get("payload")
                if payload:
                    payload_len = len(payload)
                    approx_bytes = int(payload_len * 3 / 4)
                    logger.info("INBOUND media event msg_count=%d payload_len=%d approx_bytes=%d total_obj_keys=%d", msg_count, payload_len, approx_bytes, len(obj))
                    # For deeper debugging, optionally log first/last 64 chars - but avoid logging huge content
                    start_sample = payload[:64]
                    end_sample = payload[-64:]
                    logger.debug("INBOUND media payload sample start=%s end=%s", start_sample, end_sample)
                else:
                    logger.warning("INBOUND media event without payload on msg_count=%d obj_keys=%d", msg_count, len(obj))
            elif event == "stop":
                logger.info("Stream stop event received msg_count=%d: %s", msg_count, json.dumps(obj.get('stop', {})))
            else:
                logger.debug("Unhandled Twilio media event msg_count=%d event=%s obj_keys=%d", msg_count, event, len(obj))
    except Exception as ex:
        logger.info("WS closed unexpectedly: %s", ex)
        logger.debug("twilio_media_ws: traceback: %s", traceback.format_exc())
    finally:
        try:
            await ws.close()
        except Exception:
            logger.exception("Error closing websocket")
        logger.info("WS closed (handler exit) total_msgs=%d", msg_count)


@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    """
    Serve a TTS file saved under TTS_DIR. Twilio must be able to reach this URL.
    Example file path: /tmp/tts_CALLSID_123456.wav
    """
    safe_name = Path(fname).name  # prevent ../ tricks
    file_path = TTS_DIR / safe_name
    logger.debug("serve_tts: requested fname=%s resolved_path=%s pid=%d", fname, file_path, os.getpid())
    if not file_path.exists():
        logger.warning("serve_tts: requested file not found: %s", file_path)
        raise HTTPException(status_code=404, detail="tts file not found")
    try:
        size = file_path.stat().st_size
    except Exception:
        size = None
    logger.info("Serving TTS file %s size_bytes=%s (exists=%s)", file_path, size, file_path.exists())
    # Log a little more about file permissions to rule out file access issues
    try:
        st = file_path.stat()
        logger.debug("serve_tts: file mode=%o uid=%s gid=%s mtime=%s", st.st_mode, getattr(st, 'st_uid', None), getattr(st, 'st_gid', None), datetime.utcfromtimestamp(st.st_mtime).isoformat())
    except Exception:
        logger.exception("serve_tts: unable to stat file for %s", file_path)
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=safe_name)


# --- Example handler that would be called when you want to play TTS ---------
async def handle_playback_for_call(call_sid: str, tts_bytes: bytes) -> bool:
    """
    Example orchestration: save TTS bytes to a file and attempt to play them
    into the active call by waiting for in-progress and then redirecting the call.
    Returns True if playback was triggered, False otherwise.
    """
    logger.debug("handle_playback_for_call: start call_sid=%s bytes=%d pid=%d task=%s", call_sid, len(tts_bytes), os.getpid(), task_name_or_none())
    url, fname = tts_file_url_for(call_sid)
    file_path = TTS_DIR / fname
    logger.info("handle_playback_for_call: will write tts file %s (bytes=%d)", file_path, len(tts_bytes))

    try:
        # Write file and fsync to ensure it's visible to external HTTP clients quickly
        with open(file_path, "wb") as f:
            f.write(tts_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
                logger.debug("handle_playback_for_call: fsync succeeded for %s", file_path)
            except Exception:
                logger.exception("handle_playback_for_call: fsync failed for %s", file_path)
        # Re-check file size and existence
        final_size = file_path.stat().st_size
        logger.info("handle_playback_for_call: file written ok path=%s size=%d", file_path, final_size)
    except Exception as ex:
        logger.exception("handle_playback_for_call: failed to write tts file %s: %s", file_path, ex)
        logger.debug("handle_playback_for_call: traceback: %s", traceback.format_exc())
        return False

    # Wait and attempt to play (preserve original logic)
    try:
        logger.info("handle_playback_for_call: about to play tts for call=%s url=%s", call_sid, url)
    except Exception:
        logger.exception("handle_playback_for_call: failed composing about-to-play log")

    ok = await play_tts_with_wait(call_sid, url, timeout_s=POLL_TIMEOUT_SECONDS, poll_interval=POLL_INTERVAL_SECONDS)
    if ok:
        logger.info("handle_playback_for_call: playback triggered for call %s", call_sid)
    else:
        logger.warning("handle_playback_for_call: playback could not be triggered for call %s", call_sid)
    return ok


# If you run this module directly, start uvicorn for local dev.
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ws_server_enhanced uvicorn app on port %s pid=%d", os.getenv("PORT", "10000"), os.getpid())
    uvicorn.run("ws_server_enhanced:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), log_level="info")
