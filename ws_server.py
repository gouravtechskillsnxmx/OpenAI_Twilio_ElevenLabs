"""
ws_server.py - Same logic as provided, with extensive additional logging only.
DO NOT change logic here without my explicit permission.
This version adds verbose entry/exit logs, parameter/response dumps, and exception traces
to help you diagnose why Twilio reports "Call is not in-progress" (21220) and why calls
are closed before the server can redirect them to play TTS.
"""

import os
import asyncio
import logging
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ws_server")
# Make requests from twilio and our async loops more visible
logger.setLevel(logging.DEBUG)

# --- FastAPI app ------------------------------------------------------------
app = FastAPI(title="Twilio Media + TTS Server")

# Mount a static directory for /tts files (so Twilio can fetch them via HTTP)
TTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tts_static", StaticFiles(directory=str(TTS_DIR)), name="tts_static")


# --- Helpers ----------------------------------------------------------------
async def call_get_status(call_sid: str) -> Optional[str]:
    """
    Fetch call resource and return its .status string. Returns None if fetch fails.
    """
    logger.debug("call_get_status: entering for call_sid=%s", call_sid)
    try:
        logger.info("-- BEGIN Twilio API Request (fetch call) -- call_sid=%s", call_sid)
        call = twilio_client.calls(call_sid).fetch()
        logger.info("-- END Twilio API Request (fetch call) -- call_sid=%s", call_sid)
        # Log important call attributes for debugging
        try:
            status = getattr(call, "status", None)
            direction = getattr(call, "direction", None)
            from_ = getattr(call, "from_", getattr(call, "from", None))
            to = getattr(call, "to", None)
            logger.debug("call_get_status: call attributes: status=%s direction=%s from=%s to=%s", status, direction, from_, to)
        except Exception:
            logger.exception("call_get_status: failed to read some call attributes for %s", call_sid)
        return getattr(call, "status", None)
    except Exception as ex:
        logger.exception("Exception fetching call status for %s: %s", call_sid, ex)
        return None


async def play_tts_with_wait(call_sid: str, tts_url: str, timeout_s: float = POLL_TIMEOUT_SECONDS, poll_interval: float = POLL_INTERVAL_SECONDS) -> bool:
    """
    Wait up to timeout_s for the call to become 'in-progress'. If it becomes in-progress,
    attempt to redirect Twilio by calling twilio.calls(call_sid).update(twiml=...).
    Returns True if update was issued (and Twilio returned a 2xx), False otherwise.
    """
    logger.debug("play_tts_with_wait: start call_sid=%s tts_url=%s timeout_s=%s poll_interval=%s", call_sid, tts_url, timeout_s, poll_interval)
    deadline = datetime.utcnow() + timedelta(seconds=timeout_s)
    last_status = None

    logger.info("play_tts_with_wait: call %s waiting for in-progress (deadline in %.1fs)", call_sid, (deadline - datetime.utcnow()).total_seconds())

    while datetime.utcnow() < deadline:
        try:
            status = await call_get_status(call_sid)
        except Exception as ex:
            logger.exception("play_tts_with_wait: unexpected exception while checking status for %s: %s", call_sid, ex)
            status = None

        if status:
            last_status = status
            logger.info("play_tts_with_wait: call %s current status=%s (deadline in %.1fs)", call_sid, status, (deadline - datetime.utcnow()).total_seconds())
            # Extra diagnostics: if status is not in-progress, log commonly-seen statuses
            if status != "in-progress":
                logger.debug("play_tts_with_wait: call %s not yet in-progress. common statuses: queued, ringing, in-progress, completed. current=%s", call_sid, status)
            if status == "in-progress":
                # Construct TwiML to play the TTS URL
                twiml = f"<Response><Play>{tts_url}</Play></Response>"
                try:
                    logger.info("play_tts_with_wait: attempting calls(%s).update(twiml=...)", call_sid)
                    resp = twilio_client.calls(call_sid).update(twiml=twiml)
                    # If the call is redirected successfully, Twilio returns a Call instance (200)
                    logger.info("play_tts_with_wait: update response repr=%s", repr(resp))
                    # Log returned attributes to ensure Twilio accepted the redirect
                    try:
                        logger.debug("play_tts_with_wait: update returned attrs: sid=%s status=%s to=%s from=%s", getattr(resp, 'sid', None), getattr(resp, 'status', None), getattr(resp, 'to', None), getattr(resp, 'from_', getattr(resp, 'from', None)))
                    except Exception:
                        logger.exception("play_tts_with_wait: failed to read response attributes after update for %s", call_sid)
                    return True
                except Exception as ex:
                    logger.exception("play_tts_with_wait: failed to update call %s while in-progress: %s", call_sid, ex)
                    return False
        else:
            logger.warning("play_tts_with_wait: could not read status for call %s, will retry (deadline in %.1fs)", call_sid, (deadline - datetime.utcnow()).total_seconds())
        await asyncio.sleep(poll_interval)

    logger.warning("play_tts_with_wait: timed out waiting for in-progress (last_status=%s)", last_status)
    # Final fallback: attempt a REST update once more (this will often fail with 21220 if the call is not in-progress)
    twiml = f"<Response><Play>{tts_url}</Play></Response>"
    try:
        logger.info("play_tts_with_wait: falling back to final calls(%s).update(twiml=...) attempt", call_sid)
        resp = twilio_client.calls(call_sid).update(twiml=twiml)
        logger.info("play_tts_with_wait: final update response repr=%s", repr(resp))
        return True
    except Exception as ex:
        logger.exception("play_tts_with_wait: final REST update failed for call %s: %s", call_sid, ex)
        return False


def tts_file_url_for(call_sid: str) -> str:
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
    logger.info("Received Twilio webhook /twiml_stream callSid=%s From=%s To=%s form_keys=%s", call_sid, form.get("From"), form.get("To"), list(form.keys()))
    # Additional diagnostics
    try:
        logger.debug("twiml_stream: full form data: %s", {k: (v if k.lower() != 'calltoken' else 'REDACTED') for k, v in form.items()})
    except Exception:
        logger.exception("twiml_stream: failed to log form data for call %s", call_sid)

    # Twilio expects an XML TwiML response. Use Start/Stream element.
    # Note: Twilio expects the stream url to be a wss:// endpoint.
    host = request.headers.get('host') or ''
    stream_url = f"wss://{host}/twilio-media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{stream_url}"/>
  </Start>
</Response>"""
    logger.info("Returning TwiML Start/Stream -> %s for call %s", stream_url, call_sid)
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
    logger.info("WebSocket accepted from client=%s remote=%s", getattr(client, 'host', None), getattr(client, 'port', None))
    try:
        while True:
            msg = await ws.receive_text()
            logger.debug("WS received raw (first200)=%s", msg[:200])
            # Twilio sends JSON text messages; we do a lightweight parse
            try:
                import json
                obj = json.loads(msg)
                event = obj.get("event")
                if event == "start":
                    logger.info("INBOUND stream started: %s", obj.get("start", {}))
                elif event == "media":
                    # media payload -> base64 encoded audio chunk
                    # store/process as needed; here we just log size
                    payload = obj.get("media", {}).get("payload")
                    if payload:
                        # payload is base64 string; log length and an approximate byte size
                        payload_len = len(payload)
                        approx_bytes = int(payload_len * 3 / 4)
                        logger.info("INBOUND media event: payload_len=%d approx_bytes=%d total_obj_keys=%d", payload_len, approx_bytes, len(obj))
                        # For deeper debugging, optionally log first/last 32 chars
                        logger.debug("INBOUND media payload sample start=%s end=%s", payload[:32], payload[-32:])
                elif event == "stop":
                    logger.info("Stream stop event received: %s", obj.get('stop', {}))
                else:
                    logger.debug("Unhandled Twilio media event: %s", event)
            except Exception:
                logger.exception("Failed to parse WS message")
    except Exception as ex:
        logger.info("WS closed: %s", ex)
    finally:
        try:
            await ws.close()
        except Exception:
            logger.exception("Error closing websocket")
        logger.info("WS closed (handler exit)")


@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    """
    Serve a TTS file saved under TTS_DIR. Twilio must be able to reach this URL.
    Example file path: /tmp/tts_CALLSID_123456.wav
    """
    safe_name = Path(fname).name  # prevent ../ tricks
    file_path = TTS_DIR / safe_name
    logger.debug("serve_tts: requested fname=%s resolved_path=%s", fname, file_path)
    if not file_path.exists():
        logger.warning("serve_tts: requested file not found: %s", file_path)
        raise HTTPException(status_code=404, detail="tts file not found")
    try:
        size = file_path.stat().st_size
    except Exception:
        size = None
    logger.info("Serving TTS file %s size_bytes=%s", file_path, size)
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=safe_name)


# --- Example handler that would be called when you want to play TTS ---------
async def handle_playback_for_call(call_sid: str, tts_bytes: bytes) -> bool:
    """
    Example orchestration: save TTS bytes to a file and attempt to play them
    into the active call by waiting for in-progress and then redirecting the call.
    Returns True if playback was triggered, False otherwise.
    """
    logger.debug("handle_playback_for_call: start call_sid=%s bytes=%d", call_sid, len(tts_bytes))
    url, fname = tts_file_url_for(call_sid)
    file_path = TTS_DIR / fname
    logger.info("Writing tts file %s (bytes=%d)", file_path, len(tts_bytes))
    try:
        file_path.write_bytes(tts_bytes)
        logger.debug("handle_playback_for_call: file written ok path=%s", file_path)
    except Exception as ex:
        logger.exception("handle_playback_for_call: failed to write tts file %s: %s", file_path, ex)
        return False

    # Wait and attempt to play
    # NOTE: preserving original logic (do not change) even if it contains minor bugs
    try:
        logger.info("About to play tts for call=%s tts_url=%s", call_sid, url)
    except Exception:
        logger.exception("handle_playback_for_call: failed composing about-to-play log")

    ok = await play_tts_with_wait(call_sid, url, timeout_s=POLL_TIMEOUT_SECONDS, poll_interval=POLL_INTERVAL_SECONDS)
    if ok:
        logger.info("Playback triggered for call %s", call_sid)
    else:
        logger.warning("Playback could not be triggered for call %s", call_sid)
    return ok


# If you run this module directly, start uvicorn for local dev.
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ws_server uvicorn app on port %s", os.getenv("PORT", "10000"))
    uvicorn.run("ws_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), log_level="info")
