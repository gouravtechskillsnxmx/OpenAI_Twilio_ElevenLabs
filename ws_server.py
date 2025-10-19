# ws_server.py
# Patched version — key changes:
#  - Read Twilio credentials from environment and fail fast if missing.
#  - Read NGROK_HOST or PUBLIC_HOST from environment (so you don't have to edit code each time ngrok changes).
#  - Build Stream URL using the host env var and ensure correct scheme (wss://).
#  - Improved logging and clearer Twilio exception handling to avoid "Task exception was never retrieved"
#  - Minimal, self-contained server bits (adapt to your original code as needed).

import os
import asyncio
import logging
import json
import base64
from pathlib import Path
from datetime import datetime

# Twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Websockets / HTTP server bits (you may be using a different framework)
import websockets
from aiohttp import web

# --- Configuration / env-driven values (PATCHED) ---
LOG_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() in ("1", "true", "yes") else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ws_server_patch")

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")

# Public host that Twilio and remote services will contact — prefer full URL or host-only
# Examples:
#   NGROK_HOST=1bc9f559ec09.ngrok-free.app    -> we will build wss://1bc9f559ec09.ngrok-free.app/twilio-media
#   PUBLIC_HOST=https://1bc9f559ec09.ngrok-free.app  -> used as-is
NGROK_HOST = os.getenv("NGROK_HOST")         # host without scheme
PUBLIC_HOST = os.getenv("PUBLIC_HOST")       # or full URL with scheme (https://...)
STREAM_PATH = os.getenv("STREAM_PATH", "twilio-media")  # path portion for stream endpoint

# folder for debug markers (same as your original code used tmp paths)
TMP = Path(os.getenv("TMP", "/tmp"))
if not TMP.exists():
    try:
        TMP.mkdir(parents=True, exist_ok=True)
    except Exception:
        TMP = Path(".")  # fallback to current directory

# --- Validate credentials early (PATCHED) ---
if not TWILIO_SID or not TWILIO_AUTH:
    log.error(
        "Twilio credentials not found. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables.\n"
        "Example (PowerShell): $env:TWILIO_ACCOUNT_SID='ACxxx'; $env:TWILIO_AUTH_TOKEN='your_auth_token'"
    )
    # Instead of exiting abruptly (in case you want the server to start for debugging), we raise an exception:
    raise SystemExit("Missing Twilio credentials (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN)")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# --- Utility: build a valid Stream URL from env --- 
def build_stream_url():
    """
    Build a wss://... stream url based on PUBLIC_HOST or NGROK_HOST.
    - If PUBLIC_HOST provided and already includes scheme, use it.
    - If NGROK_HOST provided, assume wss://<NGROK_HOST>/<STREAM_PATH>
    """
    if PUBLIC_HOST:
        host = PUBLIC_HOST.strip()
        # If PUBLIC_HOST has https://, replace with wss:// for websockets
        if host.startswith("https://"):
            ws = "wss://" + host[len("https://"):]
            # ensure trailing slash removal then append path
            ws = ws.rstrip("/") + "/" + STREAM_PATH.lstrip("/")
            return ws
        elif host.startswith("http://"):
            ws = "ws://" + host[len("http://"):]
            ws = ws.rstrip("/") + "/" + STREAM_PATH.lstrip("/")
            return ws
        elif host.startswith("wss://") or host.startswith("ws://"):
            return host.rstrip("/") + "/" + STREAM_PATH.lstrip("/")
        else:
            # treat as host-only
            return "wss://" + host.rstrip("/") + "/" + STREAM_PATH.lstrip("/")
    elif NGROK_HOST:
        # host-only
        return "wss://" + NGROK_HOST.rstrip("/") + "/" + STREAM_PATH.lstrip("/")
    else:
        # fallback to localhost (useful for local testing with reverse proxy)
        log.warning("No PUBLIC_HOST or NGROK_HOST configured; defaulting to localhost (wss://127.0.0.1)")
        return "wss://127.0.0.1:10000/" + STREAM_PATH.lstrip("/")

# Example usage: stream_url = build_stream_url()

# --- Twilio call-update wrapper with clear logging + exception handling (PATCHED) ---
async def update_call_twiml_async(call_sid: str, twiml: str):
    """
    Update the Twilio Call resource twiml safely from asyncio (runs in thread pool).
    Logs clear errors for authentication issues (401).
    """
    log.debug("Preparing to update call %s with new twiml", call_sid)
    loop = asyncio.get_running_loop()
    try:
        # run blocking twilio client in thread
        resp = await loop.run_in_executor(None, lambda: twilio_client.calls(call_sid).update(twiml=twiml))
        log.info("Call %s updated (sid=%s)", call_sid, getattr(resp, "sid", "<no-sid>"))
        return resp
    except TwilioRestException as e:
        # Twilio error codes are helpful; 20003 indicates auth failure
        log.error("TwilioRestException updating call %s: status=%s, code=%s, msg=%s", call_sid, e.status, getattr(e, "code", None), e.msg)
        if getattr(e, "code", None) == 20003 or e.status == 401:
            log.error("Authentication failed with Twilio: check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN. (Received 401 / code 20003)")
        # re-raise so the caller can decide (or swallow depending on desired behavior)
        raise

# --- Marker writer helper (similar to your original code) ---
def write_marker(name: str, data: dict):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = TMP / f"ws_{name}_{ts}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
        log.info("WROTE marker %s", str(filename))
    except Exception as exc:
        log.exception("Failed to write marker %s: %s", filename, exc)

# --- Example handler that Twilio Media Streams will call over websocket ---
# (Your original code likely has a more complete websocket handling loop; this is a conservative minimal example.)
async def twilio_media_ws(websocket, path):
    """
    Handle Twilio Media Streams WebSocket connection.
    Expect the remote Twilio client to start sending JSON media events with base64 payloads.
    """
    log.info("New websocket connection path=%s", path)
    call_sid = None
    try:
        async for raw in websocket:
            # Twilio sends JSON text messages with { "event":"media", "media": {...} } etc.
            try:
                msg = json.loads(raw)
            except Exception:
                log.debug("Received non-json websocket frame; ignoring")
                continue

            # write a marker so debug/markers endpoints can show it
            write_marker("media_sample", msg)

            # Example: if Twilio sent 'start' event containing callSid
            if msg.get("event") == "start":
                call_sid = msg.get("start", {}).get("callSid") or msg.get("start", {}).get("call_sid")
                write_marker("ws_start", {"timestamp": datetime.utcnow().isoformat(), "callSid": call_sid})
                log.info("Start event for call %s", call_sid)

            # For actual inbound audio events, decode and process:
            if msg.get("event") == "media" and "media" in msg:
                media = msg["media"]
                track = media.get("track")
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        audio_bytes = base64.b64decode(payload_b64)
                        # TODO: feed audio_bytes to your speech pipeline / file
                        # For debug, write small sample marker
                        write_marker("media_chunk", {"len": len(audio_bytes), "track": track})
                    except Exception as exc:
                        log.exception("Failed to decode media payload: %s", exc)
                else:
                    log.debug("media event without payload")
    except websockets.exceptions.ConnectionClosed as e:
        log.info("Websocket closed: %s", e)
    except Exception:
        log.exception("Unhandled exception in websocket handler")
    finally:
        log.info("Websocket handler finished for call_sid=%s", call_sid)

# --- Example aiohttp debug endpoint to list markers (mimics your /debug/markers) ---
async def debug_markers(request):
    """
    Return a JSON array of marker files in TMP.
    (This mirrors behaviour you used via Invoke-RestMethod against http://127.0.0.1:10000/debug/markers)
    """
    entries = []
    for p in sorted(TMP.glob("ws_*.json"))[-200:]:
        stat = p.stat()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = "<unreadable>"
        entries.append({"name": p.name, "size": stat.st_size, "mtime": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z", "data": data})
    return web.json_response({"count": len(entries), "markers": entries})

# --- Example function to queue a welcome playback (calls Twilio update to change Twiml) ---
async def queue_welcome_playback(call_sid: str, text_to_say: str = "Connecting you to your AI assistant..."):
    """
    Build minimal TwiML with <Say> and <Start><Stream> and update the Twilio call.
    This is what in your logs was failing with a 401.
    """
    stream_url = build_stream_url()
    # Build TwiML; adapt to your original format and voice selection
    twiml = (
        f"<Response>\n"
        f"  <Say voice=\"Polly.Joanna\">{text_to_say}</Say>\n"
        f"  <Start>\n"
        f"    <Stream url=\"{stream_url}\" />\n"
        f"  </Start>\n"
        f"</Response>"
    )
    log.debug("Built twiml: %s", twiml)
    try:
        resp = await update_call_twiml_async(call_sid, twiml)
        return resp
    except TwilioRestException:
        # already logged in update_call_twiml_async; return None for caller handling
        return None

# --- Application startup code (simple aiohttp server)
def create_app():
    app = web.Application()
    app.add_routes([web.get("/debug/markers", debug_markers)])
    return app

# If you want to run a simple aiohttp server for the debug endpoint:
if __name__ == "__main__":
    # Run aiohttp debug server on port 10000 (to match original debug endpoint)
    app = create_app()
    port = int(os.getenv("PORT", "10000"))
    host = os.getenv("HOST", "127.0.0.1")
    log.info("Starting debug http server on %s:%s (markers path: %s)", host, port, "/debug/markers")
    web.run_app(app, host=host, port=port)
