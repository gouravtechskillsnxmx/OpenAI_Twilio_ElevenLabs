"""
ws_server_full_logging.py - Highly instrumented Twilio Media + TTS server to aid RCA.
Features:
 - Detailed logging of incoming /twiml_stream webhooks (form + headers, redacted)
 - Twilio REST fetch/update timing and safe repr dumps
 - TTS file write with fsync and serve endpoint
 - WebSocket handler that logs ASGI scope (handshake headers), writes marker files on accept and on 'start' event,
   logs media/stop events and writes additional markers for troubleshooting
 - Health endpoints and a fallback TwiML <Play> endpoint to test audio delivery without websockets
 - Minimal behavioral changes (no functional change to how call redirect works) — only extra logging and markers
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
from starlette.websockets import WebSocketDisconnect

# --- Configuration ----------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://openai-twilio-elevenlabs.onrender.com")
TTS_DIR = Path(os.getenv("TTS_DIR", "/tmp"))
MARKER_DIR = Path(os.getenv("MARKER_DIR", "/tmp"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "0.5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("POLL_TIMEOUT_SECONDS", "5.0"))
TEST_AUDIO_URL = os.getenv("TEST_AUDIO_URL", "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")

# Validate env vars
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables must be set.")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Logging ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s")
logger = logging.getLogger("ws_server_full")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s")
console.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console)

# --- FastAPI app ------------------------------------------------------------
app = FastAPI(title="Twilio Media + TTS Server (full-logging)")

# Mount static dir and ensure markers exist
TTS_DIR.mkdir(parents=True, exist_ok=True)
MARKER_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tts_static", StaticFiles(directory=str(TTS_DIR)), name="tts_static")


# --- Helpers ---------------------------------------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def monotonic_ms():
    return int(time.monotonic() * 1000)

def mask_secret(s):
    if not s:
        return None
    s = str(s)
    if len(s) <= 8:
        return "****"
    return s[:4] + "..." + s[-4:]

def task_name_or_none():
    try:
        return asyncio.current_task().get_name()
    except Exception:
        return None

def write_marker(prefix: str, name: str, payload: dict):
    """Write a small JSON marker file to MARKER_DIR with given prefix and name."""
    try:
        fname = f"{prefix}_{name}.json"
        path = MARKER_DIR / fname
        with open(path, "w") as f:
            json.dump(payload, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                logger.exception("write_marker: fsync failed for %s", path)
        logger.info("WROTE marker %s", path)
        return str(path)
    except Exception as ex:
        logger.exception("write_marker: failed to write marker %s_%s: %s", prefix, name, ex)
        return None

# --- NEW: safe websocket close helper (single change: replace await ws.close() calls with this) ---
async def safe_close_ws(ws: WebSocket):
    """Attempt to close the websocket but ignore the RuntimeError raised if ASGI already closed it."""
    try:
        await ws.close()
    except RuntimeError as e:
        # This is expected in races where the ASGI layer already sent websocket.close
        logger.debug("safe_close_ws: ws.close() raised RuntimeError (already closed?): %s", e)
    except Exception as e:
        # Log unexpected closing errors with stack
        logger.exception("safe_close_ws: unexpected exception while closing websocket: %s", e)

# --- Twilio REST helpers --------------------------------------------------
async def call_get_status(call_sid: str) -> Optional[str]:
    start = monotonic_ms()
    logger.debug("call_get_status: entering call_sid=%s at=%s task=%s", call_sid, now_iso(), task_name_or_none())
    try:
        logger.info("BEGIN Twilio API Request (fetch call) call_sid=%s start_ms=%d", call_sid, start)
        call = twilio_client.calls(call_sid).fetch()
        elapsed = monotonic_ms() - start
        logger.info("END Twilio API Request (fetch call) call_sid=%s elapsed_ms=%d repr_len=%d", call_sid, elapsed, len(repr(call)))
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
        logger.debug("call_get_status: traceback: %s", traceback.format_exc())
        return None

async def play_tts_with_wait(call_sid: str, tts_url: str, timeout_s: float = POLL_TIMEOUT_SECONDS, poll_interval: float = POLL_INTERVAL_SECONDS) -> bool:
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
            status = None

        elapsed_loop = monotonic_ms() - loop_start
        logger.debug("play_tts_with_wait: loop=%d elapsed_ms=%d status=%s", loop_count, elapsed_loop, status)

        if status:
            last_status = status
            logger.info("play_tts_with_wait: call=%s loop=%d status=%s deadline_in_ms=%d", call_sid, loop_count, status, int((deadline - datetime.utcnow()).total_seconds() * 1000))
            if status != "in-progress":
                logger.debug("play_tts_with_wait: call=%s not yet in-progress (current=%s)", call_sid, status)
            if status == "in-progress":
                twiml = f"<Response><Play>{tts_url}</Play></Response>"
                try:
                    logger.info("play_tts_with_wait: attempting calls(%s).update(twiml=...) at=%s", call_sid, now_iso())
                    resp = twilio_client.calls(call_sid).update(twiml=twiml)
                    logger.info("play_tts_with_wait: update returned repr=%s", repr(resp))
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
                    logger.exception("play_tts_with_wait: failed to update call %s while in-progress: %s", call_sid, ex)
                    logger.debug("play_tts_with_wait: update exception traceback: %s", traceback.format_exc())
                    try:
                        logger.debug("play_tts_with_wait: context pid=%d tid=%d task=%s env_base_url=%s", os.getpid(), threading.get_ident(), task_name_or_none(), mask_secret(BASE_URL))
                        logger.debug("play_tts_with_wait: masked_twilio_sid=%s masked_token=%s", mask_secret(TWILIO_ACCOUNT_SID), mask_secret(TWILIO_AUTH_TOKEN))
                    except Exception:
                        logger.exception("play_tts_with_wait: failed to log extra context")
                    return False
        else:
            logger.warning("play_tts_with_wait: could not read status for call %s on loop %d, will retry (deadline in %.1fs)", call_sid, loop_count, (deadline - datetime.utcnow()).total_seconds())
        try:
            sleep_ms = int(poll_interval * 1000)
            logger.debug("play_tts_with_wait: sleeping for %d ms (loop=%d)", sleep_ms, loop_count)
            await asyncio.sleep(poll_interval)
        except Exception:
            logger.exception("play_tts_with_wait: sleep interrupted")
    logger.warning("play_tts_with_wait: timed out waiting for in-progress call=%s last_status=%s loops=%d", call_sid, last_status, loop_count)
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
    ts = int(datetime.utcnow().timestamp() * 1000)
    fname = f"tts_{call_sid}_{ts}.wav"
    url = f"{BASE_URL}/tts/{fname}"
    logger.debug("tts_file_url_for: call_sid=%s -> url=%s fname=%s", call_sid, url, fname)
    return url, fname

# --- Routes -----------------------------------------------------------------
@app.post("/twiml_stream")
async def twiml_stream(request: Request):
    logger.debug("twiml_stream: received request headers=%s", dict(request.headers))
    form = await request.form()
    call_sid = form.get("CallSid") or form.get("callSid") or form.get("callsid")
    logger.info("Received Twilio webhook /twiml_stream callSid=%s From=%s To=%s form_keys=%s pid=%d", call_sid, form.get("From"), form.get("To"), list(form.keys()), os.getpid())
    try:
        safe_form = {}
        for k, v in form.items():
            sk = k
            sv = v
            if 'token' in k.lower() or 'auth' in k.lower() or 'secret' in k.lower():
                sv = 'REDACTED'
            safe_form[sk] = sv
        logger.debug("twiml_stream: form (redacted)=%s", json.dumps(safe_form))
    except Exception:
        logger.exception("twiml_stream: failed to log form data for call %s", call_sid)
    host = request.headers.get('host') or ''
    stream_url = f"wss://{host}/twilio-media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{stream_url}"/>
  </Start>
</Response>"""
    logger.info("twiml_stream: Returning TwiML Start/Stream -> %s for call %s twiml_len=%d", stream_url, call_sid, len(twiml))
    return Response(content=twiml, media_type="application/xml")

@app.post("/twiml_fallback_play")
async def twiml_fallback_play(request: Request):
    """Simple TwiML <Play> fallback to test audio delivery to Twilio without websockets"""
    form = await request.form()
    call_sid = form.get("CallSid")
    logger.info("twiml_fallback_play: received test play request callSid=%s From=%s To=%s", call_sid, form.get("From"), form.get("To"))
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{TEST_AUDIO_URL}</Play>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.get("/health")
async def health():
    return PlainTextResponse("ok")

@app.websocket("/twilio-media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept()
    # --- LOG ASGI SCOPE + WRITE HANDSHAKE SCOPE MARKER --------------------------------
    try:
        scope = getattr(ws, "scope", {}) or {}
        headers = {}
        for h in scope.get("headers", []):
            try:
                k = h[0].decode() if isinstance(h[0], (bytes, bytearray)) else str(h[0])
                v = h[1].decode() if isinstance(h[1], (bytes, bytearray)) else str(h[1])
            except Exception:
                k = str(h[0]); v = str(h[1])
            kl = k.lower()
            if "authorization" in kl or "cookie" in kl or "token" in kl or "auth" in kl or "x-twilio-signature" in kl:
                v = "***REDACTED***"
            headers[k] = v
        client_info = scope.get("client")
        path_info = scope.get("path")
        logger.info("WS ASGI scope client=%s path=%s headers_sample=%s", client_info, path_info, dict(list(headers.items())[:20]))
        try:
            ts_accept = int(time.time() * 1000)
            marker = {
                "timestamp": now_iso(),
                "client": client_info,
                "path": path_info,
                "headers": headers,
                "pid": os.getpid(),
                "tid": threading.get_ident()
            }
            marker_path = f"/tmp/ws_accept_{ts_accept}_scope.json"
            with open(marker_path, "w") as mf:
                json.dump(marker, mf)
                mf.flush()
                try:
                    os.fsync(mf.fileno())
                except Exception:
                    logger.exception("failed fsync marker %s", marker_path)
            logger.info("WROTE WS accept scope marker %s", marker_path)
            write_marker("ws_handshake", str(ts_accept), marker)
        except Exception:
            logger.exception("failed to write ws accept scope marker")
    except Exception:
        logger.exception("Failed to log websocket scope on accept")
    # ------------------------------------------------------------------------------------

    client = None
    try:
        client = getattr(ws, "client", None)
    except Exception:
        pass
    logger.info("WebSocket accepted client=%s remote=%s pid=%d tid=%d", getattr(client, 'host', None), getattr(client, 'port', None), os.getpid(), threading.get_ident())

    msg_count = 0
    try:
        while True:
            msg = await ws.receive_text()
            msg_count += 1
            logger.debug("WS received raw length=%d first200=%s", len(msg), msg[:200])
            try:
                obj = json.loads(msg)
            except Exception:
                logger.exception("Failed to json.loads websocket message on msg_count=%d", msg_count)
                continue

            event = obj.get("event")
            if event == "start":
                start_payload = obj.get("start", {})
                # Attempt to extract callSid from different possible keys
                call_sid = start_payload.get("callSid") or start_payload.get("call_sid") or start_payload.get("CallSid") or start_payload.get("callSid")
                ts2 = int(time.time() * 1000)
                if call_sid:
                    marker_payload = {
                        "timestamp": now_iso(),
                        "callSid": call_sid,
                        "start_payload": start_payload,
                        "msg_count": msg_count
                    }
                    write_marker("ws_connected", call_sid, marker_payload)
                    logger.info("INBOUND stream started msg_count=%d callSid=%s -> wrote marker ws_connected_%s.json", msg_count, call_sid, call_sid)
                else:
                    marker_name = f"{ts2}"
                    write_marker("ws_start", marker_name, {"timestamp": now_iso(), "start_payload": start_payload, "msg_count": msg_count})
                    logger.info("INBOUND stream started msg_count=%d (no callSid found) -> wrote marker ws_start_%s.json", msg_count, marker_name)
            elif event == "media":
                payload = obj.get("media", {}).get("payload")
                if payload:
                    payload_len = len(payload)
                    approx_bytes = int(payload_len * 3 / 4)
                    logger.info("INBOUND media event msg_count=%d payload_len=%d approx_bytes=%d", msg_count, payload_len, approx_bytes)
                    # For deeper debugging, optionally write a small sample marker
                    if msg_count % 50 == 0:
                        try:
                            sample_marker = {"timestamp": now_iso(), "msg_count": msg_count, "sample_start": payload[:64], "sample_end": payload[-64:]}
                            write_marker("ws_media_sample", str(int(time.time()*1000)), sample_marker)
                        except Exception:
                            logger.exception("failed to write media sample marker")
                else:
                    logger.warning("INBOUND media event without payload on msg_count=%d", msg_count)
            elif event == "stop":
                logger.info("Stream stop event received msg_count=%d: %s", msg_count, json.dumps(obj.get('stop', {})))
            else:
                logger.debug("Unhandled Twilio media event msg_count=%d event=%s obj_keys=%d", msg_count, event, len(obj))
    except Exception as ex:
        logger.info("WS closed unexpectedly: %s", ex)
        logger.debug("twilio_media_ws: traceback: %s", traceback.format_exc())
    finally:
        # REPLACED direct await ws.close() with safe_close_ws(ws)
        try:
            await safe_close_ws(ws)
        except Exception:
            logger.exception("Error in safe_close_ws")
        logger.info("WS closed (handler exit) total_msgs=%d", msg_count)


@app.get("/tts/{fname}")
async def serve_tts(fname: str):
    safe_name = Path(fname).name
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
    try:
        st = file_path.stat()
        logger.debug("serve_tts: file mode=%o uid=%s gid=%s mtime=%s", st.st_mode, getattr(st, 'st_uid', None), getattr(st, 'st_gid', None), datetime.utcfromtimestamp(st.st_mtime).isoformat())
    except Exception:
        logger.exception("serve_tts: unable to stat file for %s", file_path)
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=safe_name)


# --- Example orchestration for playing TTS into a call ----------------------
async def handle_playback_for_call(call_sid: str, tts_bytes: bytes) -> bool:
    logger.debug("handle_playback_for_call: start call_sid=%s bytes=%d pid=%d task=%s", call_sid, len(tts_bytes), os.getpid(), task_name_or_none())
    url, fname = tts_file_url_for(call_sid)
    file_path = TTS_DIR / fname
    logger.info("handle_playback_for_call: will write tts file %s (bytes=%d)", file_path, len(tts_bytes))

    try:
        with open(file_path, "wb") as f:
            f.write(tts_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
                logger.debug("handle_playback_for_call: fsync succeeded for %s", file_path)
            except Exception:
                logger.exception("handle_playback_for_call: fsync failed for %s", file_path)
        final_size = file_path.stat().st_size
        logger.info("handle_playback_for_call: file written ok path=%s size=%d", file_path, final_size)
    except Exception as ex:
        logger.exception("handle_playback_for_call: failed to write tts file %s: %s", file_path, ex)
        logger.debug("handle_playback_for_call: traceback: %s", traceback.format_exc())
        return False

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

@app.get("/debug/markers")
async def list_markers():
    """Return list of recent marker files and contents (for debugging via Postman)."""
    import glob, json, os, pathlib, time
    files = sorted(glob.glob("/tmp/ws_*.json"), key=os.path.getmtime, reverse=True)
    latest = []
    for f in files[:10]:  # limit to 10 newest
        try:
            stat = os.stat(f)
            with open(f) as fp:
                data = json.load(fp)
            latest.append({
                "name": pathlib.Path(f).name,
                "size": stat.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(stat.st_mtime)),
                "data": data
            })
        except Exception as e:
            latest.append({"name": f, "error": str(e)})
    return {"count": len(latest), "markers": latest}


# If run directly, start uvicorn for local dev.
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ws_server_full_logging uvicorn app on port %s pid=%d", os.getenv("PORT", "10000"), os.getpid())
    uvicorn.run("ws_server_full_logging:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), log_level="info")
