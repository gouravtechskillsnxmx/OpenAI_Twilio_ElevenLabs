# ws_server_wscheck.py - Enhanced server that writes marker files to /tmp on WebSocket handshake and start events.
# Purpose: help verify whether Twilio successfully opens a WSS connection and when the stream 'start' event includes a call SID.
# Marker files created:
#  - /tmp/ws_handshake_<ts>.json  -> created on websocket accept (contains client host/port and timestamp)
#  - /tmp/ws_connected_<callSid>.json -> created on receiving 'start' event with callSid (contains JSON with timestamp and start payload)
#  - /tmp/ws_start_<ts>.json -> created on receiving 'start' event even if callSid missing (contains start payload)

import os
import asyncio
import logging
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import threading

from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client as TwilioClient

# Config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://openai-twilio-elevenlabs.onrender.com")
TTS_DIR = Path(os.getenv("TTS_DIR", "/tmp"))
MARKER_DIR = Path(os.getenv("MARKER_DIR", "/tmp"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "0.5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("POLL_TIMEOUT_SECONDS", "5.0"))

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set.")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s")
logger = logging.getLogger("ws_server_wscheck")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s pid=%(process)d tid=%(thread)d %(name)s %(message)s"))
if not logger.handlers:
    logger.addHandler(console)

app = FastAPI(title="Twilio Media + TTS Server (ws handshake check)")

TTS_DIR.mkdir(parents=True, exist_ok=True)
MARKER_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/tts_static", StaticFiles(directory=str(TTS_DIR)), name="tts_static")

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def write_marker_file(prefix: str, name: str, payload: dict):
    # Write a small JSON marker file to MARKER_DIR with given prefix and name.
    # Returns the path string or None on failure.
    try:
        fname = f"{prefix}_{name}.json"
        path = MARKER_DIR / fname
        with open(path, "w") as f:
            json.dump(payload, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                logger.exception("fsync failed for marker %s", path)
        logger.info("WROTE marker %s", path)
        return str(path)
    except Exception as ex:
        logger.exception("Failed to write marker file %s_%s: %s", prefix, name, ex)
        return None

@app.post("/twiml_stream")
async def twiml_stream(request: Request):
    logger.debug("twiml_stream: received request headers=%s", dict(request.headers))
    form = await request.form()
    call_sid = form.get("CallSid") or form.get("callSid") or form.get("callsid")
    logger.info("Received Twilio webhook /twiml_stream callSid=%s From=%s To=%s form_keys=%s", call_sid, form.get("From"), form.get("To"), list(form.keys()))
    safe_form = {}
    try:
        for k, v in form.items():
            sv = v
            if 'token' in k.lower() or 'auth' in k.lower() or 'secret' in k.lower():
                sv = 'REDACTED'
            safe_form[k] = sv
        logger.debug("twiml_stream: form (redacted)=%s", json.dumps(safe_form))
    except Exception:
        logger.exception("twiml_stream: failed to redact form data")
    host = request.headers.get('host') or ''
    stream_url = f"wss://{host}/twilio-media"
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{stream_url}"/>
  </Start>
</Response>""".format(stream_url=stream_url)
    logger.info("twiml_stream: Returning TwiML Start/Stream -> %s for call %s twiml_len=%d", stream_url, call_sid, len(twiml))
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-media")
async def twilio_media_ws(ws: WebSocket):
    # Accept the websocket and write a handshake marker
    await ws.accept()
    client = getattr(ws, "client", None)
    client_host = None
    client_port = None
    try:
        client_host = getattr(client, "host", None)
        client_port = getattr(client, "port", None)
    except Exception:
        pass
    ts = int(time.time() * 1000)
    handshake_name = f"{ts}"
    handshake_payload = {
        "timestamp": now_iso(),
        "client_host": client_host,
        "client_port": client_port,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }
    write_marker_file("ws_handshake", handshake_name, handshake_payload)
    logger.info("WebSocket accepted client_host=%s client_port=%s pid=%d tid=%d handshake_marker=%s", client_host, client_port, os.getpid(), threading.get_ident(), handshake_name)

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
                # Attempt to get callSid from common keys
                call_sid = start_payload.get("callSid") or start_payload.get("call_sid") or start_payload.get("callSid") or start_payload.get("call_sid") or None
                ts2 = int(time.time() * 1000)
                if call_sid:
                    marker_name = f"{call_sid}"
                    marker_payload = {
                        "timestamp": now_iso(),
                        "callSid": call_sid,
                        "start_payload": start_payload,
                        "msg_count": msg_count
                    }
                    write_marker_file("ws_connected", marker_name, marker_payload)
                    logger.info("INBOUND stream started msg_count=%d callSid=%s -> wrote marker ws_connected_%s.json", msg_count, call_sid, call_sid)
                else:
                    # Write generic start marker with timestamp
                    marker_name = f"{ts2}"
                    write_marker_file("ws_start", marker_name, {"timestamp": now_iso(), "start_payload": start_payload, "msg_count": msg_count})
                    logger.info("INBOUND stream started msg_count=%d (no callSid found) -> wrote marker ws_start_%s.json", msg_count, marker_name)
            elif event == "media":
                payload = obj.get("media", {}).get("payload")
                if payload:
                    payload_len = len(payload)
                    approx_bytes = int(payload_len * 3 / 4)
                    logger.info("INBOUND media event msg_count=%d payload_len=%d approx_bytes=%d", msg_count, payload_len, approx_bytes)
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
        try:
            await ws.close()
        except Exception:
            logger.exception("Error closing websocket")
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
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=safe_name)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ws_server_wscheck uvicorn app on port %s pid=%d", os.getenv("PORT", "10000"), os.getpid())
    uvicorn.run("ws_server_wscheck:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), log_level="info")
