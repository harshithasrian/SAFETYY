import os
import sys
import shutil
import tempfile
import json
import secrets
import string
from pathlib import Path
from datetime import date
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import analyze_video, analyze_audio_only, get_system_status
from database.db_manager import (
    get_all_incidents, get_stats, resolve_incident, clear_all_incidents,
    get_all_settings, set_setting,
    get_all_users, get_user_by_id, create_user, update_user,
    set_user_status, reset_user_password, delete_user,
    get_audit_logs, log_action, clear_audit_logs,
    verify_admin,
)

BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = BASE_DIR / "recordings"
UPLOADS_DIR    = BASE_DIR / "uploads"
DEVICES_FILE   = BASE_DIR / "devices.json"
RECORDINGS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SafeGuard AI API", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else (request.client.host or "127.0.0.1")


def _enrich(inc: dict) -> dict:
    if "aggression_score" not in inc or inc["aggression_score"] is None:
        inc["aggression_score"] = 0.0
    if "is_speech_aggressive" not in inc:
        inc["is_speech_aggressive"] = float(inc.get("aggression_score", 0)) >= 0.60
    tp = inc.get("toxic_phrases")
    if isinstance(tp, str):
        try:    inc["toxic_phrases"] = json.loads(tp)
        except: inc["toxic_phrases"] = [tp] if tp else []
    return inc


def _gen_password(n=12) -> str:
    return "".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%") for _ in range(n))


def _load_devices():
    if not DEVICES_FILE.exists(): return []
    try:    return json.loads(DEVICES_FILE.read_text())
    except: return []


def _save_devices(devices):
    DEVICES_FILE.write_text(json.dumps(devices, indent=2))


def _in_range(inc, df, dt):
    try: return df <= date.fromisoformat(inc.get("timestamp", "")[:10]) <= dt
    except: return True


# ── Auth ──────────────────────────────────────────────────
class LoginPayload(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def auth_login(payload: LoginPayload, request: Request):
    import hashlib, hmac, time, base64
    user = verify_admin(payload.username.strip(), payload.password.strip())
    if not user:
        log_action(payload.username, "Failed login", "auth", "auth", _client_ip(request))
        raise HTTPException(401, "Invalid username or password")
    secret = os.environ.get("SAFEGUARD_SECRET", "safeguard-dev-secret")
    data   = f"{user['username']}:{user['role']}:{int(time.time())}"
    sig    = hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()
    token  = base64.b64encode(f"{data}:{sig}".encode()).decode()
    log_action(user["username"], "Logged in", "auth", "auth", _client_ip(request))
    return {"token": token, "username": user["username"], "role": user["role"]}


@app.post("/auth/logout")
def auth_logout(request: Request):
    log_action("admin", "Logged out", "auth", "auth", _client_ip(request))
    return {"ok": True}


# ── Analyze ───────────────────────────────────────────────
@app.post("/analyze-video")
async def analyze_video_endpoint(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = analyze_audio_only(tmp_path, "UPLOAD") if suffix in {".wav", ".mp3"} \
                 else analyze_video(tmp_path, "UPLOAD")
    except Exception as e:
        try: os.remove(tmp_path)
        except: pass
        raise HTTPException(500, f"Analysis failed: {e}")
    try: os.remove(tmp_path)
    except: pass
    log_action("system", "Analyzed upload", file.filename, "incident", _client_ip(request))
    return result


@app.post("/analyze-recording")
async def analyze_recording_endpoint(request: Request, file: UploadFile = File(...)):
    from datetime import datetime
    suffix    = Path(file.filename or "recording.webm").suffix.lower() or ".webm"
    save_name = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    save_path = str(RECORDINGS_DIR / save_name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = analyze_video(save_path, "LIVE_RECORD")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    result["recording_filename"] = save_name
    return result


# ── Incidents ─────────────────────────────────────────────
@app.get("/incidents")
def list_incidents(
    level:     Optional[str] = Query(None),
    source:    Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to:   Optional[str] = Query(None),
    limit:     int           = Query(200),
):
    rows = [_enrich(i) for i in get_all_incidents(limit=limit, level_filter=level)]
    if source:
        rows = [i for i in rows if i.get("video_source") == source]
    if date_from or date_to:
        df = date.fromisoformat(date_from) if date_from else date.min
        dt = date.fromisoformat(date_to)   if date_to   else date.max
        rows = [i for i in rows if _in_range(i, df, dt)]
    return {"incidents": rows, "total": len(rows)}


@app.post("/incidents/{incident_id}/resolve")
def resolve(incident_id: int, request: Request):
    resolve_incident(incident_id)
    log_action("admin", "Resolved incident", f"#{incident_id}", "incident", _client_ip(request))
    return {"ok": True}


@app.delete("/incidents")
def delete_all_incidents(request: Request):
    clear_all_incidents()
    log_action("admin", "Cleared all incidents", "incidents", "incident", _client_ip(request))
    return {"ok": True}


# ── Stats ─────────────────────────────────────────────────
@app.get("/stats")
def stats():
    s       = get_stats()
    all_inc = get_all_incidents(limit=1000)
    avg     = sum(float(i.get("violence_score", 0) or 0) for i in all_inc) / len(all_inc) if all_inc else 0.0
    return {**s, "avg_risk": round(avg, 4)}


# ── Status ────────────────────────────────────────────────
@app.get("/status")
def system_status():
    return get_system_status()


# ── Recordings ────────────────────────────────────────────
@app.get("/recordings/{filename}")
def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    if not str(path.resolve()).startswith(str(RECORDINGS_DIR.resolve())):
        raise HTTPException(403, "Forbidden")
    return FileResponse(str(path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


# ── Settings ──────────────────────────────────────────────
@app.get("/settings")
def get_settings():
    return get_all_settings()


class SettingsPayload(BaseModel):
    violence_threshold: Optional[float] = None
    toxicity_threshold: Optional[float] = None
    visual_weight:      Optional[float] = None
    audio_weight:       Optional[float] = None
    frame_sample_rate:  Optional[int]   = None
    whisper_model:      Optional[str]   = None
    alert_sound:        Optional[bool]  = None


@app.post("/settings")
def save_settings(payload: SettingsPayload, request: Request):
    mapping = {
        "violence_threshold": payload.violence_threshold,
        "toxicity_threshold": payload.toxicity_threshold,
        "visual_weight":      payload.visual_weight,
        "audio_weight":       payload.audio_weight,
        "frame_sample_rate":  payload.frame_sample_rate,
        "whisper_model":      payload.whisper_model,
        "alert_sound":        ("true" if payload.alert_sound else "false")
                              if payload.alert_sound is not None else None,
    }
    for key, val in mapping.items():
        if val is not None:
            set_setting(key, str(val))
    log_action("admin", "Updated settings", "", "settings", _client_ip(request))
    return {"ok": True}


# ── Devices ───────────────────────────────────────────────
class DevicePayload(BaseModel):
    name:     str
    location: str
    rtspUrl:  str
    type:     str  = "camera"
    enabled:  bool = True


@app.get("/devices")
def list_devices():
    return {"devices": _load_devices()}


@app.post("/devices")
def add_device(payload: DevicePayload, request: Request):
    devices = _load_devices()
    new_id  = max((d["id"] for d in devices), default=0) + 1
    device  = {"id": new_id, "name": payload.name, "location": payload.location,
               "rtspUrl": payload.rtspUrl, "type": payload.type,
               "enabled": payload.enabled, "status": "offline", "lastSeen": "Never"}
    devices.append(device)
    _save_devices(devices)
    log_action("admin", "Added device", payload.name, "device", _client_ip(request))
    return device


@app.put("/devices/{device_id}")
def update_device(device_id: int, payload: DevicePayload, request: Request):
    devices = _load_devices()
    for d in devices:
        if d["id"] == device_id:
            d.update({"name": payload.name, "location": payload.location,
                      "rtspUrl": payload.rtspUrl, "type": payload.type,
                      "enabled": payload.enabled})
            _save_devices(devices)
            return d
    raise HTTPException(404, "Device not found")


@app.delete("/devices/{device_id}")
def delete_device(device_id: int, request: Request):
    devices = [d for d in _load_devices() if d["id"] != device_id]
    _save_devices(devices)
    log_action("admin", "Deleted device", f"ID {device_id}", "device", _client_ip(request))
    return {"ok": True}


@app.post("/devices/{device_id}/test")
def test_device(device_id: int, request: Request):
    devices = _load_devices()
    device  = next((d for d in devices if d["id"] == device_id), None)
    if not device:
        raise HTTPException(404, "Device not found")
    ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(device.get("rtspUrl", ""))
        ok  = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            ok = ret
        cap.release()
    except: pass
    for d in devices:
        if d["id"] == device_id:
            d["status"]   = "online" if ok else "offline"
            d["lastSeen"] = "Just now" if ok else d.get("lastSeen", "Never")
    _save_devices(devices)
    return {"ok": ok, "device_id": device_id}


# ── Users ─────────────────────────────────────────────────
class UserCreatePayload(BaseModel):
    name:     str
    email:    str
    role:     str
    password: Optional[str] = None


class UserUpdatePayload(BaseModel):
    name:  str
    email: str
    role:  str


class ResetPasswordPayload(BaseModel):
    new_password: Optional[str] = None


@app.get("/users")
def list_users():
    return {"users": get_all_users()}


@app.post("/users")
def add_user(payload: UserCreatePayload, request: Request):
    password = payload.password or _gen_password()
    try:
        user = create_user(payload.name, payload.email, password, payload.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log_action("admin", "Created user", f"{payload.name} <{payload.email}>", "user", _client_ip(request))
    return {**user, "generated_password": password if not payload.password else None}


@app.put("/users/{user_id}")
def edit_user(user_id: int, payload: UserUpdatePayload, request: Request):
    try:
        user = update_user(user_id, payload.name, payload.email, payload.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not user:
        raise HTTPException(404, "User not found")
    log_action("admin", "Updated user", f"ID {user_id}", "user", _client_ip(request))
    return user


@app.post("/users/{user_id}/disable")
def disable_user(user_id: int, request: Request):
    if not get_user_by_id(user_id):
        raise HTTPException(404, "User not found")
    set_user_status(user_id, "disabled")
    log_action("admin", "Disabled user", f"ID {user_id}", "user", _client_ip(request))
    return {"ok": True}


@app.post("/users/{user_id}/enable")
def enable_user(user_id: int, request: Request):
    if not get_user_by_id(user_id):
        raise HTTPException(404, "User not found")
    set_user_status(user_id, "active")
    log_action("admin", "Enabled user", f"ID {user_id}", "user", _client_ip(request))
    return {"ok": True}


@app.post("/users/{user_id}/reset-password")
def reset_password(user_id: int, payload: ResetPasswordPayload, request: Request):
    if not get_user_by_id(user_id):
        raise HTTPException(404, "User not found")
    new_pw = payload.new_password or _gen_password()
    reset_user_password(user_id, new_pw)
    log_action("admin", "Reset password", f"ID {user_id}", "user", _client_ip(request))
    return {"ok": True, "new_password": new_pw}


@app.delete("/users/{user_id}")
def remove_user(user_id: int, request: Request):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    delete_user(user_id)
    log_action("admin", "Deleted user", user["name"], "user", _client_ip(request))
    return {"ok": True}


# ── Audit Logs ────────────────────────────────────────────
@app.get("/audit-logs")
def get_logs(
    category: Optional[str] = Query(None),
    user:     Optional[str] = Query(None),
    limit:    int           = Query(200),
):
    return {"logs": get_audit_logs(limit=limit, category=category, user=user)}


@app.delete("/audit-logs")
def clear_logs(request: Request):
    clear_audit_logs()
    return {"ok": True}


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)