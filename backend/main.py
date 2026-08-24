"""
main.py — AnemiaScan backend (FastAPI)

Endpoints:
  POST /analyze      — upload an image, get pallor features + risk band
  GET  /screenings    — list recent screening records (for CHW dashboard)
  GET  /health         — basic liveness check

Data handling: images are analysed in-memory and discarded — only the
computed risk band, features, and a timestamp are persisted (see
SECURITY.md "analyse-and-discard" policy).
"""

import os
import sqlite3
import time
import uuid
import json
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analysis import compute_features, ml_classify, load_model, is_ml_available

DB_PATH = os.environ.get("ANEMIASCAN_DB", "screenings.db")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

app = FastAPI(title="AnemiaScan API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load ML model at startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model()   # loads model_rf.pkl if present; logs a warning if not found


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS screenings (
                id TEXT PRIMARY KEY,
                ts REAL,
                band TEXT,
                method TEXT,
                pallor_score REAL,
                erythema_index REAL,
                avg_r REAL, avg_g REAL, avg_b REAL,
                saturation REAL, value REAL
            )"""
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


@app.get("/health")
def health():
    classifier = "random_forest_v1" if is_ml_available() else "rule_based_v0"
    return {
        "status": "ok",
        "classifier": classifier,
        "storage": "supabase" if SUPABASE_URL else "sqlite_demo",
    }


def supabase_request(method: str, path: str, payload: dict | None = None):
    """Server-only Supabase REST helper. Never expose the service-role key to React."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/{path}", data=body, method=method,
        headers={"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode() or "[]")
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail="Screening store is temporarily unavailable.") from exc


def screening_payload(record_id: str, features: dict, result: dict) -> dict:
    # Exclude the internal _colour_features ndarray before persisting
    safe_features = {k: v for k, v in features.items() if not k.startswith("_")}
    return {"id": record_id, "band": result["band"], "method": result["method"], **safe_features}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 8 MB.")
    try:
        features = compute_features(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = ml_classify(features)

    record_id = str(uuid.uuid4())
    # Strip internal keys before storing / returning
    public_features = {k: v for k, v in features.items() if not k.startswith("_")}

    if SUPABASE_URL:
        supabase_request("POST", "screenings", screening_payload(record_id, features, result))
    else:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO screenings
                   (id, ts, band, method, pallor_score, erythema_index, avg_r, avg_g, avg_b, saturation, value)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record_id, time.time(),
                    result["band"], result["method"],
                    public_features["pallor_score"], public_features["erythema_index"],
                    public_features["avg_r"], public_features["avg_g"], public_features["avg_b"],
                    public_features["saturation"], public_features["value"],
                ),
            )

    return {"id": record_id, "features": public_features, "result": result}


@app.get("/screenings")
def list_screenings(limit: int = 50):
    limit = max(1, min(limit, 100))
    if SUPABASE_URL:
        rows = supabase_request("GET", f"screenings?select=*&order=created_at.desc&limit={limit}")
        return {"count": len(rows), "screenings": rows}
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM screenings ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"count": len(rows), "screenings": [dict(r) for r in rows]}
