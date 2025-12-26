from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import AppState
from .ws_client import run_ws_loop
from .scorer import score_opportunities

WS_URL = os.getenv("WS_URL", "wss://ws-feed.exchange.coinbase.com")
QUOTE_CCY = os.getenv("QUOTE_CCY", "USD").upper()
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "300"))

# Scoring / gating
MIN_QUOTE_VOL_USD_24H = float(os.getenv("MIN_QUOTE_VOL_USD_24H", "5000000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.006"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "30"))
SHORTLIST_REFRESH_SECONDS = int(os.getenv("SHORTLIST_REFRESH_SECONDS", "120"))
SHORTLIST_CANDIDATES = int(os.getenv("SHORTLIST_CANDIDATES", "80"))
IMPACT_NOTIONAL_USD = float(os.getenv("IMPACT_NOTIONAL_USD", "250"))
CORR_WINDOW_MINUTES = int(os.getenv("CORR_WINDOW_MINUTES", "120"))
CORR_THRESHOLD = float(os.getenv("CORR_THRESHOLD", "0.90"))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Coinbase Hourly Scanner", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STATE = AppState()

@app.on_event("startup")
async def _startup():
    asyncio.create_task(run_ws_loop(
        state=STATE,
        ws_url=WS_URL,
        quote_ccy=QUOTE_CCY,
        max_products=MAX_PRODUCTS,
        shortlist_size=SHORTLIST_SIZE,
        shortlist_refresh_seconds=SHORTLIST_REFRESH_SECONDS,
    ))

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

@app.get("/api/status")
async def api_status():
    import time
    now = time.time()
    return {
        "ok": True,
        "ws_connected": STATE.ws_connected,
        "ws_last_error": STATE.ws_last_error,
        "l2_last_error": STATE.l2_last_error,
        "ws_reconnects": STATE.ws_reconnects,
        "last_msg_seconds_ago": None if STATE.ws_last_msg_at is None else round(now - STATE.ws_last_msg_at, 1),
        "tracked_products": len(STATE.tracked_product_ids),
        "level2_products": len(STATE.level2_product_ids),
        "ticker_messages": STATE.ticker_messages,
        "match_messages": STATE.match_messages,
        "l2_messages": STATE.l2_messages,
        "status_messages": STATE.status_messages,
        "uptime_seconds": round(now - STATE.started_at, 1),
        "ws_url": WS_URL,
        "quote_ccy": QUOTE_CCY,
        "max_products": MAX_PRODUCTS,
    }

@app.get("/api/opportunities")
async def api_opportunities(horizon: int = 60, limit: int = 10):
    return score_opportunities(
        state=STATE,
        horizon_minutes=horizon,
        limit=limit,
        min_quote_vol_usd_24h=MIN_QUOTE_VOL_USD_24H,
        max_spread_pct=MAX_SPREAD_PCT,
        shortlist_candidates=SHORTLIST_CANDIDATES,
        impact_notional_usd=IMPACT_NOTIONAL_USD,
        corr_window_minutes=CORR_WINDOW_MINUTES,
        corr_threshold=CORR_THRESHOLD,
    )

# Make uptime tools (that use HEAD) happy
@app.head("/api/status")
async def api_status_head():
    return Response(status_code=200)

@app.head("/api/opportunities")
async def api_opportunities_head():
    return Response(status_code=200)
