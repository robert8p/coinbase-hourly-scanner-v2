from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Set

import websockets
from websockets import WebSocketClientProtocol

from .models import AppState, ProductMeta, MarketState, OrderBookState

STABLE_BASES = {"USDC", "USDT", "DAI", "EURC", "TUSD", "USDP"}


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _parse_time_to_epoch(t) -> Optional[float]:
    # Coinbase Exchange WS time strings are ISO8601.
    if not isinstance(t, str):
        return None
    try:
        from datetime import datetime, timezone
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _prune_deques(m: MarketState):
    # keep prices last 3h, spreads last 30m, trades last 30m (deque caps already)
    now = time.time()
    cutoff_prices = now - 3 * 60 * 60
    cutoff_spreads = now - 30 * 60
    cutoff_trades = now - 30 * 60
    while m.prices and m.prices[0][0] < cutoff_prices:
        m.prices.popleft()
    while m.spreads and m.spreads[0][0] < cutoff_spreads:
        m.spreads.popleft()
    while m.trades and m.trades[0][0] < cutoff_trades:
        m.trades.popleft()


async def run_ws_loop(
    state: AppState,
    ws_url: str,
    quote_ccy: str,
    max_products: int,
    shortlist_size: int = 30,
    shortlist_refresh_seconds: int = 120,
    subscribe_chunk_size: int = 100,
):
    """Single-consumer websocket loop (no competing recv coroutines).

    NOTE: The previous versions accidentally had two coroutines calling ws.recv/iterating the socket
    at the same time, which prevents subscriptions from taking effect reliably and causes reconnect loops.
    """
    backoff = 1.0
    while True:
        try:
            state.ws_last_error = None
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                state.ws_connected = True
                state.ws_reconnects += 1
                backoff = 1.0

                # Subscribe to status immediately (must subscribe within 5 seconds per docs).
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "channels": [{"name": "status"}]
                }))

                # Block until we receive the first status message and select the universe.
                product_ids = await _await_first_status_and_select_products(state, ws, quote_ccy, max_products)

                # Subscribe to ticker + matches for the tracked universe (in chunks)
                for chunk in _chunk(product_ids, subscribe_chunk_size):
                    # ticker is widely supported on Coinbase Exchange WS
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "product_ids": chunk,
                        "channels": ["ticker"]
                    }))
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "product_ids": chunk,
                        "channels": ["matches"]
                    }))
                    await asyncio.sleep(0.15)

                # Start shortlist updater for L2 (send-only task)
                l2_task = asyncio.create_task(_level2_shortlist_manager(
                    ws=ws,
                    state=state,
                    shortlist_size=shortlist_size,
                    refresh_seconds=shortlist_refresh_seconds,
                    quote_ccy=quote_ccy,
                ))

                async def _idle_watchdog():
                    # If Coinbase stops sending any messages for a while, force a reconnect.
                    while True:
                        await asyncio.sleep(15)
                        if state.ws_last_msg_at and (time.time() - state.ws_last_msg_at) > 45:
                            state.ws_last_error = "Idle timeout (no WS messages for >45s) — reconnecting"
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            return

                wd_task = asyncio.create_task(_idle_watchdog())


                try:
                    async for msg in ws:
                        state.ws_last_msg_at = time.time()
                        _handle_message(state, msg)
                finally:
                    l2_task.cancel()
                    try:
                        wd_task.cancel()
                    except Exception:
                        pass
                    try:
                        await l2_task
                    except Exception:
                        pass

        except Exception as e:
            state.ws_connected = False
            state.ws_last_error = str(e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.8, 30.0)



async def _await_first_status_and_select_products(
    state: AppState,
    ws: WebSocketClientProtocol,
    quote_ccy: str,
    max_products: int,
) -> List[str]:
    deadline = time.time() + 25
    selected: List[str] = []
    while time.time() < deadline:
        raw = await ws.recv()
        state.ws_last_msg_at = time.time()
        _handle_message(state, raw)
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if data.get("type") != "status":
            continue

        products = data.get("products") or []
        for p in products:
            pid = p.get("id")
            base = p.get("base_currency")
            quote = p.get("quote_currency")
            status = p.get("status")
            if not pid or not base or not quote:
                continue
            state.products[pid] = ProductMeta(
                product_id=pid,
                base_currency=base,
                quote_currency=quote,
                status=status or "",
            )

        candidates = [
            pid for pid, meta in state.products.items()
            if meta.quote_currency == quote_ccy
            and (meta.status or "").lower() == "online"
            and meta.base_currency not in STABLE_BASES
            and pid.endswith(f"-{quote_ccy}")
        ]
        candidates.sort()  # stable order

        # ---- SEED_MAJORS: ensure active markets are included early ----
        seed = [
            f"BTC-{quote_ccy}", f"ETH-{quote_ccy}", f"SOL-{quote_ccy}", f"XRP-{quote_ccy}", f"DOGE-{quote_ccy}",
            f"ADA-{quote_ccy}", f"AVAX-{quote_ccy}", f"LINK-{quote_ccy}", f"LTC-{quote_ccy}", f"DOT-{quote_ccy}",
        ]
        seed = [pid for pid in seed if pid in state.products]
        seen = set(seed)
        filled = seed + [pid for pid in candidates if pid not in seen]
        selected = filled[:max_products]
        # ----------------------------------------------------------------------
        state.tracked_product_ids = selected
        for pid in selected:
            state.markets.setdefault(pid, MarketState())
        return selected

    # Fallback if status doesn't arrive
    fallback = [f"BTC-{quote_ccy}", f"ETH-{quote_ccy}"]
    state.tracked_product_ids = fallback
    for pid in fallback:
        state.markets.setdefault(pid, MarketState())
    return fallback


def _handle_message(state: AppState, raw: str):
    try:
        data = json.loads(raw)
    except Exception:
        return

    mtype = data.get("type")
    if mtype == "error":
        state.ws_last_error = data.get("message") or str(data)
        return
    if mtype == "subscriptions":
        # Successful (re)subscription acknowledgement
        state.ws_last_error = None
    if mtype == "status":
        state.status_messages += 1
        return

    if mtype in ("ticker", "ticker_batch"):
        state.ticker_messages += 1
        pid = data.get("product_id")
        if not pid:
            return
        m = state.markets.get(pid)
        if m is None:
            return

        price = _safe_float(data.get("price"))
        best_bid = _safe_float(data.get("best_bid"))
        best_ask = _safe_float(data.get("best_ask"))
        volume_24h = _safe_float(data.get("volume_24h"))
        ts = _parse_time_to_epoch(data.get("time")) or time.time()

        if price is not None:
            m.last_price = price
            m.prices.append((ts, price))

        if best_bid is not None:
            m.best_bid = best_bid
        if best_ask is not None:
            m.best_ask = best_ask

        if volume_24h is not None:
            m.volume_24h_base = volume_24h

        if m.best_bid and m.best_ask and m.best_ask > 0:
            mid = (m.best_bid + m.best_ask) / 2.0
            if mid > 0:
                spread_pct = (m.best_ask - m.best_bid) / mid
                m.spreads.append((ts, spread_pct))

        m.last_ticker_at = time.time()
        _prune_deques(m)
        return

    if mtype == "match":
        state.match_messages += 1
        pid = data.get("product_id")
        if not pid:
            return
        m = state.markets.get(pid)
        if m is None:
            return
        price = _safe_float(data.get("price"))
        size = _safe_float(data.get("size"))
        side = data.get("side") or ""
        ts = _parse_time_to_epoch(data.get("time")) or time.time()
        if price is None or size is None or side not in ("buy", "sell"):
            return
        m.trades.append((ts, price, size, side))
        m.last_trade_at = time.time()
        # also update last_price / prices from trades if ticker lags
        m.last_price = price
        m.prices.append((ts, price))
        _prune_deques(m)
        return

    if mtype == "snapshot":
        state.l2_messages += 1
        pid = data.get("product_id")
        if not pid:
            return
        m = state.markets.get(pid)
        if m is None:
            return
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        book = m.book or OrderBookState()
        book.bids = {float(px): float(sz) for px, sz in bids}
        book.asks = {float(px): float(sz) for px, sz in asks}
        _recompute_best(book)
        book.last_snapshot_at = time.time()
        book.last_update_at = time.time()
        m.book = book
        return

    if mtype == "l2update":
        state.l2_messages += 1
        pid = data.get("product_id")
        if not pid:
            return
        m = state.markets.get(pid)
        if m is None or m.book is None:
            return
        changes = data.get("changes") or []
        book = m.book
        for side, px, sz in changes:
            pxf = _safe_float(px)
            szf = _safe_float(sz)
            if pxf is None or szf is None:
                continue
            if side == "buy":
                if szf == 0:
                    book.bids.pop(pxf, None)
                else:
                    book.bids[pxf] = szf
            elif side == "sell":
                if szf == 0:
                    book.asks.pop(pxf, None)
                else:
                    book.asks[pxf] = szf
        _recompute_best(book)
        book.last_update_at = time.time()
        return


def _recompute_best(book: OrderBookState):
    if book.bids:
        best_bid = max(book.bids.keys())
        book.best_bid = best_bid
        book.best_bid_size = book.bids.get(best_bid)
    else:
        book.best_bid = None
        book.best_bid_size = None
    if book.asks:
        best_ask = min(book.asks.keys())
        book.best_ask = best_ask
        book.best_ask_size = book.asks.get(best_ask)
    else:
        book.best_ask = None
        book.best_ask_size = None


def _cheap_shortlist_rank(state: AppState, top_liq: int = 120) -> List[str]:
    """Pick candidates for L2 even during warm-up.

    Use a cheap heuristic: liquidity proxy + recent drift (when available).
    This ensures we subscribe to *something* immediately so books populate.
    """
    scored: List[Tuple[float, str]] = []
    for pid in state.tracked_product_ids:
        m = state.markets.get(pid)
        if not m or m.last_price is None:
            continue

        qv = (m.volume_24h_base * m.last_price) if (m.volume_24h_base and m.last_price) else 0.0

        r15 = 0.0
        prices = list(m.prices)
        if prices:
            ts_now = prices[-1][0]
            p15 = None
            for ts, px in reversed(prices):
                if ts <= ts_now - 15 * 60:
                    p15 = px
                    break
            if p15:
                r15 = m.last_price / p15 - 1.0

        score = (0.7 * r15) + (2e-11 * qv)
        scored.append((score, pid))

    scored.sort(reverse=True)

    if not scored:
        return state.tracked_product_ids[:top_liq]

    return [pid for _, pid in scored[:top_liq]]


async def _level2_shortlist_manager(
    ws: WebSocketClientProtocol,
    state: AppState,
    shortlist_size: int,
    refresh_seconds: int,
    quote_ccy: str,
):
    # Wait a little for tickers to start flowing
    await asyncio.sleep(2)
    current: Set[str] = set()

    while True:
        try:
            # choose shortlist candidates
            candidates = _cheap_shortlist_rank(state)
            desired = set(candidates[:shortlist_size])

            # Only subscribe/unsubscribe if there are changes
            to_add = list(desired - current)
            to_remove = list(current - desired)

            # Unsubscribe removed
            if to_remove:
                for chunk in _chunk(to_remove, 100):
                    await ws.send(json.dumps({
                        "type": "unsubscribe",
                        "product_ids": chunk,
                        "channels": ["level2_batch"]
                    }))
                    await asyncio.sleep(0.15)
                for pid in to_remove:
                    ms = state.markets.get(pid)
                    if ms:
                        ms.book = None

            # Subscribe new
            if to_add:
                for chunk in _chunk(to_add, 100):
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "product_ids": chunk,
                        "channels": ["level2_batch"]
                    }))
                    await asyncio.sleep(0.15)
                for pid in to_add:
                    ms = state.markets.get(pid)
                    if ms and ms.book is None:
                        ms.book = OrderBookState()

            current = desired
            state.level2_product_ids = sorted(list(current))
            state.l2_last_error = None
        except Exception as e:
            state.l2_last_error = f"{type(e).__name__}: {e}"
            # Try again next cycle
            pass

        await asyncio.sleep(refresh_seconds)
