from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from .models import AppState, MarketState, OrderBookState



# --- Helpers moved below imports (fix for future import placement) ---
def _estimate_quote_vol_60m(m: MarketState, now: float) -> float:
    """Approximate quote volume in last 60m from trade prints (size * price)."""
    total = 0.0
    cutoff = now - 60 * 60
    for ts, px, sz, side in list(m.trades):
        if ts >= cutoff and px and sz:
            total += float(px) * float(sz)
    return total

# --- End moved helpers ---

def _find_price_at_or_before(prices: List[Tuple[float, float]], target_ts: float) -> Optional[float]:
    for ts, px in reversed(prices):
        if ts <= target_ts:
            return px
    return None


def _minute_closes(prices: List[Tuple[float, float]], window_minutes: int) -> List[Tuple[int, float]]:
    """Return (minute_epoch, close_price) for the last window_minutes.
    minute_epoch is epoch//60 for easy alignment.
    """
    if not prices:
        return []
    latest_ts = prices[-1][0]
    start_ts = latest_ts - window_minutes * 60
    out: Dict[int, float] = {}
    for ts, px in prices:
        if ts < start_ts:
            continue
        m = int(ts // 60)
        out[m] = px  # last within that minute
    mins = sorted(out.keys())
    return [(m, out[m]) for m in mins]


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, var))


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 10:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for a, b in zip(x, y):
        xa = a - mx
        yb = b - my
        num += xa * yb
        dx += xa * xa
        dy += yb * yb
    den = math.sqrt(dx * dy)
    if den == 0:
        return None
    return num / den


def _window_trade_sums(trades: List[Tuple[float, float, float, str]], cutoff_ts: float) -> Tuple[float, float, float, float]:
    """Return buy_notional, sell_notional, buy_count, sell_count in quote units."""
    buy_not = sell_not = 0.0
    buy_n = sell_n = 0.0
    for ts, px, sz, side in reversed(trades):
        if ts < cutoff_ts:
            break
        notional = px * sz
        if side == "buy":
            buy_not += notional
            buy_n += 1
        else:
            sell_not += notional
            sell_n += 1
    return buy_not, sell_not, buy_n, sell_n


def _vwap(trades: List[Tuple[float, float, float, str]], cutoff_ts: float) -> Optional[float]:
    num = 0.0
    den = 0.0
    for ts, px, sz, _side in reversed(trades):
        if ts < cutoff_ts:
            break
        num += px * sz
        den += sz
    if den == 0:
        return None
    return num / den


def _depth_within_bps(book: OrderBookState, mid: float, bps: float) -> Tuple[float, float]:
    """Return (bid_depth_usd, ask_depth_usd) within +/-bps around mid."""
    if mid <= 0:
        return 0.0, 0.0
    bid_min = mid * (1 - bps / 10000.0)
    ask_max = mid * (1 + bps / 10000.0)

    bid_depth = 0.0
    for px, sz in book.bids.items():
        if px >= bid_min:
            bid_depth += px * sz
    ask_depth = 0.0
    for px, sz in book.asks.items():
        if px <= ask_max:
            ask_depth += px * sz
    return bid_depth, ask_depth


def _obi(bid_depth: float, ask_depth: float) -> float:
    den = bid_depth + ask_depth
    if den <= 0:
        return 0.0
    return (bid_depth - ask_depth) / den


def _impact_cost(book: OrderBookState, side: str, notional_usd: float) -> Optional[float]:
    """Estimate relative impact cost (VWAP vs mid) to trade notional_usd."""
    if not book.best_bid or not book.best_ask:
        return None
    mid = (book.best_bid + book.best_ask) / 2.0
    if mid <= 0:
        return None

    remaining = notional_usd
    cost = 0.0
    filled = 0.0

    if side == "buy":
        # consume asks from low to high
        for px in sorted(book.asks.keys()):
            sz = book.asks[px]
            level_not = px * sz
            take = min(level_not, remaining)
            if take > 0:
                cost += take
                filled += take
                remaining -= take
            if remaining <= 0:
                break
        if filled <= 0:
            return None
        vwap = cost / filled
        return max(0.0, (vwap - mid) / mid)

    if side == "sell":
        # consume bids from high to low
        for px in sorted(book.bids.keys(), reverse=True):
            sz = book.bids[px]
            level_not = px * sz
            take = min(level_not, remaining)
            if take > 0:
                cost += take
                filled += take
                remaining -= take
            if remaining <= 0:
                break
        if filled <= 0:
            return None
        vwap = cost / filled
        # selling pushes price down relative to mid; treat impact as (mid - vwap)/mid
        return max(0.0, (mid - vwap) / mid)

    return None


def _microprice(book: OrderBookState) -> Optional[float]:
    if not (book.best_bid and book.best_ask and book.best_bid_size and book.best_ask_size):
        return None
    den = book.best_bid_size + book.best_ask_size
    if den <= 0:
        return None
    # classic microprice leaning towards the heavier side
    return (book.best_ask * book.best_bid_size + book.best_bid * book.best_ask_size) / den


def _market_regime_penalty(state: AppState) -> float:
    """Return a multiplier (<=1) to dampen long signals in risk-off regimes."""
    btc = state.markets.get("BTC-USD")
    if not btc or btc.last_price is None:
        return 1.0
    prices = list(btc.prices)
    if len(prices) < 10:
        return 1.0
    ts_now = prices[-1][0]
    p15 = _find_price_at_or_before(prices, ts_now - 15 * 60)
    p60 = _find_price_at_or_before(prices, ts_now - 60 * 60)
    if not p15 or not p60:
        return 1.0
    r15 = btc.last_price / p15 - 1.0
    r60 = btc.last_price / p60 - 1.0
    closes = _minute_closes(prices, 60)
    rets = []
    for i in range(1, len(closes)):
        prev = closes[i-1][1]
        cur = closes[i][1]
        if prev > 0:
            rets.append(math.log(cur/prev))
    vol = _std(rets)

    # simple regime rules
    if r15 < -0.006 and vol > 0.006:
        return 0.65
    if r60 < -0.012 and vol > 0.008:
        return 0.55
    if r15 < -0.004:
        return 0.8
    return 1.0


def score_opportunities(
    state: AppState,
    horizon_minutes: int = 60,
    limit: int = 10,
    min_quote_vol_usd_24h: float = 5_000_000,
    max_spread_pct: float = 0.006,
    shortlist_candidates: int = 80,
    impact_notional_usd: float = 250.0,
    corr_window_minutes: int = 120,
    corr_threshold: float = 0.90,
) -> Dict:
    now = time.time()
    uptime = now - state.started_at
    warmup = "warming_up" if uptime < 20 * 60 else ("partial" if uptime < 75 * 60 else "ready")
    regime_mult = _market_regime_penalty(state)

    base_rows = []
    for pid in state.tracked_product_ids:
        m = state.markets.get(pid)
        if not m or m.last_price is None:
            continue
        # stale gate (more forgiving): allow up to 15 minutes since last market update
        last_update = None
        if m.last_ticker_at:
            last_update = m.last_ticker_at
        if m.last_trade_at:
            last_update = max(last_update or 0.0, m.last_trade_at)
        if last_update and (now - last_update) > 15 * 60:
            continue

        prices = list(m.prices)
        if len(prices) < 1:
            continue
        ts_now = prices[-1][0]
        price_now = m.last_price
        p15 = _find_price_at_or_before(prices, ts_now - 15 * 60)
        p60 = _find_price_at_or_before(prices, ts_now - 60 * 60)
        ret_15m = (price_now / p15 - 1.0) if p15 else None
        ret_60m = (price_now / p60 - 1.0) if p60 else None

        # realized vol using 1m closes over 60m
        closes = _minute_closes(prices, 60)
        rlog = []
        for i in range(1, len(closes)):
            prev = closes[i-1][1]
            cur = closes[i][1]
            if prev > 0 and cur > 0:
                rlog.append(math.log(cur / prev))
        vol_60m = _std(rlog)
        vol_60m = max(vol_60m, 1e-6)

        # spread now and avg last 5m
        spread_now = None
        if m.best_bid and m.best_ask:
            mid = (m.best_bid + m.best_ask) / 2.0
            if mid > 0:
                spread_now = (m.best_ask - m.best_bid) / mid
        spread_avg_5m = None
        if m.spreads:
            cutoff = ts_now - 5 * 60
            vals = [sp for ts, sp in m.spreads if ts >= cutoff]
            if vals:
                spread_avg_5m = sum(vals) / len(vals)

        quote_vol_usd_24h = None
        if m.volume_24h_base is not None:
            quote_vol_usd_24h = m.volume_24h_base * price_now

        # base gating when ready
        if warmup == "ready":
            if spread_now is not None and spread_now > max_spread_pct:
                continue
            if quote_vol_usd_24h is not None and quote_vol_usd_24h < min_quote_vol_usd_24h:
                continue

        # base score: vol-adjusted momentum with liquidity nudge, spread penalty
        r15 = ret_15m or 0.0
        r60 = ret_60m or 0.0
        mom15 = r15 / vol_60m
        mom60 = r60 / vol_60m
        liq = math.log1p((quote_vol_usd_24h or 0.0) / 1e6)  # gentle
        sp = spread_now or 0.0

        base_score = (0.55 * mom15) + (0.25 * mom60) + (0.10 * liq) - (4.0 * sp)
        base_score *= regime_mult

        base_rows.append({
            "product_id": pid,
            "m": m,
            "price": price_now,
            "ret_15m": ret_15m,
            "ret_60m": ret_60m,
            "vol_60m": vol_60m,
            "spread_pct": spread_now,
            "spread_avg_5m": spread_avg_5m,
            "quote_vol_usd_24h": quote_vol_usd_24h,
            "base_score": base_score,
        })

    base_rows.sort(key=lambda x: x["base_score"], reverse=True)
    candidates = base_rows[:max(shortlist_candidates, limit * 8)]

    # Full scoring for candidates (adds microstructure, flow, impact)
    full = []
    for row in candidates:
        pid = row["product_id"]
        m: MarketState = row["m"]
        prices = list(m.prices)
        ts_now = prices[-1][0]
        price_now = row["price"]

        # Trades-based features
        trades = list(m.trades)
        buy5, sell5, bn5, sn5 = _window_trade_sums(trades, ts_now - 5 * 60)
        buy15, sell15, bn15, sn15 = _window_trade_sums(trades, ts_now - 15 * 60)

        tfi5 = (buy5 - sell5) / (buy5 + sell5) if (buy5 + sell5) > 0 else None
        tfi15 = (buy15 - sell15) / (buy15 + sell15) if (buy15 + sell15) > 0 else None

        # 5m volume anomaly (notional) vs 60m baseline
        buy60, sell60, _, _ = _window_trade_sums(trades, ts_now - 60 * 60)
        vol5 = (buy5 + sell5)
        vol60 = (buy60 + sell60)
        vol_anom = (vol5 / (vol60 / 12.0)) if vol60 > 0 else None

        vwap5 = _vwap(trades, ts_now - 5 * 60)
        vwap_dev = (price_now / vwap5 - 1.0) if vwap5 else None

        # Order book features (only if we have L2 book)
        obi10 = obi25 = None
        imp = None
        micro_diff = None
        if m.book and m.book.best_bid and m.book.best_ask:
            mid = (m.book.best_bid + m.book.best_ask) / 2.0
            b10, a10 = _depth_within_bps(m.book, mid, 10.0)
            b25, a25 = _depth_within_bps(m.book, mid, 25.0)
            obi10 = _obi(b10, a10)
            obi25 = _obi(b25, a25)

            ib = _impact_cost(m.book, "buy", impact_notional_usd)
            isell = _impact_cost(m.book, "sell", impact_notional_usd)
            if ib is not None and isell is not None:
                imp = (ib + isell) / 2.0
            elif ib is not None:
                imp = ib
            elif isell is not None:
                imp = isell

            mp = _microprice(m.book)
            if mp and mid > 0:
                micro_diff = (mp - mid) / mid

        # Spread blowout penalty: current vs avg 5m
        spread_now = row["spread_pct"]
        spread_avg_5m = row["spread_avg_5m"]
        spread_blow = None
        if spread_now is not None and spread_avg_5m is not None and spread_avg_5m > 0:
            spread_blow = spread_now / spread_avg_5m

        # Activity
        trades_5m = (bn5 + sn5) if (bn5 is not None and sn5 is not None) else 0.0
        tpm_5m = (trades_5m / 5.0) if trades_5m else None

        # Score assembly (heuristic, execution-aware)
        vol_60m = row["vol_60m"]
        mom15 = ((row["ret_15m"] or 0.0) / vol_60m)
        mom60 = ((row["ret_60m"] or 0.0) / vol_60m)

        score = 0.0
        score += 0.55 * mom15
        score += 0.20 * mom60
        if tfi5 is not None:
            score += 0.35 * max(-1.0, min(1.0, tfi5))
        if obi10 is not None:
            score += 0.20 * max(-1.0, min(1.0, obi10))
        if vol_anom is not None:
            score += 0.08 * math.log1p(max(0.0, vol_anom - 1.0))

        # penalties
        sp = spread_now or 0.0
        score -= 4.0 * sp
        if imp is not None:
            score -= 2.5 * imp
        if vwap_dev is not None:
            # penalize being too extended away from vwap (reversion risk)
            score -= 0.6 * abs(vwap_dev) / max(0.002, vol_60m)
        if spread_blow is not None and spread_blow > 1.4:
            score -= 0.25 * (spread_blow - 1.4)

        # regime
        score *= regime_mult

        flags = []
        if spread_now is not None and spread_now > max_spread_pct:
            flags.append("WIDE_SPREAD")
        if row["quote_vol_usd_24h"] is not None and row["quote_vol_usd_24h"] < min_quote_vol_usd_24h:
            flags.append("LOW_LIQUIDITY")
        if m.book is None:
            flags.append("NO_L2")
        if tfi5 is None:
            flags.append("NO_FLOW")
        if vwap5 is None:
            flags.append("NO_VWAP")

        drivers = []
        if row["ret_15m"] is not None:
            drivers.append(f"15m {row['ret_15m']*100:+.2f}% (vol-adj {mom15:+.2f})")
        if row["ret_60m"] is not None:
            drivers.append(f"60m {row['ret_60m']*100:+.2f}% (vol-adj {mom60:+.2f})")
        if tfi5 is not None:
            drivers.append(f"Flow 5m TFI {tfi5:+.2f}")
        if obi10 is not None:
            drivers.append(f"Book OBI(10bps) {obi10:+.2f}")
        if imp is not None:
            drivers.append(f"Impact ${impact_notional_usd:.0f} ~{imp*100:.2f}%")
        if spread_now is not None:
            drivers.append(f"Spread {spread_now*100:.3f}%")
        if vol_anom is not None:
            drivers.append(f"5m vol {vol_anom:.2f}× vs 60m avg")
        if vwap_dev is not None:
            drivers.append(f"VWAP dev {vwap_dev*100:+.2f}%")
        if micro_diff is not None:
            drivers.append(f"Microprice {micro_diff*100:+.2f}% vs mid")

        full.append({
            "product_id": pid,
            "price": round(price_now, 10),
            "ret_15m": row["ret_15m"],
            "ret_60m": row["ret_60m"],
            "vol_anom": vol_anom,
            "tfi_5m": tfi5,
            "tfi_15m": tfi15,
            "obi_10bps": obi10,
            "obi_25bps": obi25,
            "impact_cost": imp,
            "micro_diff": micro_diff,
            "vwap_5m": vwap5,
            "vwap_dev": vwap_dev,
            "spread_pct": spread_now,
            "spread_blow": spread_blow,
            "quote_vol_usd_24h": row["quote_vol_usd_24h"],
            "score": score,
            "flags": flags,
            "drivers": drivers[:8],
        })

    full.sort(key=lambda x: x["score"], reverse=True)

    # Diversify using correlation on 1m returns (greedy)
    # Build return vectors for top subset
    top_for_corr = full[:max(60, limit*10)]
    vectors: Dict[str, List[float]] = {}
    for item in top_for_corr:
        pid = item["product_id"]
        m = state.markets.get(pid)
        if not m:
            continue
        closes = _minute_closes(list(m.prices), corr_window_minutes)
        # we need aligned series length; simplest: take last N returns available
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i-1][1]
            cur = closes[i][1]
            if prev > 0 and cur > 0:
                rets.append(math.log(cur/prev))
        if len(rets) >= 20:
            # take last 90 returns max for speed
            vectors[pid] = rets[-90:]

    selected = []
    for item in full:
        pid = item["product_id"]
        if pid not in vectors:
            # if we can't compute correlation, allow but deprioritize by requiring only when list is empty
            if len(selected) < limit:
                selected.append(item)
            continue
        ok = True
        for s in selected:
            spid = s["product_id"]
            if spid in vectors:
                # align lengths
                x = vectors[pid]
                y = vectors[spid]
                n = min(len(x), len(y))
                corr = _pearson(x[-n:], y[-n:])
                if corr is not None and corr >= corr_threshold:
                    ok = False
                    break
        if ok:
            selected.append(item)
        if len(selected) >= limit:
            break

    # if diversification removed too much, fall back to top scores
    if len(selected) < limit:
        for item in full:
            if item not in selected:
                selected.append(item)
            if len(selected) >= limit:
                break

    note = ""
    if not selected:
        note = "No opportunities yet — waiting for data (normal right after deploy)."
    elif warmup != "ready":
        note = "Warm-up mode: rankings use limited history; expect instability for ~60–90 minutes."

    return {
        "asof": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
        "horizon_minutes": horizon_minutes,
        "opportunities": selected[:limit],
        "note": note,
        "meta": {
            "ws_connected": state.ws_connected,
            "last_msg_seconds_ago": None if state.ws_last_msg_at is None else round(now - state.ws_last_msg_at, 1),
            "tracked_products": len(state.tracked_product_ids),
            "level2_products": len(state.level2_product_ids),
            "ticker_messages": state.ticker_messages,
            "match_messages": state.match_messages,
            "l2_messages": state.l2_messages,
            "status_messages": state.status_messages,
            "warmup": warmup,
            "uptime_minutes": round((now - state.started_at) / 60.0, 1),
            "regime_multiplier": regime_mult,
            "base_rows_count": len(base_rows),
            "candidates_count": len(candidates),
        },
    }
