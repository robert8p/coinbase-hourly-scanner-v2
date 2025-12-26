from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional, Tuple, List
import time


@dataclass
class ProductMeta:
    product_id: str
    base_currency: str
    quote_currency: str
    status: str


@dataclass
class OrderBookState:
    # price -> size (base units). We keep only what the WS gives us (typically top 50).
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)

    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    best_bid_size: Optional[float] = None
    best_ask_size: Optional[float] = None

    last_snapshot_at: Optional[float] = None
    last_update_at: Optional[float] = None


@dataclass
class MarketState:
    # Rolling last ~3h of price points (epoch seconds, price)
    prices: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=9000))
    # Rolling last ~30m of spread samples (epoch seconds, spread_pct)
    spreads: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=3000))

    # Rolling last ~30m of trade prints (epoch seconds, price, size, side)
    trades: Deque[Tuple[float, float, float, str]] = field(default_factory=lambda: deque(maxlen=120000))

    last_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    volume_24h_base: Optional[float] = None  # base units
    last_ticker_at: Optional[float] = None
    last_trade_at: Optional[float] = None

    # Optional L2 book (we only maintain this for a shortlist to keep bandwidth manageable)
    book: Optional[OrderBookState] = None


@dataclass
class AppState:
    started_at: float = field(default_factory=lambda: time.time())
    ws_connected: bool = False
    ws_last_msg_at: Optional[float] = None
    ws_last_error: Optional[str] = None
    l2_last_error: Optional[str] = None
    ws_reconnects: int = 0

    products: Dict[str, ProductMeta] = field(default_factory=dict)
    markets: Dict[str, MarketState] = field(default_factory=dict)

    tracked_product_ids: List[str] = field(default_factory=list)   # universe (tickers + trades)
    level2_product_ids: List[str] = field(default_factory=list)    # shortlist for L2 book

    # Simple counters
    ticker_messages: int = 0
    status_messages: int = 0
    # L2 focus requested by API (top candidates). If set, shortlist manager will prioritize these.
    l2_focus_product_ids: set[str] = field(default_factory=set)
    l2_focus_expires_at: float = 0.0
    match_messages: int = 0
    l2_messages: int = 0


def set_l2_focus(self, product_ids: list[str], ttl_seconds: int = 180) -> None:
    import time
    self.l2_focus_product_ids = set(product_ids)
    self.l2_focus_expires_at = time.time() + ttl_seconds
