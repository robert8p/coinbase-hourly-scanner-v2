# Coinbase Hourly Scanner v2 (execution-aware)

A lightweight web app that:
- connects to Coinbase Exchange Market Data WebSocket
- tracks a broad universe via `ticker_batch` + `matches` (trade prints)
- keeps a smaller **L2 order-book shortlist** via `level2` to estimate depth/impact
- ranks the top 10 candidates for the next hour using a heuristic, **execution-aware** score

## What it tracks (high-level)
- Momentum (15m/60m), volatility-adjusted
- Flow (Trade Flow Imbalance / TFI over 5m/15m)
- Order-book pressure (OBI within 10bps/25bps) + microprice
- Execution cost proxies (spread + estimated impact to trade a target notional)
- Volume anomalies (5m vs 60m baseline)
- Simple regime dampener (BTC risk-off filter)
- Correlation-based diversification for the final top-10 list (greedy)

## What it is / isn’t
- ✅ A practical *scanner* for near-term candidates, with liquidity/spread/impact flags
- ❌ Not a guarantee of performance; one-hour moves are noisy

## Environment variables
Universe / feeds:
- `WS_URL` (default `wss://ws-feed.exchange.coinbase.com`)
- `QUOTE_CCY` (default `USD`)
- `MAX_PRODUCTS` (default `300`) – how many pairs to track via tickers + trades

L2 shortlist (to keep bandwidth manageable):
- `SHORTLIST_SIZE` (default `30`) – number of pairs to maintain order book for
- `SHORTLIST_REFRESH_SECONDS` (default `120`) – how often shortlist is refreshed
- `SHORTLIST_CANDIDATES` (default `80`) – how many high-scoring candidates to compute full score for

Execution-aware filters:
- `MIN_QUOTE_VOL_USD_24H` (default `5000000`)
- `MAX_SPREAD_PCT` (default `0.006` i.e. 0.6%)
- `IMPACT_NOTIONAL_USD` (default `250`) – target notional used for impact estimate

Diversification:
- `CORR_WINDOW_MINUTES` (default `120`)
- `CORR_THRESHOLD` (default `0.90`)

## Run locally (optional)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 10000
```

Open http://localhost:10000

## Notes
- Expect ~60–90 minutes for the app to "warm up" after first deploy (builds rolling history from live ticks).
- If you increase `MAX_PRODUCTS` or `SHORTLIST_SIZE` aggressively, you may hit bandwidth/CPU constraints on small hosts.
