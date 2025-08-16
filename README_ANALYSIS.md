# Analysis Page — CoinGecko v2 (Hourly OHLC + Fallback)

This version fixes the 'No day-trade results' issue by:
- Requesting **hourly candles** from `/coins/{id}/ohlc` when scanning intraday (uses `interval=hourly` if available).
- Falling back to `/coins/{id}/market_chart/range` and **aggregating to 60m/30m/15m** when OHLC is unavailable or returns too few bars.
- Reducing intraday minimum bars to **120** (daily scans still require 210).

## Setup
1) Copy `pages/02_Analysis.py` into your app's `pages/` folder.
2) Put `analysis_utils.py` in your project root (same level as your main Streamlit app file).
3) `pip install -r requirements.txt`
4) Optional: set `COINGECKO_API_KEY` for Pro rate limits.
   - Powershell: `$env:COINGECKO_API_KEY="your_key_here"`
   - macOS/Linux: `export COINGECKO_API_KEY="your_key_here"`
5) Run your app and open **📈 Market Analysis — CoinGecko Data**.

## Notes
- CoinGecko granularity: 1–90 days → hourly, >90 days → daily. We aggregate to your requested interval.
- If you still see empty tables, try reducing watchlist size or increasing intraday lookback up to 90 days.