import os
import time
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import requests
from functools import lru_cache
from datetime import datetime, timezone, timedelta

# -----------------------------
# Symbols and configuration
# -----------------------------

COINGECKO_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "ATOM": "cosmos",
    "MATIC": "polygon-pos",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "NEAR": "near",
    "HBAR": "hedera-hashgraph",
    "AAVE": "aave",
    "UNI": "uniswap",
    "INJ": "injective",
    "RNDR": "render-token",
    "FTM": "fantom",
    "ETC": "ethereum-classic",
    # AI tokens and legacy alias
    "ASI": "artificial-superintelligence-alliance",
    "FET": "fetch-ai",
}

DEFAULT_WATCHLIST = ["BTC","ETH","SOL","XRP","LINK","ADA","AVAX","MATIC","DOGE","UNI"]

# API base: use Pro if API key present
CG_TIMEOUT = 25
def _cg_base() -> str:
    key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_API_KEY")
    return "https://pro-api.coingecko.com/api/v3" if key else "https://api.coingecko.com/api/v3"

def _cg_headers() -> Dict[str, str]:
    key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_API_KEY")
    return {"accept": "application/json", **({"x-cg-pro-api-key": key} if key else {})}

def _safe_get(path: str, params: Dict) -> Optional[dict]:
    base = _cg_base()
    url = f"{base}{path}"
    for _ in range(4):
        try:
            r = requests.get(url, params=params, headers=_cg_headers(), timeout=CG_TIMEOUT)
        except Exception:
            time.sleep(1.0)
            continue
        if r.status_code == 429:
            time.sleep(1.5)
            continue
        if r.status_code >= 400:
            # print(f"CG error {r.status_code}: {r.text[:200]}")
            return None
        try:
            return r.json()
        except Exception:
            return None
    return None

# -----------------------------
# Technical Indicators
# -----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return lower, ma, upper

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# -----------------------------
# Data Fetching - CoinGecko
# -----------------------------

def _validate_interval(interval: str) -> str:
    if interval in {"1d", "60m", "30m", "15m"}:
        return interval
    return "60m"

def _days_for_lookback(lookback_days: int) -> int:
    if lookback_days <= 1:
        return 1
    return int(lookback_days)

def _to_df_from_ohlc(ohlc: List[List[float]]) -> pd.DataFrame:
    if not ohlc:
        return pd.DataFrame()
    arr = np.array(ohlc, dtype=float)
    ts = pd.to_datetime(arr[:,0], unit="ms", utc=True).tz_convert("UTC")
    df = pd.DataFrame({
        "Open": arr[:,1],
        "High": arr[:,2],
        "Low":  arr[:,3],
        "Close":arr[:,4],
    }, index=ts)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _to_df_from_prices(prices: List[List[float]]) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame()
    arr = np.array(prices, dtype=float)
    ts = pd.to_datetime(arr[:,0], unit="ms", utc=True).tz_convert("UTC")
    df = pd.DataFrame({"Price": arr[:,1]}, index=ts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _resample_prices_to_ohlc(df_prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Convert 'Price' series to OHLC
    ohlc = df_prices["Price"].resample(rule).ohlc()
    ohlc.columns = ["Open","High","Low","Close"]
    return ohlc.dropna()

def _fetch_ohlc_via_ohlc_endpoint(coin_id: str, days: int, intraday: bool) -> pd.DataFrame:
    # Try OHLC endpoint first; if intraday, request hourly granularity where supported
    params = {"vs_currency": "usd", "days": days}
    if intraday:
        params["interval"] = "hourly"  # per CG changelog, up to 90 days on Pro
    data = _safe_get(f"/coins/{coin_id}/ohlc", params)
    if data is None:
        return pd.DataFrame()
    return _to_df_from_ohlc(data)

def _fetch_ohlc_via_market_chart_range(coin_id: str, start: int, end: int, target_interval: str) -> pd.DataFrame:
    # Use market_chart/range to get raw prices and aggregate to requested interval
    data = _safe_get(f"/coins/{coin_id}/market_chart/range", {"vs_currency": "usd", "from": start, "to": end})
    if data is None or "prices" not in data:
        return pd.DataFrame()
    dfp = _to_df_from_prices(data["prices"])
    if dfp.empty:
        return pd.DataFrame()
    if target_interval == "1d":
        return _resample_prices_to_ohlc(dfp, "1D")
    elif target_interval == "60m":
        return _resample_prices_to_ohlc(dfp, "60min")
    elif target_interval == "30m":
        return _resample_prices_to_ohlc(dfp, "30min")
    elif target_interval == "15m":
        return _resample_prices_to_ohlc(dfp, "15min")
    else:
        return _resample_prices_to_ohlc(dfp, "60min")

@lru_cache(maxsize=256)
def fetch_ohlc(symbol: str, interval: str = "60m", lookback_days: int = 60) -> pd.DataFrame:
    sym = symbol.upper()
    coin_id = COINGECKO_IDS.get(sym)
    if not coin_id:
        return pd.DataFrame()

    interval = _validate_interval(interval)
    days = _days_for_lookback(int(lookback_days))
    intraday = interval in {"60m","30m","15m"}

    # First attempt: OHLC endpoint (hourly if intraday)
    df = _fetch_ohlc_via_ohlc_endpoint(coin_id, min(days, 90) if intraday else days, intraday=intraday)

    # Fallback: market_chart/range (works on free tier; granularity auto)
    needed_seconds = days * 86400
    now = int(time.time())
    start = now - needed_seconds
    if df.empty or len(df) < 120:  # if not enough bars, try range-based aggregation
        df = _fetch_ohlc_via_market_chart_range(coin_id, start, now, "1d" if interval == "1d" else interval)

    # Data hygiene
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()

# -----------------------------
# Analysis & Scoring
# -----------------------------

@dataclass
class AnalysisConfig:
    interval: str = "60m"     # '60m' for day-trade, '1d' for swing
    lookback_days: int = 60
    risk_reward: float = 2.0  # suggested RR target
    capital_per_trade_usd: float = 1000.0
    stop_buffer_atr_mult: float = 1.0

@dataclass
class AnalysisResult:
    symbol: str
    interval: str
    score: float
    bias: str
    trend: str
    momentum: str
    volatility: str
    last_price: float
    suggested_entry: float
    suggested_stop: float
    suggested_take_profit: float
    notes: str

def _trend_signal(df: pd.DataFrame) -> Tuple[str, float]:
    close = df["Close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)

    last_close = float(close.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    last_e50 = float(e50.iloc[-1])
    last_e200 = float(e200.iloc[-1])

    trend_score = 0
    if last_close > last_e20: trend_score += 1
    if last_e20 > last_e50: trend_score += 1
    if last_e50 > last_e200: trend_score += 1
    if last_close > last_e50: trend_score += 1
    if last_close > last_e200: trend_score += 1

    if trend_score >= 4:
        return "uptrend", trend_score / 5.0
    elif trend_score <= 1:
        return "downtrend", trend_score / 5.0
    else:
        return "range", trend_score / 5.0

def _momentum_signal(df: pd.DataFrame) -> Tuple[str, float]:
    macd_line, sig, hist = macd(df["Close"])
    r = rsi(df["Close"])
    last_hist = float(hist.iloc[-1])
    last_macd = float(macd_line.iloc[-1])
    last_sig = float(sig.iloc[-1])
    last_rsi = float(r.iloc[-1])

    mom = 0
    if last_hist > 0: mom += 1
    if last_macd > last_sig: mom += 1
    if 50 < last_rsi < 70: mom += 1
    if last_rsi >= 70: mom += 0.5
    if last_rsi <= 30: mom -= 0.5
    label = "bullish" if mom >= 2 else ("bearish" if mom <= 0 else "mixed")
    return label, mom / 3.5

def _volatility_signal(df: pd.DataFrame) -> Tuple[str, float, float]:
    a = atr(df, period=14)
    last_atr = float(a.iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    atr_pct = (last_atr / last_close) if last_atr > 0 else 0.0
    label = "high" if atr_pct > 0.03 else ("moderate" if atr_pct > 0.015 else "low")
    return label, atr_pct, last_atr

def _bias_from_components(trend_label: str, momentum_label: str) -> str:
    if trend_label == "uptrend" and momentum_label in ("bullish", "mixed"):
        return "long"
    if trend_label == "downtrend" and momentum_label in ("bearish", "mixed"):
        return "short"
    return "neutral"

def analyze_symbol(symbol: str, config: AnalysisConfig) -> Optional[AnalysisResult]:
    df = fetch_ohlc(symbol, interval=config.interval, lookback_days=config.lookback_days)
    # Require fewer bars for intraday to be forgiving (hourly >=120, daily >=210)
    min_bars = 120 if config.interval in {"60m","30m","15m"} else 210
    if df.empty or len(df) < min_bars:
        return None

    close = df["Close"]
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    r = rsi(close); macd_line, sig, hist = macd(close)
    bbl, bbm, bbu = bollinger_bands(close)

    trend_label, trend_score = _trend_signal(df)
    momentum_label, momentum_score = _momentum_signal(df)
    vol_label, atr_pct, atr_abs = _volatility_signal(df)

    last_price = float(close.iloc[-1])

    # Entry logic: mean reversion in trend direction
    if trend_label == "uptrend":
        suggested_entry = float(e20.iloc[-1])  # buy pullback
    elif trend_label == "downtrend":
        suggested_entry = float(e20.iloc[-1])  # short bounce
    else:
        suggested_entry = float(bbm.iloc[-1])  # range

    # Stop using ATR buffer around recent swings
    lookback_swings = df.tail(20)
    swing_low = float(lookback_swings["Low"].min())
    swing_high = float(lookback_swings["High"].max())
    if trend_label == "uptrend":
        stop = min(swing_low, suggested_entry - config.stop_buffer_atr_mult * atr_abs)
    elif trend_label == "downtrend":
        stop = max(swing_high, suggested_entry + config.stop_buffer_atr_mult * atr_abs)
    else:
        stop = suggested_entry - config.stop_buffer_atr_mult * atr_abs if momentum_label != "bearish" else suggested_entry + config.stop_buffer_atr_mult * atr_abs

    rr = max(1.0, float(config.risk_reward))
    risk_per_unit = abs(suggested_entry - stop)
    if trend_label == "downtrend":
        take_profit = suggested_entry - rr * risk_per_unit
    else:
        take_profit = suggested_entry + rr * risk_per_unit

    comp = (0.6 * trend_score + 0.3 * momentum_score + 0.1 * (1 - min(1.0, max(0.0, atr_pct / 0.05)))) * 100.0
    if trend_label == "downtrend":
        comp = 100 - comp
    comp = float(np.clip(comp, 0, 100))
    bias = _bias_from_components(trend_label, momentum_label)

    notes_parts: List[str] = []
    if trend_label == "uptrend":
        notes_parts.append("Price aligned above EMAs; look for pullbacks to EMA20.")
    elif trend_label == "downtrend":
        notes_parts.append("Price aligned below EMAs; look for bounces to EMA20 for shorts.")
    else:
        notes_parts.append("Range detected; mean-reversion favored near mid-band.")

    if momentum_label == "bullish":
        notes_parts.append("MACD hist > 0 and RSI > 50.")
    elif momentum_label == "bearish":
        notes_parts.append("MACD hist < 0 or RSI < 50.")
    else:
        notes_parts.append("Momentum mixed; consider smaller size.")

    notes_parts.append(f"Volatility {vol_label} (ATR≈{atr_pct*100:.1f}% of price).")
    notes = " ".join(notes_parts)

    return AnalysisResult(
        symbol=symbol,
        interval=config.interval,
        score=round(comp, 1),
        bias=bias,
        trend=trend_label,
        momentum=momentum_label,
        volatility=vol_label,
        last_price=round(last_price, 6),
        suggested_entry=round(float(suggested_entry), 6),
        suggested_stop=round(float(stop), 6),
        suggested_take_profit=round(float(take_profit), 6),
        notes=notes
    )

def scan_market(
    symbols: List[str],
    *,
    daytrade_interval="60m",
    swing_interval="1d",
    lookback_days_intraday=60,
    lookback_days_daily=365,
    risk_reward: float = 2.0,
    capital_per_trade: float = 1000.0,
    stop_buffer_atr_mult: float = 1.0,
) -> Dict[str, Dict[str, Optional[AnalysisResult]]]:
    results: Dict[str, Dict[str, Optional[AnalysisResult]]] = {}
    for s in symbols:
        res_day = analyze_symbol(
            s,
            AnalysisConfig(
                interval=daytrade_interval,
                lookback_days=lookback_days_intraday,
                risk_reward=risk_reward,
                capital_per_trade_usd=capital_per_trade,
                stop_buffer_atr_mult=stop_buffer_atr_mult,
            ),
        )
        res_swing = analyze_symbol(
            s,
            AnalysisConfig(
                interval=swing_interval,
                lookback_days=lookback_days_daily,
                risk_reward=risk_reward,
                capital_per_trade_usd=capital_per_trade,
                stop_buffer_atr_mult=stop_buffer_atr_mult,
            ),
        )
        results[s] = {"day": res_day, "swing": res_swing}
    return results

def leaderboard(results: dict, mode: str) -> pd.DataFrame:
    """Create a sorted DataFrame of analysis results for either 'day' or 'swing' mode.
    Returns an empty DataFrame with expected columns if there are no rows."""
    expected_cols = [
        "Symbol","Bias","Score","Trend","Momentum","Volatility",
        "Last Price","Entry","Stop","Take Profit","Interval","Notes"
    ]
    rows = []
    for symbol, modes in results.items():
        res = modes.get(mode)
        if not res:
            continue
        rows.append({
            "Symbol": symbol,
            "Bias": res.bias,
            "Score": res.score,
            "Trend": res.trend,
            "Momentum": res.momentum,
            "Volatility": res.volatility,
            "Last Price": res.last_price,
            "Entry": res.suggested_entry,
            "Stop": res.suggested_stop,
            "Take Profit": res.suggested_take_profit,
            "Interval": res.interval,
            "Notes": res.notes
        })
    if not rows:
        return pd.DataFrame(columns=expected_cols)
    df = pd.DataFrame(rows)
    if "Score" in df.columns:
        df = df.sort_values("Score", ascending=False)
    return df.reset_index(drop=True)

def suggested_position_size(capital_usd: float, entry: float, stop: float) -> Tuple[float, float]:
    risk_per_unit = abs(entry - stop)
    if entry <= 0 or risk_per_unit <= 0:
        return 0.0, 0.0
    risk_capital = 0.01 * capital_usd  # 1% risk
    qty = risk_capital / risk_per_unit
    return qty, risk_capital

# -----------------------------
# Auto-optimization (Day-trade)
# -----------------------------

def _trend_label_from_values(last_close: float, e20: float, e50: float, e200: float) -> str:
    trend_score = 0
    if last_close > e20: trend_score += 1
    if e20 > e50: trend_score += 1
    if e50 > e200: trend_score += 1
    if last_close > e50: trend_score += 1
    if last_close > e200: trend_score += 1
    if trend_score >= 4:
        return "uptrend"
    if trend_score <= 1:
        return "downtrend"
    return "range"

def _simulate_daytrade_trades(df: pd.DataFrame, rr: float = 2.0, stop_mult: float = 1.0) -> Dict[str, float]:
    if df is None or df.empty or len(df) < 220:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
    close = df["Close"].copy()
    high = df["High"].copy()
    low = df["Low"].copy()
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    atr14 = atr(df, 14)
    roll_low20 = df["Low"].rolling(20).min()
    roll_high20 = df["High"].rolling(20).max()

    r_results: List[float] = []
    i = 200
    n = len(df)
    while i < n - 2:
        price = float(close.iloc[i])
        trend = _trend_label_from_values(price, float(e20.iloc[i]), float(e50.iloc[i]), float(e200.iloc[i]))
        prev_price = float(close.iloc[i-1])
        prev_e20 = float(e20.iloc[i-1])
        cur_e20 = float(e20.iloc[i])
        cur_atr = float(atr14.iloc[i])
        taken = False
        # Long setup in uptrend on EMA20 cross up
        if trend == "uptrend" and prev_price < prev_e20 and price > cur_e20:
            entry = price
            stop = min(float(roll_low20.iloc[i]), entry - stop_mult * cur_atr)
            risk = max(1e-9, entry - stop)
            tp = entry + rr * risk
            j = i + 1
            while j < n:
                if float(low.iloc[j]) <= stop:
                    r_results.append(-1.0)
                    taken = True
                    i = j + 1
                    break
                if float(high.iloc[j]) >= tp:
                    r_results.append(rr)
                    taken = True
                    i = j + 1
                    break
                j += 1
        # Short setup in downtrend on EMA20 cross down
        if not taken and trend == "downtrend" and prev_price > prev_e20 and price < cur_e20:
            entry = price
            stop = max(float(roll_high20.iloc[i]), entry + stop_mult * cur_atr)
            risk = max(1e-9, stop - entry)
            tp = entry - rr * risk
            j = i + 1
            while j < n:
                if float(high.iloc[j]) >= stop:
                    r_results.append(-1.0)
                    taken = True
                    i = j + 1
                    break
                if float(low.iloc[j]) <= tp:
                    r_results.append(rr)
                    taken = True
                    i = j + 1
                    break
                j += 1
        if not taken:
            i += 1

    trades = len(r_results)
    if trades == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
    wins = sum(1 for r in r_results if r > 0)
    total_win = sum(r for r in r_results if r > 0)
    total_loss = -sum(r for r in r_results if r < 0)
    profit_factor = (total_win / total_loss) if total_loss > 0 else float("inf")
    expectancy = (total_win - total_loss) / trades
    return {
        "trades": trades,
        "win_rate": wins / trades,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }

def auto_optimize_daytrade(
    symbol: str,
    intervals: List[str] = ["60m", "30m", "15m"],
    lookbacks: List[int] = [30, 45, 60, 75, 90],
    rrs: List[float] = [1.5, 2.0, 2.5],
    stop_mults: List[float] = [1.0, 1.5],
) -> Optional[Tuple[str, int, float, float, Dict[str, float]]]:
    best = None
    best_key = None
    for interval in intervals:
        for lb in lookbacks:
            df = fetch_ohlc(symbol, interval=interval, lookback_days=lb)
            if df is None or df.empty or len(df) < 220:
                continue
            for rr in rrs:
                for sm in stop_mults:
                    metrics = _simulate_daytrade_trades(df, rr=rr, stop_mult=sm)
                    score = (
                        (metrics["expectancy"] * 2.0)
                        + (metrics["win_rate"])
                        + (0.2 if metrics["profit_factor"] > 1.3 else 0.0)
                        + (min(metrics["trades"], 40) / 200.0)
                    )
                    key = (interval, lb, rr, sm)
                    if best is None or score > best:
                        best = score
                        best_key = (interval, lb, rr, sm, metrics)
    if best_key is None:
        return None
    interval, lb, rr, sm, metrics = best_key
    return interval, int(lb), float(rr), float(sm), metrics