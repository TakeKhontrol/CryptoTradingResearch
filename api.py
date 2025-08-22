import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from training_utils import (
    COINGECKO_IDS as TRAIN_IDS,
    make_labeled_dataset,
    make_triple_barrier_dataset,
)
from analysis_utils import fetch_ohlc, analyze_symbol, AnalysisConfig


app = FastAPI(title="Crypto Research Pro API", version="1.0")


def _serialize_df(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    df = df.copy()
    df.insert(0, "timestamp", df.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if df.index.tz is not None else df.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return df.reset_index(drop=True).to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/dataset")
def dataset(
    symbol: str,
    interval: str = Query("60m", enum=["15m","30m","60m","1d"]),
    lookback_days: int = 60,
    horizon: int = 3,
    mode: str = Query("tb", enum=["tb","thr"]),
    threshold: float = 0.002,
    rr_target: float = 2.0,
    stop_mult: float = 1.0,
    force_refresh: bool = False,
    keep_ambiguous: bool = False,
):
    sym = symbol.upper()
    if sym not in TRAIN_IDS:
        raise HTTPException(status_code=400, detail="Unknown symbol")
    df = fetch_ohlc(sym, interval=interval, lookback_days=lookback_days, force_refresh=force_refresh)
    if df is None or df.empty:
        return JSONResponse({"symbol": sym, "rows": [], "message": "no data"})
    if mode == "tb":
        ds = make_triple_barrier_dataset(
            df,
            horizon=horizon,
            rr_target=rr_target,
            stop_atr_mult=stop_mult,
            drop_ambiguous=not keep_ambiguous,
        )
    else:
        ds = make_labeled_dataset(df, horizon=horizon, threshold=threshold)
    if ds is None or ds.empty:
        # Fallbacks
        if mode == "tb" and not keep_ambiguous:
            ds = make_triple_barrier_dataset(df, horizon=horizon, rr_target=rr_target, stop_atr_mult=stop_mult, drop_ambiguous=False)
        if ds is None or ds.empty:
            ds = make_labeled_dataset(df, horizon=horizon, threshold=max(0.0, min(threshold, 0.001)))
        if ds is None or ds.empty:
            return JSONResponse({"symbol": sym, "rows": [], "message": "no dataset after labeling"})
    # Place core fields first
    base_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in ds.columns]
    feat_cols = [c for c in ds.columns if c not in base_cols + ["label"]]
    ds = ds[base_cols + feat_cols + ["label"]]
    return {"symbol": sym, "interval": interval, "rows": _serialize_df(ds)}


@app.get("/analysis")
def analysis(
    symbol: str,
    interval: str = Query("60m", enum=["15m","30m","60m","1d"]),
    lookback_days: int = 60,
    rr: float = 2.0,
    stop_mult: float = 1.0,
    force_refresh: bool = False,
):
    sym = symbol.upper()
    cfg = AnalysisConfig(
        interval=interval,
        lookback_days=lookback_days,
        risk_reward=rr,
        stop_buffer_atr_mult=stop_mult,
        use_cache_only=False,
        force_refresh=force_refresh,
    )
    res = analyze_symbol(sym, cfg)
    if not res:
        return JSONResponse({"symbol": sym, "message": "no analysis"})
    return {
        "symbol": sym,
        "interval": res.interval,
        "score": res.score,
        "bias": res.bias,
        "trend": res.trend,
        "momentum": res.momentum,
        "volatility": res.volatility,
        "last_price": res.last_price,
        "entry": res.suggested_entry,
        "stop": res.suggested_stop,
        "take_profit": res.suggested_take_profit,
        "notes": res.notes,
    }


