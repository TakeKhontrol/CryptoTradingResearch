
import os
import json
from datetime import datetime
from typing import Any, Dict

import requests
import pandas as pd
import numpy as np

# ---------- Basic CoinGecko fetch with simple retry ----------

def fetch_market_chart(coin_id: str, vs_currency: str = "usd", days: Any = 90, interval: str = "hourly") -> Dict[str, Any]:
    """
    days: int or 'max'
    interval: 'hourly' or 'daily'
    Returns dict with 'prices' and 'total_volumes' arrays: [[ms, value], ...]
    """
    base = "https://api.coingecko.com/api/v3"
    url = f"{base}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}"
    sess = requests.Session()
    for attempt in range(3):
        r = sess.get(url, timeout=30)
        if r.status_code == 429:
            import time
            time.sleep(5)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

# ---------- Cache helpers ----------

DATA_DIR = "data_cache"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def cache_path(asset_id: str, interval: str, day_str: str) -> str:
    return os.path.join(DATA_DIR, f"{asset_id}_{interval}_{day_str}.json")

def model_path(asset_id: str, interval: str, horizon: int) -> str:
    return os.path.join(MODEL_DIR, f"{asset_id}_{interval}_{horizon}.pkl")

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Feature engineering for ML ----------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects df indexed by datetime with columns: price, volume. Returns features."""
    out = df.copy()

    # Returns & volume change
    out["ret_1"] = out["price"].pct_change()
    out["ret_5"] = out["price"].pct_change(5)
    out["vol_chg"] = out["volume"].pct_change().fillna(0)

    # MAs
    out["ma_10"] = out["price"].rolling(10).mean()
    out["ma_20"] = out["price"].rolling(20).mean()
    out["ma_ratio"] = out["ma_10"] / out["ma_20"]

    # RSI (14)
    delta = out["price"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (up / (down.replace(0, np.nan)))))
    out["rsi_14"] = rsi.fillna(50)

    # MACD (12,26,9)
    ema12 = out["price"].ewm(span=12, adjust=False).mean()
    ema26 = out["price"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_sig"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger(20,2) distance
    mid = out["price"].rolling(20).mean()
    std = out["price"].rolling(20).std()
    out["bb_dist"] = (out["price"] - mid) / (2 * std.replace(0, np.nan))

    # OBV slope (simple)
    obv = (np.sign(out["ret_1"].fillna(0)) * out["volume"]).fillna(0).cumsum()
    out["obv_slope"] = obv.diff(5)

    return out.dropna().copy()

def make_supervised(out: pd.DataFrame, horizon: int = 1, task: str = "clf"):
    fwd_ret = out["price"].pct_change(horizon).shift(-horizon)
    if task == "clf":
        y = (fwd_ret > 0).astype(int)
    else:
        y = fwd_ret
    features = ["ret_1","ret_5","vol_chg","ma_ratio","rsi_14","macd","macd_sig","macd_hist","bb_dist","obv_slope"]
    X = out[features].copy()
    valid = y.notna()
    return X[valid], y[valid]

def train_xgb_classifier(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier
    n_splits = max(3, min(n_splits, 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=0, eval_metric="logloss"
    )
    scores = []
    for train_idx, val_idx in tscv.split(X):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        scores.append(model.score(Xval, yval))
    return model, float(np.mean(scores))
