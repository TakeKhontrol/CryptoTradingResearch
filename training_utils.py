import os
import time
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import requests
import joblib

# ML (optional – only required for model training)
try:
    from sklearn.metrics import classification_report, roc_auc_score
except Exception:  # pragma: no cover - fallback when sklearn missing
    classification_report = None
    roc_auc_score = None

try:  # pragma: no cover - xgboost is optional
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    _XGB_AVAILABLE = False

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
    "ASI": "artificial-superintelligence-alliance",
    # Optional legacy mapping if needed:
    "FET": "fetch-ai",
}

DEFAULT_TRAINLIST: List[str] = []  # No pre-selection; user must choose

# -----------------------------
# Simple on-disk caching (mirrors former app module)
# -----------------------------

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

def _ohlc_cache_path(symbol: str, interval: str, lookback_days: int, day: Optional[str] = None) -> Path:
    """Return a cache path unique per symbol/interval/lookback/day."""
    day = day or datetime.utcnow().strftime("%Y-%m-%d")
    fname = f"{symbol}_{interval}_{lookback_days}_{day}.csv"
    return DATA_DIR / fname

# -----------------------------
# CoinGecko helpers
# -----------------------------

def _cg_base() -> str:
    key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_API_KEY")
    return "https://pro-api.coingecko.com/api/v3" if key else "https://api.coingecko.com/api/v3"

def _cg_headers() -> Dict[str, str]:
    key = os.environ.get("COINGECKO_API_KEY") or os.environ.get("X_CG_API_KEY")
    return {"accept": "application/json", **({"x-cg-pro-api-key": key} if key else {})}

def _safe_get(path: str, params: Dict) -> Optional[dict]:
    url = f"{_cg_base()}{path}"
    for _ in range(4):
        try:
            r = requests.get(url, params=params, headers=_cg_headers(), timeout=25)
        except Exception:
            time.sleep(1.0)
            continue
        if r.status_code == 429:
            time.sleep(1.5)
            continue
        if r.status_code >= 400:
            return None
        try:
            return r.json()
        except Exception:
            return None
    return None

def _to_df_from_prices(prices: List[List[float]]) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame()
    arr = np.array(prices, dtype=float)
    ts = pd.to_datetime(arr[:,0], unit="ms", utc=True).tz_convert("UTC")
    df = pd.DataFrame({"Price": arr[:,1]}, index=ts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _resample_prices_to_ohlc(df_prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df_prices["Price"].resample(rule).ohlc()
    ohlc.columns = ["Open","High","Low","Close"]
    return ohlc.dropna()

def _fetch_prices_range(coin_id: str, start: int, end: int) -> pd.DataFrame:
    data = _safe_get(f"/coins/{coin_id}/market_chart/range", {"vs_currency":"usd","from":start,"to":end})
    if data is None or "prices" not in data:
        return pd.DataFrame()
    return _to_df_from_prices(data["prices"])

def _fetch_prices_days(coin_id: str, days: str) -> pd.DataFrame:
    data = _safe_get(f"/coins/{coin_id}/market_chart", {"vs_currency":"usd","days":days})
    if data is None or "prices" not in data:
        return pd.DataFrame()
    return _to_df_from_prices(data["prices"])

@lru_cache(maxsize=256)
def fetch_ohlc(symbol: str, interval: str = "1d", lookback_days: int = 730) -> pd.DataFrame:
    """Fetch OHLC using range->max fallback; for ASI, fallback to legacy FET if needed."""
    sym = symbol.upper()
    coin_id = COINGECKO_IDS.get(sym)
    if not coin_id:
        return pd.DataFrame()
    cache_file = _ohlc_cache_path(sym, interval, lookback_days)
    if cache_file.exists():
        try:
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except Exception:
            cache_file.unlink(missing_ok=True)

    now = int(time.time())
    start = now - int(lookback_days) * 86400

    # 1) Try exact range
    dfp = _fetch_prices_range(coin_id, start, now)

    # 2) Fallback to days=max (useful when range fails or thin)
    if dfp.empty:
        dfp = _fetch_prices_days(coin_id, "max")

    # 3) Special fallback for ASI -> use FET legacy if still thin
    if (dfp.empty or len(dfp) < 200) and sym == "ASI":
        fet_id = COINGECKO_IDS.get("FET")
        if fet_id:
            dfp_fet = _fetch_prices_days(fet_id, "max")
            if not dfp_fet.empty:
                dfp = dfp_fet

    if dfp.empty:
        return pd.DataFrame()

    # Clip to requested window if we pulled 'max'
    dfp = dfp.loc[(dfp.index >= pd.to_datetime(start, unit="s", utc=True)) & (dfp.index <= pd.to_datetime(now, unit="s", utc=True))]

    rule = "1D" if interval == "1d" else "60min"
    df = _resample_prices_to_ohlc(dfp, rule)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if not df.empty:
        df.to_csv(cache_file)
    return df

# -----------------------------
# Feature engineering
# -----------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = (delta.where(delta>0,0)).rolling(period).mean()
    loss = (-delta.where(delta<0,0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100/(1+rs))
    return out

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)

    out["ema_10"] = _ema(close, 10)
    out["ema_20"] = _ema(close, 20)
    out["ema_50"] = _ema(close, 50)
    out["ema_200"] = _ema(close, 200)
    out["ema_spread_20_50"] = out["ema_20"] - out["ema_50"]

    out["rsi_14"] = _rsi(close, 14)

    ema12 = _ema(close, 12); ema26 = _ema(close, 26)
    out["macd_line"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd_line"], 9)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    out["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(20)
    atr14 = _atr(out, 14)
    out["atr14_norm"] = atr14 / close

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=0)
    out["bb_width"] = (2*sd20) / (ma20.replace(0, np.nan))

    return out

def make_labeled_dataset(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.002) -> pd.DataFrame:
    feats = build_features(df)
    feats = feats.dropna()
    future_ret = feats["Close"].pct_change(horizon).shift(-horizon)
    feats["label"] = (future_ret > threshold).astype(int)
    feats = feats.dropna()
    return feats

# -----------------------------
# Auto-defaults (optional)
# -----------------------------

def auto_suggest_defaults(symbol: str, interval: str = "1d") -> Tuple[int, int, float, float]:
    if interval == "1d":
        lookbacks = [365, 730, 1095]
        horizons = [1, 3, 5]
        thresholds = [0.000, 0.002, 0.005]
    else:
        lookbacks = [30, 60, 90]
        horizons = [1, 3, 6]
        thresholds = [0.000, 0.001, 0.002]

    best = (None, None, None, -1.0)
    for lb in lookbacks:
        df = fetch_ohlc(symbol, interval=interval, lookback_days=lb)
        if df.empty or len(df) < 260:  # relax slightly
            continue
        for h in horizons:
            for th in thresholds:
                ds = make_labeled_dataset(df, horizon=h, threshold=th)
                if ds.empty or ds["label"].nunique() < 2:
                    continue
                split = int(len(ds) * 0.8)
                X_train = ds.iloc[:split].drop(columns=["label","Open","High","Low","Close"]).values
                y_train = ds.iloc[:split]["label"].values
                X_test = ds.iloc[split:].drop(columns=["label","Open","High","Low","Close"]).values
                y_test = ds.iloc[split:]["label"].values

                if not _XGB_AVAILABLE:
                    continue
                params = dict(n_estimators=120, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                              objective="binary:logistic", eval_metric="auc", n_jobs=0, reg_lambda=1.0)
                if os.environ.get("XGB_USE_GPU","0") == "1":
                    params.update(tree_method="gpu_hist", predictor="gpu_predictor")
                clf = XGBClassifier(**params)
                clf.fit(X_train, y_train)
                y_prob = clf.predict_proba(X_test)[:,1]
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except Exception:
                    auc = 0.5
                if auc > best[3]:
                    best = (lb, h, float(th), float(auc))

    if best[0] is None:
        return (730, 3, 0.002, 0.5) if interval == "1d" else (60, 3, 0.001, 0.5)
    return best

# -----------------------------
# Training with XGBoost
# -----------------------------

@dataclass
class TrainConfig:
    interval: str = "1d"
    lookback_days: int = 730
    horizon: int = 3
    threshold: float = 0.002
    test_size: float = 0.2
    model_dir: str = "models"
    use_gpu: bool = False
    random_state: int = 42

def train_symbol(symbol: str, cfg: TrainConfig) -> Dict[str, str]:
    if not _XGB_AVAILABLE:
        return {"status":"error","message":"xgboost is not installed. Please `pip install xgboost`."}
    if classification_report is None or roc_auc_score is None:
        return {"status":"error","message":"scikit-learn is required. Please `pip install scikit-learn`."}

    df = fetch_ohlc(symbol, interval=cfg.interval, lookback_days=cfg.lookback_days)
    # Require a reasonable amount of bars, but don't be overly strict
    min_bars = 260 if cfg.interval == "1d" else 500
    if df.empty or len(df) < min_bars:
        return {"status":"error","message": f"Not enough data for {symbol} (got {len(df)} bars)."}

    data = make_labeled_dataset(df, horizon=cfg.horizon, threshold=cfg.threshold)
    features = [c for c in data.columns if c not in ["label","Open","High","Low","Close"]]
    X = data[features].values
    y = data["label"].values

    split = int(len(data) * (1 - cfg.test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
        eval_metric="auc", reg_lambda=1.0, n_jobs=0, random_state=cfg.random_state
    )
    if cfg.use_gpu or os.environ.get("XGB_USE_GPU","0") == "1":
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")

    clf = XGBClassifier(**params)
    es_split = int(len(X_train) * 0.85)
    X_t, X_v = X_train[:es_split], X_train[es_split:]
    y_t, y_v = y_train[:es_split], y_train[es_split:]
    fit_params = dict(eval_set=[(X_v, y_v)], verbose=False)
    if "early_stopping_rounds" in XGBClassifier.fit.__code__.co_varnames:
        fit_params["early_stopping_rounds"] = 50
    else:
        clf.set_params(early_stopping_rounds=50)
    clf.fit(X_t, y_t, **fit_params)

    y_prob = clf.predict_proba(X_test)[:,1]
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5
    report = classification_report(y_test, (y_prob>0.5).astype(int), output_dict=False)

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, f"xgb_model_{symbol}_{cfg.interval}.joblib")
    joblib.dump({"model": clf, "features": features, "params": params}, model_path)

    meta_path = os.path.join(cfg.model_dir, f"meta_{symbol}_{cfg.interval}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(pd.Series({
            "symbol": symbol,
            "interval": cfg.interval,
            "lookback_days": cfg.lookback_days,
            "horizon": cfg.horizon,
            "threshold": cfg.threshold,
            "auc": float(auc),
            "n_samples": int(len(data)),
            "model_path": model_path
        }).to_json())

    return {"status":"ok","model_path": model_path, "auc": f"{auc:.3f}", "report": report}
