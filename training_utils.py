import os
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import requests
from functools import lru_cache
from pathlib import Path

# ML
from sklearn.metrics import classification_report, roc_auc_score
import joblib

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
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

def _to_df_from_prices_and_volumes(data: dict) -> pd.DataFrame:
    prices = data.get("prices")
    if not prices:
        return pd.DataFrame()
    dfp = _to_df_from_prices(prices)
    vols = data.get("total_volumes")
    if vols:
        arr = np.array(vols, dtype=float)
        ts = pd.to_datetime(arr[:,0], unit="ms", utc=True).tz_convert("UTC")
        dfv = pd.DataFrame({"Volume": arr[:,1]}, index=ts).sort_index()
        dfv = dfv[~dfv.index.duplicated(keep="last")]
        dfp = dfp.join(dfv, how="left")
    return dfp

def _resample_prices_to_ohlc(df_prices: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df_prices["Price"].resample(rule).ohlc()
    ohlc.columns = ["Open","High","Low","Close"]
    return ohlc.dropna()

def _fetch_prices_range(coin_id: str, start: int, end: int) -> pd.DataFrame:
    data = _safe_get(f"/coins/{coin_id}/market_chart/range", {"vs_currency":"usd","from":start,"to":end})
    if data is None or "prices" not in data:
        return pd.DataFrame()
    return _to_df_from_prices_and_volumes(data)

def _fetch_prices_days(coin_id: str, days: str) -> pd.DataFrame:
    data = _safe_get(f"/coins/{coin_id}/market_chart", {"vs_currency":"usd","days":days})
    if data is None or "prices" not in data:
        return pd.DataFrame()
    return _to_df_from_prices_and_volumes(data)

@lru_cache(maxsize=256)
def fetch_ohlc(symbol: str, interval: str = "1d", lookback_days: int = 730) -> pd.DataFrame:
    """Fetch OHLC using range->max fallback; for ASI, fallback to legacy FET if needed."""
    sym = symbol.upper()
    coin_id = COINGECKO_IDS.get(sym)
    if not coin_id:
        return pd.DataFrame()

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
    # Aggregate volume if available
    if "Volume" in dfp.columns:
        vol = dfp["Volume"].resample(rule).sum()
        df["Volume"] = vol.reindex(df.index).fillna(0.0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
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
    high = out["High"]
    low = out["Low"]
    volume = out.get("Volume", pd.Series(0.0, index=out.index))

    # Returns & ROC
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    out["roc_1"] = close.pct_change(1)
    out["roc_4"] = close.pct_change(4)
    out["roc_12"] = close.pct_change(12)

    # EMAs and alignment
    out["ema_20"] = _ema(close, 20)
    out["ema_50"] = _ema(close, 50)
    out["ema_200"] = _ema(close, 200)
    out["ema_align"] = ((close > out["ema_20"]).astype(int) + (out["ema_20"] > out["ema_50"]).astype(int) + (out["ema_50"] > out["ema_200"]).astype(int))
    out["ema_spread_20_50"] = (out["ema_20"] - out["ema_50"]) / close

    # RSI + slope
    rsi = _rsi(close, 14)
    out["rsi_14"] = rsi
    out["rsi_slope"] = rsi.diff(3)
    out["rsi_band_50_70"] = ((rsi >= 50) & (rsi <= 70)).astype(int)

    # MACD
    ema12 = _ema(close, 12); ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    # Volatility & ATR bands
    out["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(20)
    atr14 = _atr(out, 14)
    out["atr_pct"] = atr14 / close
    out["atr_band_high"] = out["ema_20"] + 1.0 * atr14
    out["atr_band_low"] = out["ema_20"] - 1.0 * atr14
    out["vol_pctile_60d"] = close.pct_change().rolling(60).std().rank(pct=True)

    # Structure: Bollinger z-score, min/max distances, Donchian
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=0)
    out["bb_z"] = (close - ma20) / (sd20.replace(0, np.nan))
    out["dist_min20"] = (close - low.rolling(20).min()) / close
    out["dist_max20"] = (high.rolling(20).max() - close) / close
    out["donchian_up_20"] = high.rolling(20).max()
    out["donchian_dn_20"] = low.rolling(20).min()
    out["donchian_range_20"] = (out["donchian_up_20"] - out["donchian_dn_20"]) / close

    # Seasonality (for intraday)
    if close.index.freq is not None or True:
        h = out.index.tz_convert("UTC").hour if out.index.tz is not None else out.index.hour
        dow = out.index.dayofweek
        out["hod_sin"] = np.sin(2*np.pi*h/24)
        out["hod_cos"] = np.cos(2*np.pi*h/24)
        out["dow_sin"] = np.sin(2*np.pi*dow/7)
        out["dow_cos"] = np.cos(2*np.pi*dow/7)

    # Liquidity/flow
    obv = (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
    out["obv"] = obv
    out["obv_z"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std(ddof=0).replace(0, np.nan))
    out["vol_z"] = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std(ddof=0).replace(0, np.nan))

    # Normalize select features by volatility
    for col in ["ema_spread_20_50","bb_z","dist_min20","dist_max20","donchian_range_20","macd_hist"]:
        out[col] = out[col] / (out["atr_pct"].replace(0, np.nan))

    return out

def make_labeled_dataset(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.002) -> pd.DataFrame:
    feats = build_features(df)
    feats = feats.dropna()
    future_ret = feats["Close"].pct_change(horizon).shift(-horizon)
    feats["label"] = (future_ret > threshold).astype(int)
    feats = feats.dropna()
    return feats

def make_triple_barrier_dataset(
    df: pd.DataFrame,
    *,
    horizon: int = 3,
    rr_target: float = 2.0,
    stop_atr_mult: float = 1.0,
    drop_ambiguous: bool = True,
) -> pd.DataFrame:
    """Create features with triple-barrier labels.
    Label = 1 if TP hit before SL within horizon, else 0 if SL hit first.
    Rows where neither barrier is hit within horizon are dropped by default.
    Risk is defined as stop_atr_mult * ATR(14) at entry; TP distance = rr_target * risk.
    """
    feats = build_features(df).dropna().copy()
    df2 = df.loc[feats.index]
    close = df2["Close"].values
    high = df2["High"].values
    low = df2["Low"].values
    atr = _atr(df2, 14).values

    n = len(feats)
    labels = np.full(n, np.nan, dtype=float)
    for i in range(0, n - horizon):
        entry = close[i]
        risk = stop_atr_mult * (atr[i] if not np.isnan(atr[i]) else 0.0)
        if risk <= 0 or np.isnan(risk):
            continue
        tp = entry + rr_target * risk
        sl = entry - risk
        win = False
        loss = False
        for j in range(i + 1, min(n, i + 1 + horizon)):
            if high[j] >= tp:
                win = True
                break
            if low[j] <= sl:
                loss = True
                break
        if win:
            labels[i] = 1.0
        elif loss:
            labels[i] = 0.0
        else:
            if not drop_ambiguous:
                labels[i] = 0.0

    feats["label"] = labels
    feats = feats.dropna().copy()
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
    # Profit-aware labeling and evaluation
    use_triple_barrier: bool = True
    rr_target: float = 2.0
    stop_atr_mult: float = 1.0
    cost_r: float = 0.05  # costs in R units per trade (applied once)
    min_trades_eval: int = 25

def train_symbol(symbol: str, cfg: TrainConfig) -> Dict[str, str]:
    if not _XGB_AVAILABLE:
        return {"status":"error","message":"xgboost is not installed. Please `pip install xgboost`."}

    df = fetch_ohlc(symbol, interval=cfg.interval, lookback_days=cfg.lookback_days)
    # Require a reasonable amount of bars, but don't be overly strict
    min_bars = 260 if cfg.interval == "1d" else 500
    if df.empty or len(df) < min_bars:
        return {"status":"error","message": f"Not enough data for {symbol} (got {len(df)} bars)."}

    if cfg.use_triple_barrier:
        data = make_triple_barrier_dataset(
            df,
            horizon=cfg.horizon,
            rr_target=cfg.rr_target,
            stop_atr_mult=cfg.stop_atr_mult,
            drop_ambiguous=True,
        )
    else:
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
    # Compatibility with different xgboost versions
    try:
        clf.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False, early_stopping=50)
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping
            clf.fit(
                X_t,
                y_t,
                eval_set=[(X_v, y_v)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=50, save_best=True)],
            )
        except TypeError:
            clf.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

    y_prob = clf.predict_proba(X_test)[:,1]

    def _select_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, rr: float, cost_r: float, min_trades: int = 25) -> Tuple[float, dict]:
        candidates = np.linspace(0.4, 0.8, 17)
        best_thr = 0.5
        best_score = -1e9
        best_stats = {"trades":0,"win_rate":0.0,"expectancy":0.0,"profit_factor":0.0}
        for thr in candidates:
            picks = y_pred_proba >= thr
            trades = int(picks.sum())
            if trades < min_trades:
                continue
            wins = int(((y_true == 1) & picks).sum())
            losses = trades - wins
            total_win_r = wins * rr
            total_loss_r = losses * 1.0
            # Apply per-trade costs
            total_cost = trades * cost_r
            expectancy = ((total_win_r - total_loss_r) - total_cost) / max(trades,1)
            profit_factor = (total_win_r / max(total_loss_r,1e-9)) if total_loss_r > 0 else float("inf")
            win_rate = wins / trades if trades > 0 else 0.0
            score = expectancy + 0.05 * win_rate + 0.02 * min(trades, 200)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_stats = {
                    "trades": trades,
                    "win_rate": float(win_rate),
                    "expectancy": float(expectancy),
                    "profit_factor": float(profit_factor),
                }
        return best_thr, best_stats

    # Evaluate both AUC and profit-optimized threshold
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    best_thr, thr_stats = _select_best_threshold(y_test, y_prob, cfg.rr_target, cfg.cost_r, cfg.min_trades_eval)
    decision = (y_prob >= best_thr).astype(int)
    report = classification_report(y_test, decision, output_dict=False)

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, f"xgb_model_{symbol}_{cfg.interval}.joblib")
    joblib.dump({"model": clf, "features": features, "params": params, "best_threshold": best_thr, "rr_target": cfg.rr_target}, model_path)

    meta_path = os.path.join(cfg.model_dir, f"meta_{symbol}_{cfg.interval}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(pd.Series({
            "symbol": symbol,
            "interval": cfg.interval,
            "lookback_days": cfg.lookback_days,
            "horizon": cfg.horizon,
            "threshold": cfg.threshold,
            "auc": float(auc),
            "best_threshold": float(best_thr),
            "thr_trades": int(thr_stats.get("trades",0)),
            "thr_win_rate": float(thr_stats.get("win_rate",0.0)),
            "thr_expectancy": float(thr_stats.get("expectancy",0.0)),
            "thr_profit_factor": float(thr_stats.get("profit_factor",0.0)),
            "n_samples": int(len(data)),
            "model_path": model_path
        }).to_json())

    return {
        "status":"ok",
        "model_path": model_path,
        "auc": f"{auc:.3f}",
        "best_threshold": f"{best_thr:.2f}",
        "thr_stats": thr_stats,
        "report": report,
    }