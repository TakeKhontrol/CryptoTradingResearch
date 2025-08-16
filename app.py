
import os
import json
from datetime import datetime
import time

import streamlit as st
import pandas as pd
import numpy as np

from crypto_trading_app import (
    fetch_market_chart,
    save_json,
    load_json,
    cache_path,
    model_path,
    build_features,
    make_supervised,
    train_xgb_classifier,
)

st.set_page_config(page_title="Crypto Research Pro (with ML)", layout="wide")

st.title("🧠 Crypto Research Pro — ML Training (Free API + Cache)")
st.caption("Trains an XGBoost model inside the app using cached CoinGecko data to stay within free plan limits.")

# ---- Simple asset list (CoinGecko IDs) ----
ASSETS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "Avalanche (AVAX)": "avalanche-2",
    "Chainlink (LINK)": "chainlink",
    "Polygon (MATIC)": "matic-network",
    "Cardano (ADA)": "cardano",
    "Dogecoin (DOGE)": "dogecoin",
    "Binance Coin (BNB)": "binancecoin",
    "Render (RENDER)": "render-token",
}

col1, col2 = st.columns(2)
with col1:
    asset_label = st.selectbox("Asset", list(ASSETS.keys()), index=0)
    asset_id = ASSETS[asset_label]

with col2:
    interval = st.selectbox(
        "Training interval",
        ["hourly_90d", "daily_all"],
        index=0,
        help="Use 90 days of hourly bars for more samples, or multi-year daily bars."
    )

st.divider()

# ---- Cache controls ----
today_str = datetime.utcnow().strftime("%Y-%m-%d")
data_file = cache_path(asset_id, interval, today_str)

st.subheader("Data Cache")
left, right = st.columns([2,1])
with left:
    use_cached_only = st.checkbox("Use cached data only (no fetch if missing)", value=True)
with right:
    fetch_btn = st.button("Fetch / Refresh Today")

if fetch_btn or (not os.path.exists(data_file) and not use_cached_only):
    with st.spinner("Fetching data from CoinGecko (1 call, free tier)..."):
        # Decide days/interval for API
        if interval == "hourly_90d":
            data = fetch_market_chart(asset_id, vs_currency="usd", days=90, interval="hourly")
        else:
            # 'max' daily history
            data = fetch_market_chart(asset_id, vs_currency="usd", days="max", interval="daily")
        save_json(data_file, data)
        st.success(f"Cached: {data_file}")

if os.path.exists(data_file):
    st.success(f"Using cached file: {data_file}")
else:
    st.warning("No cache for today yet. Either enable fetching or press Fetch / Refresh Today.")

st.divider()

# ---- Build DataFrame from cache for preview & features ----
df = None
if os.path.exists(data_file):
    raw = load_json(data_file)
    df = pd.DataFrame(raw["prices"], columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop(columns=["ts"], inplace=True)

    if "total_volumes" in raw and raw["total_volumes"]:
        vol = pd.DataFrame(raw["total_volumes"], columns=["ts","volume"])
        vol["date"] = pd.to_datetime(vol["ts"], unit="ms")
        vol.set_index("date", inplace=True)
        df["volume"] = vol["volume"]
    else:
        df["volume"] = 0.0

    st.subheader("Preview (last 10 rows)")
    st.dataframe(df.tail(10))

# ---- ML Training ----
st.divider()
st.subheader("Train XGBoost (in-app)")

colA, colB, colC = st.columns(3)
with colA:
    horizon = st.selectbox("Prediction horizon (bars ahead)",
                           [1, 4, 24],
                           index=1,
                           help="For hourly: 1=next hour, 4=~4h, 24=~1 day; For daily: days ahead.")
with colB:
    folds = st.number_input("TimeSeries CV folds", min_value=3, max_value=10, value=5, step=1)
with colC:
    retrain = st.button("Train / Retrain model")

mdl_file = model_path(asset_id, interval, int(horizon))

if retrain:
    if df is None or df.empty:
        st.error("No data available. Fetch or enable cache first.")
        st.stop()

    with st.spinner("Building features..."):
        feats = build_features(df)
        X, y = make_supervised(feats, horizon=int(horizon), task="clf")

    st.write(f"Samples available: **{len(X)}**")
    if len(X) < 400 and interval == "hourly_90d":
        st.warning("Less than ~400 samples. Consider horizon=1 and hourly_90d for more rows.")
    if len(X) < 120 and interval == "daily_all":
        st.warning("Less than ~120 samples. Try hourly_90d to gain more samples.")

    with st.spinner("Training XGBoost (walk-forward CV)..."):
        model, cv_score = train_xgb_classifier(X, y, n_splits=int(folds))

    import pickle
    bundle = {"model": model, "features": X.columns.tolist(), "interval": interval, "horizon": int(horizon)}
    with open(mdl_file, "wb") as f:
        pickle.dump(bundle, f)
    st.success(f"Model saved → {mdl_file}")
    st.info(f"Cross-validated accuracy ≈ {cv_score:.3f}")

# ---- Inference (if model exists) ----
if os.path.exists(mdl_file) and df is not None:
    import pickle
    with open(mdl_file, "rb") as f:
        bundle = pickle.load(f)
    feats = build_features(df)
    latest = feats.iloc[[-1]][bundle["features"]]
    prob_up = float(bundle["model"].predict_proba(latest)[0,1])
    st.subheader("Live Inference")
    st.metric(label=f"P(up) next {bundle['horizon']} bar(s)", value=f"{prob_up:.2%}")

st.caption("The app fetches once per day per asset. Training uses cached data only, so it stays within free-API limits.")
