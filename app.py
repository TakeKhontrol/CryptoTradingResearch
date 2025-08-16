
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

# Import training utilities
from training_utils import (
    COINGECKO_IDS, TrainConfig, train_symbol, auto_suggest_defaults, fetch_ohlc
)

st.set_page_config(page_title="Crypto Research Pro (with ML)", layout="wide")

st.title("🧠 Crypto Research Pro — ML Training (Free API + Cache)")
st.caption("Trains an XGBoost model inside the app using cached CoinGecko data to stay within free plan limits.")

# ---- Sidebar with training controls ----
with st.sidebar:
    st.header("Training Settings")
    
    # Symbol selection
    syms = st.multiselect("Symbols to Train", options=list(COINGECKO_IDS.keys()), default=[])
    
    # Training parameters
    interval = st.selectbox("Interval", options=["1d","60m"], index=0)
    
    # Auto-suggest defaults
    auto = st.checkbox("Auto-suggest optimized defaults (first selected symbol)", value=True)
    if auto and len(syms) > 0:
        with st.spinner(f"Finding defaults using {syms[0]} ({interval}) ..."):
            lb, h, th, auc = auto_suggest_defaults(syms[0], interval=interval)
            st.caption(f"Suggested: lookback={lb} | horizon={h} | threshold={th} | AUC≈{auc:.3f}")
    else:
        lb, h, th = (730, 3, 0.002) if interval == "1d" else (60, 3, 0.001)
    
    lookback = st.number_input("Lookback (days)", min_value=30, max_value=1460, value=int(lb), step=10)
    horizon = st.number_input("Bars ahead (horizon)", min_value=1, max_value=30, value=int(h), step=1)
    threshold = st.number_input("Target threshold (return > x)", value=float(th), step=0.001, format="%.3f")
    test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    model_dir = st.text_input("Model output dir", value="models")
    use_gpu = st.checkbox("Use GPU (if available)", value=False)
    
    # Training button
    train_btn = st.button("🚀 Train Selected", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("**Model:** XGBoost (binary:logistic), early stopping on validation split, metric AUC.")

# ---- Main content area ----
if train_btn:
    if len(syms) == 0:
        st.warning("Select at least one symbol to train.")
    else:
        cfg = TrainConfig(interval=interval, lookback_days=int(lookback), horizon=int(horizon), threshold=float(threshold),
                          test_size=float(test_size), model_dir=model_dir, use_gpu=use_gpu)
        results = []
        for s in syms:
            with st.spinner(f"Training {s} with XGBoost ..."):
                res = train_symbol(s, cfg)
                results.append({"Symbol": s, **{k:v for k,v in res.items() if k!='report'}})
                if res.get("status") == "ok":
                    st.success(f"{s}: AUC={res['auc']} | Saved → {res['model_path']}")
                else:
                    st.warning(f"{s}: {res.get('message','Unknown error')}")
        st.subheader("Training Summary")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
else:
    # ---- Data visualization and analysis ----
    st.subheader("Data Overview")
    
    # Show available symbols and their status
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Available Symbols:**")
        st.write(f"Total: {len(COINGECKO_IDS)} symbols available")
        st.write("Select symbols in the sidebar to train models")
    
    with col2:
        st.write("**Training Status:**")
        if len(syms) > 0:
            st.write(f"Selected: {', '.join(syms)}")
        else:
            st.write("No symbols selected")
    
    st.divider()
    
    # Show recent model files if they exist
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') or f.endswith('.joblib')]
        if model_files:
            st.subheader("Existing Models")
            st.write(f"Found {len(model_files)} trained models:")
            for model_file in sorted(model_files)[:10]:  # Show first 10
                st.write(f"• {model_file}")
            if len(model_files) > 10:
                st.write(f"... and {len(model_files) - 10} more")
        else:
            st.info("No trained models found. Use the sidebar to train your first model.")
    
    # Show data cache status
    st.divider()
    st.subheader("Data Cache Status")
    
    if os.path.exists("data_cache"):
        cache_files = [f for f in os.listdir("data_cache") if f.endswith('.json')]
        if cache_files:
            st.write(f"Found {len(cache_files)} cached data files:")
            for cache_file in sorted(cache_files)[:5]:  # Show first 5
                st.write(f"• {cache_file}")
            if len(cache_files) > 5:
                st.write(f"... and {len(cache_files) - 5} more")
        else:
            st.write("No cached data files found")
    else:
        st.write("Data cache directory not found")
    
    st.info("Choose symbols in the sidebar and click **Train Selected** to train models with advanced settings.")
    st.caption("Data: CoinGecko range endpoint with max/legacy fallbacks → aggregated to 1d or 60m.")

st.caption("The app fetches once per day per asset. Training uses cached data only, so it stays within free-API limits.")
