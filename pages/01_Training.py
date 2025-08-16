import streamlit as st
import pandas as pd
from training_utils import (
    COINGECKO_IDS, DEFAULT_TRAINLIST, TrainConfig, train_symbol, auto_suggest_defaults, fetch_ohlc
)

st.set_page_config(page_title="Crypto Trading: Training (XGBoost + CoinGecko)", page_icon="🧠", layout="wide")
st.title("🧠 Model Training — XGBoost + CoinGecko Data")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Training Settings")
    syms = st.multiselect("Symbols", options=list(COINGECKO_IDS.keys()), default=DEFAULT_TRAINLIST)
    interval = st.selectbox("Interval", options=["1d","60m"], index=0)

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
    go = st.button("🚀 Train Selected", type="primary", use_container_width=True)

st.write("**Model:** XGBoost (binary:logistic), early stopping on validation split, metric AUC.")
st.write("**Note:** No symbols are pre-selected. Choose the tickers you want to train.")

if go:
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
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
else:
    st.info("Choose symbols (ASI supported; legacy FET fallback for history) and click **Train Selected**.")
    st.caption("Data: CoinGecko range endpoint with max/legacy fallbacks → aggregated to 1d or 60m.")