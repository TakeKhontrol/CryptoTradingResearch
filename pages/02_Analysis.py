import streamlit as st
import pandas as pd
import numpy as np
from analysis_utils import (
    DEFAULT_WATCHLIST, COINGECKO_IDS, scan_market, leaderboard, AnalysisConfig, analyze_symbol, suggested_position_size
)

st.set_page_config(page_title="Crypto Trading: Analysis (CoinGecko)", page_icon="📈", layout="wide")
st.title("📈 Market Analysis — CoinGecko Data")

with st.sidebar:
    st.header("Settings")
    watchlist = st.multiselect("Watchlist", options=list(COINGECKO_IDS.keys()), default=DEFAULT_WATCHLIST)
    st.caption("Data source: CoinGecko. Set COINGECKO_API_KEY for higher limits (optional).")
    daytrade_interval = st.selectbox("Day-trade timeframe", options=["60m","30m","15m"], index=0)
    swing_interval = st.selectbox("Swing timeframe", options=["1d"], index=0)
    lookback_intra = st.slider("Intraday lookback (days, <= 90 recommended)", min_value=1, max_value=90, value=60, step=1)
    lookback_daily = st.slider("Daily lookback (days)", min_value=30, max_value=1460, value=365, step=5)
    rr = st.slider("Risk/Reward target", min_value=1.0, max_value=4.0, value=2.0, step=0.25)
    capital = st.number_input("Capital per trade (USD)", min_value=100.0, value=1000.0, step=50.0)

    if daytrade_interval in ("30m","15m"):
        st.caption("Note: 30m/15m resampled from hourly (CoinGecko granularity).")

    generate = st.button("🚀 Generate Analysis", type="primary", use_container_width=True)

if generate:
    with st.spinner("Fetching OHLC from CoinGecko and computing indicators..."):
        results = scan_market(
            watchlist, 
            daytrade_interval=daytrade_interval,
            swing_interval=swing_interval,
            lookback_days_intraday=lookback_intra,
            lookback_days_daily=lookback_daily
        )

    tab1, tab2, tab3 = st.tabs(["Day Trading", "Swing Trading", "Per-Coin Details"])

    with tab1:
        st.subheader("Top Day-Trade Opportunities")
        df_day = leaderboard(results, mode="day")
        if df_day.empty:
            st.warning("No day-trade results. Increase lookback (<=90d), confirm API access, or try fewer symbols.")
        else:
            st.dataframe(df_day, use_container_width=True, hide_index=True)
            csv = df_day.to_csv(index=False).encode()
            st.download_button("Download CSV (Day)", data=csv, file_name="day_trade_leaderboard.csv", mime="text/csv")

    with tab2:
        st.subheader("Top Swing Opportunities")
        df_swing = leaderboard(results, mode="swing")
        if df_swing.empty:
            st.warning("No swing results. Increase daily lookback or check API limits.")
        else:
            st.dataframe(df_swing, use_container_width=True, hide_index=True)
            csv2 = df_swing.to_csv(index=False).encode()
            st.download_button("Download CSV (Swing)", data=csv2, file_name="swing_trade_leaderboard.csv", mime="text/csv")

    with tab3:
        st.subheader("Per-Coin Drilldown")
        for sym in watchlist:
            with st.expander(f"🔍 {sym} details", expanded=False):
                day = results.get(sym, {}).get("day")
                sw = results.get(sym, {}).get("swing")
                if not day and not sw:
                    st.info("No data available.")
                    continue

                if day:
                    st.markdown("**Day-trade**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Bias", day.bias)
                    c2.metric("Score", day.score)
                    c3.metric("Last", f"{day.last_price:.6f}")
                    c4.metric("Entry", f"{day.suggested_entry:.6f}")
                    c5.metric("Stop", f"{day.suggested_stop:.6f}")
                    qty, risk = suggested_position_size(capital, day.suggested_entry, day.suggested_stop)
                    st.caption(f"Suggested size ≈ {qty:.4f} units (~${risk:.2f} risk @1%). TP ≈ {day.suggested_take_profit:.6f}")
                    st.write(day.notes)

                if sw:
                    st.markdown("**Swing**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Bias", sw.bias)
                    c2.metric("Score", sw.score)
                    c3.metric("Last", f"{sw.last_price:.6f}")
                    c4.metric("Entry", f"{sw.suggested_entry:.6f}")
                    c5.metric("Stop", f"{sw.suggested_stop:.6f}")
                    qty2, risk2 = suggested_position_size(capital, sw.suggested_entry, sw.suggested_stop)
                    st.caption(f"Suggested size ≈ {qty2:.4f} units (~${risk2:.2f} risk @1%). TP ≈ {sw.suggested_take_profit:.6f}")
                    st.write(sw.notes)

    st.success("Analysis complete.")
else:
    st.info("Configure settings in the sidebar and click **Generate Analysis** to run the analysis.")
    st.caption("Powered by CoinGecko. For higher rate limits, set COINGECKO_API_KEY in your environment.")