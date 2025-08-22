import os
import io
import json
from datetime import datetime
from typing import List, Dict
import zipfile

import streamlit as st
import pandas as pd

from training_utils import (
    COINGECKO_IDS as TRAIN_IDS,
    make_labeled_dataset,
    make_triple_barrier_dataset,
    fetch_ohlc as train_fetch_ohlc,
)
from analysis_utils import fetch_ohlc as analysis_fetch_ohlc


st.set_page_config(page_title="Dataset Export — Meme Coins", page_icon="🧾", layout="wide")
st.title("🧾 Export Labeled Datasets — Meme Coins")
st.caption("Build and export labeled feature datasets per coin for training bots or external apps.")


EXPORT_DIR = os.path.join("data_cache", "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

MEME_DEFAULTS: List[str] = [
    "PEPE","SHIB","FLOKI","BONK","WIF","BABYDOGE","BRETT","BOME","MEW","DOG","MOG","TURBO","LADYS","POPCAT","PONKE"
]
MEME_AVAILABLE = [s for s in MEME_DEFAULTS if s in TRAIN_IDS]


def _min_bars_for_interval(interval: str) -> int:
    return 120 if interval in {"60m","30m","15m"} else 210


with st.sidebar:
    st.header("Export Settings")

    all_symbols = list(TRAIN_IDS.keys())
    symbols = st.multiselect("Symbols", options=all_symbols, default=MEME_AVAILABLE)

    interval = st.selectbox("Interval", options=["60m","30m","15m","1d"], index=0)
    if interval == "1d":
        lookback = st.slider("Lookback (days)", min_value=60, max_value=1460, value=365, step=5)
    else:
        lookback = st.slider("Lookback (days, <=90)", min_value=7, max_value=90, value=60, step=1)

    label_mode = st.radio("Labeling", options=["Triple barrier (recommended)", "Simple threshold"], index=0)
    horizon = st.number_input("Horizon (bars)", min_value=1, max_value=30, value=3, step=1)
    if label_mode.startswith("Triple"):
        rr_target = st.slider("RR target", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        stop_mult = st.slider("Stop buffer (×ATR)", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
        threshold = None
    else:
        threshold = st.number_input("Return threshold (label=1 if pct_change > x)", value=0.002, step=0.001, format="%.3f")
        rr_target = None
        stop_mult = None

    fmt_opts = st.multiselect("Formats", options=["CSV","JSONL"], default=["CSV"])  # keep dependencies light
    make_zip = st.checkbox("Combine outputs into ZIP", value=True)
    save_to_disk = st.checkbox("Also save to data_cache/exports", value=True)
    allow_thin = st.checkbox("Allow thin datasets (relax min bars)", value=False,
                             help="When enabled, export proceeds even if fewer than 120 (intraday) / 210 (daily) bars are available.")
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False,
                                help="Bypass in-memory cache to refetch latest OHLC before exporting.")
    keep_ambiguous = st.checkbox("Keep ambiguous (no TP/SL hit) as 0", value=False,
                                 help="If enabled, rows where neither barrier hits within horizon are labeled 0 instead of being dropped.")
    use_training_fetcher = st.checkbox("Use training fetcher (match training data)", value=True,
                                       help="When enabled, uses the same OHLC fetch path as the Training page for consistency.")

    export_btn = st.button("📦 Build & Export", type="primary", use_container_width=True)


def _dataset_for_symbol(symbol: str, interval: str, lookback_days: int, horizon: int, label_mode: str,
                        threshold: float | None, rr_target: float | None, stop_mult: float | None,
                        *, relax_min_bars: bool = False, force_refresh_flag: bool = False,
                        keep_ambig: bool = False, use_train_fetch: bool = True) -> pd.DataFrame:
    # Fetch OHLC using selected source
    if use_train_fetch:
        if force_refresh_flag:
            try:
                train_fetch_ohlc.cache_clear()
            except Exception:
                pass
        df = train_fetch_ohlc(symbol, interval=interval, lookback_days=int(lookback_days))
    else:
        df = analysis_fetch_ohlc(symbol, interval=interval, lookback_days=int(lookback_days), force_refresh=force_refresh_flag)
    min_bars = _min_bars_for_interval(interval)
    if df is None or df.empty or (len(df) < min_bars and not relax_min_bars):
        return pd.DataFrame()
    if label_mode.startswith("Triple"):
        ds = make_triple_barrier_dataset(
            df,
            horizon=int(horizon),
            rr_target=float(rr_target),
            stop_atr_mult=float(stop_mult),
            drop_ambiguous=not keep_ambig,
        )
    else:
        ds = make_labeled_dataset(df, horizon=int(horizon), threshold=float(threshold))
    if ds is None or ds.empty:
        return pd.DataFrame()
    ds = ds.copy()
    ds.insert(0, "symbol", symbol)
    ds.insert(1, "timestamp", ds.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") if ds.index.tz is not None else ds.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    # Reorder core OHLCV up front when present
    base_cols = [c for c in ["symbol","timestamp","Open","High","Low","Close","Volume"] if c in ds.columns]
    feat_cols = [c for c in ds.columns if c not in base_cols + ["label"]]
    cols = base_cols + feat_cols + ["label"]
    return ds[cols]


if export_btn:
    if force_refresh:
        try:
            if use_training_fetcher:
                train_fetch_ohlc.cache_clear()
            else:
                analysis_fetch_ohlc.cache_clear()
        except Exception:
            pass
    if not symbols:
        st.warning("Select at least one symbol.")
    else:
        with st.spinner("Fetching data and building datasets..."):
            built: Dict[str, pd.DataFrame] = {}
            skipped: List[str] = []
            diag_rows: List[Dict[str, int]] = []
            for s in symbols:
                if use_training_fetcher:
                    df_preview = train_fetch_ohlc(s, interval=interval, lookback_days=int(lookback))
                else:
                    df_preview = analysis_fetch_ohlc(s, interval=interval, lookback_days=int(lookback), force_refresh=force_refresh)
                bars = 0 if df_preview is None or df_preview.empty else len(df_preview)
                diag_rows.append({"Symbol": s, "Bars": int(bars)})
                ds = _dataset_for_symbol(
                    s, interval, lookback, horizon, label_mode, threshold, rr_target, stop_mult,
                    relax_min_bars=allow_thin, force_refresh_flag=force_refresh,
                    keep_ambig=keep_ambiguous, use_train_fetch=use_training_fetcher,
                )
                # Fallbacks: if triple-barrier yielded empty, try keep_ambiguous=True, then simple threshold
                if ds.empty and label_mode.startswith("Triple"):
                    ds = _dataset_for_symbol(
                        s, interval, lookback, horizon, label_mode, threshold, rr_target, stop_mult,
                        relax_min_bars=allow_thin, force_refresh_flag=False,  # already fetched above
                        keep_ambig=True, use_train_fetch=use_training_fetcher,
                    )
                if ds.empty and label_mode.startswith("Triple"):
                    # Simple threshold fallback with a tiny threshold to avoid all zeros
                    ds = _dataset_for_symbol(
                        s, interval, lookback, horizon, "Simple threshold", 0.000, rr_target, stop_mult,
                        relax_min_bars=allow_thin, force_refresh_flag=False,
                        keep_ambig=False, use_train_fetch=use_training_fetcher,
                    )
                if ds.empty:
                    skipped.append(s)
                else:
                    built[s] = ds

        if diag_rows:
            diag_df = pd.DataFrame(diag_rows).sort_values("Bars", ascending=False)
            st.caption("Diagnostics — bars fetched per symbol")
            st.dataframe(diag_df, use_container_width=True, hide_index=True)
        if skipped:
            st.info(f"No dataset after labeling: {', '.join(skipped)}. Tips: increase lookback, switch to 'Simple threshold', or enable 'Allow thin datasets'.")

        if not built:
            st.error("No datasets built. Try increasing lookback or changing interval.")
        else:
            ts_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            label_tag = "tb" if label_mode.startswith("Triple") else "thr"
            files_for_zip: List[tuple[str, bytes]] = []

            for sym, ds in built.items():
                st.subheader(f"{sym} — {len(ds)} rows")
                # Quick stats
                pos = int((ds["label"] == 1).sum())
                neg = int((ds["label"] == 0).sum())
                st.caption(f"label=1: {pos} | label=0: {neg}")
                st.dataframe(ds.head(20), use_container_width=True, hide_index=True)

                base_name = f"{sym}_{interval}_{lookback}d_{label_tag}_h{horizon}_{ts_tag}"

                if "CSV" in fmt_opts:
                    csv_bytes = ds.to_csv(index=False).encode()
                    st.download_button(
                        f"Download CSV — {sym}", data=csv_bytes, file_name=base_name + ".csv", mime="text/csv"
                    )
                    if make_zip:
                        files_for_zip.append((base_name + ".csv", csv_bytes))
                    if save_to_disk:
                        with open(os.path.join(EXPORT_DIR, base_name + ".csv"), "wb") as f:
                            f.write(csv_bytes)

                if "JSONL" in fmt_opts:
                    jsonl_bytes = ds.to_json(orient="records", lines=True).encode()
                    st.download_button(
                        f"Download JSONL — {sym}", data=jsonl_bytes, file_name=base_name + ".jsonl", mime="application/json"
                    )
                    if make_zip:
                        files_for_zip.append((base_name + ".jsonl", jsonl_bytes))
                    if save_to_disk:
                        with open(os.path.join(EXPORT_DIR, base_name + ".jsonl"), "wb") as f:
                            f.write(jsonl_bytes)

            if make_zip and files_for_zip:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, data in files_for_zip:
                        zf.writestr(fname, data)
                mem.seek(0)
                zip_name = f"datasets_{interval}_{lookback}d_{label_tag}_h{horizon}_{ts_tag}.zip"
                st.download_button("Download All as ZIP", data=mem, file_name=zip_name, mime="application/zip")
                if save_to_disk:
                    with open(os.path.join(EXPORT_DIR, zip_name), "wb") as f:
                        f.write(mem.getvalue())

            st.success("Export complete.")
else:
    st.info("Pick meme coins, choose labeling and timeframe, then click Build & Export.")
    st.caption("Outputs include symbol, timestamp, OHLCV (when available), engineered features, and label.")


