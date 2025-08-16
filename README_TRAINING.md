# Training Module — XGBoost + CoinGecko v2 (ASI-ready)

### Key changes
- **No pre-selected tickers**: you must choose symbols before training.
- **ASI supported** with a **legacy FET fallback** for history if ASI returns thin data.
- **Robust fetch**: range → days=max → legacy FET (ASI only).
- **XGBoost** classifier with early stopping and AUC metric.
- **Auto-defaults** available (first selected symbol), but they run only after you pick at least one symbol.

### Replace the old training page
If you still see the old training page from your previous “app module,” remove it so only this one remains:
- Delete or rename the old file in your `pages/` folder (e.g., `pages/Training.py`, `pages/02_Training.py`, or similar).
- Keep **`pages/01_Training.py`** from this package as your single training UI.

### Setup
1) Copy `pages/01_Training.py` into your app's `pages/` folder.
2) Put `training_utils.py` in your project root (same folder as your main Streamlit app file).
3) `pip install -r requirements_training.txt`
4) Launch the app, select symbols, optionally use auto-suggest, and **Train Selected**.

### Tips
- For **ASI**, if you select long lookbacks and get “Not enough data,” the module automatically falls back to **FET** historical prices when needed.
- For GPU training, set `XGB_USE_GPU=1` or tick the checkbox.