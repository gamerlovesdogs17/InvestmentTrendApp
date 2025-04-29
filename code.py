import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from pytz import timezone

# ─── HELPERS ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    """Download and return 1-minute intraday data."""
    df = yf.download(ticker, period="1d", interval="1m").dropna()
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = np.arange(len(df))
    y = df["Close"].values
    if len(x) < 2:
        raise ValueError("Not enough data for trend calculation")
    slope, intercept = np.polyfit(x, y, 1)
    df["Trend"] = slope * x + intercept

    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["Bollinger Upper"] = m + 2 * s
    df["Bollinger Lower"] = m - 2 * s

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

# Patterns to detect (placeholders – you’d fill in real logic)
PATTERNS = [
    "Head and shoulders",
    "Double top",
    "Triple top",
    "Rising wedge",
    "Inverse head and shoulders",
    "Double bottom",
    "Unique three river",
    "Falling wedge",
]

def detect_pattern(df: pd.DataFrame):
    """
    Stub: Detect the most recent pattern in df.
    Returns (pattern_name:str, index_of_pattern:int).
    """
    # TODO: replace with real detection logic
    return "None", None

def get_market_status(et_now):
    if et_now.weekday() < 5 and time(9,30) <= et_now.time() <= time(16,0):
        return "⏰ Market Open"
    if et_now.weekday() < 5:
        return "⏰ After Hours Trading"
    return "⏰ Market Closed"

def get_24h_status(et_now):
    # Closed Fri 20:00 ET --> Sun 20:00 ET
    if (et_now.weekday() == 4 and et_now.time() >= time(20,0)) \
    or et_now.weekday() in (5,6 and et_now.time() < time(20,0)):
        return "🌐 24h Markets Closed"
    return "🌐 24h Markets Open"

# ─── SESSION STATE SETUP ───────────────────────────────────────────────────────

if "started" not in st.session_state:
    st.session_state.started = False

# ─── SETTINGS SCREEN ───────────────────────────────────────────────────────────

if not st.session_state.started:
    st.title("📈 Intraday Trend & Pattern Scanner")
    ticker_input = st.text_input("Ticker (e.g. AAPL)", value="")
    rsi_input = st.checkbox("Show RSI", True)
    bb_input  = st.checkbox("Show Bollinger Bands", True)
    refresh_input = st.slider("Refresh every N minutes", 1, 5, 1)

    if st.button("▶️ Start Chart"):
        st.session_state.ticker  = ticker_input.upper()
        st.session_state.rsi_on  = rsi_input
        st.session_state.bb_on   = bb_input
        st.session_state.refresh = refresh_input
        st.session_state.started = True
        st.experimental_rerun()

# ─── CHART SCREEN ──────────────────────────────────────────────────────────────

else:
    # back button
    if st.button("← Back to Settings"):
        st.session_state.started = False
        st.experimental_rerun()

    ticker  = st.session_state.ticker
    rsi_on  = st.session_state.rsi_on
    bb_on   = st.session_state.bb_on
    refresh = st.session_state.refresh

    try:
        df = get_intraday(ticker)
        df = compute_indicators(df)
    except Exception as e:
        st.error(f"No intraday data available for '{ticker}'.")
        st.stop()

    pattern, idx = detect_pattern(df)

    first = float(df["Close"].iloc[0])
    last  = float(df["Close"].iloc[-1])
    price_color = "green" if last >= first else "red"
    signal = "BUY" if last >= first else "SELL"

    # Build figure
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(12,6))
    ax1.plot(df.index, df["Close"], color=price_color, label="Price")
    ax1.plot(df.index, df["Trend"], "--", color="tab:blue", label="Trend")
    if bb_on:
        ax1.plot(df.index, df["Bollinger Upper"], "--", label="Bollinger Upper")
        ax1.plot(df.index, df["Bollinger Lower"], "--", label="Bollinger Lower")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")

    if rsi_on:
        ax2.plot(df.index, df["RSI"], color="orange", label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.5)
        ax2.axhline(30, linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left")

    fig.tight_layout()

    # layout: left=Signal+Market, mid=Chart, right=Info
    c1, c2, c3 = st.columns([1,3,1])
    with c2:
        st.pyplot(fig)

    et_now = datetime.now(timezone("US/Eastern"))

    with c1:
        st.markdown(
            f"<div style='background:{price_color};color:white;padding:12px;text-align:center;font-size:24px;border-radius:8px'>{signal}</div>",
            unsafe_allow_html=True
        )
        status = get_market_status(et_now)
        status24 = get_24h_status(et_now)
        st.info(f"{status}  ——  {status24}")

    with c3:
        # Trend Info
        updown = "Uptrend" if last>=first else "Downtrend"
        st.markdown(
            f"<div style='background:#1E3A8A;color:white;padding:12px;border-radius:8px'>"
            f"🌟 {updown} Detected<br><small>trend: price is {'rising' if last>=first else 'falling'}.</small>"
            f"</div>",
            unsafe_allow_html=True
        )
        # Pattern Info
        st.markdown(
            f"<div style='background:#7C6E5A;color:white;padding:12px;border-radius:8px'>"
            f"🔍 Pattern: {pattern}<br><small>at idx {idx}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"*Last refresh:* {datetime.now().strftime('%H:%M:%S')} — *next in {refresh} min*")
    st.experimental_memo.clear()  # force reload on rerun
    st.experimental_rerun()
