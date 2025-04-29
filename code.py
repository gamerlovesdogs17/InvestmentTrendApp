import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- Helper functions ---
def get_intraday(ticker: str) -> pd.DataFrame:
    df = (yf.download(ticker, period="1d", interval="1m")
          .dropna())
    return df

@st.cache_data(ttl=60)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Linear trend
    x = np.arange(len(df))
    if len(x) > 1:
        slope, intercept = np.polyfit(x, df["Close"].values, 1)
        df["Trend"] = slope * x + intercept
    else:
        df["Trend"] = df["Close"]
    # Bollinger Bands
    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["BB_upper"] = m + 2*s
    df["BB_lower"] = m - 2*s
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# stub - replace with real logic

def detect_pattern(df: pd.DataFrame):
    # Identify most recent of head&shoulders, double/triple top, wedges, bottoms...
    # For now, always return None
    return None, None


def get_market_status():
    now = datetime.now()
    # US equity hours 9:30-16:00
    if now.weekday() < 5 and now.time() >= datetime.strptime("09:30", "%H:%M").time() and now.time() <= datetime.strptime("16:00", "%H:%M").time():
        return "Market Open"
    elif now.weekday() < 5:
        return "After Hours"
    else:
        return "Closed"


def get_24h_status():
    # Some global stocks trade Mon 00:00 to Fri 23:59
    now = datetime.now()
    if now.weekday() < 5:
        return "24h Open"
    else:
        return "24h Closed"

# --- App state setup ---
if 'started' not in st.session_state:
    st.session_state.started = False


def start_chart():
    st.session_state.started = True


def back_to_settings():
    st.session_state.started = False

# --- UI ---
st.set_page_config(layout="wide")
st.title("📈 Intraday Trend & Pattern Scanner")

if not st.session_state.started:
    # Settings
    st.text_input("Ticker (e.g. AAPL)", key="ticker")
    st.checkbox("Show RSI", key="rsi_on")
    st.checkbox("Show Bollinger Bands", key="bb_on")
    st.slider("Refresh every N minutes", 1, 5, 1, key="refresh")
    st.button("▶️ Start Chart", on_click=start_chart)
    st.stop()

# --- Chart screen ---
ticker = st.session_state.ticker
rsi_on = st.session_state.rsi_on
bb_on = st.session_state.bb_on
refresh = st.session_state.refresh
st.button("← Back to Settings", on_click=back_to_settings)

# Fetch & compute
df = get_intraday(ticker)
df = compute_indicators(df)
pattern, idx = detect_pattern(df)
first = float(df["Close"].iloc[0])
last  = float(df["Close"].iloc[-1])

# Build figure
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.set_size_inches(12, 6)
# Price + trend
ax1.plot(df.index, df["Close"], color="green" if last>=first else "red", label="Price")
ax1.plot(df.index, df["Trend"], linestyle="--", label="Trend")
if bb_on:
    ax1.plot(df.index, df["BB_upper"], linestyle=":", label="Bollinger Upper")
    ax1.plot(df.index, df["BB_lower"], linestyle=":", label="Bollinger Lower")
if pattern and idx is not None:
    ax1.axvline(df.index[idx], linestyle="--", alpha=0.7)
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
# RSI
if rsi_on:
    ax2.plot(df.index, df["RSI"], color="orange", label="RSI")
    ax2.axhline(70, ":--", alpha=0.5)
    ax2.axhline(30, ":--", alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left")
# Layout columns
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.subheader("Signal & Market")
    sig_color = "green" if last>=first else "red"
    sig = "BUY" if last>=first else "SELL"
    st.markdown(
        f"<div style='background:#333; padding:1rem; border-radius:8px;'>"
        f"<h1 style='color:{sig_color}; text-align:center; margin:0;'>{sig}</h1>"
        f"<p style='color:#eee; text-align:center; margin:0;'>"
        f"{get_market_status()} &mdash; {get_24h_status()}</p></div>",
        unsafe_allow_html=True
    )

with col2:
    st.pyplot(fig, clear_figure=True)

with col3:
    st.subheader("Info Panels")
    # Trend panel
    trend_text = "Uptrend Detected" if last>=first else "Downtrend Detected"
    st.markdown(
        f"<div style='background:#1f3b5f; padding:1rem; border-radius:8px;'>"
        f"<strong style='color:#fff;'>{trend_text}</strong><br>"
        f"<span style='color:#eee;'>trend: price is {'rising' if last>=first else 'falling'}.</span>"
        f"</div>", unsafe_allow_html=True
    )
    st.markdown("")
    # Pattern panel
    pat = pattern or "None"
    st.markdown(
        f"<div style='background:#5f531f; padding:1rem; border-radius:8px;'>"
        f"<strong style='color:#fff;'>{pat} Detected</strong><br>"
        f"<span style='color:#eee;'>pattern: {pattern or 'No'}.</span>"
        f"</div>", unsafe_allow_html=True
    )

# Footer refresh info
now = datetime.now().strftime('%H:%M:%S')
st.caption(f"Last refresh: {now} — next in {refresh} min")
