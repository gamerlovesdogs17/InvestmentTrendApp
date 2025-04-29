import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

st.set_page_config(layout="wide", page_title="Intraday Trend & Pattern Scanner")

# ----------------------------------------
# Helper functions
# ----------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    """Fetch the last 1 day of 1-minute data via yfinance."""
    df = (
        yf.download(ticker, period="1d", interval="1m")
        .dropna()
    )
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute linear trend, Bollinger Bands, and RSI."""
    x = np.arange(len(df))
    # linear trend
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + intercept

    # Bollinger Bands (20-period)
    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["Bollinger Upper"] = m + 2 * s
    df["Bollinger Lower"] = m - 2 * s

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - 100 / (1 + rs)

    return df.dropna()

def detect_pattern(df: pd.DataFrame):
    """Very simple example: look for last double-top or triple-top."""
    close = df["Close"].values
    peaks = np.where((close[1:-1] > close[:-2]) & (close[1:-1] > close[2:]))[0] + 1
    if len(peaks) >= 3 and abs(close[peaks[-1]] - close[peaks[-3]]) < 0.02 * close[peaks[-3]]:
        return "Triple top", peaks[-3]
    if len(peaks) >= 2 and abs(close[peaks[-1]] - close[peaks[-2]]) < 0.015 * close[peaks[-2]]:
        return "Double top", peaks[-2]
    return "None", None

def get_signal(df: pd.DataFrame):
    first, last = df["Close"].iloc[0], df["Close"].iloc[-1]
    return "BUY" if last > first else "SELL" if last < first else "HOLD"

def get_market_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    if now.weekday() >= 5 or now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour > 16:
        return "After Hours Trading"
    else:
        return "Regular Trading"

def get_24h_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    # Sunday 8pm ET (weekday=6, hour>=20) to Friday 8pm ET (weekday=4, hour>=20)
    if (now.weekday() == 6 and now.hour >= 20) or now.weekday() < 5 or (now.weekday() == 5 and now.hour < 20):
        return "24h Markets Open"
    else:
        return "24h Markets Closed"

# ----------------------------------------
# Settings Screen
# ----------------------------------------

if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.title("📈 Intraday Trend & Pattern Scanner")
    ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL", key="ticker")
    rsi_on = st.checkbox("Show RSI", value=True, key="rsi_on")
    bb_on  = st.checkbox("Show Bollinger Bands", value=True, key="bb_on")
    refresh = st.slider("Refresh every N minutes", 1, 5, 1, key="refresh")
    if st.button("▶️ Start Chart"):
        st.session_state.started = True
        st.experimental_rerun()
    st.stop()

# ----------------------------------------
# Chart Screen
# ----------------------------------------

st.title("📈 Intraday Trend & Pattern Scanner")
if st.button("← Back to Settings"):
    st.session_state.started = False
    st.experimental_rerun()

ticker  = st.session_state.ticker
rsi_on  = st.session_state.rsi_on
bb_on   = st.session_state.bb_on
refresh = st.session_state.refresh

# --- fetch & fallback ---
df_new = get_intraday(ticker)
if df_new.empty:
    if "last_df" in st.session_state:
        df = st.session_state.last_df.copy()
        stale = True
    else:
        st.error(f"No intraday data available for '{ticker}'.")
        st.stop()
else:
    df = df_new
    st.session_state.last_df = df.copy()
    stale = False

# compute
df = compute_indicators(df)
pattern, idx = detect_pattern(df)
signal = get_signal(df)

# draw
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios":[2,1]})
colors = "green" if signal=="BUY" else "red" if signal=="SELL" else "gray"
ax1.plot(df.index, df["Close"], color=colors, label="Price")
ax1.plot(df.index, df["Trend"], "--", color="tab:blue", label="Trend")
if bb_on:
    ax1.plot(df.index, df["Bollinger Upper"], ":", color="tab:orange", label="Bollinger Upper")
    ax1.plot(df.index, df["Bollinger Lower"], ":", color="tab:green",  label="Bollinger Lower")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.set_title(f"{ticker} – Daily Change: {df['Close'].iloc[-1] - df['Close'].iloc[0]:+.2f}")

if rsi_on:
    ax2.plot(df.index, df["RSI"], color="tab:orange", label="RSI")
    ax2.axhline(70, "--", alpha=0.5)
    ax2.axhline(30, "--", alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left")

st.pyplot(fig, use_container_width=True)

# status & signal
col1, col2 = st.columns([1,3])
with col1:
    st.markdown(f"""
    <div style="padding:1rem;background-color:{'green' if signal=='BUY' else 'red' if signal=='SELL' else 'gray'};color:#fff;text-align:center;border-radius:0.25rem">
      <h3 style="margin:0">{signal}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0.75rem;background-color:#1f2937;color:#fff;border-radius:0.25rem">
      ⏰ {get_market_status()} &nbsp; — &nbsp; 🌐 {get_24h_status()}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="padding:1rem;background-color:#1e3a8a;color:#fff;border-radius:0.5rem;margin-bottom:1rem">
      🌟 <strong>Trend Detected</strong><br/>trend: price is {'rising' if df['Trend'].iloc[-1] > df['Trend'].iloc[0] else 'falling'}.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:1rem;background-color:#78350f;color:#fff;border-radius:0.5rem">
      🔍 <strong>Pattern Detected</strong><br/>pattern: {pattern}{'' if idx is None else f' (at idx {idx})'}.
    </div>
    """, unsafe_allow_html=True)

# stale warning
if stale:
    st.warning("🔃 No new data received — showing last available intraday data.")

# next refresh
st.markdown(f"*Last refresh: {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S')} — next in {refresh} min*")
st.experimental_rerun()
