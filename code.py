import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

# --- Helper functions -----------------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str):
    """Download 1‐day, 1-minute data from Yahoo."""
    data = yf.download(ticker, period="1d", interval="1m", progress=False)
    data = data.dropna(how="any")
    return data

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute linear trend, Bollinger Bands, RSI."""
    # trend
    x = np.arange(len(df))
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + intercept

    # Bollinger
    rolling = df["Close"].rolling(20)
    df["BB_mid"] = rolling.mean()
    df["BB_std"] = rolling.std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def detect_pattern(df: pd.DataFrame):
    """
    Dummy “most recent pattern detector.”
    Replace with your own logic, but must return:
      - pattern_name: str
      - index_of_pattern_end: int
    """
    # example: if price made two distinct peaks above trend line → double top
    closes = df["Close"].values
    tops = np.argwhere((closes[1:-1] > closes[:-2]) & (closes[1:-1] > closes[2:])).flatten() + 1
    if len(tops) >= 2:
        return "Double top", tops[-1]
    return "None", len(df) - 1

def get_market_status(now=None):
    """Regular NYSE hours: 9:30–16:00 ET."""
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    if now.weekday() >= 5:
        return "Closed"
    t = now.time()
    if t < datetime.strptime("09:30", "%H:%M").time():
        return "Pre-Market"
    if t < datetime.strptime("16:00", "%H:%M").time():
        return "Open Trading"
    return "After Hours"

def get_24h_status(now=None):
    """
    Some exchanges run Sun 20:00 ET – Fri 20:00 ET.
    For simplicity we’ll mirror that.
    """
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    # day 6 = Saturday, 4 = Friday
    if now.weekday() == 5 or (now.weekday() == 4 and now.time() >= datetime.strptime("20:00","%H:%M").time()):
        return "24h Closed"
    return "24h Open"


# --- Session State Setup --------------------------------------------

for key, val in {
    "started": False,
    "ticker": "",
    "rsi_on": True,
    "bb_on": True,
    "refresh": 1
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --- UI: Settings Screen --------------------------------------------

st.set_page_config(layout="wide")
st.title("📈 Intraday Trend & Pattern Scanner")

if not st.session_state.started:
    ticker_input = st.text_input("Ticker (e.g. AAPL)").upper()
    rsi_input    = st.checkbox("Show RSI", value=True)
    bb_input     = st.checkbox("Show Bollinger Bands", value=True)
    refresh_input = st.slider("Refresh every N minutes", 1, 5, 1)

    if st.button("▶ Start Chart"):
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
        else:
            st.session_state.started = True
            st.session_state.ticker  = ticker_input
            st.session_state.rsi_on  = rsi_input
            st.session_state.bb_on   = bb_input
            st.session_state.refresh = refresh_input
            st.experimental_rerun()

else:
    # --- Back Button & Load Settings -------------------------------
    if st.button("← Back to Settings"):
        st.session_state.started = False
        st.experimental_rerun()

    ticker = st.session_state.ticker
    rsi_on  = st.session_state.rsi_on
    bb_on   = st.session_state.bb_on
    refresh = st.session_state.refresh

    # --- Data Fetch & Indicators -----------------------------------
    df = get_intraday(ticker)
    df = compute_indicators(df)
    pattern, pat_idx = detect_pattern(df)
    first = float(df["Close"].iloc[0])
    last  = float(df["Close"].iloc[-1])

    # --- PLOT: Shared X-Axis with tight layout ---------------------
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    # Price + Trend + Bollinger
    color = "green" if last >= first else "red"
    ax1.plot(df.index, df["Close"], color=color, linewidth=1.5, label="Price")
    ax1.plot(df.index, df["Trend"], "--", linewidth=1, label="Trend")
    if bb_on:
        ax1.plot(df.index, df["BB_upper"], ":", linewidth=1, label="Boll Upper")
        ax1.plot(df.index, df["BB_lower"], ":", linewidth=1, label="Boll Lower")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # RSI Panel
    if rsi_on:
        ax2.plot(df.index, df["RSI"], color="orange", linewidth=1.5, label="RSI")
        ax2.axhline(70, "--", alpha=0.5)
        ax2.axhline(30, "--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left")
        ax2.grid(True)

    # X-axis datetime formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.suptitle(f"{ticker} – Daily Change: {last - first:+.2f}", fontsize=14)

    st.pyplot(fig)

    # --- Signal Box -------------------------------------------------
    # Compare last vs. trend endpoint
    trend_end = df["Trend"].iloc[-1]
    sig = (
        "BUY"  if last > trend_end else
        "SELL" if last < trend_end else
        "HOLD"
    )
    st.markdown("### Signal")
    st.warning(sig, icon="⚡️")

    # --- Market Status Box -----------------------------------------
    status_str = f"{get_market_status()}   --------   {get_24h_status()}"
    st.markdown("### Market Status")
    st.info(status_str)

    # --- Info Panels ------------------------------------------------
    updown = "price is rising" if last >= first else "price is falling"
    st.markdown("### Info Panels")
    st.success(f"Uptrend Detected\ntrend: {updown}.")
    st.success(f"{pattern} Detected\npattern: {pattern}.")

    # --- Auto-refresh Timer ----------------------------------------
    now = datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S")
    st.write(f"*Last refresh:* {now} — *next in* {refresh} min")
    st.experimental_rerun(interval=refresh * 60)
