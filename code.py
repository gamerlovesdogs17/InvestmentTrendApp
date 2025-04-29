import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- HELPERS ---------------------------------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    df = (yf.download(ticker, period="1d", interval="1m", progress=False)
            .dropna())
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = np.arange(len(df))
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"]    = slope * x + intercept
    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["BB_upper"] = m + 2*s
    df["BB_lower"] = m - 2*s

    # RSI
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain/avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    return df

def detect_pattern(df: pd.DataFrame):
    c     = df["Close"].values
    peaks = np.argwhere((c[1:-1]>c[:-2])&(c[1:-1]>c[2:])).flatten()+1
    if len(peaks)>=3:
        return "Triple top", peaks[-1]
    if len(peaks)==2:
        return "Double top", peaks[-1]
    return "None", len(df)-1

def get_market_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    wd  = now.weekday()
    t   = now.time()
    if wd>=5:
        return "Closed"
    if t < datetime.strptime("09:30","%H:%M").time():
        return "Pre-Market"
    if t < datetime.strptime("16:00","%H:%M").time():
        return "Open Trading"
    return "After Hours"

def get_24h_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    wd  = now.weekday()
    t   = now.time()
    # 24h Markets close Fri 20:00ET → Sun 20:00ET
    if wd==5 or (wd==4 and t>=datetime.strptime("20:00","%H:%M").time()):
        return "24h Closed"
    return "24h Open"

# --- SESSION STATE DEFAULTS ------------------------------------------------

for k,v in {
    "started": False,
    "ticker":   "",
    "rsi_on":   True,
    "bb_on":    True,
    "refresh":  1
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- PAGE LAYOUT ------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("📈 Intraday Trend & Pattern Scanner")

# --- SETTINGS SCREEN -------------------------------------------------------

if not st.session_state.started:
    ticker_in = st.text_input("Ticker (e.g. AAPL)").upper()
    rsi_in    = st.checkbox("Show RSI", True)
    bb_in     = st.checkbox("Show Bollinger Bands", True)
    ref_in    = st.slider("Refresh every N minutes", 1, 5, 1)

    if st.button("▶ Start Chart"):
        if not ticker_in:
            st.error("Enter a ticker to proceed.")
        else:
            st.session_state.update({
                "ticker":  ticker_in,
                "rsi_on":  rsi_in,
                "bb_on":   bb_in,
                "refresh": ref_in,
                "started": True
            })
            st.experimental_rerun()

# --- CHART SCREEN ----------------------------------------------------------

else:
    # back‐to‐settings
    if st.button("← Back to Settings"):
        st.session_state.started = False
        st.experimental_rerun()

    # load
    ticker  = st.session_state.ticker
    rsi_on  = st.session_state.rsi_on
    bb_on   = st.session_state.bb_on
    refresh = st.session_state.refresh

    df       = get_intraday(ticker)
    df       = compute_indicators(df)
    pattern, idx = detect_pattern(df)
    first    = float(df["Close"].iloc[0])
    last     = float(df["Close"].iloc[-1])

    # --- PLOT -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,1, sharex=True, figsize=(14,7),
        gridspec_kw={"height_ratios":[3,1]},
        constrained_layout=True
    )
    clr = "green" if last>=first else "red"
    ax1.plot(df.index, df["Close"], color=clr, label="Price")
    ax1.plot(df.index, df["Trend"], "--", label="Trend")
    if bb_on:
        ax1.plot(df.index, df["BB_upper"], ":", label="Boll Upper")
        ax1.plot(df.index, df["BB_lower"], ":", label="Boll Lower")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if rsi_on:
        ax2.plot(df.index, df["RSI"], color="orange", linewidth=1.5, label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.5)
        ax2.axhline(30, linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left")
        ax2.grid(True)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.suptitle(f"{ticker} – Daily Change: {last-first:+.2f}")

    st.pyplot(fig, use_container_width=True)

    # --- SIGNAL & STATUS ---------------------------------------------------
    trend_end = df["Trend"].iloc[-1]
    sig       = "BUY"  if last>trend_end else \
                "SELL" if last<trend_end else "HOLD"
    color_map = {"BUY":"green","SELL":"red","HOLD":"gold"}
    st.markdown("### Signal")
    st.markdown(
        f"<div style='background:{color_map[sig]};"
        f"padding:1em;color:white;text-align:center;"
        f"font-size:1.5em'>{sig}</div>",
        unsafe_allow_html=True
    )

    status = get_market_status()
    m24    = get_24h_status()
    st.markdown("### Market Status")
    st.info(f"{status}   ———   {m24}", icon="⏰")

    # --- INFO PANELS ------------------------------------------------------
    st.markdown("### Info Panels")
    trend_txt = "rising" if last>=first else "falling"
    st.markdown(f"#### 🌟 Trend Detected\ntrend: price is {trend_txt}.")
    st.markdown(f"#### 🔍 Pattern Detected\npattern: {pattern}. (most recent at idx {idx})")

    # --- AUTO REFRESH ------------------------------------------------------
    now = datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S")
    st.caption(f"Last refresh: {now} — next in {refresh} min")
    st_autorefresh(interval=refresh*60*1000, key="auto")
