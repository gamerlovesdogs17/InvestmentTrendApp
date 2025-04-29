import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

# --- RERUN SHIM ------------------------------------------------------------
try:
    rerun = st.experimental_rerun
except AttributeError:
    from streamlit.runtime.scriptrunner.script_runner import RerunException
    def rerun(*args, **kwargs):
        # Streamlit 1.45+ RerunException needs a 'rerun_data' arg
        raise RerunException({})

# --- HELPERS ---------------------------------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="1m", progress=False).dropna()
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = np.arange(len(df))
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + intercept

    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["BB_upper"] = m + 2*s
    df["BB_lower"] = m - 2*s

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    return df

def detect_pattern(df: pd.DataFrame):
    c = df["Close"].values
    peaks = np.argwhere((c[1:-1]>c[:-2])&(c[1:-1]>c[2:])).flatten()+1
    if len(peaks)>=3:
        return "Triple top", peaks[-1]
    if len(peaks)==2:
        return "Double top", peaks[-1]
    return "None", len(df)-1

def get_market_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    if now.weekday()>=5:
        return "Closed"
    t = now.time()
    if t < datetime.strptime("09:30","%H:%M").time():
        return "Pre-Market"
    if t < datetime.strptime("16:00","%H:%M").time():
        return "Open Trading"
    return "After Hours"

def get_24h_status(now=None):
    tz = pytz.timezone("US/Eastern")
    now = now or datetime.now(tz)
    wd = now.weekday()
    t  = now.time()
    if wd==5 or (wd==4 and t>=datetime.strptime("20:00","%H:%M").time()):
        return "24h Closed"
    return "24h Open"

# --- SESSION STATE DEFAULTS ------------------------------------------------

for key, val in {
    "started": False,
    "ticker":   "",
    "rsi_on":   True,
    "bb_on":    True,
    "refresh":  1
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- PAGE LAYOUT ------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("📈 Intraday Trend & Pattern Scanner")

# --- SETTINGS SCREEN -------------------------------------------------------

if not st.session_state.started:
    ticker_input = st.text_input("Ticker (e.g. AAPL)").upper()
    rsi_input    = st.checkbox("Show RSI", True)
    bb_input     = st.checkbox("Show Bollinger Bands", True)
    ref_input    = st.slider("Refresh every N minutes", 1, 5, 1)

    if st.button("▶ Start Chart"):
        if not ticker_input:
            st.error("Please enter a ticker.")
        else:
            st.session_state.ticker   = ticker_input
            st.session_state.rsi_on   = rsi_input
            st.session_state.bb_on    = bb_input
            st.session_state.refresh  = ref_input
            st.session_state.started  = True
            rerun()

# --- CHART SCREEN ----------------------------------------------------------

else:
    if st.button("← Back to Settings"):
        st.session_state.started = False
        rerun()

    ticker   = st.session_state.ticker
    rsi_on   = st.session_state.rsi_on
    bb_on    = st.session_state.bb_on
    refresh  = st.session_state.refresh

    df        = get_intraday(ticker)
    df        = compute_indicators(df)
    pattern, idx = detect_pattern(df)
    first     = float(df["Close"].iloc[0])
    last      = float(df["Close"].iloc[-1])

    # --- PLOT -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,1, sharex=True, figsize=(14,7),
        gridspec_kw={"height_ratios":[3,1]},
        constrained_layout=True
    )
    color = "green" if last>=first else "red"
    ax1.plot(df.index, df["Close"], color=color, label="Price")
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

    st.pyplot(fig)

    # --- SIGNAL & MARKET STATUS ------------------------------------------
    trend_end = df["Trend"].iloc[-1]
    signal    = "BUY" if last>trend_end else "SELL" if last<trend_end else "HOLD"
    sig_color = "green" if signal=="BUY" else "red" if signal=="SELL" else "yellow"
    st.markdown("### Signal")
    st.markdown(
        f"<div style='background:{sig_color};padding:1em;color:white;text-align:center;font-size:1.5em'>{signal}</div>",
        unsafe_allow_html=True
    )

    status = get_market_status()
    m24    = get_24h_status()
    st.markdown("### Market Status")
    st.info(f"{status}   --------   {m24}")

    # --- INFO PANELS ------------------------------------------------------
    st.markdown("### Info Panels")
    trend_txt = "price is rising" if last>=first else "price is falling"
    st.success(f"Uptrend Detected\ntrend: {trend_txt}")
    st.success(f"{pattern} Detected\npattern: {pattern}")

    # --- AUTO REFRESH ------------------------------------------------------
    now = datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S")
    st.write(f"*Last refresh:* {now} — *next in* {refresh} min")
    rerun(interval=refresh*60)
