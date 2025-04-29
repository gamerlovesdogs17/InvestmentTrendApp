import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import pytz

# ----- force a full rerun (Streamlit 1.45+)
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException
    def rerun():
        raise RerunException({})
except ImportError:
    def rerun():
        st.experimental_rerun()


# ----- data & indicator funcs -----------------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    df = (
        yf.download(ticker, period="1d", interval="1m", progress=False)
          .dropna()
    )
    return df

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = np.arange(len(df))
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + intercept

    m20 = df["Close"].rolling(20).mean()
    s20 = df["Close"].rolling(20).std()
    df["BollingerUpper"] = m20 + 2 * s20
    df["BollingerLower"] = m20 - 2 * s20

    df["RSI"] = compute_rsi(df["Close"])
    return df

def detect_pattern(df: pd.DataFrame) -> tuple[str, int | None]:
    """
    Stub for eight patterns:
      Head&Shoulders, Double Top, Triple Top, Rising Wedge,
      Inverse H&S, Double Bottom, Unique Three River, Falling Wedge.
    Returns (name, index_of_pattern_start) or ("None", None).
    """
    # TODO: replace with real detection
    return "None", None


# ----- market status funcs --------------------------------------------------

def get_market_status() -> str:
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern).time()
    if time(9,30) <= now <= time(16,0):
        return "Market Open"
    if time(16,0) < now <= time(20,0):
        return "After Hours Trading"
    return "Market Closed"

def get_24h_status() -> str:
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    wd, t = now.weekday(), now.time()
    # 24h for some stocks: Sun 20:00 ET → Fri 20:00 ET
    if wd < 4:  # Mon–Thu
        return "Open"
    if wd == 4:  # Fri
        return "Open" if t < time(20,0) else "Closed"
    if wd == 6:  # Sun
        return "Open" if t >= time(20,0) else "Closed"
    return "Closed"


# ----- session state init ---------------------------------------------------

if "started" not in st.session_state:
    st.session_state.started = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "rsi_on" not in st.session_state:
    st.session_state.rsi_on = True
if "bb_on" not in st.session_state:
    st.session_state.bb_on = True
if "refresh" not in st.session_state:
    st.session_state.refresh = 1


# ----- SETTINGS SCREEN ------------------------------------------------------

if not st.session_state.started:
    st.title("📈 Intraday Trend & Pattern Scanner")
    ticker_input = st.text_input("Ticker (e.g. AAPL)", st.session_state.ticker)
    rsi_input    = st.checkbox("Show RSI", st.session_state.rsi_on)
    bb_input     = st.checkbox("Show Bollinger Bands", st.session_state.bb_on)
    refresh_input= st.slider("Refresh every N minutes", 1, 5, st.session_state.refresh)

    if st.button("▶️ Start Chart"):
        st.session_state.ticker  = ticker_input.upper()
        st.session_state.rsi_on  = rsi_input
        st.session_state.bb_on   = bb_input
        st.session_state.refresh = refresh_input
        st.session_state.started = True
        rerun()


# ----- CHART SCREEN ---------------------------------------------------------

else:
    st.title("📈 Intraday Trend & Pattern Scanner")
    if st.button("← Back to Settings"):
        st.session_state.started = False
        rerun()

    ticker = st.session_state.ticker
    rsi_on = st.session_state.rsi_on
    bb_on  = st.session_state.bb_on
    refresh= st.session_state.refresh

    df = get_intraday(ticker)
    if df.empty:
        st.error(f"No intraday data available for '{ticker}'.")
        st.stop()

    df = compute_indicators(df)
    pattern, idx = detect_pattern(df)
    first = float(df["Close"].iloc[0])
    last  = float(df["Close"].iloc[-1])

    # ——— plot ——————————————————————————————————————————————————————
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,6), sharex=True)
    price_color = "green" if last >= first else "red"
    ax1.plot(df.index, df["Close"], color=price_color, label="Price", linewidth=1)
    ax1.plot(df.index, df["Trend"],    "--",       label="Trend", linewidth=1)
    if bb_on:
        ax1.plot(df.index, df["BollingerUpper"], ":", label="Boll Upper", linewidth=1)
        ax1.plot(df.index, df["BollingerLower"], ":", label="Boll Lower", linewidth=1)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize="small")

    if rsi_on:
        ax2.plot(df.index, df["RSI"], color="orange", label="RSI", linewidth=1.2)
        ax2.axhline(70, "--", alpha=0.5)
        ax2.axhline(30, "--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left", fontsize="small")

    st.pyplot(fig, use_container_width=True)

    # ——— SIGNAL & STATUS PANELS —————————————————————————————————————
    col1, col2 = st.columns([1,3])
    with col1:
        sig = "BUY" if last > first else "SELL"
        sig_color = "#0a0" if sig=="BUY" else "#a00"
        st.markdown(
            f"<div style='background:{sig_color};color:#fff;padding:20px;text-align:center;"
            f"font-size:24px;font-weight:bold;border-radius:4px'>{sig}</div>",
            unsafe_allow_html=True,
        )

        mkt = get_market_status()
        m24 = get_24h_status()
        st.markdown(
            "<div style='background:#112;color:#dde;padding:10px;border-radius:4px'>"
            f"⏰ {mkt}<br>─── 24h Markets {m24}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        # trend panel
        trend_text = "rising" if last >= first else "falling"
        st.markdown(
            "<div style='background:#023;color:#eef;padding:12px;border-radius:4px'>"
            f"<strong>🌟 Trend Detected</strong><br>trend: price is {trend_text}."
            "</div>",
            unsafe_allow_html=True,
        )
        # pattern panel
        link = "" if pattern=="None" else "🔗"
        st.markdown(
            "<div style='background:#432;color:#ffd;padding:12px;border-radius:4px'>"
            f"<strong>🔍 Pattern Detected {link}</strong><br>"
            f"pattern: {pattern}.{'' if idx is None else f' (at idx {idx})'}"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write(
        f"*Last refresh:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
        f"*next in {refresh} min*"
    )

    # schedule auto‐refresh
    # (Streamlit doesn’t support a native timer; you can use streamlit-autorefresh or JS)
