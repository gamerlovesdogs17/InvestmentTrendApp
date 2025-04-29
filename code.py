import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import pytz

# ── Full-page rerun shim for Streamlit 1.45+ ────────────────────────────────
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException
    def rerun():
        raise RerunException({})
except ImportError:
    def rerun():
        st.experimental_rerun()

# ── Data Fetching & Indicators ──────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="1m", progress=False).dropna()
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
    return df.dropna()

def detect_pattern(df: pd.DataFrame) -> tuple[str, int | None]:
    # Stub: replace with your actual pattern logic
    close = df["Close"].values
    peaks = np.where((close[1:-1] > close[:-2]) & (close[1:-1] > close[2:]))[0] + 1
    if len(peaks) >= 3 and abs(close[peaks[-1]] - close[peaks[-3]]) < 0.02 * close[peaks[-3]]:
        return "Triple top", peaks[-3]
    if len(peaks) >= 2 and abs(close[peaks[-1]] - close[peaks[-2]]) < 0.015 * close[peaks[-2]]:
        return "Double top", peaks[-2]
    return "None", None

# ── Market Hours ────────────────────────────────────────────────────────────

def get_market_status() -> str:
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz).time()
    if time(9, 30) <= now <= time(16, 0):
        return "Market Open"
    if time(16, 0) < now <= time(20, 0):
        return "After Hours Trading"
    return "Market Closed"

def get_24h_status() -> str:
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    wd, t = now.weekday(), now.time()
    # Sun 20:00 → Fri 20:00
    if wd < 4:
        return "24h Markets Open"
    if wd == 4:
        return "24h Open" if t < time(20, 0) else "24h Closed"
    if wd == 6:
        return "24h Open" if t >= time(20, 0) else "24h Closed"
    return "24h Closed"

# ── Session State Initialization ───────────────────────────────────────────

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

# ── Settings Screen ────────────────────────────────────────────────────────

if not st.session_state.started:
    st.title("📈 Intraday Trend & Pattern Scanner")
    t_in   = st.text_input("Ticker (e.g. AAPL)", st.session_state.ticker)
    r_in   = st.checkbox("Show RSI", st.session_state.rsi_on)
    bb_in  = st.checkbox("Show Bollinger Bands", st.session_state.bb_on)
    rf_in  = st.slider("Refresh every N minutes", 1, 5, st.session_state.refresh)
    if st.button("▶️ Start Chart"):
        st.session_state.ticker  = t_in.upper().strip()
        st.session_state.rsi_on  = r_in
        st.session_state.bb_on   = bb_in
        st.session_state.refresh = rf_in
        st.session_state.started = True
        rerun()
    st.stop()

# ── Chart Screen ───────────────────────────────────────────────────────────

st.title("📈 Intraday Trend & Pattern Scanner")
if st.button("← Back to Settings"):
    st.session_state.started = False
    rerun()

ticker  = st.session_state.ticker
rsi_on  = st.session_state.rsi_on
bb_on   = st.session_state.bb_on
refresh = st.session_state.refresh

# Fetch & fallback
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

# Compute
df = compute_indicators(df)
pattern, idx = detect_pattern(df)
first, last = df["Close"].iloc[0], df["Close"].iloc[-1]
signal = "BUY" if last > first else "SELL" if last < first else "HOLD"

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                               gridspec_kw={"height_ratios":[2,1]})
price_color = "green" if signal=="BUY" else "red" if signal=="SELL" else "gray"
ax1.plot(df.index, df["Close"], color=price_color, label="Price")
ax1.plot(df.index, df["Trend"], "--", color="tab:blue", label="Trend")
if bb_on:
    ax1.plot(df.index, df["BollingerUpper"], ":", color="tab:orange", label="Boll Upper")
    ax1.plot(df.index, df["BollingerLower"], ":", color="tab:green",  label="Boll Lower")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left", fontsize="small")
ax1.set_title(f"{ticker}  Daily Change: {last - first:+.2f}")

if rsi_on:
    ax2.plot(df.index, df["RSI"], color="tab:orange", label="RSI")
    ax2.axhline(70, "--", alpha=0.5)
    ax2.axhline(30, "--", alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize="small")

st.pyplot(fig, use_container_width=True)

# Panels
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown(
        f"<div style='background:{price_color};color:#fff;padding:16px;"
        "text-align:center;border-radius:6px'><h2>{signal}</h2></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='background:#1e293b;color:#fff;padding:10px;border-radius:6px'>"
        f"⏰ {get_market_status()}<br>──────────<br>🌐 {get_24h_status()}"
        "</div>",
        unsafe_allow_html=True
    )

with col2:
    trend_txt = "rising" if last >= first else "falling"
    st.markdown(
        "<div style='background:#0e3c86;color:#fff;padding:12px;border-radius:6px;"
        "margin-bottom:12px'>🌟 <strong>Trend Detected</strong><br>"
        f"trend: price is {trend_txt}."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='background:#4b3b0a;color:#ffd;padding:12px;border-radius:6px'>"
        "🔍 <strong>Pattern Detected</strong><br>"
        f"pattern: {pattern}{'' if idx is None else f' (at idx {idx})'}."
        "</div>",
        unsafe_allow_html=True
    )

# Stale warning & next refresh
if stale:
    st.warning("🔃 No new data — showing last available intraday snapshot.")
st.markdown(f"*Last refresh: {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S')} — next in {refresh} min*")

# Auto-refresh
st.experimental_rerun()
