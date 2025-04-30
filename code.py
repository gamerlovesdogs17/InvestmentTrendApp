import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import pytz

# ----- data & indicators ------------------------------------------------------

@st.cache_data(ttl=60)
def get_intraday(ticker: str) -> pd.DataFrame:
    try:
        df = (
            yf.download(ticker, period="1d", interval="1m", progress=False)
              .dropna()  # drop rows with any NaN
        )
    except Exception:
        # any error (including rate-limit), return empty to trigger fallback
        return pd.DataFrame()
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
    # linear trend
    slope, intercept = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + intercept
    # Bollinger Bands
    m20 = df["Close"].rolling(20).mean()
    s20 = df["Close"].rolling(20).std()
    df["BollingerUpper"] = m20 + 2 * s20
    df["BollingerLower"] = m20 - 2 * s20
    # RSI
    df["RSI"] = compute_rsi(df["Close"])
    return df

def detect_pattern(df: pd.DataFrame) -> tuple[str, int | None]:
    # stub for: Head & Shoulders, Double/Triple Top, Wedges, etc.
    return "None", None

# ----- market status ----------------------------------------------------------

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
    # Sun 20:00 ET → Fri 20:00 ET
    if wd < 4:
        return "Open"
    if wd == 4:
        return "Open" if t < time(20,0) else "Closed"
    if wd == 6:
        return "Open" if t >= time(20,0) else "Closed"
    return "Closed"

# ----- session_state init ----------------------------------------------------

for key, val in {
    "started": False,
    "ticker":    "",
    "rsi_on":    True,
    "bb_on":     True,
    "refresh":   1,
    "last_df":   pd.DataFrame(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ----- SETTINGS SCREEN -------------------------------------------------------

if not st.session_state.started:
    st.title("📈 Intraday Trend & Pattern Scanner")
    t_in  = st.text_input("Ticker (e.g. AAPL)", st.session_state.ticker)
    r_in  = st.checkbox("Show RSI", st.session_state.rsi_on)
    bb_in = st.checkbox("Show Bollinger Bands", st.session_state.bb_on)
    rf_in = st.slider("Refresh every N minutes", 1, 5, st.session_state.refresh)

    if st.button("▶️ Start Chart"):
        st.session_state.ticker  = t_in.upper().strip()
        st.session_state.rsi_on  = r_in
        st.session_state.bb_on   = bb_in
        st.session_state.refresh = rf_in
        st.session_state.started = True
        st.experimental_rerun()
    st.stop()

# ----- CHART SCREEN ----------------------------------------------------------

st.title("📈 Intraday Trend & Pattern Scanner")
if st.button("← Back to Settings"):
    st.session_state.started = False
    st.experimental_rerun()

ticker = st.session_state.ticker
df_new  = get_intraday(ticker)

# fallback logic
if df_new.empty:
    if not st.session_state.last_df.empty:
        df, stale = st.session_state.last_df.copy(), True
    else:
        st.error(f"No intraday data available for '{ticker}'.")
        st.stop()
else:
    df, stale = df_new, False
    st.session_state.last_df = df.copy()

# compute
df = compute_indicators(df)
pattern, idx = detect_pattern(df)
first = float(df["Close"].iloc[0])
last  = float(df["Close"].iloc[-1])

# — plotting —
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,6), sharex=True)
price_col = "green" if last>=first else "red"
ax1.plot(df.index, df["Close"], color=price_col, label="Price")
ax1.plot(df.index, df["Trend"], "--", label="Trend")
if st.session_state.bb_on:
    ax1.plot(df.index, df["BollingerUpper"], ":", label="Boll Upper")
    ax1.plot(df.index, df["BollingerLower"], ":", label="Boll Lower")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left", fontsize="small")

if st.session_state.rsi_on:
    ax2.plot(df.index, df["RSI"], color="orange", label="RSI")
    ax2.axhline(70, "--", alpha=0.5)
    ax2.axhline(30, "--", alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize="small")

st.pyplot(fig, use_container_width=True)

# — info panels —
c1, c2 = st.columns([1,3], gap="medium")
with c1:
    sig = "BUY" if last>first else "SELL"
    bg  = "#0a0" if sig=="BUY" else "#a00"
    st.markdown(
        f"<div style='background:{bg};color:#fff;padding:16px;"
        "text-align:center;font-size:24px;font-weight:bold;border-radius:4px'>"
        f"{sig}</div>", unsafe_allow_html=True
    )

    mkt = get_market_status()
    m24 = get_24h_status()
    st.markdown(
        "<div style='background:#112;color:#dde;padding:12px;border-radius:4px'>"
        f"⏰ {mkt}  ───  24h Markets {m24}"
        "</div>", unsafe_allow_html=True
    )

with c2:
    trend_txt = "rising" if last>=first else "falling"
    st.markdown(
        "<div style='background:#023;color:#eef;padding:12px;border-radius:4px'>"
        f"<strong>🌟 Trend Detected</strong><br>trend: price is {trend_txt}."
        "</div>", unsafe_allow_html=True
    )

    link = "" if pattern=="None" else " 🔗"
    st.markdown(
        "<div style='background:#432;color:#ffd;padding:12px;border-radius:4px'>"
        f"<strong>🔍 Pattern Detected{link}</strong><br>"
        f"pattern: {pattern}{'' if idx is None else f' (at idx {idx})'}"
        "</div>", unsafe_allow_html=True
    )

# refresh timer & stale warning
st.write(
    f"*Last refresh:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – "
    f"*next in {st.session_state.refresh} min*"
)
if stale:
    st.warning("🔃 No new data received — showing last available intraday data.")
