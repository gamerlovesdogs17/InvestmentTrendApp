import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

st.set_page_config(layout="wide")

@st.cache_data
def get_intraday(ticker):
    df = yf.download(
        tickers=ticker,
        period="1d",
        interval="5m",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"[Error] No price data found. Ticker may be invalid or delisted.")
    return df

@st.cache_data
def compute_indicators(df):
    # if not enough data, fill with NaN
    if len(df) < 2:
        for col in ("Trend","BB_mid","BB_std","BB_upper","BB_lower","RSI"):
            df[col] = np.nan
        return df

    x = np.arange(len(df))
    slope, inter = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + inter

    df["BB_mid"]   = df["Close"].rolling(20).mean()
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2*df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2*df["BB_std"]

    delta = df["Close"].diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ma_up   = up.ewm(span=14, adjust=False).mean()
    ma_down = down.ewm(span=14, adjust=False).mean()
    rs = ma_up/ma_down
    df["RSI"] = 100 - (100/(1+rs))

    return df

def detect_pattern(df):
    closes = df["Close"].values
    hi = np.where((closes>np.roll(closes,1)) & (closes>np.roll(closes,-1)))[0]
    lo = np.where((closes<np.roll(closes,1)) & (closes<np.roll(closes,-1)))[0]

    # Double top
    if len(hi)>=2:
        h1,h2 = hi[-2],hi[-1]
        if abs(closes[h1]-closes[h2])/closes[h1]<0.01:
            return "Double top", h2

    # Triple top
    if len(hi)>=3:
        h1,h2,h3 = hi[-3],hi[-2],hi[-1]
        ref = closes[h1]
        if all(abs(closes[h]-ref)/ref<0.01 for h in (h2,h3)):
            return "Triple top", h3

    # Head & Shoulders
    if len(hi)>=3:
        l,m,r = closes[hi[-3]],closes[hi[-2]],closes[hi[-1]]
        if m>l and m>r and abs(l-r)/l<0.02:
            return "Head & shoulders", hi[-2]

    # Rising wedge
    if len(hi)>=2 and len(lo)>=2:
        sh,_ = np.polyfit(hi, closes[hi],1)
        sl,_ = np.polyfit(lo, closes[lo],1)
        if sh>0 and sl>0 and sl<sh:
            return "Rising wedge", hi[-1]

    # Inverse H&S
    if len(lo)>=3:
        l,m,r = closes[lo[-3]],closes[lo[-2]],closes[lo[-1]]
        if m<l and m<r and abs(l-r)/l<0.02:
            return "Inverse H&S", lo[-2]

    # Double bottom
    if len(lo)>=2:
        b1,b2 = lo[-2],lo[-1]
        if abs(closes[b1]-closes[b2])/closes[b1]<0.01:
            return "Double bottom", b2

    # Unique three river
    if len(lo)>=3:
        l,m,r = closes[lo[-3]],closes[lo[-2]],closes[lo[-1]]
        if l>m<r:
            return "Unique three river", lo[-1]

    # Falling wedge
    if len(hi)>=2 and len(lo)>=2:
        sh,_ = np.polyfit(hi, closes[hi],1)
        sl,_ = np.polyfit(lo, closes[lo],1)
        if sh<0 and sl<0 and sl>sh:
            return "Falling wedge", lo[-1]

    return "None", None

def get_market_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.weekday()<5 and 9<=now.hour<16:
        return "Market Open"
    if now.weekday()<5:
        return "After Hours Trading"
    return "Market Closed"

def get_24h_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    # Sunday 8pm ET – Friday 8pm ET open
    if (now.weekday()==6 and now.hour>=20) or now.weekday()<4 or (now.weekday()==4 and now.hour<20):
        return "24h Markets Open"
    return "24h Markets Closed"

st.title("📈 Intraday Trend & Pattern Scanner")

# session state for toggling screens
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    ticker      = st.text_input("Ticker", "AAPL").upper()
    rsi_on      = st.checkbox("Show RSI", True)
    bb_on       = st.checkbox("Show Bollinger Bands", True)
    refresh_min = st.slider("Refresh every N minutes", 1, 5, 1)
    if st.button("Start Chart"):
        st.session_state.started = True
        st.session_state.ticker = ticker
        st.session_state.rsi_on  = rsi_on
        st.session_state.bb_on   = bb_on
        st.session_state.refresh = refresh_min
    st.stop()

# --- chart screen ---
ticker   = st.session_state.ticker
rsi_on   = st.session_state.rsi_on
bb_on    = st.session_state.bb_on
refresh  = st.session_state.refresh

if st.button("← Back to Settings"):
    st.session_state.started = False
    st.experimental_rerun()  # on very new Streamlit this may no longer exist

try:
    df     = get_intraday(ticker)
    df     = compute_indicators(df)
    pattern, pat_idx = detect_pattern(df)
    first  = float(df["Close"].iloc[0])
    last   = float(df["Close"].iloc[-1])
    sig    = (
        "BUY"  if last > df["Trend"].iloc[-1] else
        "SELL" if last < df["Trend"].iloc[-1] else
        "HOLD"
    )
except Exception as e:
    st.error(e)
    st.stop()

col1, col2, col3 = st.columns([1,4,1])

with col1:
    st.markdown("## Signal & Market")
    st.markdown(f"""
    <div style="background:#2c2f33;padding:1rem;border-radius:0.5rem;text-align:center;">
      <h2 style="color:white;margin:0;">{sig}</h2>
      <p style="color:#b0bec5;margin:0;">
        {get_market_status()}<br>
        --------<br>
        {get_24h_status()}
      </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True,
                                   gridspec_kw={"height_ratios":[2,1]})
    color = "green" if last>=first else "red"
    ax1.plot(df.index, df["Close"], color=color, label="Price")
    ax1.plot(df.index, df["Trend"], "--", linewidth=1, label="Trend")
    if bb_on:
        ax1.plot(df.index, df["BB_upper"], ":", label="Bollinger Upper")
        ax1.plot(df.index, df["BB_lower"], ":", label="Bollinger Lower")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.set_title(f"{ticker} – Daily Change: {last-first:+.2f}")

    if rsi_on:
        ax2.plot(df.index, df["RSI"], color="#FF9800", label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.5)
        ax2.axhline(30, linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.legend(loc="upper left")

    plt.xticks(rotation=30)
    plt.tight_layout(pad=2)
    st.pyplot(fig)

    next_in = refresh*60 - (datetime.now().second % (refresh*60))
    st.markdown(f"*Last refresh:* {datetime.now():%H:%M:%S} — *next in* {next_in//60} min")

with col3:
    st.markdown("## Info Panels")
    st.markdown(f"""
      <div style="background:#233044;padding:1rem;border-radius:0.5rem;">
        <strong style="color:white;">Uptrend Detected</strong><br>
        <span style="color:#b0bec5;">trend: price is rising.</span>
      </div>
      <div style="background:#4a442f;padding:1rem;border-radius:0.5rem;margin-top:1rem;">
        <strong style="color:#e6e183;">{pattern} Detected</strong><br>
        <span style="color:#b8a94a;">pattern: {pattern}.</span>
      </div>
    """, unsafe_allow_html=True)
