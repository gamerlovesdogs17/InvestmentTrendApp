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
    # Guarantee Trend always present
    if df.empty or len(df) < 2:
        df["Trend"]    = np.nan
        df["BB_mid"]   = np.nan
        df["BB_std"]   = np.nan
        df["BB_upper"] = np.nan
        df["BB_lower"] = np.nan
        df["RSI"]      = np.nan
        return df

    x = np.arange(len(df))
    # Linear trend
    slope, inter = np.polyfit(x, df["Close"].values, 1)
    df["Trend"] = slope * x + inter

    # Bollinger Bands
    df["BB_mid"]   = df["Close"].rolling(20).mean()
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # RSI
    delta   = df["Close"].diff()
    up      = delta.clip(lower=0)
    down    = -delta.clip(upper=0)
    ma_up   = up.ewm(span=14, adjust=False).mean()
    ma_down = down.ewm(span=14, adjust=False).mean()
    rs      = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def detect_pattern(df):
    """Detect the most recent classic reversal/continuation pattern."""
    closes = df["Close"].values
    # simple pivot finder
    piv_hi = np.where((closes > np.roll(closes,1)) & (closes > np.roll(closes,-1)))[0]
    piv_lo = np.where((closes < np.roll(closes,1)) & (closes < np.roll(closes,-1)))[0]

    # Double Top
    if len(piv_hi) >= 2:
        h1, h2 = piv_hi[-2], piv_hi[-1]
        if abs(closes[h1] - closes[h2]) / closes[h1] < 0.01:
            return "Double top", h2

    # Triple Top
    if len(piv_hi) >= 3:
        h1, h2, h3 = piv_hi[-3], piv_hi[-2], piv_hi[-1]
        ref = closes[h1]
        if all(abs(closes[h] - ref)/ref < 0.01 for h in (h2,h3)):
            return "Triple top", h3

    # Head & Shoulders
    if len(piv_hi) >= 3:
        l,m,r = closes[piv_hi[-3]], closes[piv_hi[-2]], closes[piv_hi[-1]]
        if m>l and m>r and abs(l-r)/l<0.02:
            return "Head & shoulders", piv_hi[-2]

    # Rising Wedge
    if len(piv_hi)>=2 and len(piv_lo)>=2:
        sh,_ = np.polyfit(piv_hi, closes[piv_hi],1)
        sl,_ = np.polyfit(piv_lo, closes[piv_lo],1)
        if sh>0 and sl>0 and sl<sh:
            return "Rising wedge", piv_hi[-1]

    # Inverse H&S
    if len(piv_lo) >= 3:
        l,m,r = closes[piv_lo[-3]], closes[piv_lo[-2]], closes[piv_lo[-1]]
        if m<l and m<r and abs(l-r)/l<0.02:
            return "Inverse H&S", piv_lo[-2]

    # Double Bottom
    if len(piv_lo) >= 2:
        b1, b2 = piv_lo[-2], piv_lo[-1]
        if abs(closes[b1] - closes[b2]) / closes[b1] < 0.01:
            return "Double bottom", b2

    # Unique Three River
    if len(piv_lo) >= 3:
        l,m,r = closes[piv_lo[-3]], closes[piv_lo[-2]], closes[piv_lo[-1]]
        if l>m and r>m:
            return "Unique three river", piv_lo[-1]

    # Falling Wedge
    if len(piv_hi)>=2 and len(piv_lo)>=2:
        sh,_ = np.polyfit(piv_hi, closes[piv_hi],1)
        sl,_ = np.polyfit(piv_lo, closes[piv_lo],1)
        if sh<0 and sl<0 and sl>sh:
            return "Falling wedge", piv_lo[-1]

    return "None", None

def get_market_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.weekday() < 5 and now.hour>=9 and now.hour<16:
        return "Market Open"
    if now.weekday()<5 and (now.hour>=16 or now.hour<9):
        return "After Hours Trading"
    return "Market Closed"

def get_24h_status():
    # stock 24h sessions: Sun 8pm ET – Fri 8pm ET
    now = datetime.now(pytz.timezone("US/Eastern"))
    start = now.replace(hour=20, minute=0, second=0)
    end   = now.replace(hour=20, minute=0, second=0)
    # if between Fri 20:00 and Sun 20:00 ET => closed
    if now.weekday()==4 and now.hour>=20 or now.weekday()==5 or (now.weekday()==6 and now.hour<20):
        return "24h Markets Closed"
    return "24h Markets Open"


# ──────────────────────────────────────────────────────────────────────────
# App UI
# ──────────────────────────────────────────────────────────────────────────

st.title("📈 Intraday Trend & Pattern Scanner")

# Sidebar controls
ticker = st.text_input("Ticker", "AAPL").upper()
rsi_on = st.checkbox("Show RSI", value=True)
bb_on  = st.checkbox("Show Bollinger Bands", value=True)
refresh_min = st.slider("Refresh every N minutes", 1, 5, 1)

if "started" not in st.session_state:
    st.session_state.started = False

if st.button("Start Chart"):
    st.session_state.started = True

if st.session_state.started:
    # Fetch & compute
    try:
        df = get_intraday(ticker)
        df = compute_indicators(df)
        pattern, pat_idx = detect_pattern(df)
        first = float(df["Close"].iloc[0])
        last  = float(df["Close"].iloc[-1])
        # Buy/Sell/Hold signal vs trend line
        sig = (
            "BUY"  if last > df["Trend"].iloc[-1]
            else "SELL" if last < df["Trend"].iloc[-1]
            else "HOLD"
        )
    except Exception as e:
        st.error(e)
        st.stop()

    # Layout columns
    col1, col2, col3 = st.columns([1,3,1])

    # ── Left pane: Signal & market status
    with col1:
        st.markdown("## Signal & Market")
        st.markdown(f"<div style='"
                    f"background:#2c2f33;padding:1rem;border-radius:0.5rem;text-align:center;'>"
                    f"<h2 style='color:white;margin:0;'>{sig}</h2>"
                    f"<p style='color:#b0bec5;margin:0;'>"
                    f"After Hours Trading --<br>24h Markets Open"
                    f"</p></div>",
                    unsafe_allow_html=True)

    # ── Middle pane: the two‐panel chart
    with col2:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5), sharex=True,
                                       gridspec_kw={"height_ratios":[2,1]})
        # Price + trend + BB
        price_color = "green" if last >= first else "red"
        ax1.plot(df.index, df["Close"], color=price_color, label="Price")
        ax1.plot(df.index, df["Trend"], "--", linewidth=1, label="Trend")
        if bb_on:
            ax1.plot(df.index, df["BB_upper"], ":", linewidth=1, label="Bollinger Upper")
            ax1.plot(df.index, df["BB_lower"], ":", linewidth=1, label="Bollinger Lower")
        ax1.set_ylabel("Price (USD)")
        ax1.legend(loc="upper left")
        ax1.set_title(f"{ticker} – Daily Change: {last-first:+.2f}")

        # RSI
        if rsi_on:
            ax2.plot(df.index, df["RSI"], color="orange", label="RSI")
            ax2.axhline(70, "--", alpha=0.5)
            ax2.axhline(30, "--", alpha=0.5)
            ax2.set_ylabel("RSI")
            ax2.legend(loc="upper left")

        plt.xticks(rotation=30)
        plt.tight_layout(pad=2)
        st.pyplot(fig)

        # refresh timer
        next_in = refresh_min*60 - (datetime.now().second % (refresh_min*60))
        st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}  —  next in {next_in//60} min")

    # ── Right pane: Info panels
    with col3:
        st.markdown("## Info Panels")
        st.markdown(f"<div style='background:#233044;padding:1rem;border-radius:0.5rem;'>"
                    f"<strong style='color:white;'>Uptrend Detected</strong><br>"
                    f"<span style='color:#b0bec5;'>trend: price is rising.</span>"
                    f"</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='background:#4a442f;padding:1rem;border-radius:0.5rem;margin-top:1rem;'>"
                    f"<strong style='color:#e6e183;'>{pattern} Detected</strong><br>"
                    f"<span style='color:#b8a94a;'>pattern: {pattern}.</span>"
                    f"</div>",
                    unsafe_allow_html=True)

    # Auto‐refresh
    st.experimental_rerun()
    st.experimental_set_query_params()  # dummy to suppress Streamlit warning

else:
    st.write("👈 Enter a ticker and click **Start Chart**")
