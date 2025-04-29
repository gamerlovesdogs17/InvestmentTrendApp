import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ----- Data Functions -----
@st.cache_data
def fetch_data(ticker, period="1d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No price data found. Ticker may be invalid or delisted.")
    return df

@st.cache_data
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def compute_bollinger(prices, window=20, num_std=2):
    ma  = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return ma + num_std*std, ma - num_std*std

# ----- Pattern Detection -----
def detect_pattern(prices):
    """
    Scan for the most recent occurrence of classic chart patterns:
    - Head & Shoulders
    - Double Top, Triple Top
    - Inverse Head & Shoulders
    - Double Bottom, Unique Three River
    - Rising Wedge, Falling Wedge
    Returns pattern name or None.
    """
    s = prices.values
    idx = np.arange(len(s))
    peaks    = [i for i in range(1,len(s)-1) if s[i]>s[i-1] and s[i]>s[i+1]]
    troughs  = [i for i in range(1,len(s)-1) if s[i]<s[i-1] and s[i]<s[i+1]]
    detections = []
    
    # Head & Shoulders (three peaks, middle highest)
    if len(peaks)>=3:
        p1,p2,p3 = peaks[-3:]
        if s[p2]>s[p1] and s[p2]>s[p3] and abs(s[p1]-s[p3])<0.02*s[p2]:
            detections.append(('Head and shoulders', p3))
    # Double Top
    if len(peaks)>=2:
        p1,p2 = peaks[-2:]
        if abs(s[p1]-s[p2])<0.01*np.mean([s[p1],s[p2]]):
            detections.append(('Double top', p2))
    # Triple Top
    if len(peaks)>=3:
        last3 = peaks[-3:]
        if np.std(s[last3])<0.01*np.mean(s[last3]):
            detections.append(('Triple top', last3[-1]))
    # Inverse Head & Shoulders
    if len(troughs)>=3:
        t1,t2,t3 = troughs[-3:]
        if s[t2]<s[t1] and s[t2]<s[t3] and abs(s[t1]-s[t3])<0.02*np.mean([s[t1],s[t3]]):
            detections.append(('Inverse head and shoulders', t3))
    # Double Bottom
    if len(troughs)>=2:
        t1,t2 = troughs[-2:]
        if abs(s[t1]-s[t2])<0.01*np.mean([s[t1],s[t2]]):
            detections.append(('Double bottom', t2))
    # Unique Three River (three rising troughs)
    if len(troughs)>=3:
        t1,t2,t3 = troughs[-3:]
        if s[t1]<s[t2]<s[t3]:
            detections.append(('Unique three river', t3))
    # Rising Wedge (positive trend but second half slope < first)
    if len(s)>10:
        mid = len(s)//2
        m_all,_ = np.polyfit(idx, s, 1)
        m1,_    = np.polyfit(idx[:mid], s[:mid], 1)
        m2,_    = np.polyfit(idx[mid:], s[mid:], 1)
        if m_all>0 and m2<m1:
            detections.append(('Rising wedge', peaks[-1] if peaks else mid))
        if m_all<0 and m2>m1:
            detections.append(('Falling wedge', troughs[-1] if troughs else mid))

    # pick the detection with greatest index (most recent)
    if detections:
        pattern, _ = max(detections, key=lambda x: x[1])
        return pattern
    return None

# ----- Session State -----
if 'started' not in st.session_state:
    st.session_state.started = False
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""

# ----- Input Screen -----
if not st.session_state.started:
    st.title("Investment Trend App")
    symbol_input    = st.text_input("Enter stock ticker:", value="AAPL").upper()
    show_rsi_input  = st.checkbox("Show RSI", value=True)
    show_boll_input = st.checkbox("Show Bollinger Bands", value=True)
    refresh_input   = st.slider("Refresh every N minutes", 1, 5, 1)
    if st.button("Start Chart"):
        st.session_state.started   = True
        st.session_state.symbol    = symbol_input
        st.session_state.show_rsi  = show_rsi_input
        st.session_state.show_boll = show_boll_input
        st.session_state.refresh   = refresh_input
    st.stop()

# ----- Chart Screen -----
if st.button("Stop Chart"):
    st.session_state.started = False
    st.stop()

# auto-refresh
st_autorefresh(interval=st.session_state.refresh * 60 * 1000, key="auto")

# Fetch data
try:
    df = fetch_data(st.session_state.symbol)
except Exception as e:
    st.error(e)
    st.stop()

closes = df['Close']
times  = df.index
first  = closes.iloc[0].item()
last   = closes.iloc[-1].item()

# Trend
idx = np.arange(len(closes))
m, b = np.polyfit(idx, closes.values, 1)
trend_name    = 'Uptrend' if m>0 else 'Downtrend'
trend_message = f"Detected trend: price is {'rising' if m>0 else 'falling'}."

# Pattern
pattern = detect_pattern(closes)
pattern_name    = pattern if pattern else 'None'
pattern_message = (f"Detected pattern: {pattern_name}." if pattern else "No recognizable chart pattern detected.")

# Signal (pattern takes priority)
bullish = ['Inverse head and shoulders','Double bottom','Unique three river','Falling wedge']
bearish = ['Head and shoulders','Double top','Triple top','Rising wedge']
if pattern in bullish:
    signal = 'BUY'
elif pattern in bearish:
    signal = 'SELL'
else:
    signal = 'BUY' if last>first else 'SELL' if last<first else 'HOLD'

# Indicators
upper, lower = compute_bollinger(closes) if st.session_state.show_boll else (None,None)
rsi = compute_rsi(closes) if st.session_state.show_rsi else None

# ----- Layout -----
sig_col, chart_col, info_col = st.columns([1.5,4,2.5])

with sig_col:
    st.markdown("### Signal")
    if signal=='BUY': st.success(signal)
    elif signal=='SELL': st.error(signal)
    else: st.warning(signal)

with chart_col:
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), sharex=True)
    # price line colored green/red
    color = 'green' if last>=first else 'red'
    ax1.plot(times, closes, color=color, label='Price')
    ax1.plot(times, m*idx+b, '--', color='orange', label='Trend')
    if st.session_state.show_boll:
        ax1.plot(times, upper, '--', alpha=0.5, label='Bollinger Upper')
        ax1.plot(times, lower, '--', alpha=0.5, label='Bollinger Lower')
    ax1.set_title(f"{st.session_state.symbol} – Daily Change: {last-first:+.2f}")
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    if rsi is not None:
        ax2.plot(times, rsi, label='RSI')
        ax2.axhline(y=70, linestyle='--', alpha=0.3)
        ax2.axhline(y=30, linestyle='--', alpha=0.3)
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')} — next in {st.session_state.refresh} min")

with info_col:
    st.markdown('### Trend Info')
    st.info(f"**{trend_name}**\n{trend_message}")
    st.markdown('### Pattern Info')
    st.warning(f"**{pattern_name}**\n{pattern_message}")
