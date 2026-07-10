import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import json
import urllib.parse
import requests
import numpy as np
from datetime import datetime

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Quant Metals Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (QUANT TRADING THEME) ---
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #12161D !important;
        border-left: 3px solid #FFD700;
        border-radius: 2px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #E0E6ED !important; }
    h1, h2, h3, h4 { font-weight: 300 !important; letter-spacing: 1px; text-transform: uppercase; font-family: 'Courier New', Courier, monospace; }
    .streamlit-expanderHeader { background-color: rgba(18, 22, 29, 0.5) !important; border-radius: 2px; font-family: 'Courier New', Courier, monospace; }
    .stCaption { font-family: 'Courier New', Courier, monospace; color: #FFD700 !important; }
    
    /* Custom Bank Quote Box */
    .bank-quote-box {
        background: linear-gradient(135deg, #1a1e24 0%, #12161d 100%);
        border: 1px solid #00E5FF;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
    }
    .bank-quote-title { color: #888; font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 5px; font-family: 'Courier New', monospace;}
    .bank-quote-price { color: #00E5FF; font-size: 2.5rem; font-weight: bold; margin: 0; font-family: 'Courier New', monospace;}
    .bank-quote-sub { color: #E0E6ED; font-size: 0.85rem; margin-top: 5px; font-family: 'Courier New', monospace;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nltk()

try:
    api_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=api_key)
except Exception:
    groq_client = None

# --- TELEGRAM BOT SETUP ---
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

# --- CACHED MACRO & FOREX DATA ---
@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        myr = yf.Ticker("MYR=X").history(period="1d")['Close'].iloc[-1]
        dxy = yf.Ticker("DX-Y.NYB").history(period="2d")['Close']
        tnx = yf.Ticker("^TNX").history(period="2d")['Close']
        return myr, dxy.iloc[-1], ((dxy.iloc[-1] - dxy.iloc[-2])/dxy.iloc[-2])*100, tnx.iloc[-1], ((tnx.iloc[-1] - tnx.iloc[-2])/tnx.iloc[-2])*100
    except: return 4.70, 100.0, 0.0, 4.0, 0.0

USD_TO_MYR, DXY_PRICE, DXY_PCT, TNX_PRICE, TNX_PCT = get_macro_data()

# --- THE METALS DATABASE ---
BASE_COMMODITIES = {
    "PRECIOUS METALS": {
        "Gold": {"ticker": "GC=F", "search": "gold", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035},
        "Silver": {"ticker": "SI=F", "search": "silver", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035},
        "Platinum": {"ticker": "PL=F", "search": "platinum", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035},
        "Palladium": {"ticker": "PA=F", "search": "palladium", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035},
    },
    "INDUSTRIAL METALS": {
        "Copper": {"ticker": "HG=F", "search": "copper", "multiplier": 1.0, "unit": "Pound", "kg_per_unit": 0.453592},
        "Aluminum": {"ticker": "ALI=F", "search": "aluminum", "multiplier": 1.0, "unit": "Metric Ton", "kg_per_unit": 1000.0},
    }
}

if 'custom_foods' not in st.session_state: st.session_state.custom_foods = {}
if 'deleted_foods' not in st.session_state: st.session_state.deleted_foods = {}
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}
if 'active_cat' not in st.session_state: st.session_state.active_cat = "PRECIOUS METALS"
if 'active_asset' not in st.session_state: st.session_state.active_asset = "Gold"

COMMODITIES = {}
for cat, foods in BASE_COMMODITIES.items():
    for comm, data in foods.items():
        if (cat, comm) not in st.session_state.deleted_foods:
            if cat not in COMMODITIES: COMMODITIES[cat] = {}
            COMMODITIES[cat][comm] = data

for cat, foods in st.session_state.custom_foods.items():
    for comm, data in foods.items():
        if cat not in COMMODITIES: COMMODITIES[cat] = {}
        COMMODITIES[cat][comm] = data

# --- INTELLIGENCE FUNCTIONS ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_financial_data(ticker, multiplier):
    if ticker == "NONE": return 0.0, 0.0, 0.0, 50.0, pd.DataFrame() 
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="6mo")
        if hist.empty or len(hist) < 15: return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()
        hist['50_MA'] = hist['Close'].rolling(window=50).mean() 
        hist['RSI'] = calculate_rsi(hist)
        raw_today = hist['Close'].iloc[-1]
        raw_yesterday = hist['Close'].iloc[-2]
        return raw_today * multiplier, ((raw_today - raw_yesterday) / raw_yesterday) * 100, hist['50_MA'].iloc[-1] * multiplier, hist['RSI'].iloc[-1], hist.tail(90)
    except: return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()

def get_news_data(search_term):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={urllib.parse.quote(f'\"{search_term}\" (inflation OR fed OR central bank OR shortage OR strike)')}&hl=en-US&gl=US&ceid=US:en")
        articles = [{"Headline": e.title, "Threat Score": sia.polarity_scores(e.title)['compound']} for e in feed.entries[:10]]
        return (sum(a['Threat Score'] for a in articles) / len(articles) if articles else 0.0), articles
    except: return 0.0, []

def calculate_master_threat(price_pct, sentiment, rsi, dxy_pct, tnx_pct, is_osint_only):
    score = 0
    if is_osint_only:
        if sentiment < 0: score = min(100, int(abs(sentiment) * 100))
    else:
        if abs(price_pct) > 2.0: score += 20 
        if sentiment > 0.30: score += 15 
        if sentiment < -0.30: score += 15 
        if rsi > 70 or rsi < 30: score += 20 
        if abs(dxy_pct) > 0.5: score += 15 
        if abs(tnx_pct) > 2.0: score += 15 
        
    if score >= 70: return score, "VOLATILITY 1 [EXTREME]"
    if score >= 40: return score, "VOLATILITY 2 [ELEVATED]"
    return score, "VOLATILITY 3 [STABLE]"

def get_quant_action_signal(rsi, sentiment, threat_score):
    if rsi < 35 and sentiment > -0.2: 
        return "🟢 STRONG BUY", "Asset is technically oversold (discounted) with stable global sentiment."
    elif rsi > 70: 
        return "🔴 TAKE PROFITS", "Asset is technically overbought. Risk of sudden price correction is high."
    elif threat_score > 60: 
        return "🟡 HOLD / CAUTION", "High market volatility detected. Wait for macro stabilization before entry."
    else: 
        return "🟢 ACCUMULATE", "Market is stable. Optimal environment for Dollar-Cost Averaging (DCA)."

def get_ai_brief(commodity, articles, price_change, rsi, dxy_pct, tnx_pct, threat_score, is_osint_only):
    if not groq_client: return "SYS_ERR: AI CORE OFFLINE."
    headlines = [art['Headline'] for art in articles[:3]] if articles else ["No news."]
    prompt = f"Act as a Quantitative Intelligence Analyst. Target: {commodity}. Volatility: {threat_score}/100. "
    if is_osint_only: prompt += f"Base summary on News: {headlines}. Write 2 tactical sentences. Tone: Cold, professional."
    else: prompt += f"Price Change: {price_change:.2f}%. RSI: {rsi:.1f}. DXY Change: {dxy_pct:.2f}%. 10-Yr Yield Change: {tnx_pct:.2f}%. Headlines: {headlines}. Write 2 tactical sentences summarizing the investment risk/opportunity. Tone: Cold, professional."
    try: return groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile").choices[0].message.content
    except: return "SYS_ERR: AI CORE UNRESPONSIVE."

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; margin-bottom: 10px;">
    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
    </svg>
    <h3 style="color: #E0E6ED; margin: 10px 0 0 0; font-size: 1.1rem; letter-spacing: 2px;">QUANT.CORE</h3>
</div>
""", unsafe_allow_html=True)

# QUICK ACCESS BUTTONS
st.sidebar.markdown("### ⚡ QUICK ACCESS")
col_q1, col_q2 = st.sidebar.columns(2)
if col_q1.button("🥇 GOLD", use_container_width=True):
    st.session_state.active_cat = "PRECIOUS METALS"
    st.session_state.active_asset = "Gold"
    st.rerun()
if col_q2.button("🥈 SILVER", use_container_width=True):
    st.session_state.active_cat = "PRECIOUS METALS"
    st.session_state.active_asset = "Silver"
    st.rerun()

st.sidebar.divider()

if not COMMODITIES: st.stop()

st.sidebar.markdown("### ⌕ MANUAL OVERRIDE")
selected_category = st.sidebar.selectbox("1. SELECT SECTOR", list(COMMODITIES.keys()), index=list(COMMODITIES.keys()).index(st.session_state.active_cat) if st.session_state.active_cat in COMMODITIES else 0)
selected_commodity = st.sidebar.selectbox("2. SELECT ASSET", list(COMMODITIES[selected_category].keys()), index=list(COMMODITIES[selected_category].keys()).index(st.session_state.active_asset) if st.session_state.active_asset in COMMODITIES[selected_category] else 0)

# Update state so buttons stay synced
st.session_state.active_cat = selected_category
st.session_state.active_asset = selected_commodity

details = COMMODITIES[selected_category][selected_commodity]
is_osint_only = details.get("ticker") == "NONE"

st.sidebar.divider()
st.sidebar.markdown("### ⛙ MACRO ECONOMICS")
st.sidebar.metric("US DOLLAR INDEX (DXY)", f"{DXY_PRICE:.2f}", f"{DXY_PCT:.2f}%", help="A stronger US Dollar makes globally traded metals more expensive, usually driving prices down.")
st.sidebar.metric("10-YR TREASURY YIELD", f"{TNX_PRICE:.2f}%", f"{TNX_PCT:.2f}%", help="Higher yields make bonds more attractive than non-yielding metals like Gold, usually driving prices down.")

# --- MAIN DASHBOARD UI ---
st.title("❖ QUANTITATIVE ASSET TERMINAL")

# Fetch Data
price_usd, price_change, trend_ma, rsi, price_history = get_financial_data(details["ticker"], details.get("multiplier", 1.0))
avg_sentiment, news_articles = get_news_data(details["search"])
threat_score, threat_level = calculate_master_threat(price_change, avg_sentiment, rsi, DXY_PCT, TNX_PCT, is_osint_only)
action_signal, action_reason = get_quant_action_signal(rsi, avg_sentiment, threat_score)

# Standardized Gram Math
price_myr = price_usd * USD_TO_MYR
grams_per_unit = details.get("kg_per_unit", 1.0) * 1000
price_per_gram_usd = price_usd / grams_per_unit if grams_per_unit > 0 else 0
price_per_gram_myr = price_myr / grams_per_unit if grams_per_unit > 0 else 0

# Header
col_head1, col_head2 = st.columns([5, 1])
with col_head1: st.header(f"⌖ ACTIVE ASSET: {selected_commodity}")
with col_head2:
    st.write("")
    if st.button("REMOVE ASSET", type="tertiary"):
        if selected_category in st.session_state.custom_foods and selected_commodity in st.session_state.custom_foods[selected_category]:
            del st.session_state.custom_foods[selected_category][selected_commodity]
        st.rerun()

# --- ANALYST ACTION SIGNAL BANNER ---
if not is_osint_only:
    if "BUY" in action_signal or "ACCUMULATE" in action_signal:
        st.success(f"**QUANTITATIVE SIGNAL: {action_signal}** — {action_reason}")
    elif "SELL" in action_signal:
        st.error(f"**QUANTITATIVE SIGNAL: {action_signal}** — {action_reason}")
    else:
        st.warning(f"**QUANTITATIVE SIGNAL: {action_signal}** — {action_reason}")

# --- LOCAL RETAIL BANK QUOTE BOARD ---
if selected_commodity == "Gold" and price_per_gram_myr > 0:
    uob_estimate = price_per_gram_myr * 1.025 # Spot + 2.5% UOB Premium
    st.markdown(f"""
    <div class="bank-quote-box">
        <div class="bank-quote-title">🇲🇾 LIVE SYNTHETIC RETAIL QUOTE (UOB MALAYSIA)</div>
        <div class="bank-quote-price">RM {uob_estimate:.2f} / g</div>
        <div class="bank-quote-sub">Derived from Global Spot (RM {price_per_gram_myr:.2f}) + Standard 2.5% Bank Spread</div>
    </div>
    """, unsafe_allow_html=True)
elif selected_commodity == "Silver" and price_per_gram_myr > 0:
    rhb_estimate = price_per_gram_myr * 1.035 # Spot + 3.5% RHB Premium
    st.markdown(f"""
    <div class="bank-quote-box">
        <div class="bank-quote-title" style="color: #00E5FF;">🇲🇾 LIVE SYNTHETIC RETAIL QUOTE (RHB MALAYSIA)</div>
        <div class="bank-quote-price" style="color: #E0E6ED;">RM {rhb_estimate:.2f} / g</div>
        <div class="bank-quote-sub">Derived from Global Spot (RM {price_per_gram_myr:.2f}) + Standard 3.5% Bank Spread</div>
    </div>
    """, unsafe_allow_html=True)

# --- 1-GLANCE METRICS ROW ---
if not is_osint_only:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: 
        st.metric(label=f"GLOBAL SPOT ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%", help="Live global futures market price.")
        st.caption(f"> ${price_per_gram_usd:.2f} / g")
    with col2: 
        st.metric(label="GLOBAL SPOT (MYR)", value=f"RM {price_myr:.2f}", help="Global price converted to MYR using live forex rates. Does not include local bank premiums.")
        st.caption(f"> RM {price_per_gram_myr:.2f} / g")
    with col3: 
        st.metric(label="TECHNICAL RSI", value=f"{rsi:.1f}", delta="OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL", delta_color="off", help="Relative Strength Index (14-Day). Above 70 means the asset is overpriced (Panic Buying). Below 30 means it is underpriced (Discounted).")
    with col4: 
        st.metric(label="OSINT SENTIMENT", value=f"{avg_sentiment:.2f}", delta="BULLISH" if avg_sentiment > 0 else "BEARISH", delta_color="normal", help="AI analysis of global news headlines. Scores range from -1.0 (Extreme Bearish Panic) to +1.0 (Extreme Bullish Euphoria).")
    with col5: 
        st.metric(label="MARKET VOLATILITY", value=f"{threat_score}/100", delta=threat_level, delta_color="inverse" if "VOLATILITY 1" in threat_level else "off", help="Master algorithm combining Price Action, RSI, News Sentiment, US Dollar Index, and Treasury Yields to predict sudden market movements.")
else:
    st.warning("[OSINT-ONLY MODE] Tracking via Global News Sentiment only. Financial data unavailable.")

# AI Brief & Charts
st.markdown("### ⎔ AI QUANTITATIVE ANALYSIS")
with st.spinner('DECRYPTING INTEL...'):
    st.info(get_ai_brief(selected_commodity, news_articles, price_change, rsi, DXY_PCT, TNX_PCT, threat_score, is_osint_only))

col_chart, col_news = st.columns([2, 1])
with col_chart:
    if not is_osint_only and not price_history.empty:
        st.markdown("### ◱ 90-DAY TECHNICAL ANALYSIS")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=price_history.index, open=price_history['Open']*details["multiplier"], high=price_history['High']*details["multiplier"], low=price_history['Low']*details["multiplier"], close=price_history['Close']*details["multiplier"], name="Price"))
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history['50_MA']*details["multiplier"], line=dict(color='#FFD700', width=1.5), name="50-Day MA"))
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
with col_news:
    st.markdown("### ▧ LIVE OSINT CHATTER")
    if news_articles: st.dataframe(pd.DataFrame(news_articles).style.map(lambda val: f'color: {"#00E5FF" if val > 0 else "#FF3366"}', subset=['Threat Score']), hide_index=True)
    else: st.info("NO IMMEDIATE THREATS DETECTED.")

st.divider()

# --- AI INTERROGATION MODE ---
st.markdown(f"### ⎚ AI INTERROGATION TERMINAL: {selected_commodity}")
st.caption("QUERY THE SYSTEM REGARDING HISTORICAL TRENDS OR LIVE DATA.")

if selected_commodity not in st.session_state.chat_history:
    st.session_state.chat_history[selected_commodity] = []

for message in st.session_state.chat_history[selected_commodity]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"ENTER QUERY..."):
    st.session_state.chat_history[selected_commodity].append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("PROCESSING..."):
            context = f"Context for {selected_commodity}: Price Change: {price_change}%, RSI: {rsi}, Volatility Score: {threat_score}. News: {[a['Headline'] for a in articles[:3]] if news_articles else 'None'}. User Question: {prompt}. Tone: Cold, professional, financial intelligence."
            try:
                response = groq_client.chat.completions.create(messages=[{"role": "user", "content": context}], model="llama-3.3-70b-versatile").choices[0].message.content
                st.markdown(response)
                st.session_state.chat_history[selected_commodity].append({"role": "assistant", "content": response})
            except:
                st.error("SYS_ERR: AI CORE UNRESPONSIVE.")
