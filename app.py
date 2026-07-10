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
st.set_page_config(page_title="Metals Quant Terminal", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (HIGH-DENSITY TERMINAL THEME) ---
st.markdown("""
<style>
    /* Ultra-dense, sleek metric cards */
    div[data-testid="stMetric"] {
        background-color: #0B0E14 !important;
        border-top: 2px solid #FFD700;
        border-radius: 4px;
        padding: 10px 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetric"] label { color: #8892B0 !important; font-size: 0.85rem !important; font-weight: 600; letter-spacing: 1px;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #E0E6ED !important; font-size: 1.6rem !important; font-weight: 700;}
    
    h1, h2, h3, h4 { font-weight: 400 !important; letter-spacing: 1px; text-transform: uppercase; font-family: 'Courier New', Courier, monospace; }
    
    /* Quick Access Buttons */
    .stButton>button {
        border: 1px solid #FFD700;
        background-color: #12161D;
        color: #FFD700;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton>button:hover { background-color: #FFD700; color: #12161D; }
    
    .stCaption { font-family: 'Courier New', Courier, monospace; color: #00E5FF !important; }
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

# --- STATE MANAGEMENT FOR QUICK ACCESS ---
if 'active_cat' not in st.session_state: st.session_state.active_cat = "PRECIOUS METALS"
if 'active_com' not in st.session_state: st.session_state.active_com = "Gold"

def set_asset(cat, com):
    st.session_state.active_cat = cat
    st.session_state.active_com = com

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
COMMODITIES = BASE_COMMODITIES.copy()
for cat, foods in st.session_state.custom_foods.items():
    if cat not in COMMODITIES: COMMODITIES[cat] = {}
    COMMODITIES[cat].update(foods)

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

def get_quant_signal(rsi, dxy_pct, sentiment):
    """Calculates an instant Buy/Hold/Sell signal based on technicals and macro."""
    score = 0
    if rsi < 35: score += 2      # Oversold = Buy
    elif rsi > 65: score -= 2    # Overbought = Sell
    
    if dxy_pct < -0.3: score += 1 # Dollar dropping = Metals up
    elif dxy_pct > 0.3: score -= 1
    
    if sentiment > 0.2: score += 1 # Bullish news
    elif sentiment < -0.2: score -= 1
    
    if score >= 2: return "STRONG BUY 🟢", "#00E5FF"
    if score == 1: return "ACCUMULATE ↗️", "#00cc96"
    if score == 0: return "HOLD ⚖️", "#FFD700"
    if score == -1: return "REDUCE ↘️", "#ff9900"
    if score <= -2: return "SELL / AVOID 🔴", "#FF3366"
    return "HOLD ⚖️", "#FFD700"

def get_ai_brief(commodity, signal, price_change, rsi, dxy_pct, price_myr_gram):
    if not groq_client: return "SYS_ERR: AI CORE OFFLINE."
    prompt = f"""
    You are a Senior Quantitative Analyst. 
    Target: {commodity}. 
    Current Quant Signal: {signal}.
    Data: Price Change {price_change:.2f}%, RSI {rsi:.1f}, US Dollar Change {dxy_pct:.2f}%, Local Price RM{price_myr_gram:.2f}/g.
    Write a strict, 2-sentence financial advice brief justifying the '{signal}' recommendation. Do not use emojis. Tone: Ruthless, professional, Wall Street.
    """
    try: return groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile").choices[0].message.content
    except: return "SYS_ERR: AI CORE UNRESPONSIVE."

# --- SIDEBAR (Minimized by default for focus) ---
st.sidebar.markdown("### ⌖ MANUAL OVERRIDE")
selected_category = st.sidebar.selectbox("SECTOR", list(COMMODITIES.keys()), index=list(COMMODITIES.keys()).index(st.session_state.active_cat))
selected_commodity = st.sidebar.selectbox("TARGET", list(COMMODITIES[selected_category].keys()), index=list(COMMODITIES[selected_category].keys()).index(st.session_state.active_com))
st.session_state.active_cat = selected_category
st.session_state.active_com = selected_commodity
details = COMMODITIES[st.session_state.active_cat][st.session_state.active_com]

# --- MAIN DASHBOARD UI ---
# 1. QUICK ACCESS BAR
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
with col_q1: 
    if st.button("🥇 QUICK ACCESS: GOLD", use_container_width=True): set_asset("PRECIOUS METALS", "Gold"); st.rerun()
with col_q2: 
    if st.button("🥈 QUICK ACCESS: SILVER", use_container_width=True): set_asset("PRECIOUS METALS", "Silver"); st.rerun()
with col_q3: 
    if st.button("⚙️ QUICK ACCESS: PLATINUM", use_container_width=True): set_asset("PRECIOUS METALS", "Platinum"); st.rerun()
with col_q4: 
    if st.button("⚡ QUICK ACCESS: COPPER", use_container_width=True): set_asset("INDUSTRIAL METALS", "Copper"); st.rerun()

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Fetch Data
price_usd, price_change, trend_ma, rsi, price_history = get_financial_data(details["ticker"], details.get("multiplier", 1.0))
avg_sentiment, news_articles = get_news_data(details["search"])

# --- CORE MATH: PRICE PER GRAM ---
grams_per_unit = details.get("kg_per_unit", 1.0) * 1000
price_per_gram_usd = price_usd / grams_per_unit if grams_per_unit > 0 else 0
price_myr = price_usd * USD_TO_MYR
price_per_gram_myr = price_myr / grams_per_unit if grams_per_unit > 0 else 0

# Local Retail Premium Logic
retail_label = "N/A"
retail_price = 0.0
if st.session_state.active_com == "Gold":
    retail_label = "UOB BANK EST. (MYR/g)"
    retail_price = price_per_gram_myr * 1.025 # 2.5% Premium
elif st.session_state.active_com == "Silver":
    retail_label = "RHB BANK EST. (MYR/g)"
    retail_price = price_per_gram_myr * 1.035 # 3.5% Premium
else:
    retail_label = "STANDARD RETAIL (MYR/g)"
    retail_price = price_per_gram_myr * 1.05 # Generic 5% Premium

# Calculate Quant Signal
signal_text, signal_color = get_quant_signal(rsi, DXY_PCT, avg_sentiment)

# 2. HEADER & ACTION SIGNAL
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown(f"<h1 style='margin-bottom: 0;'>❖ {st.session_state.active_com}</h1>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 15px;">
        <span style="font-family: 'Courier New', monospace; font-size: 1rem; color: #8892B0;">QUANT SIGNAL: </span>
        <span style="font-family: 'Courier New', monospace; font-size: 1.8rem; font-weight: bold; color: {signal_color}; border: 2px solid {signal_color}; padding: 5px 15px; border-radius: 4px;">{signal_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# 3. HIGH-DENSITY PRICING BLOCK (Focus on Grams)
st.markdown("#### ⛙ PRICING & LOCAL RETAIL")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1: st.metric(label=f"GLOBAL SPOT ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%")
with col_p2: st.metric(label="GLOBAL SPOT (MYR)", value=f"RM {price_myr:.2f}", delta=f"{price_change:.2f}%")
with col_p3: st.metric(label="PURE SPOT (MYR / GRAM)", value=f"RM {price_per_gram_myr:.2f}", delta="RAW VALUE", delta_color="off")
with col_p4: st.metric(label=retail_label, value=f"RM {retail_price:.2f}", delta="INCLUDES PREMIUM", delta_color="inverse")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# 4. HIGH-DENSITY MACRO BLOCK
st.markdown("#### ◱ TECHNICALS & MACRO DRIVERS")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1: st.metric(label="TECHNICAL RSI (14D)", value=f"{rsi:.1f}", delta=">70 OVERBOUGHT | <30 OVERSOLD", delta_color="off")
with col_m2: st.metric(label="US DOLLAR INDEX (DXY)", value=f"{DXY_PRICE:.2f}", delta=f"{DXY_PCT:.2f}% (Inversely correlated)", delta_color="inverse")
with col_m3: st.metric(label="10-YR TREASURY YIELD", value=f"{TNX_PRICE:.2f}%", delta=f"{TNX_PCT:.2f}%", delta_color="inverse")
with col_m4: st.metric(label="OSINT SENTIMENT", value=f"{avg_sentiment:.2f}", delta=">0 BULLISH | <0 BEARISH", delta_color="normal")

st.divider()

# 5. CHARTS & AI ANALYST ADVICE
col_c1, col_c2 = st.columns([7, 3])

with col_c1:
    if not price_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=price_history.index, open=price_history['Open']*details["multiplier"], high=price_history['High']*details["multiplier"], low=price_history['Low']*details["multiplier"], close=price_history['Close']*details["multiplier"], name="Price"))
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history['50_MA']*details["multiplier"], line=dict(color='#FFD700', width=1.5), name="50-Day MA"))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

with col_c2:
    st.markdown("### ⎔ ANALYST ADVICE")
    with st.spinner('GENERATING BRIEF...'):
        ai_summary = get_ai_brief(st.session_state.active_com, signal_text, price_change, rsi, DXY_PCT, price_per_gram_myr)
        st.info(ai_summary)
    
    st.markdown("### ▧ OSINT CHATTER")
    if news_articles: 
        df = pd.DataFrame(news_articles[:4]) # Show top 4 to save space
        st.dataframe(df.style.map(lambda val: f'color: {"#00E5FF" if val > 0 else "#FF3366"}', subset=['Threat Score']), hide_index=True, use_container_width=True)
    else: st.caption("NO RELEVANT NEWS DETECTED.")
