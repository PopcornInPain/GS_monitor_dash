import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from groq import Groq
import json
import urllib.parse
import requests
import numpy as np
from datetime import datetime

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Metals Intel Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
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
    h1, h2, h3 { font-weight: 300 !important; letter-spacing: 1px; text-transform: uppercase; font-family: 'Courier New', Courier, monospace; }
    .streamlit-expanderHeader { background-color: rgba(18, 22, 29, 0.5) !important; border-radius: 2px; font-family: 'Courier New', Courier, monospace; }
    .stCaption { font-family: 'Courier New', Courier, monospace; color: #FFD700 !important; }
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

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
        return True
    except: return False

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
        "Gold": {"ticker": "GC=F", "search": "gold", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035, "lat": -26.2, "lon": 28.0, "region": "Johannesburg, South Africa"},
        "Silver": {"ticker": "SI=F", "search": "silver", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035, "lat": -12.0, "lon": -77.0, "region": "Lima, Peru"},
        "Platinum": {"ticker": "PL=F", "search": "platinum", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035, "lat": -25.6, "lon": 27.4, "region": "Rustenburg, South Africa"},
        "Palladium": {"ticker": "PA=F", "search": "palladium", "multiplier": 1.0, "unit": "Troy Oz", "kg_per_unit": 0.0311035, "lat": 69.3, "lon": 88.2, "region": "Norilsk, Russia"},
    },
    "INDUSTRIAL METALS": {
        "Copper": {"ticker": "HG=F", "search": "copper", "multiplier": 1.0, "unit": "Pound", "kg_per_unit": 0.453592, "lat": -22.8, "lon": -68.9, "region": "Antofagasta, Chile"},
        "Aluminum": {"ticker": "ALI=F", "search": "aluminum", "multiplier": 1.0, "unit": "Metric Ton", "kg_per_unit": 1000.0, "lat": -22.9, "lon": -43.1, "region": "Rio de Janeiro, Brazil"},
    }
}

if 'custom_foods' not in st.session_state: st.session_state.custom_foods = {}
if 'deleted_foods' not in st.session_state: st.session_state.deleted_foods = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = {}

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

def get_weather_data(lat, lon):
    if not lat or not lon or lat == 0.0: return None
    try:
        res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&timezone=auto").json()
        return {"temp": res['current_weather']['temperature'], "rain": sum(res['daily']['precipitation_sum'][:7])}
    except: return None

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

def get_ai_brief(commodity, articles, price_change, rsi, dxy_pct, tnx_pct, weather, threat_score, is_osint_only):
    if not groq_client: return "SYS_ERR: AI CORE OFFLINE."
    headlines = [art['Headline'] for art in articles[:5]] if articles else ["No news."]
    weather_txt = f"Temp: {weather['temp']}C, Rain: {weather['rain']}mm" if weather else "N/A"
    
    prompt = f"Act as a Quantitative Intelligence Analyst for Precious Metals. Target: {commodity}. Volatility Score: {threat_score}/100. "
    if is_osint_only:
        prompt += f"CRITICAL: NO FINANCIAL DATA EXISTS. Base summary on News: {headlines}. Write 2 tactical sentences. Tone: Cold, professional, financial intelligence."
    else:
        prompt += f"Price Change: {price_change:.2f}%. RSI: {rsi:.1f}. US Dollar Index Change: {dxy_pct:.2f}%. 10-Yr Yield Change: {tnx_pct:.2f}%. Mining Hub Weather: {weather_txt}. Headlines: {headlines}. Write 2 tactical sentences summarizing the investment risk/opportunity. Tone: Cold, professional, financial intelligence."
        
    try: return groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile").choices[0].message.content
    except: return "SYS_ERR: AI CORE UNRESPONSIVE."

def ai_auto_discover(food_name, existing_categories):
    if not groq_client: return None, "AI is offline."
    prompt = f"""Find global data for: "{food_name}". Return ONLY valid JSON.
    {{"category": "MUST be one of: {existing_categories}. Or invent a new one (UPPERCASE, NO EMOJIS).", "ticker": "Yahoo Finance futures ticker (e.g. GC=F). If it DOES NOT TRADE on futures, return 'NONE'", "search": "1 word search term for news", "unit": "Trading unit (e.g. Troy Oz). If NONE, put 'Kg'", "kg_per_unit": Float kg in unit. If NONE, put 1.0, "is_cents": true if US Cents, false if USD. If NONE, put false, "lat": Float latitude of primary mining region. If unknown, put 0.0, "lon": Float longitude. If unknown, put 0.0, "region": String name of region. If unknown, put "Unknown"}}"""
    try: return json.loads(groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile").choices[0].message.content.replace("```json", "").replace("```", "").strip()), "Success"
    except: return None, "Failed"

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; margin-bottom: 20px;">
    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
    </svg>
    <h3 style="color: #E0E6ED; margin: 10px 0 0 0; font-size: 1.2rem; letter-spacing: 2px;">QUANT.CORE</h3>
</div>
""", unsafe_allow_html=True)

if not COMMODITIES: st.stop()

search_mode = st.sidebar.radio("NAVIGATION MODE", ["BROWSE BY SECTOR", "SEARCH ALL TARGETS"])

if search_mode == "BROWSE BY SECTOR":
    selected_category = st.sidebar.selectbox("1. SELECT SECTOR", list(COMMODITIES.keys()))
    selected_commodity = st.sidebar.selectbox("2. SELECT TARGET", list(COMMODITIES[selected_category].keys()))
else:
    flat_foods = {f"{food} ({cat})": (cat, food) for cat, foods in COMMODITIES.items() for food in foods.keys()}
    search_selection = st.sidebar.selectbox("SEARCH TARGET", sorted(list(flat_foods.keys())))
    selected_category, selected_commodity = flat_foods[search_selection]

details = COMMODITIES[selected_category][selected_commodity]
is_osint_only = details.get("ticker") == "NONE"

try:
    center_lat = float(details.get("lat", 0.0))
    if pd.isna(center_lat): center_lat = 0.0
except: center_lat = 0.0
try:
    center_lon = float(details.get("lon", 0.0))
    if pd.isna(center_lon): center_lon = 0.0
except: center_lon = 0.0
center_lat = float(max(-89.9, min(89.9, center_lat)))
center_lon = float(max(-179.9, min(179.9, center_lon)))

st.sidebar.divider()
st.sidebar.markdown("### ⛙ MACRO ECONOMICS")
st.sidebar.metric("US DOLLAR INDEX (DXY)", f"{DXY_PRICE:.2f}", f"{DXY_PCT:.2f}%")
st.sidebar.metric("10-YR TREASURY YIELD", f"{TNX_PRICE:.2f}%", f"{TNX_PCT:.2f}%")

st.sidebar.divider()

st.sidebar.markdown("### ⌖ AUTO-DISCOVER")
st.sidebar.caption("> INITIATE AI TARGET ACQUISITION")
new_food_name = st.sidebar.text_input("ENTER ASSET NAME:")

if st.sidebar.button("DEPLOY TRACKER"):
    if new_food_name:
        with st.sidebar.status("ACQUIRING DATA..."):
            ai_data, status = ai_auto_discover(new_food_name, list(BASE_COMMODITIES.keys()))
            if ai_data:
                cat = ai_data["category"]
                if cat not in st.session_state.custom_foods: st.session_state.custom_foods[cat] = {}
                st.session_state.custom_foods[cat][new_food_name.title()] = {
                    "ticker": ai_data["ticker"], "search": ai_data["search"], "multiplier": 0.01 if ai_data.get("is_cents") else 1.0,
                    "unit": ai_data["unit"], "kg_per_unit": ai_data["kg_per_unit"], "lat": ai_data.get("lat", 0.0), "lon": ai_data.get("lon", 0.0), "region": ai_data.get("region", "Unknown")
                }
                st.rerun()

# --- MAIN DASHBOARD UI ---
st.title("❖ GLOBAL METALS & ASSET INTELLIGENCE")

# --- BIGGER, BLURRED CINEMATIC PYDECK MAP ---
active_data = []
global_data = []

for cat, foods in COMMODITIES.items():
    for name, d in foods.items():
        try:
            l_lat = float(d.get("lat", 0.0))
            l_lon = float(d.get("lon", 0.0))
            if not pd.isna(l_lat) and not pd.isna(l_lon) and (l_lat != 0.0 or l_lon != 0.0):
                if name == selected_commodity: active_data.append({"name": name, "lat": l_lat, "lon": l_lon})
                else: global_data.append({"name": name, "lat": l_lat, "lon": l_lon})
        except: pass

layers = []
if global_data:
    layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(global_data), get_position='[lon, lat]', get_color='[255, 215, 0, 100]', get_radius=200000, pickable=True))
if active_data:
    layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(active_data), get_position='[lon, lat]', get_color='[0, 229, 255, 255]', get_radius=300000, pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(active_data), get_position='[lon, lat]', get_color='[0, 229, 255, 60]', get_radius=1000000, pickable=False))

view_state = pdk.ViewState(latitude=center_lat if center_lat != 0.0 else 20.0, longitude=center_lon if center_lon != 0.0 else 0.0, zoom=2.5 if (center_lat != 0.0 or center_lon != 0.0) else 1, pitch=45, bearing=0)
st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=layers, tooltip={"text": "{name}"}), use_container_width=True)

# Fetch Data
price_usd, price_change, trend_ma, rsi, price_history = get_financial_data(details["ticker"], details.get("multiplier", 1.0))
avg_sentiment, news_articles = get_news_data(details["search"])
weather = get_weather_data(center_lat, center_lon)

price_myr = price_usd * USD_TO_MYR
kg_per_unit = details.get("kg_per_unit", 1.0)
price_per_kg_usd = price_usd / kg_per_unit if kg_per_unit > 0 else 0
price_per_kg_myr = price_myr / kg_per_unit if kg_per_unit > 0 else 0
std_unit = "L" if "gallon" in details.get("unit", "").lower() or "liter" in details.get("unit", "").lower() else "kg"

# --- LOCAL RETAIL BANK ESTIMATES (UOB & RHB) ---
price_per_gram_myr = price_myr / 31.1034768 if "Troy Oz" in details["unit"] else 0

threat_score, threat_level = calculate_master_threat(price_change, avg_sentiment, rsi, DXY_PCT, TNX_PCT, is_osint_only)

col_head1, col_head2 = st.columns([5, 1])
with col_head1: st.header(f"⌖ ASSET ACQUIRED: {selected_commodity}")
with col_head2:
    st.write("")
    if st.button("REMOVE ASSET", type="tertiary"):
        if selected_category in st.session_state.custom_foods and selected_commodity in st.session_state.custom_foods[selected_category]:
            del st.session_state.custom_foods[selected_category][selected_commodity]
        else: st.session_state.deleted_foods.append((selected_category, selected_commodity))
        st.rerun()

if "VOLATILITY 1" in threat_level: st.error(f"[EXTREME VOLATILITY] {selected_commodity} Score is {threat_score}/100. High risk of major price action.")
elif "VOLATILITY 2" in threat_level: st.warning(f"[ELEVATED VOLATILITY] {selected_commodity} Score is {threat_score}/100. Anomalies detected in macro inputs.")

if is_osint_only: st.warning("[OSINT-ONLY MODE] Tracking via Global News Sentiment only. Financial data unavailable.")

# Metrics Row
if not is_osint_only:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: 
        st.metric(label=f"GLOBAL SPOT ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%")
        st.caption(f"> ${price_per_kg_usd:.4f} / {std_unit}")
    with col2: 
        st.metric(label="GLOBAL SPOT (MYR)", value=f"RM {price_myr:.2f}")
        st.caption(f"> RM {price_per_kg_myr:.4f} / {std_unit}")
    with col3: st.metric(label="TECHNICAL RSI", value=f"{rsi:.1f}", delta="OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL", delta_color="off")
    with col4: st.metric(label="OSINT SENTIMENT", value=f"{avg_sentiment:.2f}", delta="BULLISH" if avg_sentiment > 0 else "BEARISH", delta_color="normal")
    with col5: st.metric(label="MARKET VOLATILITY", value=f"{threat_score}/100", delta=threat_level, delta_color="inverse" if "VOLATILITY 1" in threat_level else "off")
else:
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="OSINT SENTIMENT", value=f"{avg_sentiment:.2f}", delta="BULLISH" if avg_sentiment > 0 else "BEARISH", delta_color="normal")
    with col2: st.metric(label="MARKET VOLATILITY", value=f"{threat_score}/100", delta=threat_level, delta_color="inverse" if "VOLATILITY 1" in threat_level else "off")
    with col3: st.empty()

# --- MALAYSIAN RETAIL BANK MODULE ---
if selected_commodity == "Gold" and price_per_gram_myr > 0:
    st.info(f"🇲🇾 **LOCAL RETAIL ESTIMATE (UOB MALAYSIA):** Estimated Bank Selling Price: **RM {price_per_gram_myr * 1.025:.2f} / gram** *(Based on Spot + 2.5% Premium)*")
elif selected_commodity == "Silver" and price_per_gram_myr > 0:
    st.info(f"🇲🇾 **LOCAL RETAIL ESTIMATE (RHB MALAYSIA):** Estimated Bank Selling Price: **RM {price_per_gram_myr * 1.035:.2f} / gram** *(Based on Spot + 3.5% Premium)*")

if weather and center_lat != 0.0: st.info(f"[+] MINING HUB CLIMATE INTEL ({details.get('region', 'Unknown')}): Current Temp: {weather['temp']}°C | 7-Day Rainfall: {weather['rain']}mm")

# AI Brief
st.markdown("### ⎔ AI QUANTITATIVE ANALYSIS")
with st.spinner('DECRYPTING INTEL...'):
    ai_summary = get_ai_brief(selected_commodity, news_articles, price_change, rsi, DXY_PCT, TNX_PCT, weather, threat_score, is_osint_only)
    st.info(ai_summary)

# Charts & News
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
            context = f"Context for {selected_commodity}: Price Change: {price_change}%, RSI: {rsi}, Weather: {weather}, Volatility Score: {threat_score}. News: {[a['Headline'] for a in articles[:3]] if news_articles else 'None'}. User Question: {prompt}. Tone: Cold, professional, financial intelligence."
            try:
                response = groq_client.chat.completions.create(messages=[{"role": "user", "content": context}], model="llama-3.3-70b-versatile").choices[0].message.content
                st.markdown(response)
                st.session_state.chat_history[selected_commodity].append({"role": "assistant", "content": response})
            except:
                st.error("SYS_ERR: AI CORE UNRESPONSIVE.")
