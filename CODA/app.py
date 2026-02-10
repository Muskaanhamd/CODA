import streamlit as st
import pickle
import time
import os
import requests
import wikipedia
import re
from dotenv import load_dotenv

# --- 1. SETUP & CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY") 

# --- 2. THE ENGINES ---

def is_valid_news_claim(text):
    """
    Gatekeeper: Blocks personal chat, names, and short non-news sentences.
    Returns (is_valid, error_message)
    """
    words = text.strip().split()
    
    # Check 1: Length (News claims need context, usually > 5 words)
    if len(words) < 6:
        return False, "Input is too short. Please provide a full news headline or claim."

    # Check 2: Personal Pronouns (Reject 'I am', 'My name', etc.)
    personal_triggers = {"i", "me", "my", "mine", "i'm", "am", "hello", "hi"}
    first_few_words = [w.lower().replace("'", "") for w in words[:3]]
    if any(p in first_few_words for p in personal_triggers):
        return False, "CODA is for news verification, not personal statements or chat."

    # Check 3: Entity Density (News must mention people, places, or orgs)
    # We look for capitalized words that aren't the very first word.
    entities = [w for w in words[1:] if w[0].isupper() and len(w) > 1]
    if len(entities) < 1:
        return False, "This looks like a general statement. News claims usually involve specific names or entities."

    return True, ""

def extract_precise_keywords(text):
    """Strips fluff to find the 'Subject' and 'Object' of the news."""
    fluff = {"denies", "links", "official", "statement", "report", "claims", "says", "verified", "news"}
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    filtered = [e for e in entities if e.lower() not in fluff]
    
    if len(filtered) >= 2:
        return f'"{filtered[0]} {filtered[1]}"'
    return filtered[0] if filtered else text[:50]

def get_live_news(query):
    """Queries NewsAPI for real-time reporting from trusted domains."""
    trusted_sources = "reuters.com,apnews.com,bbc.co.uk,nytimes.com,wsj.com,theguardian.com"
    url = f"https://newsapi.org/v2/everything?q={query}&domains={trusted_sources}&apiKey={NEWS_API_KEY}&pageSize=3"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
    except:
        return []
    return []

def get_wiki_context(text):
    """Uses Wikipedia for background info on the primary entity."""
    try:
        subject = re.findall(r'\b[A-Z][a-z]+\b', text)
        if subject:
            search_results = wikipedia.search(subject[0])
            if search_results:
                return {"title": search_results[0], "summary": wikipedia.summary(search_results[0], sentences=1)}
    except:
        return None
    return None

def get_fact_check_results(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("claims", [])
    except:
        return []
    return []

# --- 3. ML MODEL LOADING ---
@st.cache_resource
def load_coda_brain():
    try:
        model = pickle.load(open('coda_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_coda_brain()

# --- 4. UI SETUP ---
st.set_page_config(page_title="CODA | Intelligence Matrix", page_icon="🌀", layout="wide")
st.title("🛡️ CODA: Project Intelligence Matrix")
st.markdown("---")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

user_input = st.text_area("Input News Content for Verification:", placeholder="Paste news headline or article snippet here...", height=150)

# --- 5. EXECUTION LOGIC ---
if st.button("🚀 Run Deep Analysis"):
    if not user_input.strip():
        st.warning("Please enter text first.")
    else:
        # --- GATEKEEPER CHECK ---
        is_valid, error_msg = is_valid_news_claim(user_input)
        
        if not is_valid:
            st.error(f"🚫 {error_msg}")
            st.session_state.analysis_done = False
        else:
            with st.spinner("CODA is cross-referencing multi-layer intelligence..."):
                refined_query = extract_precise_keywords(user_input)
                
                # Linguistic Analysis
                transformed_input = vectorizer.transform([user_input])
                st.session_state.prediction = model.predict(transformed_input)[0]
                st.session_state.prob = model.predict_proba(transformed_input)[0][1]
                
                # Real-Time Verification
                st.session_state.news = get_live_news(refined_query)
                st.session_state.fact_results = get_fact_check_results(refined_query)
                st.session_state.wiki = get_wiki_context(user_input)
                st.session_state.analysis_done = True

# --- 6. DISPLAY RESULTS ---
if st.session_state.analysis_done:
    st.markdown("### 📊 CODA Intelligence Report")
    col_ml, col_news, col_wiki = st.columns(3)

    with col_ml:
        st.subheader("🧠 Linguistic Layer")
        status = "Suspicious" if st.session_state.prediction == 1 else "Neutral"
        if status == "Neutral": st.success(f"Verdict: {status}")
        else: st.error(f"Verdict: {status}")
        st.metric("Manipulation Score", f"{st.session_state.prob*100:.1f}%")

    with col_news:
        st.subheader("📰 Live News Layer")
        if st.session_state.news:
            for art in st.session_state.news:
                st.write(f"✅ **{art['source']['name']}**: {art['title'][:70]}...")
                st.caption(f"[Read Article]({art['url']})")
        else:
            st.warning("No coverage found in trusted news outlets. High risk of fabricated claim.")

    with col_wiki:
        st.subheader("📚 Knowledge Graph")
        if st.session_state.wiki:
            st.info(f"Subject: {st.session_state.wiki['title']}")
            st.caption(st.session_state.wiki['summary'])
        else:
            st.write("No matching background context found.")

    st.markdown("---")
    if st.session_state.fact_results:
        with st.expander("🔍 Specific Fact-Check Database Matches"):
            for claim in st.session_state.fact_results[:2]:
                st.write(f"**Claim:** {claim['text']}")
                st.write(f"**Verdict:** {claim['claimReview'][0]['textualRating']}")

    with st.expander("🛠️ Technical System Logs"):
        st.write(f"System State: 🟢 Active")
        st.write(f"Refined Search Query: {extract_precise_keywords(user_input)}")

st.markdown("---")
st.caption("CODA System v1.2 | Developed for Project PS-1.4")