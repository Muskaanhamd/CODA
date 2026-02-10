import streamlit as st
import pickle
import time
import os
import requests
import wikipedia
import re
import pandas as pd
from dotenv import load_dotenv

# --- 1. SETUP & CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY") 

# Expanded Global Trust Matrix
TRUSTED_DOMAINS = [
    "reuters.com", "apnews.com", "afp.com", "bbc.co.uk", "nytimes.com", 
    "wsj.com", "theguardian.com", "bloomberg.com", "aljazeera.com", 
    "scmp.com", "timesofindia.indiatimes.com", "ndtv.com", "snopes.com", 
    "politifact.com", "factcheck.org", "fullfact.org"
]

# --- 2. THE ENGINES ---

def save_user_feedback(input_text, coda_verdict, user_vote, sources_found):
    """Logs user feedback to a CSV file for the Admin Dashboard."""
    feedback_label = "Correct" if user_vote == 1 else "Incorrect"
    new_data = {
        "timestamp": [time.ctime()],
        "input_text": [input_text],
        "coda_verdict": [coda_verdict],
        "user_feedback": [feedback_label],
        "sources": [", ".join(sources_found) if sources_found else "None"]
    }
    df = pd.DataFrame(new_data)
    # Append to CSV; create with header if file doesn't exist
    df.to_csv("coda_feedback_log.csv", mode='a', index=False, header=not os.path.exists("coda_feedback_log.csv"))

def is_valid_news_claim(text):
    """Gatekeeper: Blocks personal chat and fluff."""
    words = text.strip().split()
    if len(words) < 6:
        return False, "Input too short for analysis."
    
    personal_triggers = {"i", "me", "my", "mine", "i'm", "am", "hello", "hi"}
    if any(p in [w.lower().replace("'", "") for w in words[:3]] for p in personal_triggers):
        return False, "CODA detected a personal statement. Please provide a news claim."

    entities = [w for w in words[1:] if w[0].isupper() and len(w) > 1]
    if not entities:
        return False, "No specific entities (names/places) detected. Try a more specific headline."

    return True, ""

def extract_precise_keywords(text):
    """Strips fluff to find the 'Subject' and 'Object' of the news."""
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    if len(entities) >= 2:
        return f'"{entities[0]} {entities[1]}"'
    return entities[0] if entities else text[:50]

def get_live_news_with_verdict(query):
    """Queries NewsAPI and calculates a consensus verdict."""
    domain_str = ",".join(TRUSTED_DOMAINS)
    url = f"https://newsapi.org/v2/everything?q={query}&domains={domain_str}&apiKey={NEWS_API_KEY}&pageSize=10"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            
            # Domain Consensus Logic
            found_domains = set()
            for art in articles:
                url_lower = art['url'].lower()
                for d in TRUSTED_DOMAINS:
                    if d in url_lower:
                        found_domains.add(d)
            
            count = len(found_domains)
            if count >= 3:
                verdict = ("Verified Fact", "Green", f"Confirmed by {count} major outlets.")
            elif count >= 1:
                verdict = ("Uncertain", "Orange", f"Reported by {count} source(s). Verify further.")
            else:
                verdict = ("High Risk", "Red", "No matches found in trusted news matrix.")
                
            return articles, (verdict, list(found_domains))
    except:
        pass
    return [], (("Error", "Grey", "API connection failed"), [])

def get_wiki_context(text):
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
        res = requests.get(url)
        if res.status_code == 200:
            return res.json().get("claims", [])
    except:
        return []
    return []

# --- 3. ML MODEL ---
@st.cache_resource
def load_coda_brain():
    try:
        model = pickle.load(open('coda_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_coda_brain()

# --- 4. UI ---
st.set_page_config(page_title="CODA | Intelligence Matrix", page_icon="🌀", layout="wide")
st.title("🌀 CODA: Intelligence Matrix")
st.markdown("---")

user_input = st.text_area("Verification Input:", placeholder="Paste headline here...", height=100)

if st.button("Run Deep Analysis"):
    if not user_input.strip():
        st.warning("Input required.")
    else:
        is_valid, error_msg = is_valid_news_claim(user_input)
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner("Analyzing multi-layer intelligence..."):
                refined_query = extract_precise_keywords(user_input)
                
                # Layer 1: ML
                transformed = vectorizer.transform([user_input])
                st.session_state.ml_res = (model.predict(transformed)[0], model.predict_proba(transformed)[0][1])
                
                # Layer 2: Live News & Consensus
                st.session_state.news_data = get_live_news_with_verdict(refined_query)
                
                # Layer 3: Context
                st.session_state.fact_results = get_fact_check_results(refined_query)
                st.session_state.wiki = get_wiki_context(user_input)
                st.session_state.analysis_done = True

# --- 5. DISPLAY RESULTS ---
if st.session_state.get('analysis_done'):
    col_ml, col_news, col_wiki = st.columns(3)

    with col_ml:
        st.subheader("🛡️ Linguistic Layer")
        pred, prob = st.session_state.ml_res
        if pred == 0: st.success("Verdict: Neutral")
        else: st.error("Verdict: Suspicious")
        st.metric("Manipulation Score", f"{prob*100:.1f}%")

    with col_news:
        st.subheader("📰 Consensus Layer")
        articles, (verdict, found_sources) = st.session_state.news_data
        status, color, detail = verdict
        
        if color == "Green": st.success(f"**{status}**")
        elif color == "Orange": st.warning(f"**{status}**")
        else: st.error(f"**{status}**")
        
        st.caption(detail)
        for art in articles[:2]:
            st.markdown(f"**{art['source']['name']}**: [{art['title'][:50]}...]({art['url']})")

    with col_wiki:
        st.subheader("📚 Context Layer")
        if st.session_state.wiki:
            st.info(f"**{st.session_state.wiki['title']}**")
            st.caption(st.session_state.wiki['summary'])
        else:
            st.write("No historical data found.")

    # --- FEEDBACK COLLECTOR ---
    st.markdown("---")
    st.subheader("📝 Help CODA Learn")
    st.write("Does this intelligence report seem accurate to you?")
    
    # options="thumbs" returns 0 for down, 1 for up
    user_sentiment = st.feedback("thumbs")
    
    if user_sentiment is not None:
        # Extract current status from the consensus layer
        current_status = st.session_state.news_data[1][0][0]
        found_sources = st.session_state.news_data[1][1]
        
        save_user_feedback(user_input, current_status, user_sentiment, found_sources)
        st.toast("Thank you! Feedback logged in the Intelligence Matrix.", icon="🧠")

    # --- EXPANDERS ---
    st.markdown("---")
    if st.session_state.fact_results:
        with st.expander("🔍 Fact-Check Database Hits"):
            for claim in st.session_state.fact_results[:2]:
                st.write(f"**Claim:** {claim['text']} \n**Verdict:** {claim['claimReview'][0]['textualRating']}")

    with st.expander("🛠️ Technical Logs"):
        st.write(f"Refined Query: `{extract_precise_keywords(user_input)}`")
        st.write(f"Trusted Domains Checked: {len(TRUSTED_DOMAINS)}")
        st.write(f"Identified Sources: {', '.join(st.session_state.news_data[1][1]) if st.session_state.news_data[1][1] else 'None'}")

st.markdown("---")
st.caption("CODA System v1.3 | Developed for Project PS-1.4")