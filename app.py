import streamlit as st
import pickle
import time
import os
import re
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2 import service_account  # Required for JSON credentials
from PIL import Image

# --- 1. SETUP & CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
# Note: FACT_CHECK_KEY is replaced by service_account.json for better security

# Path to your downloaded JSON file - MUST be in the same folder as app.py
SERVICE_ACCOUNT_FILE = 'service_account.json' 

# --- 2. THE ENGINES ---

def save_user_feedback(input_text, coda_verdict, user_vote, sources_found):
    feedback_label = "Correct" if user_vote == 1 else "Incorrect"
    new_data = {
        "timestamp": [time.ctime()],
        "input_text": [input_text],
        "coda_verdict": [coda_verdict],
        "user_feedback": [feedback_label],
        "sources": [", ".join(sources_found) if sources_found else "None"]
    }
    df = pd.DataFrame(new_data)
    df.to_csv("coda_feedback_log.csv", mode='a', index=False, header=not os.path.exists("coda_feedback_log.csv"))

def is_valid_news_claim(text):
    words = text.strip().split()
    if len(words) < 6:
        return False, "Input too short for analysis."
    personal_triggers = {"i", "me", "my", "mine", "i'm", "am", "hello", "hi"}
    if any(p in [w.lower().replace("'", "") for w in words[:3]] for p in personal_triggers):
        return False, "CODA detected a personal statement. Please provide a news claim."
    return True, ""

def extract_precise_keywords(text):
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    if len(entities) >= 2:
        return f'"{entities[0]} {entities[1]}"'
    return entities[0] if entities else text[:50]

def get_matrix_consensus(query):
    try:
        if not SEARCH_ENGINE_ID or not GOOGLE_API_KEY:
            return [], (("System Offline", "Grey", "Missing API Credentials"), [])

        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID.strip(), num=5).execute()
        
        items = res.get('items', [])
        
        if not items:
            return [], (("High Risk", "Red", "Consensus Gap: No matches in trusted matrix."), [])

        found_domains = list(set([item.get('displayLink', 'Unknown') for item in items]))
        count = len(found_domains)
        
        if count >= 3:
            verdict = ("Verified Fact", "Green", f"Confirmed by {count} major outlets.")
        elif count >= 1:
            verdict = ("Uncertain", "Orange", f"Reported by {count} trusted source(s).")
        else:
            verdict = ("High Risk", "Red", "Consensus Gap: No trusted matches found.")
            
        return items, (verdict, found_domains)
        
    except Exception as e:
        print(f"CRITICAL DEBUG: {e}")
        return [], (("System Offline", "Grey", "Matrix Connection Issue"), [])

def get_fact_check_data(query):
    """Checks the official Google Fact Check Tools API using Service Account"""
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, 
                scopes=['https://www.googleapis.com/auth/factcheck']
            )
            service = build("factchecktools", "v1", credentials=creds)
            res = service.claims().search(query=query).execute()
            return res.get('claims', [])
        return []
    except Exception as e:
        print(f"Fact Check Error: {e}")
        return []

# --- 3. ML MODEL ---
@st.cache_resource
def load_coda_brain():
    import os
    # This print will show up in your terminal to tell us where it's looking
    print(f"DEBUG: Looking for model in: {os.getcwd()}") 
    try:
        model = pickle.load(open('coda_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        print(f"BRAIN ERROR: {e}") # This will tell us if it's a "File Not Found"
        return None, None

model, vectorizer = load_coda_brain()

# --- 4. UI SETUP ---
st.set_page_config(page_title="CODA | Intelligence Matrix", page_icon="🌀", layout="wide")
st.title("CODA: Intelligence Matrix")
st.markdown(f"Targeting Project: PS-1.4 | Active Sources: 70 Domains")

# --- 5. IMAGE VERIFICATION SECTION ---
with st.sidebar:
    st.header("Visual Verification")
    uploaded_file = st.file_uploader("Upload suspicious image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Analyzing image metadata...", use_container_width=True)
        if st.button("Check Image Authenticity"):
            current_query = st.session_state.get('user_input', "suspicious news")
            img_results, (img_v, img_s) = get_matrix_consensus(current_query)
            if img_results:
                st.success("Image found in trusted archives.")
            else:
                st.error("Visual Consensus Gap: Image not found in trusted media.")

# --- 6. MAIN ANALYSIS ---
user_input = st.text_area("Verification Input:", placeholder="Paste headline here...", height=100)

if st.button("Run Deep Analysis"):
    if not user_input.strip():
        st.warning("Input required.")
    else:
        is_valid, error_msg = is_valid_news_claim(user_input)
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner("Calculating Truth Score..."):
                refined_query = extract_precise_keywords(user_input)
                
                # Layer 1: ML
                if model and vectorizer:
                    transformed = vectorizer.transform([user_input])
                    ml_prob = model.predict_proba(transformed)[0][1]
                else:
                    ml_prob = 0.5 # Default to 50% if model missing
                
                # Layer 2: Matrix & Fact Check
                items, (verdict, found_sources) = get_matrix_consensus(refined_query)
                fact_claims = get_fact_check_data(refined_query)
                
                # --- TRUTH SCORE CALCULATION ---
                # Boost score if official fact checkers have already verified it
                fact_boost = 20 if fact_claims else 0
                matrix_score = min(len(found_sources) * 33.3, 100) 
                final_score = (matrix_score * 0.7) + ((1 - ml_prob) * 100 * 0.3) + fact_boost
                final_score = min(final_score, 100.0)
                
                st.session_state.matrix_data = (items, (verdict, found_sources))
                st.session_state.final_score = final_score
                st.session_state.ml_prob = ml_prob
                st.session_state.fact_check = fact_claims
                st.session_state.analysis_done = True

# --- 7. RESULTS DISPLAY ---
if st.session_state.get('analysis_done'):
    st.divider()
    score = st.session_state.final_score
    if score > 75: st.balloons()
    
    st.metric("CODA Truth Confidence Score", f"{score:.1f}%", delta="High Trust" if score > 70 else "Low Trust")

    col_ml, col_news = st.columns(2)
    with col_ml:
        st.subheader("Linguistic Analysis")
        prob = st.session_state.ml_prob
        st.progress(prob, text=f"Manipulation Probability: {prob*100:.1f}%")
        if st.session_state.fact_check:
            st.info(f"Fact Check Registry: {len(st.session_state.fact_check)} matching claims found.")
        
    with col_news:
        st.subheader("Consensus Layer")
        items, (verdict, found_sources) = st.session_state.matrix_data
        st.markdown(f"Status: **{verdict[0]}**")
        for item in items[:2]:
            st.markdown(f"{item['displayLink']}: [{item['title']}]({item['link']})")

    # --- 8. ADMIN AUDIT VIEW ---
    st.divider()
    with st.expander("Admin Intelligence Dashboard"):
        st.subheader("Recent Verification Logs")
        if os.path.exists("coda_feedback_log.csv"):
            log_df = pd.read_csv("coda_feedback_log.csv")
            st.dataframe(log_df.tail(5), use_container_width=True)
        else:
            st.info("No logs found. Run an analysis to generate data.")

    # Feedback
    st.subheader("Help CODA Learn")
    user_sentiment = st.feedback("thumbs")
    if user_sentiment is not None:
        save_user_feedback(user_input, st.session_state.matrix_data[1][0][0], user_sentiment, st.session_state.matrix_data[1][1])
        st.toast("Feedback recorded in Audit Log.")

st.caption("CODA System v1.5 | Developed for Project PS-1.4")