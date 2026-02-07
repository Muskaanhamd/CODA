import streamlit as st
import pickle
import time
import os
import requests
import spacy
import wikipedia
import subprocess
from dotenv import load_dotenv

# --- 1. BOOTLOADER: Ensure spaCy is ready ---
@st.cache_resource
def load_nlp_resources():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

nlp = load_nlp_resources()
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# --- 2. FACT-CHECKING ENGINE (Multi-Source) ---
def get_fact_check_results(query):
    clean_query = query.split('.')[0][:100].strip() 
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={clean_query}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            all_claims = response.json().get("claims", [])
            query_words = set(clean_query.lower().split())
            return [c for c in all_claims if len(query_words.intersection(set(c.get('text', '').lower().split()))) >= 2]
    except Exception:
        return []
    return []

def get_wiki_verification(query):
    try:
        search_results = wikipedia.search(query)
        if search_results:
            return {"title": search_results[0], "summary": wikipedia.summary(search_results[0], sentences=1)}
    except:
        return None

# --- 3. ML MODEL LOADING ---
@st.cache_resource
def load_coda_brain():
    model = pickle.load(open('coda_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_coda_brain()

# --- 4. UI SETUP ---
st.set_page_config(page_title="CODA | Misinformation Intelligence", page_icon="➰", layout="wide")
st.title(" CODA: Project Intelligence Matrix")
st.markdown("---")

user_input = st.text_area("Input Content for Verification:", placeholder="Paste text here...", height=150)

if st.button(" Run Deep Analysis"):
    if not user_input.strip():
        st.warning("Please enter text first.")
    else:
        # A. CLAIM SPOTTING (spaCy)
        doc = nlp(user_input)
        is_claim = len(doc.ents) > 0 and any(t.pos_ == "VERB" for t in doc)
        
        # B. LINGUISTIC ANALYSIS
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        prob = model.predict_proba(transformed_input)[0][1]

        # DISPLAY RESULTS IN COLUMNS
        st.markdown("###  CODA Analysis Report")
        col_ml, col_check, col_wiki = st.columns(3)

        with col_ml:
            st.write("**Linguistic Layer**")
            if prediction == 0:
                st.success("Verdict: Neutral")
            else:
                st.error("Verdict: Suspicious")
            st.metric("Manipulation Score", f"{prob*100:.1f}%")

        with col_check:
            st.write("**Fact-Check Layer**")
            fact_results = get_fact_check_results(user_input)
            if fact_results:
                st.warning(f"Found {len(fact_results)} Debunks")
                st.caption(f"Rating: {fact_results[0]['claimReview'][0]['textualRating']}")
            else:
                st.success("No active debunks found.")

        with col_wiki:
            st.write("**Knowledge Graph (Wiki)**")
            wiki = get_wiki_verification(user_input)
            if wiki:
                st.info(f"Context: {wiki['title']}")
                st.caption(wiki['summary'])
            else:
                st.write("No matching entries.")

        # ... (rest of your analysis logic above)

        # FINAL EXPLAINABILITY
        st.markdown("---")
        if is_claim:
            st.write(" **CODA Insight:** This statement contains specific entities and actions, making it a high-priority factual claim.")
        else:
            st.write("ℹ **CODA Insight:** This text appears to be an opinion or subjective statement.")

        # MOVED INSIDE THE BUTTON BLOCK:
        with st.expander(" Technical System Logs"):
            st.write(f"NLP Engine: spaCy {spacy.__version__}")
            st.write(f"Model State: Brain Loaded (coda_model.pkl)")
            st.write(f"Sources Queried: Google API, Wikipedia, Local Linguistic Model")
            st.write(f"Current Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.caption("CODA System v1.0 | Project for PS-1.4")