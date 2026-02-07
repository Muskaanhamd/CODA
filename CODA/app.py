import streamlit as st
import pickle
import time
import os
import requests
import wikipedia
from dotenv import load_dotenv

# --- 1. SETUP & CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# --- 2. FACT-CHECKING ENGINE (Native Python) ---
def get_fact_check_results(query):
    # Search the first sentence to reduce noise
    clean_query = query.split('.')[0][:100].strip() 
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={clean_query}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            all_claims = response.json().get("claims", [])
            query_words = set(clean_query.lower().split())
            # Relevance Filter: Match at least 2 words
            return [c for c in all_claims if len(query_words.intersection(set(c.get('text', '').lower().split()))) >= 2]
    except:
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
st.set_page_config(page_title="CODA | Intelligence Matrix", page_icon="➰", layout="wide")
st.title("CODA: Intelligence Matrix")
st.markdown("---")

# Persistent state for results
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

user_input = st.text_area("Input Content for Verification:", placeholder="Paste text here...", height=150)

# --- 5. EXECUTION LOGIC ---
if st.button("Run Deep Analysis"):
    if not user_input.strip():
        st.warning("Please enter text first.")
    else:
        with st.spinner("CODA is cross-referencing multi-layer intelligence..."):
            # A. Native Claim Detection (Replaces spaCy)
            # We look for keywords like capitalized names or numbers to signify a 'claim'
            words = user_input.split()
            has_entities = any(w[0].isupper() for w in words if len(w) > 1)
            st.session_state.is_claim = has_entities and len(words) > 3
            
            # B. Linguistic Analysis (ML Model)
            transformed_input = vectorizer.transform([user_input])
            st.session_state.prediction = model.predict(transformed_input)[0]
            st.session_state.prob = model.predict_proba(transformed_input)[0][1]
            
            # C. Verification (APIs)
            st.session_state.fact_results = get_fact_check_results(user_input)
            st.session_state.wiki = get_wiki_verification(user_input)
            
            st.session_state.analysis_done = True

# --- 6. DISPLAY RESULTS ---
if st.session_state.analysis_done:
    st.markdown("CODA Analysis Report")
    col_ml, col_check, col_wiki = st.columns(3)

    with col_ml:
        st.write("Linguistic Layer")
        if st.session_state.prediction == 0:
            st.success("Verdict: Neutral")
        else:
            st.error("Verdict: Suspicious")
        st.metric("Manipulation Score", f"{st.session_state.prob*100:.1f}%")

    with col_check:
        st.write("**Fact-Check Layer**")
        if st.session_state.fact_results:
            st.warning(f"Found {len(st.session_state.fact_results)} Debunks")
            st.caption(f"Rating: {st.session_state.fact_results[0]['claimReview'][0]['textualRating']}")
        else:
            st.success("No active debunks found.")

    with col_wiki:
        st.write("**Knowledge Graph (Wiki)**")
        if st.session_state.wiki:
            st.info(f"Context: {st.session_state.wiki['title']}")
            st.caption(st.session_state.wiki['summary'])
        else:
            st.write("No matching entries found.")

    st.markdown("---")
    if st.session_state.is_claim:
        st.info(" CODA Insight: This statement contains specific entities, suggesting a high-priority factual claim.")
    else:
        st.write("CODA Insight: This text appears to be more subjective or conversational.")

    with st.expander("Technical System Logs"):
        st.write(f"Model State: Brain Loaded (coda_model.pkl)")
        st.write(f"Sources Queried: Google API, Wikipedia, Random Forest Model")
        st.write(f"Status:  System Operational")

st.markdown("---")
st.caption("CODA System v1.0 | Project for PS-1.4")