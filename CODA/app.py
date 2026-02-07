import streamlit as st
import pickle
import time
import os
import requests
from dotenv import load_dotenv

# --- NEW: 1. API Key Setup ---
# This loads the variable from your .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# --- NEW: 2. Fact Check Logic ---
def get_fact_check_results(query):
    # TWEAK 1: Only take the first sentence or the first 50 characters. 
    # Long sentences confuse the Fact-Check API.
    clean_query = query.split('.')[0][:50].strip() 
    
    # TWEAK 2: We use the 'query' parameter specifically.
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={clean_query}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # We filter: Only show results if the claim actually mentions our keywords
            all_claims = data.get("claims", [])
            return all_claims
    except Exception as e:
        st.error(f"Fact-Check Error: {e}")
    return []

# --- 3. Existing Load "Brain" Section ---
@st.cache_resource
def load_coda_brain():
    model = pickle.load(open('coda_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_coda_brain()

# --- 4. Page Setup ---
st.set_page_config(page_title="CODA | Misinformation Intelligence", page_icon="🛡️")
st.title("🛡️ CODA")
st.subheader("Cross-Platform Misinformation Intelligence System")
st.markdown("---")

# --- 5. User Input ---
user_input = st.text_area("Paste the news article or social media post below:", 
                         placeholder="e.g., NASA confirms Earth will stop rotating...",
                         height=200)

if st.button("Analyze Content"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Step A: Linguistic Analysis
        with st.spinner("CODA is analyzing linguistic patterns..."):
            time.sleep(1)
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)
            probability = model.predict_proba(transformed_input)[0][1]

            st.markdown("### 📊 Intelligence Report")
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 0:
                    st.success("Verdict: Linguistically Neutral")
                else:
                    st.error("Verdict: Linguistically Suspicious")
            with col2:
                st.metric("Manipulation Score", f"{probability*100:.1f}%")

        # --- NEW: Step B: Fact Verification Layer ---
        st.markdown("---")
        st.subheader("🔍 Fact Verification Layer")
        with st.spinner("Searching trusted global fact-check databases..."):
            # We search for the first 100 characters to find specific claims
            fact_results = get_fact_check_results(user_input[:100])

            if fact_results:
                st.warning("Related Fact-Checks Found:")
                for claim in fact_results[:2]: # Show only the top 2 matches
                    claim_text = claim.get('text', 'No claim text')
                    rating = claim['claimReview'][0].get('textualRating', 'Unknown')
                    publisher = claim['claimReview'][0]['publisher'].get('name', 'Source')
                    
                    st.write(f"**Claim:** {claim_text}")
                    st.write(f"**Verdict:** {rating} (Source: {publisher})")
            else:
                st.success("No previous fact-checks found for this claim.")

        # --- Explainability Section ---
        st.info("**Why this result?**")
        if prediction[0] == 1:
            st.write("The model detected patterns common in sensationalist reporting.")
        else:
            st.write("The language follows standard informative reporting patterns.")

st.markdown("---")
st.caption("CODA System v1.0 | Project for PS-1.4")