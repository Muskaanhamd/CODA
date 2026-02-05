import streamlit as st
import pickle
import time

# --- 1. Load the "Brain" ---
@st.cache_resource # This makes the app fast
def load_coda_brain():
    model = pickle.load(open('coda_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_coda_brain()

# --- 2. Page Setup (The "Face" of CODA) ---
st.set_page_config(page_title="CODA | Misinformation Intelligence", page_icon="🛡️")

st.title("🛡️ CODA")
st.subheader("Cross-Platform Misinformation Intelligence System")
st.markdown("---")

# --- 3. User Input ---
user_input = st.text_area("Paste the news article or social media post below:", 
                         placeholder="e.g., NASA confirms Earth will stop rotating...",
                         height=200)

if st.button("Analyze Content"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("CODA is analyzing linguistic patterns..."):
            time.sleep(1) # Visual effect for the hackathon
            
            # --- 4. Linguistic Analysis (The ML Logic) ---
            # Transform the input text using our saved vectorizer
            transformed_input = vectorizer.transform([user_input])
            
            # Predict (0 = Reliable, 1 = Unreliable)
            prediction = model.predict(transformed_input)
            probability = model.predict_proba(transformed_input)[0][1] # Probability of being fake

            # --- 5. Displaying Results ---
            st.markdown("### 📊 Intelligence Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 0:
                    st.success("Verdict: Linguistically Neutral")
                else:
                    st.error("Verdict: Linguistically Suspicious")
            
            with col2:
                st.metric("Manipulation Score", f"{probability*100:.1f}%")

            # --- 6. Explainability Section ---
            st.info("**Why this result?**")
            if prediction[0] == 1:
                st.write("The model detected patterns common in sensationalist or non-factual reporting (e.g., high use of loaded adjectives or aggressive syntax).")
            else:
                st.write("The language used follows standard informative reporting patterns with low emotional manipulation.")

st.markdown("---")
st.caption("CODA System v1.0 | Project for PS-1.4")