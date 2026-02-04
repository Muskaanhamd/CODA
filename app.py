import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("coda_model.pkl", "rb"))
vectorizer = pickle.load(open("coda_vectorizer.pkl", "rb"))

# App title
st.set_page_config(page_title="CODA Fake News Detector")
st.title("📰 CODA – Fake News Detection App")
st.write("Enter a news article below to check whether it is **Real** or **Fake**.")

# Text input
user_input = st.text_area("Paste the news text here:")

# Prediction
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.success("✅ This news looks REAL")
        else:
            st.error("🚨 This news looks FAKE")
