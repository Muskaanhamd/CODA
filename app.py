import streamlit as st
import pickle
import re
import spacy

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()



# -------------------- Load Model --------------------
with open("coda_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("coda_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------- Content Filter --------------------
REPORTING_VERBS = {
    "said", "reported", "announced", "claimed", "stated",
    "according", "revealed", "confirmed", "denied",
    "causes", "affects", "leads", "results"
}

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}

def is_article_like(text):
    text = text.strip()

    if len(text.split()) < 50:
        return False, "Input is too short to be an article"

    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    if len(sentences) < 3:
        return False, "Input does not contain enough sentences"

    doc = nlp(text)

    pronouns = [t.text.lower() for t in doc if t.pos_ == "PRON"]
    if pronouns:
        first_person_ratio = sum(p in FIRST_PERSON for p in pronouns) / len(pronouns)
        if first_person_ratio > 0.2:
            return False, "Input appears to be a personal statement"

    if len(doc.ents) == 0:
        return False, "No named entities found; not article-like"

    verbs = {t.lemma_.lower() for t in doc if t.pos_ == "VERB"}
    if not verbs.intersection(REPORTING_VERBS):
        return False, "No informational or reporting verbs found"

    return True, "Accepted"

# -------------------- Streamlit UI --------------------
st.title("CODA – Contextual Auditor")

user_input = st.text_area("Paste an article, news, or informational content here")

if st.button("Analyze"):
    eligible, reason = is_article_like(user_input)

    if not eligible:
        st.error(reason)
        st.stop()

    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max()

    st.subheader("Result")
    st.write("Prediction:", "Fake News" if prediction == 1 else "Not Flagged as Fake")
    st.write("Confidence:", round(confidence, 2))



