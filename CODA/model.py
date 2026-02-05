import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def train_model():
    # 1. Load the LIAR-2 dataset from Hugging Face
    print("Loading LIAR-2 dataset...")
    dataset = load_dataset("chengxuphd/liar2")
    
    # 2. Convert to a format Python can read (Pandas)
    train_df = pd.DataFrame(dataset['train'])
    
    # 3. Clean the labels: LIAR has 6 labels. 
    # For now, let's make it simple: 0 = True-ish, 1 = False-ish
    # (True, Mostly-True, Half-True -> 0 | False, Barely-True, Pants-on-fire -> 1)
    def binary_label(label):
        if label in [0, 1, 2]: # Adjust based on dataset mapping
            return 0 # Reliable
        return 1 # Unreliable
    
    train_df['label'] = train_df['label'].apply(binary_label)

    # 4. Turn Text into Numbers (Vectorization)
    # We use TF-IDF: it looks for 'unique' words in fake news
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(train_df['statement'])
    y = train_df['label']

    # 5. The Brain (Logistic Regression)
    # It's fast, explainable, and perfect for a 2-day hackathon
    model = LogisticRegression()
    model.fit(X, y)
    
    # 6. Save the Brain and the Vectorizer to files
    # This way we don't have to 're-train' every time we run the website
    with open('coda_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("CODA Linguistic Brain Trained and Saved!")

if __name__ == "__main__":
    train_model()