# app.py
import streamlit as st
import joblib
import re

import nltk
nltk.download('stopwords', quiet=True)

# ----------------------------
# Helper: Text Preprocessing
# ----------------------------
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model_multiclass.pkl")
    return model

model = load_model()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)
    pred = model.predict([cleaned])[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(pred, "Unknown")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Apple & Google Tweet Sentiment Analyzer", page_icon="🔍")
st.title("Twitter Sentiment Analyzer")
st.markdown("""
    This app analyzes the sentiment of tweets about **Apple** and **Google** products.  
    Enter a tweet below to see if it's **Positive**, **Negative**, or **Neutral**!
""")

# Input box
user_input = st.text_area(
    "Enter a tweet about Apple or Google:",
    placeholder="e.g., I love my new iPhone 15!",
    height=100
)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        with st.spinner("Analyzing..."):
            sentiment = predict_sentiment(user_input)
        
        # Display result with color
        if sentiment == "Positive":
            st.success(f"Sentiment: **{sentiment}**")
        elif sentiment == "Negative":
            st.error(f"Sentiment: **{sentiment}**")
        else:
            st.info(f"Sentiment: **{sentiment}**")

        

# Footer
st.markdown("---")
#st.caption("Powered by Scikit-learn + TF-IDF • Trained on CrowdFlower Twitter data")