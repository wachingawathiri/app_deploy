# app.py
import re
import pandas as pd  # (only needed if you later add CSV features)
import streamlit as st

# NLTK for preprocessing + lexicon sentiment
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ---- one-time downloads (safe to re-run) ----
def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)

for res in ["corpora/stopwords", "tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4", "sentiment/vader_lexicon"]:
    safe_nltk_download(res)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> str:
    """
    Lowercase, strip URLs/handles, keep hashtag words (drop '#'),
    remove punctuation/numbers, tokenize, remove stopwords/short tokens,
    and lemmatize.
    """
    if not isinstance(text, str):
        return ""

    t = text.lower()

    # remove urls, mentions
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)

    # keep hashtag text but drop '#'
    t = re.sub(r"#", "", t)

    # keep only letters and spaces
    t = re.sub(r"[^a-z\s]", " ", t)

    # tokenize
    tokens = word_tokenize(t)

    # remove stopwords and very short tokens
    tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]

    # lemmatize
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return " ".join(tokens).strip()

def label_from_compound(c: float, pos_th=0.05, neg_th=-0.05) -> str:
    if c >= pos_th:
        return "Positive"
    elif c <= neg_th:
        return "Negative"
    else:
        return "Neutral"

# --------------------- UI (single page, no sidebar/tabs) ---------------------
st.set_page_config(page_title="Sentiment on Apple & Google", layout="centered")

st.title("Sentiment Analyzer")
st.write("""
    This app analyzes the sentiment of tweets about **Apple** and **Google** products.  
    Enter a tweet below to see if it's **Positive**, **Negative**, or **Neutral**!
""")

user_text = st.text_area(
    "Enter a tweet about Apple or Google:",
    placeholder="e.g., The new Google Pixel camera is amazing!",
    height=120
)

analyze = st.button("Analyze", type="primary")

if analyze:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        clean = preprocess_text(user_text)
        # If cleaning empties everything, fall back to the raw text so VADER still scores
        scores = vader.polarity_scores(clean if clean else user_text)
        label = label_from_compound(scores["compound"])

        # Color-coded result
        if label == "Positive":
            st.success(f"Sentiment: **{label}**")
        elif label == "Negative":
            st.error(f"Sentiment: **{label}**")
        else:
            st.info(f"Sentiment: **{label}**")

       

st.markdown("---")
