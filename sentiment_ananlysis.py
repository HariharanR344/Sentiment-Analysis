import pickle
import re
import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer


# LOAD ALL ARTIFACTS

@st.cache_resource
def load_artifacts():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("stopwords.pkl", "rb") as f:
        stop_words = pickle.load(f)

    with open("dashboard_data.pkl", "rb") as f:
        df = pickle.load(f)

    return tfidf, model, stop_words, df


tfidf, model, stop_words, df = load_artifacts()
lemmatizer = WordNetLemmatizer()


# CLEAN TEXT

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Predictor", "Analysis Dashboard"])


# PAGE 1: SENTIMENT PREDICTOR

if page == "Sentiment Predictor":

    st.title("🧠 ChatGPT Review Sentiment Predictor")

    review = st.text_area("Enter your review")

    if st.button("Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review")
        else:
            cleaned = clean_text(review)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.success(f"Predicted Sentiment: **{prediction}**")

# PAGE 2: ANALYSIS SUMMARY 
else:
    st.title("📊 Sentiment Analysis Summary")

    total_reviews = len(df)

    # 1️ Overall Sentiment
    st.header("1. Overall Sentiment of User Reviews")
    sentiment_dist = df["sentiment"].value_counts(normalize=True) * 100
    st.write(sentiment_dist.round(2).astype(str) + " %")

    st.info(
        f"Out of {total_reviews} reviews, "
        f"{sentiment_dist.idxmax()} sentiment is dominant."
    )

    # 2️ Sentiment vs Rating
    st.header("2. Sentiment vs Rating Mismatch")
    mismatch = df[(df["rating"] <= 2) & (df["sentiment"] != "Negative")]

    st.write(
        f"Most 1-2 star ratings are negative, "
        f"but **{len(mismatch)} reviews** show a mismatch."
    )
    st.dataframe(mismatch[["rating", "review", "sentiment"]].head())

    # 3️ Keywords by Sentiment
    st.header("3. Keywords Associated with Each Sentiment")
    for s in df["sentiment"].unique():
        words = " ".join(df[df["sentiment"] == s]["clean_review"]).split()
        top_words = pd.Series(words).value_counts().head(10)
        st.subheader(f"{s} Reviews – Top Keywords")
        st.write(top_words)


    # 5️ Verified Purchase
    st.header("5. Verified Users vs Sentiment")
    verified_summary = df.groupby(["verified_purchase", "sentiment"]).size()
    st.write(verified_summary)

    st.info(
        "Verified users tend to leave more structured and balanced reviews "
        "compared to non-verified users."
    )

    # 6️ Review Length vs Sentiment
    st.header("6. Review Length vs Sentiment")
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))
    length_summary = df.groupby("sentiment")["review_length"].mean()

    st.write(length_summary.round(2))

    st.info(
        "Negative reviews are generally longer, "
        "indicating users explain problems in more detail."
    )

    # 7️ Location-Based Sentiment
    st.header("7. Location-Based Sentiment")
    location_summary = df.groupby(["location", "sentiment"]).size().unstack().fillna(0)
    st.dataframe(location_summary.head())

    # 8️ Platform-Based Sentiment
    st.header("8. Platform Comparison (Web vs Mobile)")
    platform_summary = df.groupby(["platform", "sentiment"]).size()
    st.write(platform_summary)

    st.info(
        "Platform-wise analysis helps identify user experience issues "
        "specific to Web or Mobile platforms."
    )

    # 9️ ChatGPT Version Impact
    st.header("9. ChatGPT Version vs Sentiment")
    version_summary = df.groupby(["version", "sentiment"]).size()
    st.write(version_summary)

    st.info(
        "Certain versions show increased negative sentiment, "
        "indicating possible regression after releases."
    )

    # 10 Common Negative Themes
    st.header("10. Common Negative Feedback Themes")
    negative_words = " ".join(
        df[df["sentiment"] == "Negative"]["clean_review"]
    ).split()

    top_negative_words = pd.Series(negative_words).value_counts().head(15)
    st.write(top_negative_words)

    st.error(
        "Frequent negative themes include bugs, slow performance, "
        "poor responses, and reliability issues."
    )
